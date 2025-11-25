import argparse
import copy
import json
import os
import time

import h5py
import torch.backends.cudnn as cudnn
import torch.optim
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

from datasets import CaptionDataset, FederatedPMDDataset, CaptionEvalDataset
from models import CNN_Encoder
from transformer import Transformer
from utils import AverageMeter, clip_gradient, accuracy, get_eval_score, save_checkpoint


def check_weight_update(model, new_weights, model_name="Model"):
    state_dict = model.state_dict()
    mismatched_keys = []
    for key in new_weights.keys():
        if not state_dict[key].dtype.is_floating_point:
            continue
        if not torch.allclose(state_dict[key].float(), new_weights[key].float(), atol=1e-6):
            mismatched_keys.append(key)
    if mismatched_keys:
        print(f"❗ {model_name} did NOT update for keys: {mismatched_keys[:3]}... ({len(mismatched_keys)} total)")
        return False
    else:
        print(f"✅ {model_name} successfully updated all weights.")
        return True


def average_weights(w_list):
    avg_weights = copy.deepcopy(w_list[0])
    for key in avg_weights.keys():
        avg_weights[key] = torch.stack([w_[key].float() for w_ in w_list], dim=0).mean(dim=0)
    return avg_weights


def train_model(args_, train_dataset_, val_loader_, g_encoder_, g_decoder_, criterion_, file_name_):
    best_bleu4 = 0.
    best_cider = 0.
    bleu4_per_rnd = []
    cider_per_rnd = []
    val_loss_per_rnd = []
    val_acc_per_rnd = []
    train_acc_per_rnd = []
    train_loss_per_rnd = []
    epochs_since_improvement = 0

    encoder_lr = args_.encoder_lr
    decoder_lr = args_.decoder_lr

    client_datasets = [FederatedPMDDataset(train_dataset_, client_id, args_.num_clients)
                       for client_id in range(args_.num_clients)]
    print(f"Created {len(client_datasets)} IID client datasets")

    for epoch_g in range(args_.epoch_global):
        print(f"Global Epoch [{epoch_g + 1}/{args_.epoch_global}]")

        clients_encoder_weights_list = []
        clients_decoder_weights_list = []
        clients_train_loss = 0.0
        clients_train_accuracy = 0.0

        for client_id in range(args_.num_clients):
            print(f"Training Client {client_id}")

            client_dataset = client_datasets[client_id]
            client_dataloader = DataLoader(client_dataset, batch_size=args_.batch_size, shuffle=True,
                                           num_workers=args_.workers, pin_memory=True)

            local_encoder = copy.deepcopy(g_encoder_)
            local_decoder = copy.deepcopy(g_decoder_)

            local_encoder.train()
            local_decoder.train()

            local_encoder_optimizer = torch.optim.Adam(params=filter(
                lambda p: p.requires_grad, local_encoder.parameters()), lr=encoder_lr, weight_decay=0.0001)
            local_decoder_optimizer = torch.optim.Adam(params=filter(
                lambda p: p.requires_grad, local_decoder.parameters()), lr=decoder_lr, weight_decay=0.0001)

            client_epoch_loss = 0.0
            client_epoch_accuracy = 0.0

            for epoch_l in range(args_.epoch_local):
                print(f"Client {client_id} - Local Epoch [{epoch_l + 1}/{args_.epoch_local}]")

                batch_time = AverageMeter()
                data_time = AverageMeter()
                losses = AverageMeter()
                top5accs = AverageMeter()

                start = time.time()

                for i_t, (pmd_image, caps, caplens) in enumerate(client_dataloader):
                    data_time.update(time.time() - start)

                    pmd_image = pmd_image.to(device)
                    caps = caps.to(device)
                    caplens = caplens.to(device)

                    feat = local_encoder(pmd_image)
                    scores, caps_sorted, decode_lengths, alphas, sort_ind = local_decoder(feat, caps, caplens)
                    targets = caps_sorted[:, 1:]
                    scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
                    targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

                    loss = criterion_(scores, targets)
                    dec_alphas = alphas["dec_enc_attns"]
                    alpha_trans_c = args_.alpha_c / (args_.n_heads * args_.decoder_layers)
                    for layer in range(args_.decoder_layers):
                        cur_layer_alphas = dec_alphas[layer]
                        for h in range(args_.n_heads):
                            cur_head_alpha = cur_layer_alphas[:, h, :, :]
                            loss += alpha_trans_c * ((1. - cur_head_alpha.sum(dim=1)) ** 2).mean()

                    local_decoder_optimizer.zero_grad()
                    local_encoder_optimizer.zero_grad()
                    loss.backward()

                    if args_.grad_clip is not None:
                        clip_gradient(local_decoder_optimizer, args_.grad_clip)
                        clip_gradient(local_encoder_optimizer, args_.grad_clip)

                    local_decoder_optimizer.step()
                    local_encoder_optimizer.step()

                    top5 = accuracy(scores, targets, 5)
                    losses.update(loss.item(), sum(decode_lengths))
                    top5accs.update(top5, sum(decode_lengths))
                    batch_time.update(time.time() - start)
                    start = time.time()
                    if i_t % args_.print_freq == 0:
                        print("Epoch: {}/{} step: {}/{} Loss: {} AVG_Loss: {} Top-5 Accuracy: {} Batch_time: {}s"
                              "".format(epoch_l + 1, args_.epoch_local, i_t + 1, len(client_dataloader), losses.val,
                                        losses.avg, top5accs.val, batch_time.val))
                print('Epoch: [{0}] LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}\n'.format(
                    epoch_l + 1, loss=losses, top5=top5accs))

                client_epoch_loss += losses.avg
                client_epoch_accuracy += top5accs.avg

            client_epoch_loss /= args_.epoch_local
            client_epoch_accuracy /= args_.epoch_local

            clients_train_loss += client_epoch_loss
            clients_train_accuracy += client_epoch_accuracy

            clients_encoder_weights_list.append(copy.deepcopy(local_encoder.state_dict()))
            clients_decoder_weights_list.append(copy.deepcopy(local_decoder.state_dict()))

        avg_train_loss = clients_train_loss / args_.num_clients
        avg_train_accuracy = clients_train_accuracy / args_.num_clients

        train_acc_per_rnd.append(avg_train_accuracy)
        train_loss_per_rnd.append(clients_train_loss)

        print(f"Global Epoch [{epoch_g + 1}/{args_.epoch_global}] - Avg Loss: {avg_train_loss:.4f}  - "
              f"Avg Accuracy: {avg_train_accuracy:.4f}")

        global_encoder_weights = average_weights(clients_encoder_weights_list)
        global_decoder_weights = average_weights(clients_decoder_weights_list)
        g_encoder_.load_state_dict(global_encoder_weights)
        g_decoder_.load_state_dict(global_decoder_weights)
        check_weight_update(g_encoder_, global_encoder_weights, "Encoder")
        check_weight_update(g_decoder_, global_decoder_weights, "Decoder")

        metrics, val_loss, val_acc = validate(args, val_loader_=val_loader, encoder_=g_encoder_, decoder_=g_decoder_,
                                              criterion_=criterion)
        recent_bleu4 = metrics["Bleu_4"]
        recent_cider = metrics["CIDEr"]
        bleu4_per_rnd.append(recent_bleu4)
        cider_per_rnd.append(recent_cider)
        val_loss_per_rnd.append(val_loss)
        val_acc_per_rnd.append(val_acc)

        is_best_bleu = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        is_best_cider = recent_cider > best_cider
        best_cider = max(recent_cider, best_cider)
        if epochs_since_improvement == args.stop_criteria:
            print("the model has not improved in the last {} epochs".format(args.stop_criteria))
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 5 == 0:
            encoder_lr *= 0.8
            decoder_lr *= 0.8
        if not is_best_bleu:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        save_checkpoint(file_name_, epoch_g, epochs_since_improvement, g_encoder_, g_decoder_, metrics, is_best_bleu,
                        is_best_cider, args.photon, args.image_size, args.data_name, args.num_clients)

    return train_acc_per_rnd, train_loss_per_rnd, bleu4_per_rnd, cider_per_rnd, val_loss_per_rnd, val_acc_per_rnd


def validate(args_, val_loader_, encoder_, decoder_, criterion_):
    decoder_.eval()
    encoder_.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()
    hypotheses = list()

    with torch.no_grad():
        for i_c, (pmd_image, caps, caplens, allcaps) in enumerate(val_loader_):

            pmd_image = pmd_image.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            feat = encoder_(pmd_image)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder_(feat, caps, caplens)

            targets = caps_sorted[:, 1:]
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            loss = criterion_(scores, targets)

            dec_alphas = alphas["dec_enc_attns"]
            alpha_trans_c = args_.alpha_c / (args_.n_heads * args_.decoder_layers)
            for layer in range(args_.decoder_layers):
                cur_layer_alphas = dec_alphas[layer]
                for h in range(args_.n_heads):
                    cur_head_alpha = cur_layer_alphas[:, h, :, :]
                    loss += alpha_trans_c * ((1. - cur_head_alpha.sum(dim=1)) ** 2).mean()

            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)
            start = time.time()

            allcaps = allcaps[sort_ind.to(torch.device("cpu"))]
            for j_a in range(allcaps.shape[0]):
                img_caps = allcaps[j_a].tolist()
                img_captions = list(
                    map(lambda c: [w_ for w_ in c if w_ not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))
                references.append(img_captions)

            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j_c, p in enumerate(preds):
                temp_preds.append(preds[j_c][:decode_lengths[j_c]])
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

    metrics_ = get_eval_score(references, hypotheses)
    print("EVA LOSS: {} TOP-5 Accuracy {} BLEU-1 {} BLEU2 {} BLEU3 {} BLEU-4 {} METEOR {} ROUGE_L {} CIDEr {}".format
          (losses.avg, top5accs.avg, metrics_["Bleu_1"], metrics_["Bleu_2"], metrics_["Bleu_3"], metrics_["Bleu_4"],
           metrics_["METEOR"], metrics_["ROUGE_L"], metrics_["CIDEr"]))
    return metrics_, losses.avg, top5accs.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image_Captioning')
    # Data parameters
    parser.add_argument('--data_folder', default="D:\PycharmProjects\PMD-PCI\DATA")
    parser.add_argument('--file_name', default="flickr30k_5_cap_5_mw_freq")
    parser.add_argument('--data_name', default="flickr30k", )
    parser.add_argument('--photon', default=10.00, type=float, help='photon number')
    parser.add_argument('--image_size', default=224, type=int, help='image size')
    parser.add_argument('--num_clients', default=10, type=int, help='number of clients')
    # Model parameters
    parser.add_argument('--emb_dim', type=int, default=300, help='dimension of word embeddings.')
    parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention.')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--encoder_layers', type=int, default=6, help='the number of layers of encoder in Transformer.')
    parser.add_argument('--decoder_layers', type=int, default=6, help='the number of layers of decoder in Transformer.')
    # Training parameters
    parser.add_argument('--epoch_global', type=int, default=20,
                        help='number of epochs to train for (if early stopping is not triggered).')
    parser.add_argument('--epoch_local', type=int, default=5,
                        help='number of epochs to train for (if early stopping is not triggered).')
    parser.add_argument('--stop_criteria', type=int, default=10,
                        help='training stop if epochs_since_improvement == stop_criteria')
    parser.add_argument('--batch_size', type=int, default=2, help='batch_size')
    parser.add_argument('--print_freq', type=int, default=100, help='print training/validation stats every __ batches.')
    parser.add_argument('--workers', type=int, default=4, help='for data-loading; right now, only 1 works with h5pys.')
    parser.add_argument('--encoder_lr', type=float, default=1e-4, help='learning rate for encoder if fine-tuning.')
    parser.add_argument('--decoder_lr', type=float, default=1e-4, help='learning rate for decoder.')
    parser.add_argument('--grad_clip', type=float, default=5., help='clip gradients at an absolute value of.')
    parser.add_argument('--alpha_c', type=float, default=2.,
                        help='regularization parameter for doubly stochastic attention, as in the paper.')
    parser.add_argument('--fine_tune_encoder_all', type=bool, default=True,
                        help='whether fine-tune all the parameters')
    parser.add_argument('--cnn_name', default='resnet101')
    args = parser.parse_args()

    print(args)

    device = torch.device("cuda")
    cudnn.benchmark = True
    print(device)

    word_map_file = os.path.join(args.data_folder, 'WORDMAP_' + args.file_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    file_name = args.cnn_name + '_' + str(args.image_size)

    if args.fine_tune_encoder_all:
        file_name = file_name + '_fine_tune_all'
    encoder = CNN_Encoder()
    if args.fine_tune_encoder_all:
        encoder.fine_tune_all(args.fine_tune_encoder_all)
    decoder = Transformer(vocab_size=len(word_map), embed_dim=args.emb_dim, encoder_layers=args.encoder_layers,
                          decoder_layers=args.decoder_layers, dropout=args.dropout, n_heads=args.n_heads)
    file_name = file_name + '_' + str(args.encoder_layers) + '_' + str(args.decoder_layers)
    print(file_name)

    decoder = decoder.to(device)
    encoder = encoder.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    train_dataset = CaptionDataset(args.data_folder, args.file_name, args.data_name, 'TRAIN', args.photon,
                                   args.image_size)
    val_loader = torch.utils.data.DataLoader(
        CaptionEvalDataset(args.data_folder, args.file_name, args.data_name, 'VAL', args.photon, args.image_size),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    t_acc_p, t_loss_p, bleu4_p, cider_p, v_loss_p, v_acc_p = train_model(
        args, train_dataset_=train_dataset, val_loader_=val_loader, g_encoder_=encoder, g_decoder_=decoder,
        criterion_=criterion, file_name_=file_name)

    with h5py.File(os.path.join('models/' + args.data_name + '/' + str(args.image_size) + '/' +
                                'models_' + "{:.2f}".format(args.photon) + '/' + str(args.num_clients) + '/'
                                + 'metrics.hdf5'), 'a') as h_:
        h_.attrs['train_loss'] = t_loss_p
        h_.attrs['train_acc'] = t_acc_p
        h_.attrs['val_loss'] = v_loss_p
        h_.attrs['val_acc'] = v_acc_p
        h_.attrs['val_bleu4'] = bleu4_p
        h_.attrs['val_cider'] = cider_p

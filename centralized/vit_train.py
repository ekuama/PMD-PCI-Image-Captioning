import argparse
import json
import os
import time

import h5py
import timm
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau

from datasets import CaptionDataset
from transformer_v import Transformer
from utils_v import AverageMeter, clip_gradient, accuracy, get_eval_score, save_checkpoint


def train(args_, train_loader_, encoder_, decoder_, criterion_, encoder_optimizer_, decoder_optimizer_, epoch_):
    decoder_.train()
    encoder_.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()
    start = time.time()
    for i_t, (pmd_image, caps, caplens) in enumerate(train_loader_):
        data_time.update(time.time() - start)
        pmd_image = pmd_image.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)
        feat = encoder_.forward_features(pmd_image)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder_(feat, caps, caplens)
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
        decoder_optimizer_.zero_grad()
        encoder_optimizer_.zero_grad()
        loss.backward()
        if args_.grad_clip is not None:
            clip_gradient(decoder_optimizer_, args_.grad_clip)
            clip_gradient(encoder_optimizer_, args_.grad_clip)
        decoder_optimizer_.step()
        encoder_optimizer_.step()
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)
        start = time.time()
        if i_t % args_.print_freq == 0:
            print("Epoch: {}/{} step: {}/{} Loss: {} AVG_Loss: {} Top-5 Accuracy: {} Batch_time: {}s".format(
                epoch_ + 1, args_.epochs, i_t + 1, len(train_loader_), losses.val, losses.avg, top5accs.val,
                batch_time.val))
    print('Epoch: [{0}] LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}\n'.format(epoch + 1, loss=losses,
                                                                                         top5=top5accs))
    return losses.avg, top5accs.avg


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
            feat = encoder_.forward_features(pmd_image)
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
    parser.add_argument('--file_name', default="flickr8k_5_cap_5_mw_freq")
    parser.add_argument('--data_name', default="flickr8k", )
    parser.add_argument('--photon', default=10.00, type=float, help='photon number')
    parser.add_argument('--image_size', default=224, type=int, help='image size')
    # Model parameters
    parser.add_argument('--emb_dim', type=int, default=300, help='dimension of word embeddings.')
    parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention.')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--encoder_layers', type=int, default=2, help='the number of layers of encoder in Transformer.')
    parser.add_argument('--decoder_layers', type=int, default=2, help='the number of layers of decoder in Transformer.')
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train for (if early stopping is not triggered).')
    parser.add_argument('--stop_criteria', type=int, default=10,
                        help='training stop if epochs_since_improvement == stop_criteria')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--print_freq', type=int, default=100, help='print training/validation stats every __ batches.')
    parser.add_argument('--workers', type=int, default=0, help='for data-loading; right now, only 1 works with h5pys.')
    parser.add_argument('--encoder_lr', type=float, default=5e-5, help='learning rate for encoder if fine-tuning.')
    parser.add_argument('--decoder_lr', type=float, default=1e-4, help='learning rate for decoder.')
    parser.add_argument('--grad_clip', type=float, default=5., help='clip gradients at an absolute value of.')
    parser.add_argument('--alpha_c', type=float, default=2.,
                        help='regularization parameter for doubly stochastic attention, as in the paper.')
    parser.add_argument('--fine_tune_encoder_all', type=bool, default=True,
                        help='whether fine-tune all the parameters')
    parser.add_argument('--cnn_name', default='vit')
    args = parser.parse_args()
    print(args)
    start_epoch = 0
    best_bleu4 = 0.
    best_cider = 0.
    epochs_since_improvement = 0
    device = torch.device("cuda")
    cudnn.benchmark = True
    print(device)

    word_map_file = os.path.join(args.data_folder, 'WORDMAP_' + args.file_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    file_name = args.cnn_name + '_' + str(args.image_size)
    if args.fine_tune_encoder_all:
        file_name = file_name + '_fine_tune_all'
    encoder = timm.create_model("vit_base_patch16_224", pretrained=True)
    encoder.head = nn.Identity()
    if args.fine_tune_encoder_all:
        for param in encoder.parameters():
            param.requires_grad = True
    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                         lr=args.encoder_lr)
    encoder_scheduler = ReduceLROnPlateau(encoder_optimizer, mode='max', factor=0.1, patience=10)
    decoder = Transformer(vocab_size=len(word_map), embed_dim=args.emb_dim, dropout=args.dropout,
                          encoder_layers=args.encoder_layers, n_heads=args.n_heads,
                          decoder_layers=args.decoder_layers)
    file_name = file_name + '_' + str(args.encoder_layers) + '_' + str(args.decoder_layers)
    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                         lr=args.decoder_lr)
    decoder_scheduler = ReduceLROnPlateau(decoder_optimizer, mode='max', factor=0.1, patience=10)
    print(file_name)
    decoder = decoder.to(device)
    encoder = encoder.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(args.data_folder, args.file_name, args.data_name, 'TRAIN', args.photon, args.image_size),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(args.data_folder, args.file_name, args.data_name, 'VAL', args.photon, args.image_size),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    bleu4_per_rnd = []
    cider_per_rnd = []
    val_loss_per_rnd = []
    val_acc_per_rnd = []
    train_acc_per_rnd = []
    train_loss_per_rnd = []

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = train(args, train_loader_=train_loader, encoder_=encoder, decoder_=decoder,
                                      criterion_=criterion, encoder_optimizer_=encoder_optimizer,
                                      decoder_optimizer_=decoder_optimizer, epoch_=epoch)
        train_acc_per_rnd.append(train_acc)
        train_loss_per_rnd.append(train_loss)
        metrics, val_loss, val_acc = validate(args, val_loader_=val_loader, encoder_=encoder, decoder_=decoder,
                                              criterion_=criterion)
        recent_bleu4 = metrics["Bleu_4"]
        recent_cider = metrics["CIDEr"]
        bleu4_per_rnd.append(recent_bleu4)
        cider_per_rnd.append(recent_cider)
        val_loss_per_rnd.append(val_loss)
        val_acc_per_rnd.append(val_acc)
        if args.fine_tune_encoder or args.fine_tune_encoder_all:
            encoder_scheduler.step(recent_bleu4)
        decoder_scheduler.step(recent_bleu4)
        is_best_bleu4 = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        is_best_cider = recent_cider > best_cider
        best_cider = max(recent_cider, best_cider)
        if not is_best_bleu4:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0
        if epochs_since_improvement == args.stop_criteria:
            print("the model has not improved in the last {} epochs".format(args.stop_criteria))
            break
        save_checkpoint(file_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, encoder_scheduler, decoder_scheduler, metrics, is_best_bleu4, is_best_cider,
                        args.photon, args.image_size, args.data_name)
    with h5py.File(os.path.join('models/' + args.data_name + '_new/' + str(args.image_size) + '/' +
                                'models_' + "{:.2f}".format(args.photon) + '/' + 'metrics_vit.hdf5'), 'a') as h_:
        h_.attrs['train_loss'] = train_loss_per_rnd
        h_.attrs['train_acc'] = train_acc_per_rnd
        h_.attrs['val_loss'] = val_loss_per_rnd
        h_.attrs['val_acc'] = val_acc_per_rnd
        h_.attrs['val_bleu4'] = bleu4_per_rnd
        h_.attrs['val_cider'] = cider_per_rnd

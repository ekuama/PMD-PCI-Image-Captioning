import argparse
import json
import os

import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from tqdm import tqdm

from datasets import CaptionDataset
from utils import get_eval_score


def evaluate(args_, mode_):
    beam_size = args_.beam_size
    Caption_End = False
    loader = torch.utils.data.DataLoader(
        CaptionDataset(args_.data_folder, args_.file_name, 'TEST', mode_),
        batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    references = list()
    hypotheses = list()

    with torch.no_grad():
        if mode_ == "drpe":
            for i, (real_image, imag_image, caps, caplens, allcaps) in enumerate(
                    tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):
                k = beam_size
                real_image = real_image.to(device)
                imag_image = imag_image.to(device)
                if args_.cnn_name == 'vit':
                    encoder_out = encoder_1.forward_features(torch.cat((real_image, imag_image), dim=1))
                else:
                    encoder_out = encoder_1(real_image, imag_image)
                if args_.cnn_name == 'vit':
                    encoder_dim = encoder_out.size(-1)
                    encoder_out = encoder_out.expand(k, -1, encoder_dim)
                else:
                    enc_image_size = encoder_out.size(1)
                    encoder_dim = encoder_out.size(-1)
                    encoder_out = encoder_out.expand(k, enc_image_size, enc_image_size, encoder_dim)
                k_prev_words = torch.LongTensor([[word_map['<start>']] * 52] * k).to(device)
                seqs = torch.LongTensor([[word_map['<start>']]] * k).to(device)
                top_k_scores = torch.zeros(k, 1).to(device)
                complete_seqs = []
                complete_seqs_scores = []
                step = 1
                while True:
                    cap_len = torch.LongTensor([52]).repeat(k, 1)
                    scores, _, _, _, _ = decoder(encoder_out, k_prev_words, cap_len)
                    scores = scores[:, step - 1, :].squeeze(1)
                    scores = F.log_softmax(scores, dim=1)
                    scores = top_k_scores.expand_as(scores) + scores
                    if step == 1:
                        top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
                    else:
                        top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

                    prev_word_inds = top_k_words // vocab_size
                    next_word_inds = top_k_words % vocab_size

                    seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
                    incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                       next_word != word_map['<end>']]
                    complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
                    if len(complete_inds) > 0:
                        Caption_End = True
                        complete_seqs.extend(seqs[complete_inds].tolist())
                        complete_seqs_scores.extend(top_k_scores[complete_inds])
                    k -= len(complete_inds)
                    if k == 0:
                        break
                    seqs = seqs[incomplete_inds]
                    encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
                    top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                    k_prev_words = k_prev_words[incomplete_inds]
                    k_prev_words[:, :step + 1] = seqs
                    if step > 50:
                        break
                    step += 1
                assert Caption_End
                if complete_seqs_scores:
                    indices = complete_seqs_scores.index(max(complete_seqs_scores))
                    seq = complete_seqs[indices]
                else:
                    seq = seqs[0]

                img_caps = allcaps[0].tolist()
                img_captions = list(
                    map(lambda c_: [w for w in c_ if
                                    w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                        img_caps))
                references.append(img_captions)

                hypotheses.append(
                    [w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
                assert len(references) == len(hypotheses)
        else:
            for i, (pmd_img, caps, caplens, allcaps) in enumerate(
                    tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):
                k = beam_size
                pmd_img = pmd_img.to(device)

                if args_.cnn_name == 'vit':
                    encoder_out = encoder_1.forward_features(pmd_img)
                else:
                    encoder_out = encoder_1(pmd_img)
                if args_.cnn_name == 'vit':
                    encoder_dim = encoder_out.size(-1)
                    encoder_out = encoder_out.expand(k, -1, encoder_dim)
                else:
                    enc_image_size = encoder_out.size(1)
                    encoder_dim = encoder_out.size(-1)
                    encoder_out = encoder_out.expand(k, enc_image_size, enc_image_size, encoder_dim)
                k_prev_words = torch.LongTensor([[word_map['<start>']] * 52] * k).to(device)
                seqs = torch.LongTensor([[word_map['<start>']]] * k).to(device)
                top_k_scores = torch.zeros(k, 1).to(device)
                complete_seqs = []
                complete_seqs_scores = []
                step = 1
                while True:
                    cap_len = torch.LongTensor([52]).repeat(k, 1)
                    scores, _, _, _, _ = decoder(encoder_out, k_prev_words, cap_len)
                    scores = scores[:, step - 1, :].squeeze(1)
                    scores = F.log_softmax(scores, dim=1)
                    scores = top_k_scores.expand_as(scores) + scores
                    if step == 1:
                        top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
                    else:
                        top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

                    prev_word_inds = top_k_words // vocab_size
                    next_word_inds = top_k_words % vocab_size

                    seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
                    incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                       next_word != word_map['<end>']]
                    complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
                    if len(complete_inds) > 0:
                        Caption_End = True
                        complete_seqs.extend(seqs[complete_inds].tolist())
                        complete_seqs_scores.extend(top_k_scores[complete_inds])
                    k -= len(complete_inds)
                    if k == 0:
                        break
                    seqs = seqs[incomplete_inds]
                    encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
                    top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                    k_prev_words = k_prev_words[incomplete_inds]
                    k_prev_words[:, :step + 1] = seqs
                    if step > 50:
                        break
                    step += 1
                assert Caption_End
                if complete_seqs_scores:
                    indices = complete_seqs_scores.index(max(complete_seqs_scores))
                    seq = complete_seqs[indices]
                else:
                    seq = seqs[0]
                img_caps = allcaps[0].tolist()
                img_captions = list(
                    map(lambda c_: [w for w in c_ if
                                    w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                        img_caps))
                references.append(img_captions)
                hypotheses.append(
                    [w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
                assert len(references) == len(hypotheses)

    metrics_ = get_eval_score(references, hypotheses)

    return metrics_


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image_Captioning')
    parser.add_argument('--data_folder', default="D:\PycharmProjects\PMD-PCI\DATA")
    parser.add_argument('--file_name', default="flickr30k_5_cap_5_mw_freq")
    parser.add_argument('--data_name', default="flickr30k")
    parser.add_argument('--cnn_name', default="resnet101", help='which model does cnn use?')
    parser.add_argument('--beam_size', type=int, default=3, help='beam_size.')
    parser.add_argument('--image_size', default=224, type=int, help='image size')
    parser.add_argument("--base_path", default="")
    parser.add_argument("--checkpoint_name", default="")
    args = parser.parse_args()

    word_map_file = os.path.join(args.data_folder, 'WORDMAP_' + args.file_name + '.json')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    print(device)

    mode = ["orig", "drpe", "aes"]
    for m in mode:
        checkpoint = (args.base_path + '/' + args.data_name + '/' + args.image_size + '/models_' + m +
                      "/" + args.checkpoint_name)
        print(checkpoint)
        checkpoint = torch.load(checkpoint, map_location=str(device))
        decoder = checkpoint['decoder']
        decoder = decoder.to(device)
        decoder.eval()
        encoder_1 = checkpoint['encoder']
        encoder_1 = encoder_1.to(device)
        encoder_1.eval()
        with open(word_map_file, 'r') as j:
            word_map = json.load(j)
        vocab_size = len(word_map)
        rev_word_map = {v: k for k, v in word_map.items()}

        metrics = evaluate(args, m)
        print("beam size {}: BLEU-1 {} BLEU-2 {} BLEU-3 {} BLEU-4 {} METEOR {} ROUGE_L {} CIDEr {}".format
              (args.beam_size, metrics["Bleu_1"], metrics["Bleu_2"], metrics["Bleu_3"],
               metrics["Bleu_4"], metrics["METEOR"], metrics["ROUGE_L"], metrics["CIDEr"]))

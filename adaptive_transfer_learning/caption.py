import argparse
import json
import os

import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm

from datasets import CaptionTestDataset


def caption(args_, word_map_, pn_, encoder_1, decoder):
    beam_size = args_.beam_size
    vocab_size = len(word_map_)
    loader = torch.utils.data.DataLoader(CaptionTestDataset(args_.data_folder, args_.file_name, args_.data_name,
                                                            pn_, args_.image_size),
                                         batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    image_names_ = list()
    sequences_ = list()
    with torch.no_grad():
        for i, (pmd_img, name) in enumerate(tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):
            k = beam_size
            pmd_img = pmd_img.to(device)
            Caption_End = False
            infinite_pred = False
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
                    infinite_pred = True
                    break
                step += 1
            if infinite_pred is not True:
                assert Caption_End
                if complete_seqs_scores:
                    indices = complete_seqs_scores.index(max(complete_seqs_scores))
                    seq_ = complete_seqs[indices]
                else:
                    seq_ = seqs[0]
            else:
                seq_ = seqs[0][:20]
                seq_ = [seq_[i].item() for i in range(len(seq_))]
            image_names_.append(name)
            sequences_.append(seq_)
    return image_names_, sequences_


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image_Captioning')
    parser.add_argument('--data_folder', '-d', default="")
    parser.add_argument('--file_name', default="flickr8k_5_cap_5_mw_freq")
    parser.add_argument('--data_name', default="flickr8k")
    parser.add_argument('--beam_size', '-b', type=int, default=3, help='beam size for beam search')
    parser.add_argument('--cnn_name', default="resnet101", help='which model does cnn use?')
    parser.add_argument('--image_size', default=224, type=int, help='image size')
    parser.add_argument("--base_path", default="")
    parser.add_argument("--checkpoint_name", default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    word_map_file = os.path.join(args.data_folder, 'WORDMAP_' + args.file_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}

    photon_numbers = [10, 100, 1000, 10000, 100000, 50, 500, 5000, 50000, 500000]

    for photon_number in photon_numbers:
        caps_3 = list()
        print("-------------------------------------------------------------------------------")
        print("photon_number:", photon_number)
        checkpoint_path = (args.base_path + '/' + args.data_name + '/' + args.image_size + '/models_'
                           + "{:.2f}".format(photon_number) + "/" + args.checkpoint_name)
        checkpoint = torch.load(checkpoint_path, map_location=str(device))
        decoder_ = checkpoint['decoder']
        decoder_ = decoder_.to(device)
        decoder_.eval()
        encoder_ = checkpoint['encoder']
        encoder_ = encoder_.to(device)
        encoder_.eval()
        image_names, sequences = caption(args, word_map, photon_number, encoder_, decoder_)
        for seq in sequences:
            hypo_caps = [w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
            words = ' '.join(rev_word_map[f] for f in hypo_caps)
            caps_3.append(words)
        name_dict = {'Name': image_names, 'Beam 3': caps_3}
        caption_nic = pd.DataFrame(name_dict)
        folder = 'results_csv/' + args.data_name + '/' + str(args.image_size) + '/' + 'models_' + "{:.2f}".format(
            photon_number)
        if not os.path.exists(folder):
            os.makedirs(folder)
        name_csv = checkpoint_path.split('/')[-1][:-8] + '.csv'
        caption_nic.to_csv(folder + '/' + name_csv, index=False)

import os

import torch

from eval_func.bleu.bleu import Bleu
from eval_func.cider.cider import Cider
from eval_func.meteor.meteor import Meteor
from eval_func.rouge.rouge import Rouge


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(file_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                    decoder_optimizer, encoder_scheduler, decoder_scheduler, metrics, is_best_bleu, is_best_cider,
                    photon, image_size, data_name, n_heads):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'metrics': metrics,
             'encoder': encoder,
             'decoder': decoder,
             'photon': photon,
             'image_size': image_size,
             'data_name': data_name}
    filename = 'checkpoint_' + file_name + '.pth.tar'
    folder = 'models/' + data_name + '_new/' + str(image_size) + '/' + 'models_' + "{:.2f}".format(
        photon)
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(state, folder + '/' + filename)
    if is_best_bleu:
        torch.save(state, folder + '/BEST_BLEU4_' + filename)
    if is_best_cider:
        torch.save(state, folder + '/BEST_CIDEr_' + filename)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def get_eval_score(references, hypotheses):
    scorers = [(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]), (Meteor(), "METEOR"), (Rouge(), "ROUGE_L"),
               (Cider(), "CIDEr")]
    hypo = [[' '.join(hypo)] for hypo in [[str(x) for x in hypo] for hypo in hypotheses]]
    ref = [[' '.join(reft) for reft in reftmp] for reftmp in
           [[[str(x) for x in reft] for reft in reftmp] for reftmp in references]]
    score = []
    method = []
    for scorer, method_i in scorers:
        score_i, scores_i = scorer.compute_score(ref, hypo)
        score.extend(score_i) if isinstance(score_i, list) else score.append(score_i)
        method.extend(method_i) if isinstance(method_i, list) else method.append(method_i)
        print("{} {}".format(method_i, score_i))
    score_dict = dict(zip(method, score))
    return score_dict

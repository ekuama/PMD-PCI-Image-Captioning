import argparse
import os

import h5py as hp
import numpy as np
import torch
from skimage.transform import resize


# Poisson-Multinomial Distribution-based PCI #####################################

def PMD_PCI(img_, photon_num, image_size, image_channel, seed, norm):
    img_ = img_.transpose(1, 2, 0)
    total_val = np.sum(img_)
    img_norm = img_ / total_val
    img_norm = np.reshape(img_norm, [1, -1])

    rng = np.random.default_rng(seed)

    img_pci = rng.multinomial(photon_num, img_norm)
    img_pci = np.reshape(img_pci, [image_size, image_size, image_channel])

    if norm:

        img_pci_final = (img_pci - np.min(img_pci)) / (np.max(img_pci) - np.min(img_pci))
        img_pci_final = img_pci_final * 255

        img_pci_final = img_pci_final.astype(np.uint8)

    else:
        img_pci_final = img_pci.astype(np.uint8)

    return img_pci_final


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description='PMD PCI reading')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--image_channel', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', default="flickr8k", choices=['flickr8k', 'flickr30k', 'coco'])
    parser.add_argument('--data_folder', default="DATA")

    arg = parser.parse_args()

    photon_numbers = [10 ** 1, 10 ** 1.5, 50, 10 ** 2, 10 ** 2.5, 500, 10 ** 3, 10 ** 3.5, 5000, 10 ** 4, 10 ** 4.5,
                      50000, 10 ** 5, 10 ** 5.5, 500000]

    x_train = hp.File(os.path.join(arg.data_folder, 'TRAIN_IMAGES_' + arg.dataset + '_5_cap_5_mw_freq.hdf5'), 'r')
    x_train = x_train['images']
    x_val = hp.File(os.path.join(arg.data_folder, 'VAL_IMAGES_' + arg.dataset + '_5_cap_5_mw_freq.hdf5'), 'r')
    x_val = x_val['images']
    x_test = hp.File(os.path.join(arg.data_folder, 'TEST_IMAGES_' + arg.dataset + '_5_cap_5_mw_freq.hdf5'), 'r')
    x_test = x_test['images']

    for ph_num in photon_numbers:
        print('Dataset: ', arg.dataset, 'Photon number: ', ph_num)
        average = torch.zeros(3, device=device)
        standard_dev = torch.zeros(3, device=device)
        for tr in range(x_train.shape[0]):
            img = x_train[tr]
            img = img.transpose(1, 2, 0)
            if arg.image_size != 256:
                img = resize(img, (arg.image_size, arg.image_size), order=1, preserve_range=True)
            pmd_img = PMD_PCI(img, ph_num, arg.image_size, arg.image_channel, arg.seed, True)
            pmd_img = pmd_img.transpose(2, 0, 1)
            pmd_image = torch.FloatTensor(pmd_img / 255.0).to(device)
            average += pmd_image.mean([1, 2])
            standard_dev += pmd_image.std([1, 2])
        for tr in range(x_val.shape[0]):
            img = x_val[tr]
            img = img.transpose(1, 2, 0)
            if arg.image_size != 256:
                img = resize(img, (arg.image_size, arg.image_size), order=1, preserve_range=True)
            pmd_img = PMD_PCI(img, ph_num, arg.image_size, arg.image_channel, arg.seed, True)
            pmd_img = pmd_img.transpose(2, 0, 1)
            pmd_image = torch.FloatTensor(pmd_img / 255.0).to(device)
            average += pmd_image.mean([1, 2])
            standard_dev += pmd_image.std([1, 2])
        for tr in range(x_test.shape[0]):
            img = x_test[tr]
            img = img.transpose(1, 2, 0)
            if arg.image_size != 256:
                img = resize(img, (arg.image_size, arg.image_size), order=1, preserve_range=True)
            pmd_img = PMD_PCI(img, ph_num, arg.image_size, arg.image_channel, arg.seed, True)
            pmd_img = pmd_img.transpose(2, 0, 1)
            pmd_image = torch.FloatTensor(pmd_img / 255.0).to(device)
            average += pmd_image.mean([1, 2])
            standard_dev += pmd_image.std([1, 2])
        total = x_test.shape[0] + x_val.shape[0] + x_train.shape[0]
        print('Average value', (average / total).cpu(), 'Standard deviation value', (standard_dev / total).cpu())

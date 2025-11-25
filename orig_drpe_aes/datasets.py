import json
import os

import h5py
import numpy as np
import torch
import torchvision.transforms as transforms
from Crypto.Cipher import AES
from PIL import Image
from torch.utils.data import Dataset


def encrypt_image(image_, key_, iv_=None):
    height, width, channels = image_.shape
    image_array = image_.tobytes()
    padding_length = AES.block_size - len(image_array) % AES.block_size
    image_array += bytes(padding_length * ".", "UTF-8")

    if iv_ is None:
        aes = AES.new(key_, AES.MODE_ECB)
    else:
        aes = AES.new(key_, AES.MODE_CBC, iv_)

    encrypted_image = aes.encrypt(image_array)
    encrypted_image = encrypted_image[:-padding_length]

    return Image.frombytes("RGB", (width, height), encrypted_image, "raw", "RGB")


def stats(mode):
    if mode == "aes":
        mean_p, std_p = [0.5000, 0.5000, 0.5000], [0.2898, 0.2898, 0.2898]
    elif mode == "drpe_real":
        mean_p, std_p = [-0.0013, -0.0012, -0.0011], [0.3632, 0.3452, 0.3235]
        # mean_p, std_p = [-5.1658e-06, -4.7697e-06, -4.3577e-06], [0.0014, 0.0014, 0.0013]
    elif mode == "drpe_imag":
        # mean_p, std_p = [6.2095e-07, 5.8147e-07, 3.6990e-07], [0.0014, 0.0014, 0.0013]
        mean_p, std_p = [1.1001e-04, 9.1772e-05, 6.0525e-05], [0.3609, 0.3429, 0.3215]
    else:
        mean_p, std_p = [0.4584, 0.4466, 0.4045], [0.2341, 0.2250, 0.2294]
    return mean_p, std_p


class CaptionDataset(Dataset):
    def __init__(self, data_folder, file_name, split, mode):
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}
        self.mode = mode

        self.n_x = np.load(os.path.join(data_folder, 'n_x_224.npy'))
        self.b_x = np.load(os.path.join(data_folder, 'b_x_224.npy'))
        self.key = b"\xcf\xf2\xb3xXW\xc7\x7f\xd5\xedx\xb2mvp\xcbE\x02\x17fck?\xedV\xc7\xcc'\xef\x95y\xe3"
        self.iv = b'Y\xce\x86\x9d\x06\x9cFr.\x12\xceR7D\x15\xce'

        self.r = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + file_name + '_224.hdf5'), 'r')
        self.r_imgs = self.r['images']

        self.cpi = 5

        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + file_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + file_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        if mode == 'drpe':
            mean_, std_ = stats('drpe_real')
            mean2_, std2_ = stats('drpe_imag')
            normalize = transforms.Normalize(mean=mean_, std=std_)
            self.transform = transforms.Compose([normalize])
            normalize2 = transforms.Normalize(mean=mean2_, std=std2_)
            self.transform2 = transforms.Compose([normalize2])
        elif mode == 'aes':
            mean_, std_ = stats('aes')
            normalize = transforms.Normalize(mean=mean_, std=std_)
            self.transform = transforms.Compose([normalize])
        else:
            mean_, std_ = stats('orig')
            normalize = transforms.Normalize(mean=mean_, std=std_)
            self.transform = transforms.Compose([normalize])

        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        caption = torch.LongTensor(self.captions[i])
        caplen = torch.LongTensor([self.caplens[i]])
        if self.mode == 'drpe':
            image = self.r_imgs[i // self.cpi].transpose(1, 2, 0)
            img = image / 255.0
            red = img[:, :, 0]
            green = img[:, :, 1]
            blue = img[:, :, 2]
            red = np.fft.ifft2(np.fft.fft2(red * np.exp(2j * np.pi * self.n_x)) * np.exp(2j * np.pi * self.b_x))
            green = np.fft.ifft2(np.fft.fft2(green * np.exp(2j * np.pi * self.n_x)) * np.exp(2j * np.pi * self.b_x))
            blue = np.fft.ifft2(np.fft.fft2(blue * np.exp(2j * np.pi * self.n_x)) * np.exp(2j * np.pi * self.b_x))
            red, green, blue = (np.expand_dims(red, axis=2), np.expand_dims(green, axis=2),
                                np.expand_dims(blue, axis=2))
            encrypt_ = np.concatenate((red, green, blue), axis=2)
            real_ = np.real(encrypt_)
            imag_ = np.imag(encrypt_)
            real_ = real_.transpose(2, 0, 1)
            imag_ = imag_.transpose(2, 0, 1)
            real_img = torch.FloatTensor(real_)
            imag_img = torch.FloatTensor(imag_)
            real_img = self.transform(real_img)
            imag_img = self.transform2(imag_img)
            if self.split == 'TRAIN':
                return real_img, imag_img, caption, caplen
            else:
                all_captions = torch.LongTensor(
                    self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
                return real_img, imag_img, caption, caplen, all_captions
        elif self.mode == 'aes':
            image = self.r_imgs[i // self.cpi].transpose(1, 2, 0)
            e_i = encrypt_image(image, self.key, self.iv)
            e_i = np.array(e_i).transpose(2, 0, 1)
            e_i = e_i / 255.0
            e_img = torch.FloatTensor(e_i)
            e_img = self.transform(e_img)
            if self.split == 'TRAIN':
                return e_img, caption, caplen
            else:
                all_captions = torch.LongTensor(
                     self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
                return e_img, caption, caplen, all_captions
        else:
            image = torch.FloatTensor(self.r_imgs[i // self.cpi] / 255.0)
            image = self.transform(image)
            if self.split == 'TRAIN':
                return image, caption, caplen
            else:
                all_captions = torch.LongTensor(
                    self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
                return image, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size


class CaptionTestDataset(Dataset):
    def __init__(self, data_folder, file_name, mode):
        self.mode = mode
        self.r = h5py.File(os.path.join(data_folder, 'TEST_IMAGES_' + file_name + '_224.hdf5'), 'r')
        self.r_imgs = self.r['images']

        self.n_x = np.load(os.path.join(data_folder, 'n_x_224.npy'))
        self.b_x = np.load(os.path.join(data_folder, 'b_x_224.npy'))
        self.key = b"\xcf\xf2\xb3xXW\xc7\x7f\xd5\xedx\xb2mvp\xcbE\x02\x17fck?\xedV\xc7\xcc'\xef\x95y\xe3"
        self.iv = b'Y\xce\x86\x9d\x06\x9cFr.\x12\xceR7D\x15\xce'

        with open(os.path.join(data_folder, 'TEST_NAME_' + file_name + '.json'), 'r') as j:
            self.names = json.load(j)

        if mode == 'drpe':
            mean_, std_ = stats('drpe_real')
            mean2_, std2_ = stats('drpe_imag')
            normalize = transforms.Normalize(mean=mean_, std=std_)
            self.transform = transforms.Compose([normalize])
            normalize2 = transforms.Normalize(mean=mean2_, std=std2_)
            self.transform2 = transforms.Compose([normalize2])
        elif mode == 'aes':
            mean_, std_ = stats('aes')
            normalize = transforms.Normalize(mean=mean_, std=std_)
            self.transform = transforms.Compose([normalize])
        else:
            mean_, std_ = stats('orig')
            normalize = transforms.Normalize(mean=mean_, std=std_)
            self.transform = transforms.Compose([normalize])

        self.dataset_size = len(self.r_imgs)

    def __getitem__(self, i):
        names = self.names[i]
        if self.mode == 'drpe':
            image = self.r_imgs[i].transpose(1, 2, 0)
            img = image / 255.0
            red = img[:, :, 0]
            green = img[:, :, 1]
            blue = img[:, :, 2]
            red = np.fft.ifft2(np.fft.fft2(red * np.exp(2j * np.pi * self.n_x)) * np.exp(2j * np.pi * self.b_x))
            green = np.fft.ifft2(np.fft.fft2(green * np.exp(2j * np.pi * self.n_x)) * np.exp(2j * np.pi * self.b_x))
            blue = np.fft.ifft2(np.fft.fft2(blue * np.exp(2j * np.pi * self.n_x)) * np.exp(2j * np.pi * self.b_x))
            red, green, blue = (np.expand_dims(red, axis=2), np.expand_dims(green, axis=2),
                                np.expand_dims(blue, axis=2))
            encrypt_ = np.concatenate((red, green, blue), axis=2)
            real_ = np.real(encrypt_)
            imag_ = np.imag(encrypt_)
            real_ = real_.transpose(2, 0, 1)
            imag_ = imag_.transpose(2, 0, 1)
            real_img = torch.FloatTensor(real_)
            imag_img = torch.FloatTensor(imag_)
            real_img = self.transform(real_img)
            imag_img = self.transform2(imag_img)
            return real_img, imag_img, names
        elif self.mode == 'aes':
            image = self.r_imgs[i].transpose(1, 2, 0)
            e_i = encrypt_image(image, self.key, self.iv)
            e_i = np.array(e_i).transpose(2, 0, 1)
            e_i = e_i / 255.0
            e_img = torch.FloatTensor(e_i)
            e_img = self.transform(e_img)
            return e_img, names
        else:
            image = torch.FloatTensor(self.r_imgs[i] / 255.0)
            image = self.transform(image)
            return image, names

    def __len__(self):
        return self.dataset_size


class CaptionFedBaseDataset(Dataset):
    def __init__(self, data_folder, file_name, split, mode):
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}
        self.mode = mode

        self.n_x = np.load(os.path.join(data_folder, 'n_x_224.npy'))
        self.b_x = np.load(os.path.join(data_folder, 'b_x_224.npy'))
        self.key = b"\xcf\xf2\xb3xXW\xc7\x7f\xd5\xedx\xb2mvp\xcbE\x02\x17fck?\xedV\xc7\xcc'\xef\x95y\xe3"
        self.iv = b'Y\xce\x86\x9d\x06\x9cFr.\x12\xceR7D\x15\xce'

        self.r = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + file_name + '_224.hdf5'), 'r')
        self.r_imgs = self.r['images']

        self.cpi = 5

        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + file_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + file_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        if mode == 'drpe':
            mean_, std_ = stats('drpe_real')
            mean2_, std2_ = stats('drpe_imag')
            normalize = transforms.Normalize(mean=mean_, std=std_)
            self.transform = transforms.Compose([normalize])
            normalize2 = transforms.Normalize(mean=mean2_, std=std2_)
            self.transform2 = transforms.Compose([normalize2])
        elif mode == 'aes':
            mean_, std_ = stats('aes')
            normalize = transforms.Normalize(mean=mean_, std=std_)
            self.transform = transforms.Compose([normalize])
        else:
            mean_, std_ = stats('orig')
            normalize = transforms.Normalize(mean=mean_, std=std_)
            self.transform = transforms.Compose([normalize])

        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        caption = torch.LongTensor(self.captions[i])
        caplen = torch.LongTensor([self.caplens[i]])
        if self.mode == 'drpe':
            image = self.r_imgs[i // self.cpi].transpose(1, 2, 0)
            img = image / 255.0
            red = img[:, :, 0]
            green = img[:, :, 1]
            blue = img[:, :, 2]
            red = np.fft.ifft2(np.fft.fft2(red * np.exp(2j * np.pi * self.n_x)) * np.exp(2j * np.pi * self.b_x))
            green = np.fft.ifft2(np.fft.fft2(green * np.exp(2j * np.pi * self.n_x)) * np.exp(2j * np.pi * self.b_x))
            blue = np.fft.ifft2(np.fft.fft2(blue * np.exp(2j * np.pi * self.n_x)) * np.exp(2j * np.pi * self.b_x))
            red, green, blue = (np.expand_dims(red, axis=2), np.expand_dims(green, axis=2),
                                np.expand_dims(blue, axis=2))
            encrypt_ = np.concatenate((red, green, blue), axis=2)
            real_ = np.real(encrypt_)
            imag_ = np.imag(encrypt_)
            real_ = real_.transpose(2, 0, 1)
            imag_ = imag_.transpose(2, 0, 1)
            real_img = torch.FloatTensor(real_)
            imag_img = torch.FloatTensor(imag_)
            real_img = self.transform(real_img)
            imag_img = self.transform2(imag_img)
            return real_img, imag_img, caption, caplen
        elif self.mode == 'aes':
            image = self.r_imgs[i // self.cpi].transpose(1, 2, 0)
            e_i = encrypt_image(image, self.key, self.iv)
            e_i = np.array(e_i).transpose(2, 0, 1)
            e_i = e_i / 255.0
            e_img = torch.FloatTensor(e_i)
            e_img = self.transform(e_img)
            return e_img, caption, caplen
        else:
            image = torch.FloatTensor(self.r_imgs[i // self.cpi] / 255.0)
            image = self.transform(image)
            return image, caption, caplen

    def __len__(self):
        return self.dataset_size


class FederatedDataset(Dataset):
    def __init__(self, base_dataset, client_id, num_clients):
        self.base_dataset = base_dataset

        dataset_size = len(base_dataset)
        items_per_client = dataset_size // num_clients

        start_idx = client_id * items_per_client
        if client_id == num_clients - 1:
            end_idx = dataset_size
        else:
            end_idx = start_idx + items_per_client

        self.indices = list(range(start_idx, end_idx))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        return self.base_dataset[base_idx]

import json
import os

import h5py
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


def stats(photon_number):
    if photon_number == 10:
        mean_p, std_p = [9.3929e-05, 5.8962e-05, 4.6408e-05], [0.0095, 0.0074, 0.0063]
    elif photon_number == 50:
        mean_p, std_p = [0.0004, 0.0003, 0.0003], [0.0201, 0.0174, 0.0166]
    elif photon_number == 100:
        mean_p, std_p = [0.0008, 0.0006, 0.0006], [0.0278, 0.0253, 0.0236]
    elif photon_number == 500:
        mean_p, std_p = [0.0026, 0.0024, 0.0021], [0.0424, 0.0411, 0.0385]
    elif photon_number == 1000:
        mean_p, std_p = [0.0036, 0.0034, 0.0030], [0.0428, 0.0411, 0.0387]
    elif photon_number == 5000:
        mean_p, std_p = [0.0130, 0.0122, 0.0110], [0.0692, 0.0670, 0.0635]
    elif photon_number == 10000:
        mean_p, std_p = [0.0221, 0.0208, 0.0187], [0.0837, 0.0811, 0.0768]
    elif photon_number == 50000:
        mean_p, std_p = [0.0595, 0.0559, 0.0506], [0.1060, 0.1025, 0.0977]
    elif photon_number == 100000:
        mean_p, std_p = [0.0864, 0.0812, 0.0736], [0.1151, 0.1111, 0.1064]
    else:
        mean_p, std_p = [0.1764, 0.1663, 0.1510], [0.1394, 0.1338, 0.1306]
    return mean_p, std_p


class CaptionDataset(Dataset):
    def __init__(self, data_folder, file_name, data_name, split, pn, image_size):
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}
        folder = data_folder + '/' + data_name + '/' + str(image_size)
        self.r = h5py.File(
            os.path.join(folder, self.split + '_IMAGES_' + file_name + '_' + "{:.2f}".format(pn) + 'pn.hdf5'), 'r')
        self.r_imgs = self.r['images']
        self.cpi = 5
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + file_name + '.json'), 'r') as j:
            self.captions = json.load(j)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + file_name + '.json'), 'r') as j:
            self.caplens = json.load(j)
        mean_, std_ = stats(pn)
        normalize = transforms.Normalize(mean=mean_, std=std_)
        self.transform = transform = transforms.Compose([normalize])
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        pmd_image = torch.FloatTensor(self.r_imgs[i // self.cpi] / 255.0)
        pmd_image = self.transform(pmd_image)
        caption = torch.LongTensor(self.captions[i])
        caplen = torch.LongTensor([self.caplens[i]])
        return pmd_image, caption, caplen

    def __len__(self):
        return self.dataset_size


class FederatedPMDDataset(Dataset):
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


class CaptionEvalDataset(Dataset):
    def __init__(self, data_folder, file_name, data_name, split, pn, image_size):
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}
        folder = data_folder + '/' + data_name + '/' + str(image_size)
        self.r = h5py.File(
            os.path.join(folder, self.split + '_IMAGES_' + file_name + '_' + "{:.2f}".format(pn) + 'pn.hdf5'), 'r')
        self.r_imgs = self.r['images']
        self.cpi = 5
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + file_name + '.json'), 'r') as j:
            self.captions = json.load(j)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + file_name + '.json'), 'r') as j:
            self.caplens = json.load(j)
        mean_, std_ = stats(pn)
        normalize = transforms.Normalize(mean=mean_, std=std_)
        self.transform = transform = transforms.Compose([normalize])
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        pmd_image = torch.FloatTensor(self.r_imgs[i // self.cpi] / 255.0)
        pmd_image = self.transform(pmd_image)
        caption = torch.LongTensor(self.captions[i])
        caplen = torch.LongTensor([self.caplens[i]])
        if self.split == 'TRAIN':
            return pmd_image, caption, caplen
        else:
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return pmd_image, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size


class CaptionTestDataset(Dataset):

    def __init__(self, data_folder, file_name, data_name, pn, image_size):
        folder = data_folder + '/' + data_name + '/' + str(image_size)
        self.r = h5py.File(os.path.join(folder, 'TEST_IMAGES_' + file_name + '_' + "{:.2f}".format(pn) + 'pn.hdf5'),
                           'r')
        self.r_imgs = self.r['images']

        with open(os.path.join(data_folder, 'TEST_NAME_' + file_name + '.json'), 'r') as j:
            self.names = json.load(j)

        mean_, std_ = stats(pn)
        normalize = transforms.Normalize(mean=mean_, std=std_)
        self.transform = transform = transforms.Compose([normalize])

        self.dataset_size = len(self.r_imgs)

    def __getitem__(self, i):
        pmd_image = torch.FloatTensor(self.r_imgs[i] / 255.0)
        pmd_image = self.transform(pmd_image)
        names = self.names[i]
        return pmd_image, names

    def __len__(self):
        return self.dataset_size

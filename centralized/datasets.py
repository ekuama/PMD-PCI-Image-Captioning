import json
import os

import h5py
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


def stats(data_name, photon_number):
    if data_name == 'coco':
        if photon_number == 10:
            mean_p, std_p = [1.0109e-04, 5.6547e-05, 4.1663e-05], [0.0099, 0.0073, 0.0059]
        elif photon_number == 50:
            mean_p, std_p = [0.0004, 0.0003, 0.0003], [0.0205, 0.0173, 0.0163]
        elif photon_number == 100:
            mean_p, std_p = [0.0008, 0.0006, 0.0006], [0.0281, 0.0252, 0.0233]
        elif photon_number == 500:
            mean_p, std_p = [0.0026, 0.0024, 0.0021], [0.0426, 0.0414, 0.0386]
        elif photon_number == 1000:
            mean_p, std_p = [0.0037, 0.0034, 0.0030], [0.0428, 0.0411, 0.0386]
        elif photon_number == 5000:
            mean_p, std_p = [0.0131, 0.0124, 0.0112], [0.0697, 0.0678, 0.0641]
        elif photon_number == 10000:
            mean_p, std_p = [0.0224, 0.0212, 0.0190], [0.0846, 0.0823, 0.0777]
        elif photon_number == 50000:
            mean_p, std_p = [0.0608, 0.0573, 0.0519], [0.1072, 0.1042, 0.0992]
        elif photon_number == 100000:
            mean_p, std_p = [0.0886, 0.0836, 0.0756], [0.1159, 0.1127, 0.1078]
        else:
            mean_p, std_p = [0.1835, 0.1736, 0.1573], [0.1382, 0.1346, 0.1316]
    elif data_name == "flickr8k":
        if photon_number == 10:
            mean_p, std_p = [9.4404e-05, 5.9742e-05, 4.5147e-05], [0.0095, 0.0075, 0.0062]
        elif photon_number == 50:
            mean_p, std_p = [0.0004, 0.0003, 0.0003], [0.0200, 0.0175, 0.0165]
        elif photon_number == 100:
            mean_p, std_p = [0.0008, 0.0006, 0.0006], [0.0277, 0.0254, 0.0236]
        elif photon_number == 500:
            mean_p, std_p = [0.0026, 0.0025, 0.0022], [0.0431, 0.0425, 0.0394]
        elif photon_number == 1000:
            mean_p, std_p = [0.0036, 0.0034, 0.0030], [0.0426, 0.0416, 0.0389]
        elif photon_number == 5000:
            mean_p, std_p = [0.0130, 0.0126, 0.0113], [0.0697, 0.0685, 0.0646]
        elif photon_number == 10000:
            mean_p, std_p = [0.0221, 0.0215, 0.0191], [0.0842, 0.0829, 0.0780]
        elif photon_number == 50000:
            mean_p, std_p = [0.0600, 0.0581, 0.0521], [0.1069, 0.1049, 0.0996]
        elif photon_number == 100000:
            mean_p, std_p = [0.0873, 0.0845, 0.0757], [0.1155, 0.1131, 0.1079]
        else:
            mean_p, std_p = [0.1799, 0.1747, 0.1570], [0.1377, 0.1337, 0.1307]
    else:
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
        mean_, std_ = stats(data_name, pn)
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
        mean_, std_ = stats(image_size, pn)
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

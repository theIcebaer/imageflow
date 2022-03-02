import os
import glob
import random
import numpy as np
import torch
from pathlib import Path
import h5py
import torchvision.transforms
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset
from imageflow.utils import load_birl
from torchvision.transforms import ToTensor, Grayscale, Compose
from torchvision.io import read_image

from PIL import Image, ImageOps


class MnistDataset(Dataset):
    """
    This dataset assumes the data is given as hdf5 dataset in the following format:
    /source/samples
    /target/samples
    /v_field/samples
    /deformations/samples
    """

    def __init__(self, file_path, access_mode='r', noise=None):
        file_path = Path(file_path)
        self.file = h5py.File(file_path, access_mode)
        self.noise = noise

    def __len__(self):
        # print(len(self.file["source"]))
        return len(self.file["source"])

    def __getitem__(self, idx):
        v_field = torch.tensor(self.file['v_field'][idx])
        source = torch.tensor(self.file['source'][idx])
        target = torch.tensor(self.file['target'][idx])
        condition = torch.cat((source, target), dim=0)
        if self.noise:
            v_field += self.noise * torch.randn_like(v_field)
            condition += self.noise * torch.randn_like(condition)
        return v_field, condition


def vxm_mnist_generator(torch_dataloader, batch_size=32):
    data_shape = (2, 28, 28)

    for vfield, cond in torch_dataloader:
        zero_field = np.zeros_like(vfield)
        inputs = [cond[:, 0, ..., np.newaxis], cond[:, 1, ..., np.newaxis]]
        outputs = [cond[:, 1, ...], zero_field]

        yield inputs, outputs


class MnistSingleNumber(Dataset):

    def __init__(self, data, targets, transform=None):
        self.data = data / torch.max(data)
        self.targets = targets
        self.transform = transform


    def __len__(self):
        # print(self.data.shape[0])
        return len(self.data)

    def __getitem__(self, item):
        idx = random.sample(range(self.data.shape[0]), 2)
        source = self.data[idx[0]]
        target = self.data[idx[1]]
        if self.transform:
            source = self.transform(source)
            target = self.transform(target)

        cond = torch.stack((source, target), dim=0)
        return cond


class BirlData(IterableDataset):
    """
    Dataset for training on Birl data. we keep lung lesion 3, lung lobes 4 and mammary gland 2 for testing purposes.
    TODO implement landmark data usage.
    """
    def __init__(self, birl_path, sample_res=(64, 64), mode='train', color='grayscale', scale=5, types="all"):
        super(BirlData).__init__()
        self.data_path = birl_path
        self.mode = mode  # mode either train validation or test
        self.sample_res = sample_res
        self.images, self.csv_lists = load_birl(birl_path, scale=scale, mode=mode)
        self.color = color
        self.scale = scale
        if types == 'all':
            self.types = ['lung-lesion', 'lung-lobes', 'mammary-gland']
        elif type(types) == list:
            self.types = types
        elif type(types) == int:
            self.types = [self.types[types]]
        elif type(types) == str:
            self.types = [types]
        else:
            raise AttributeError("No valid types specified. Use types='all' for all image types, or select types with "
                                 "types=['lung-lesion', ...], types='lung-lesion', or types=0.")

    def __iter__(self):
        return self

    def __next__(self):
        # roll for image set
        if self.mode == 'test':
            image_type = torch.randint(3, (1,))
        else:
            image_type = torch.randint(6, (1,))
        # roll for images to build a pair
        image_nrs = torch.randint(5, (2,))
        # roll for pixels to crop from
        shape = self.images[image_type][0].size
        val_shape = [int(1.2 * x) for x in self.sample_res]
        train_shape = [int(x) for x in shape]
        if self.mode == 'validation':
            left = torch.randint(val_shape[0] - self.sample_res[0], (1,))
            upper = torch.randint(val_shape[1] - self.sample_res[1], (1,))
            right = left + self.sample_res[0]
            lower = upper + self.sample_res[1]
        elif self.mode == 'train':
            # left = torch.randint(low=val_shape[0], high=train_shape[0]-self.sample_res[0], size=(1,))
            # upper = torch.randint(low=val_shape[1], high=train_shape[1] - self.sample_res[1], size=(1,))
            left = torch.randint(low=0, high=train_shape[0]-self.sample_res[0], size=(1,))
            upper = torch.randint(low=0, high=train_shape[1] - self.sample_res[1], size=(1,))
            right = left + self.sample_res[0]
            lower = upper + self.sample_res[1]
        else:
            left = torch.randint(low=0, high=shape[0] - self.sample_res[0], size=(1,))
            upper = torch.randint(low=0, high=shape[1] - self.sample_res[0], size=(1,))
            right = left + self.sample_res[0]
            lower = upper + self.sample_res[1]
        area = (int(left), int(upper), int(right), int(lower))

        # crop image
        transform = ToTensor()
        if self.color == 'grayscale':
            ims = [transform(ImageOps.grayscale(self.images[image_type][nr].crop(area))) for nr in image_nrs]
        else:  # image == 'rgb'
            ims = [transform((self.images[image_type][nr].crop(area))) for nr in image_nrs]
        ims = [im - torch.mean(im) for im in ims]

        sample = torch.stack(ims)
        sample = torch.reshape(sample, (2, *self.sample_res))
        return sample


from torchvision.transforms import Pad
class FireDataset(Dataset):
    def __init__(self, path, size=182):
        self.dir_path = os.path.join(path, "FIRE")
        self.file_names = glob.glob(str(os.path.join(self.dir_path, f"Images/size_{size}/*.jpg")))
        # print(self.file_names)
        if size == 182:
            self.transform = Compose([Grayscale(), Pad(padding=1)])
        else:
            self.transform = Grayscale()

    def __len__(self):
        return int(len(self.file_names) / 2)

    def __getitem__(self, idx):
        src_idx = idx * 2
        tar_idx = src_idx + 1
        src_path = self.file_names[src_idx]
        tar_path = self.file_names[tar_idx]
        source = read_image(src_path)/255
        target = read_image(tar_path)/255
        if self.transform:
            source = self.transform(source)
            target = self.transform(target)
        label = self.file_names[src_idx][0]
        cond = torch.stack((source, target)).squeeze()
        return cond


class CovidXDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.file_names = glob.glob(os.path.join(self.path, "CovidXCT/2A_images/*"))
        self.transform = torchvision.transforms.Resize((512, 512))

    def __len__(self):
        print("data len", len(self.file_names))
        return len(self.file_names)//10

    def __getitem__(self, idx):
        src_idx = idx * 10
        if src_idx + 100 < len(self.file_names):
            tar_idx = src_idx + 100
        else:
            tar_idx = 0+src_idx+100 - len(self.file_names)
        src_path = self.file_names[src_idx]
        tar_path = self.file_names[tar_idx]
        source = read_image(src_path)/255
        target = read_image(tar_path)/255
        if self.transform:
            source = self.transform(source)
            target = self.transform(target)
        cond = torch.stack((source, target)).squeeze()
        return cond

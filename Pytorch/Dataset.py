import os
import skimage.io as io
import torch.utils.data as data
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        img, mask = img.resize((256, 256), resample=Image.BILINEAR), mask.resize((256, 256), resample=Image.BILINEAR)
        h, w = img.size
        new_h, new_w = self.size, self.size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        img = img.crop((left, top, left + new_w, top + new_h))
        mask = mask.crop((left, top, left + new_w, top + new_h))

        return {'image': img, 'mask': mask}


class RandomFlip(object):
    def __init__(self, prob):
        self.prob = prob
        self.flip = transforms.RandomHorizontalFlip(1.)

    def __call__(self, sample):
        if np.random.random_sample() < self.prob:
            img, mask = sample['image'], sample['mask']
            img = self.flip(img)
            mask = self.flip(mask)
            return {'image': img, 'mask': mask}
        else:
            return sample


class ToTensor(object):
    def __init__(self):
        self.tensor = transforms.ToTensor()

    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        img, mask = self.tensor(img), self.tensor(mask)
        return {'image': img, 'mask': mask}


class DUTS_dataset(data.Dataset):
    def __init__(self, root_dir, train=True, data_augmentation=True):
        self.image_list = sorted(os.listdir('{}/DUTS-{}-Image'.format(root_dir, 'TR' if train else 'TE')))
        self.mask_list = sorted(os.listdir('{}/DUTS-{}-Mask'.format(root_dir, 'TR' if train else 'TE')))
        self.transform = transforms.Compose(
            [RandomFlip(0.5),
             RandomCrop(224),
             ToTensor()])
        self.root_dir = root_dir
        self.train = train
        self.data_augmentation = data_augmentation

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        img_name = '{}/DUTS-{}-Image/{}'.format(self.root_dir, 'TR' if self.train else 'TE', self.image_list[item])
        mask_name = '{}/DUTS-{}-Mask/{}'.format(self.root_dir, 'TR' if self.train else 'TE', self.mask_list[item])
        img = Image.open(img_name)
        mask = Image.open(mask_name)
        img = img.convert('RGB')
        mask = mask.convert('L')
        sample = {'image': img, 'mask': mask}

        if self.data_augmentation:
            sample = self.transform(sample)
        return sample


if __name__ == '__main__':
    # os.chdir('..')
    # print(os.getcwd())
    ds = DUTS_dataset('../DUTS-TR')
    pil = transforms.ToPILImage()
    out = ds[0]
    pil(out['image']).show()
    pil(out['mask']).show()

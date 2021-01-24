import torch
import PIL
from PIL import Image
import os
import pandas as pd
import math
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
import torchvision.datasets as tvdataset
from datasets.tfs import get_cassava_transform
from sklearn.model_selection import GroupKFold, StratifiedKFold
import glob


import albumentations as A
from albumentations import Compose
from albumentations.pytorch import ToTensor
import pdb

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class ImageLoader(torch.utils.data.Dataset):

    def __init__(self, root, transform=None, target_transform=None, train=False, loader=pil_loader, tta=None,bbox=False,pdata=None):


        self.root = root
        self.imgs = pdata
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.train = train
        self.tta = tta
        self.isbbox = bbox

        print('num of data:{}'.format(len(self.imgs)))



    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        item = self.imgs.iloc[index]
        file_path = item['image_id']
        target = item['label']

        img = self.loader(os.path.join(self.root, file_path))

        if self.tta is None:
            img = self.transform(img)
        elif self.tta == 2:
            img_1 = self.transform(img)
            img_2 = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            img_2 = self.transform(img_2)
            img = torch.stack((img_1, img_2), dim=0)
        else:
            imgs = [self.transform(img) for i in range(self.tta)]
            img = torch.stack(imgs, dim=0)


        return img, target

    def __len__(self):
        return len(self.imgs)


def get_dataset(conf):

    datadir = 'data/cassava'

    if conf and 'datadir' in conf:
        datadir = conf.datadir
    conf['num_class'] = 5

    trainpd,valpd = None,None

    if conf.testing:
        imgdir = datadir + '/test_images'
        imgfile = glob.glob(imgdir+'/*.jpg')

    else:
        if 'foldid' in conf:
            traindata = pd.read_csv(datadir+'/train.csv')
            folds = StratifiedKFold(n_splits=5).split(np.arange(traindata.shape[0]), traindata.label.values)
            trainidx,validx = list(folds)[conf.foldid]
            trainpd = traindata.loc[trainidx,:].reset_index(drop=True)
            valpd = traindata.loc[validx,:].reset_index(drop=True)
        imgdir = datadir + '/train_images'
        transform_train,transform_test = get_cassava_transform(conf)
        ds_train = ImageLoader(imgdir, train=True, transform=transform_train,pdata=trainpd)

    if conf.tta is None or conf.tta==2:
        ds_test = ImageLoader(imgdir, train=False, transform=transform_test,pdata=valpd,tta=conf.tta)
    else:
        ds_test = ImageLoader(imgdir, train=False, transform=Compose([
            A.Resize(400, 300),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            A.augmentations.transforms.RGBShift (r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=0.5),
            #A.augmentations.transforms.ChannelDropout (channel_drop_range=(1, 1), fill_value=0, always_apply=False, p=0.5),
            A.augmentations.transforms.GridDistortion (num_steps=5, distort_limit=0.3, interpolation=1, border_mode=4, always_apply=False, p=0.5),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            # A.CoarseDropout(p=0.5),
            A.Cutout(p=0.5), 
            ToTensor()]),pdata=valpd,tta=conf.tta)


    return ds_train,ds_test

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
        ds_test = ImageLoader(imgdir, train=False, transform=transform_train,pdata=valpd,tta=conf.tta)


    return ds_train,ds_test

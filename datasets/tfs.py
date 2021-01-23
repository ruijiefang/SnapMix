
import torchvision.transforms as transforms
from utils import *

import albumentations as A
from albumentations import Compose
from albumentations.pytorch import ToTensor

resizedict = {'224':256,'448':512,'112':128}
#resizedict = {'224':256,'448':550,'112':128}

def get_coco_transform(conf=None):

    return get_voc_transform(conf)

def get_voc_transform(conf=None):

    resize = 256
    cropsize = 224


    if conf and 'cropsize' in conf:
        cropsize = conf.cropsize
        resize = resizedict[str(cropsize)]

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    if resize == cropsize:
        tflist = [transforms.RandomResizedCrop(cropsize)]
    else:
        tflist = [transforms.Resize(resize),transforms.RandomCrop(cropsize)]

    if 'commtfs' in conf:

        resize = 640
        cropsize = 576
        conf['resize'] = resize
        conf['cropsize'] = cropsize
        transform_train = transforms.Compose([transforms.Resize((resize, resize)),
                                               transforms.RandomChoice([transforms.RandomCrop(640),
                                               transforms.RandomCrop(576),
                                               transforms.RandomCrop(512),
                                               transforms.RandomCrop(384),
                                               transforms.RandomCrop(320)]),
                                               transforms.Resize((cropsize, cropsize)),
                                               transforms.ToTensor(),
                                               normalize])

        transform_test = transforms.Compose([transforms.Resize((resize, resize)),
                                             transforms.CenterCrop(cropsize),
                                             transforms.ToTensor(),
                                             normalize])
    else:

        transform_train = transforms.Compose(tflist + [
                transforms.RandomCrop(cropsize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize])

        transform_test = transforms.Compose([
                             transforms.Resize(resize),
                             transforms.CenterCrop(cropsize),
                             transforms.ToTensor(),
                             normalize
                             ])

    return transform_train,transform_test

def get_aircraft_transform(conf=None):
    return get_cub_transform(conf)

def get_cassava_transform(conf=None):
    return get_cub_transform(conf)

def get_car_transform(conf=None):
    return get_cub_transform(conf)

def get_mit67_transform(conf=None):
    return get_cub_transform(conf)

def get_fmd_transform(conf=None):
    return get_cub_transform(conf)

def get_gtos_transform(conf=None):
    return get_cub_transform(conf)

def get_nabirds_transform(conf=None):
    return get_cub_transform(conf)

def get_cub_transform_v1(conf=None):

    resize = 256
    cropsize = 224

    transform_train = transforms.Compose([
        transforms.Scale((550, 550)),
        transforms.RandomCrop(448, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])



    transform_test = transforms.Compose([
                            transforms.Scale((550, 550)),
                            transforms.CenterCrop(448),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

    return transform_train,transform_test
def get_cub_transform(conf=None):

    resize = (400, 300)
    #cropsize = 224


    #if conf and 'cropsize' in conf:
    #    cropsize = conf.cropsize
    #    resize = resizedict[str(cropsize)]

    #if 'warp' in conf:#
#
#        if conf.warp:
#            print('using warping')
#            resize = (resize,resize)
#            cropsize = (cropsize,cropsize)



    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    #if resize == cropsize:
    #    tflist = [transforms.RandomResizedCrop(cropsize)]
    #else:
    #    tflist = [transforms.Resize(resize),transforms.RandomCrop(cropsize)]
    custom_transforms = Compose([
        A.Resize(resize[0], resize[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        A.augmentations.transforms.RGBShift (r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=0.5),
        A.augmentations.transforms.ChannelDropout (channel_drop_range=(1, 1), fill_value=0, always_apply=False, p=0.5),
        A.augmentations.transforms.GridDistortion (num_steps=5, distort_limit=0.3, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        A.CoarseDropout(p=0.5),
        A.Cutout(p=0.5), 
        ToTensor()])
    #transform_train = transforms.Compose(tflist + [
    #            transforms.ToTensor(),
    #            normalize])
    transform_test = transforms.Compose([
                             transforms.Resize(resize),
                             #transforms.CenterCrop(cropsize),
                             transforms.ToTensor(),
                             normalize
                             ])

    return custom_transforms,transform_test



def get_cifar_transform(conf=None):

    cropsize = 32

    if  conf and 'cropsize' in conf:
        cropsize = conf.cropsize
    else:
        conf['cropsize'] = cropsize

    normalize = transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
        )

    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    return transform_train,transform_test




def get_imagenet_transform(conf=None):

    resize = 256
    cropsize = 224

    if conf and 'resize' in conf:
        resize = conf.resize

    if conf and 'cropsize' in conf:
        cropsize = conf.cropsize

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    if conf is not None and 'extra_tfs' in conf:

        jittering = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)

        lighting = Lighting(alphastd=0.1,
                            eigval=[0.2175, 0.0188, 0.0045],
                            eigvec=[[-0.5675, 0.7192, 0.4009],
                                    [-0.5808, -0.0045, -0.8140],
                                    [-0.5836, -0.6948, 0.4203]])

        transform_train = transforms.Compose([
                transforms.RandomResizedCrop(cropsize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                jittering,
                lighting,
                normalize,
            ])
    else:
        transform_train = transforms.Compose([
                transforms.RandomResizedCrop(cropsize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])


    transform_test = transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(cropsize),
                transforms.ToTensor(),
                normalize,
            ])


    return transform_train,transform_test

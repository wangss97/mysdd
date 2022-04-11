import os
from PIL import Image
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from torch.utils.data import DataLoader
import imgaug.augmenters as iaa

import cv2
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from perlin import *
import utils



# note: these are half-widths in [0, 0.5]
# ((h_min, h_max), (w_min, w_max))
WIDTH_BOUNDS_PCT = {'bottle':((0.03, 0.4), (0.03, 0.4)), 'cable':((0.05, 0.4), (0.05, 0.4)), 'capsule':((0.03, 0.15), (0.03, 0.4)), 
                    'hazelnut':((0.03, 0.35), (0.03, 0.35)), 'metal_nut':((0.03, 0.4), (0.03, 0.4)), 'pill':((0.03, 0.2), (0.03, 0.4)), 
                    'screw':((0.03, 0.12), (0.03, 0.12)), 'toothbrush':((0.03, 0.4), (0.03, 0.2)), 'transistor':((0.03, 0.4), (0.03, 0.4)), 
                    'zipper':((0.03, 0.4), (0.03, 0.2)), 
                    'carpet':((0.03, 0.4), (0.03, 0.4)), 'grid':((0.03, 0.4), (0.03, 0.4)), 
                    'leather':((0.03, 0.4), (0.03, 0.4)), 'tile':((0.03, 0.4), (0.03, 0.4)), 'wood':((0.03, 0.4), (0.03, 0.4))}

MIN_OVERLAP_PCT = {'bottle': 0.25,  'capsule':0.25, 
                   'hazelnut':0.25, 'metal_nut':0.25, 'pill':0.25, 
                   'screw':0.25, 'toothbrush':0.25, 
                   'zipper':0.25}

MIN_OBJECT_PCT = {'bottle': 0.7,  'capsule':0.7, 
                  'hazelnut':0.7, 'metal_nut':0.5, 'pill':0.7, 
                  'screw':.5, 'toothbrush':0.25, 
                  'zipper':0.7}

NUM_PATCHES = {'bottle':3, 'cable':3, 'capsule':3, 'hazelnut':3, 'metal_nut':3, 
               'pill':3, 'screw':4, 'toothbrush':3, 'transistor':3, 'zipper':4,
               'carpet':4, 'grid':4, 'leather':4, 'tile':4, 'wood':4}

# k, x0 pairs
INTENSITY_LOGISTIC_PARAMS = {'bottle':(1/12, 24), 'cable':(1/12, 24), 'capsule':(1/2, 4), 'hazelnut':(1/12, 24), 'metal_nut':(1/3, 7), 
            'pill':(1/3, 7), 'screw':(1, 3), 'toothbrush':(1/6, 15), 'transistor':(1/6, 15), 'zipper':(1/6, 15),
            'carpet':(1/3, 7), 'grid':(1/3, 7), 'leather':(1/3, 7), 'tile':(1/3, 7), 'wood':(1/6, 15)}

# bottle is aligned but it's symmetric under rotation
UNALIGNED_OBJECTS = ['bottle', 'hazelnut', 'metal_nut', 'screw']

# non-aligned objects get extra time
EPOCHS = {'bottle':320, 'cable':320, 'capsule':320, 'hazelnut':560, 'metal_nut':560, 
          'pill':320, 'screw':560, 'toothbrush':320, 'transistor':320, 'zipper':320,
          'carpet':320, 'grid':320, 'leather':320, 'tile':320, 'wood':320}

# brightness, threshold pairs
BACKGROUND = {'bottle':(200, 60), 'screw':(200, 60), 'capsule':(200, 60), 'zipper':(200, 60), 
              'hazelnut':(20, 20), 'pill':(20, 20), 'toothbrush':(20, 20), 'metal_nut':(20, 20)}


CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
OBJECTS = ['bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 
            'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
TEXTURES = ['carpet', 'grid', 'leather', 'tile', 'wood']


class MVTecTrainDataset_GOAD(Dataset):
    def __init__(self, data_dir='/root/test/wss/datasets/mvtec_anomaly_detection', category='bottle', pic_size=256,positive_aug_ratio=5,negative_aug_ratio=5):
        super(MVTecTrainDataset_GOAD, self).__init__()
        assert category in CLASS_NAMES, 'class_name: {}, should be in {}'.format(category, CLASS_NAMES)
        self.data_dir = data_dir
        self.category = category
        self.positive_aug_ratio = positive_aug_ratio
        self.negative_aug_ratio = negative_aug_ratio
        self.pic_size = pic_size

        self.resize_transform = T.Resize(self.pic_size)

        self.norm_transform = T.Compose([
                T.ToTensor(), 
                T.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])])  #ImageNet的均值和标准差, BGR

        # load dataset
        self.paths = self.load_dataset_folder()

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90,90))])

    
    def load_dataset_folder(self):
        x_paths = []

        img_dir = os.path.join(self.data_dir, self.category, 'train')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:
            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png')])
            x_paths.extend(img_fpath_list)

        return x_paths

    def get_augmented_negative(self, image):
        image_defect = np.array(image, dtype=np.uint8)
        mask = np.zeros([1,self.pic_size, self.pic_size], dtype=np.float32)
        label = 0

        return self.norm_transform(image), self.norm_transform(image_defect), torch.tensor(mask), label
    
    def get_augmented_positive(self, image):
        def _get_object_mask():
            src_object_mask = np.ones_like(image[...,0:1])
            if self.category in BACKGROUND.keys():
                background, threshold = BACKGROUND.get(self.category)
                src_object_mask &= np.uint8(np.abs(image.mean(axis=-1, keepdims=True) - background) > threshold)
                src_object_mask[...,0] = cv2.medianBlur(src_object_mask[...,0], 7)  # remove grain from threshold choice
                return src_object_mask
            else:
                return np.ones_like(image[...,0:1])

        def _get_perlin_mask():
            # 获得柏林噪声, 柏林噪声用于指定一幅图中哪些区域是正常区域, 哪些是缺陷区域
            perlin_scale = 6
            min_perlin_scale = 0
            perlin_scalex = 2 ** np.random.randint(min_perlin_scale, perlin_scale)
            perlin_scaley = 2 ** np.random.randint(min_perlin_scale, perlin_scale)
            perlin_noise = rand_perlin_2d_np((self.pic_size,self.pic_size), (perlin_scalex, perlin_scaley))  #噪声元素值范围是[-1,1], 噪声形状是self.pic_shape,二维
            perlin_noise = self.rot(image = perlin_noise)
            # 噪声二值化
            perlin_noise_bin = np.where(perlin_noise > 0.5, 1., 0.)
            perlin_noise_bin = perlin_noise_bin[:,:,np.newaxis].astype(np.float32)  #[h, w, 1]
            return perlin_noise_bin
        
        def _get_defect_source():
            # trans_matrix = np.random.normal(loc=0, scale=1,size=(2,3))
            trans_matrix = np.random.uniform(-0.5,0.5,size=(2,3))
            defect_source = cv2.warpAffine(image, trans_matrix, (self.pic_size,self.pic_size))

            defect_source = cv2.cvtColor(defect_source, cv2.COLOR_BGR2HLS)
            defect_source = defect_source.astype(np.float32)

            hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            hls_image = hls_image.astype(np.float32)

            h, l, s = np.random.random()-0.5,np.random.random()-0.5,np.random.random()-0.5
            h_mean, l_mean, s_mean = np.mean(hls_image[...,0]),np.mean(hls_image[...,1]),np.mean(hls_image[...,2])
            defect_source[...,0] += h*h_mean
            defect_source[...,1] += l*l_mean
            defect_source[...,2] += s*s_mean
            defect_source = np.clip(defect_source, 0, 255).astype(np.uint8)
            defect_source = cv2.cvtColor(defect_source, cv2.COLOR_HLS2BGR)
            return defect_source

        object_mask = _get_object_mask()
        perlin_mask = _get_perlin_mask()
        defect_source = _get_defect_source()
        mask = object_mask * perlin_mask

        image = image.astype(np.float32)
        defect_source = defect_source.astype(np.float32)
        cv2.imwrite(f"./tmp/source.jpg",np.squeeze(defect_source).astype(np.uint8))

        
        if np.all(mask==0):
            return self.norm_transform(image.astype(np.uint8)),self.norm_transform(image.astype(np.uint8)), torch.tensor(mask).permute((2,0,1)), 0

        image_norm_part = image * (1-mask)
        image_defect_part = image * mask
        defect = defect_source * mask

        beta = np.random.uniform(0,0.5)
        defect_image = image_norm_part + image_defect_part * beta + defect* (1-beta)
        
        return self.norm_transform(image.astype(np.uint8)), self.norm_transform(defect_image.astype(np.uint8)), torch.tensor(mask).permute((2,0,1)), 1

    def __len__(self):
        return len(self.paths) * (self.positive_aug_ratio + self.negative_aug_ratio)

    def __getitem__(self, idx):
        length = len(self.paths)
        negative_length = length * self.negative_aug_ratio
        sample_idx = idx % length

        imgPath = self.paths[sample_idx]
        img = cv2.imread(imgPath, flags=cv2.IMREAD_COLOR)
        img = cv2.resize(img, (self.pic_size, self.pic_size))

        if idx < negative_length:
            image, image_defect, mask, label = self.get_augmented_negative(img)
        else:
            image, image_defect, mask, label = self.get_augmented_positive(img)
        
        return image, image_defect, mask, label, imgPath



if __name__ == '__main__':

    class_name = 'bottle'

    dataset = MVTecTrainDataset_GOAD(data_dir='/root/test/wss/datasets/mvtec_anomaly_detection/', category=class_name,positive_aug_ratio=1,negative_aug_ratio=0)

    images = []
    for i in range(1):
        image, image_def, mask, label, path = dataset.__getitem__(i)
        image = image_def.permute(1,2,0).unsqueeze(0).numpy()
        print(image.shape)
        images.append(image)

    images = np.concatenate(images, axis=0)
    print(images.shape)
    images = utils.toImg(images)
    for i, image in enumerate(images):
        # cv2.imwrite(f"./tmp/img{i}.jpg",image)
        pass
        

    quit()

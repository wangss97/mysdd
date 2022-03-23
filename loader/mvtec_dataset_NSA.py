import os
import tarfile
from PIL import Image
from tqdm import tqdm
import urllib.request
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from torch.utils.data import DataLoader
import cv2

from loader.self_sup_tasks import patch_ex

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


class MVTecTestDataset_NSA(Dataset):
    def __init__(self, data_dir='/root/test/wss/datasets/mvtec_anomaly_detection', pic_size=256, categroy='bottle') -> None:
        super(MVTecTestDataset_NSA, self).__init__()
        self.pic_shape = (pic_size, pic_size)
        self.categroy = categroy
        self.data_dir = data_dir
        self.data_list, self.mask_list = self.get_data_list()

        positive_count, negative_count, defect_type, count_perType = self.get_statistics()
        print("datasets info:")
        print(f"categroy:{self.categroy}, positive count:{positive_count}, negative count:{negative_count}")
        for i in range(len(defect_type)):
            print(f"{defect_type[i]}:{count_perType[i]} ", end='')
        print()

        if categroy in UNALIGNED_OBJECTS:
            self.crop_transform = T.Compose([
                    T.RandomRotation(5),
                    T.CenterCrop(230), 
                    T.RandomCrop(224),
                    T.Resize(pic_size)])
        elif categroy in OBJECTS:
            # no rotation for aligned objects
            self.crop_transform = T.Compose([
                    T.CenterCrop(230),
                    T.RandomCrop(224),
                    T.Resize(pic_size)])
        else:  # texture
            self.crop_transform = T.Compose([
                    T.RandomVerticalFlip(), 
                    T.RandomCrop(256)])

        self.resize_transform = T.Resize(pic_size)

        self.norm_transform = T.Compose([
                T.ToTensor(), 
                T.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])])  #ImageNet的均值和标准差, BGR

    def __len__(self):
        return len(self.data_list)

    def get_statistics(self):
        ''' 获得数据集数量信息 '''
        positive_count = len([item for item in self.data_list if 'good' in item])
        negative_count = len(self.data_list) - positive_count
        defect_type = np.unique([str.split(item,'/')[-2] for item in self.data_list])
        count_perType = []
        for type in defect_type:
            count_perType.append(len([item for item in self.data_list if str.split(item,'/')[-2]==type]))
        return positive_count, negative_count, defect_type, count_perType

    def get_data_list(self):
        ''' 获得数据路径列表 和 像素标签路径列表 '''
        test_dir = os.path.join(self.data_dir, self.categroy, 'test')
        groundtruth_dir = os.path.join(self.data_dir, self.categroy, 'ground_truth')
        defecttype_dirs = os.listdir(test_dir)
        data_list = []
        mask_list = []
        for tdir in defecttype_dirs:
            data_fdir = os.path.join(test_dir, tdir)
            type_data_list = os.listdir(data_fdir)
            data_list += [os.path.join(data_fdir, item) for item in type_data_list]

            label_fdir = os.path.join(groundtruth_dir, tdir)
            type_mask_list = [item[:-4]+'_mask'+item[-4:] for item in type_data_list]
            mask_list += [os.path.join(label_fdir, item) for item in type_mask_list]
        return data_list, mask_list
    
    def __getitem__(self, idx):
        ''' 测试时,返回样本， 像素标签， 图片标签， 图片路径， 像素标签路径 '''
        image_path = self.data_list[idx]
        mask_path = self.mask_list[idx]

        image = self.resize_transform(Image.open(image_path).convert('RGB'))
        image = self.crop_transform(image)
        image = np.asarray(image)[...,::-1].copy()  #BGR
        image = self.norm_transform(image)

        defecttype = str.split(mask_path, '/')[-2]
        if defecttype == 'good':
            mask = torch.zeros([1,self.pic_shape[0],self.pic_shape[1]])
            label = 0
        else:
            # 转换为二值图像
            mask = self.resize_transform(Image.open(mask_path).convert('1'))
            mask = self.crop_transform(mask)
            mask = np.asarray(mask)
            mask = np.where(mask < 0.5, 0., 1.)
            mask = torch.tensor(mask[np.newaxis, ...])
            label = 1
        return image, mask, label, image_path


class MVTecTrainDataset_NSA(Dataset):
    def __init__(self, data_dir='/root/test/wss/datasets/mvtec_anomaly_detection', categroy='bottle', pic_size=256,positive_aug_ratio=5,negative_aug_ratio=5, self_sup_args={}):
        super(MVTecTrainDataset_NSA, self).__init__()
        assert categroy in CLASS_NAMES, 'class_name: {}, should be in {}'.format(categroy, CLASS_NAMES)
        self.data_dir = data_dir
        self.class_name = categroy
        self.positive_aug_ratio = positive_aug_ratio
        self.negative_aug_ratio = negative_aug_ratio
        self.low_res = pic_size
        self.self_sup_args = self_sup_args

        self.self_sup_args.update({'gamma_params':(2, 0.05, 0.03), 'resize':True, 
                        #    'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'logistic-intensity'})
                           'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'binary'})
        if self.class_name in TEXTURES:
            self.self_sup_args.update({'resize_bounds': (.5, 2)})
        self.self_sup_args.update({'width_bounds_pct': WIDTH_BOUNDS_PCT.get(self.class_name),
                                                    'intensity_logistic_params': INTENSITY_LOGISTIC_PARAMS.get(self.class_name),
                                                    'num_patches': NUM_PATCHES.get(self.class_name),
                                                    'min_object_pct': MIN_OBJECT_PCT.get(self.class_name),
                                                    'min_overlap_pct': MIN_OVERLAP_PCT.get(self.class_name)})
        # set transforms
        # load data
        if self.class_name in UNALIGNED_OBJECTS:
            self.crop_transform = T.Compose([
                    T.RandomRotation(5),
                    T.CenterCrop(230), 
                    T.RandomCrop(224),
                    T.Resize(self.low_res)])
        elif self.class_name in OBJECTS:
            # no rotation for aligned objects
            self.crop_transform = T.Compose([
                    T.CenterCrop(230),
                    T.RandomCrop(224),
                    T.Resize(self.low_res)])
        else:  # texture
            self.crop_transform = T.Compose([
                    T.RandomVerticalFlip(), 
                    T.RandomCrop(256)])

        self.resize_transform = T.Resize(self.low_res)

        self.norm_transform = T.Compose([
                T.ToTensor(), 
                T.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])])  #ImageNet的均值和标准差, BGR


        # load dataset
        self.x, self.paths = self.load_dataset_folder()

        self.prev_idx = np.random.randint(len(self.x))

    
    def load_dataset_folder(self):
        x_paths = []

        img_dir = os.path.join(self.data_dir, self.class_name, 'train')

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

        xs = []
        for path in x_paths:
            xs.append(self.resize_transform(Image.open(path).convert('RGB'))) 

        return list(xs), x_paths

    def get_augmented_negative(self, image):
        image = self.crop_transform(image)
        image = np.asarray(image)[...,::-1].copy()    #转换成BGR
        image_defect = np.array(image, dtype=np.uint8)
        mask = np.zeros([1,self.low_res, self.low_res], dtype=np.float32)
        label = 0

        return self.norm_transform(image), self.norm_transform(image_defect), torch.tensor(mask), label
    
    def get_augmented_positive(self, image, defect_source):
        image = self.crop_transform(image)
        defect_source = self.crop_transform(defect_source)
        image = np.asarray(image)
        defect_source = np.asarray(defect_source)

        image_defect, mask = patch_ex(image, defect_source, **self.self_sup_args)

        image = image[...,::-1].copy()  #BGR
        image_defect = image_defect[...,::-1].copy()  #BGR
        mask = torch.tensor(mask[None, ..., 0]).float()
        label = 1
        
        return self.norm_transform(image), self.norm_transform(image_defect), mask, label

    def __len__(self):
        return len(self.x) * (self.positive_aug_ratio +self.negative_aug_ratio)

    def __getitem__(self, idx):
        length = len(self.x)
        negative_length = length * self.negative_aug_ratio
        sample_idx = idx % length

        x, imgPath = self.x[sample_idx], self.paths[sample_idx]
        defect_source = self.x[self.prev_idx]

        if idx < negative_length:
            image, image_defect, mask, label = self.get_augmented_negative(x)
        else:
            image, image_defect, mask, label = self.get_augmented_positive(x, defect_source)

        self.prev_idx = sample_idx
        
        return image, image_defect, mask, label, imgPath



if __name__ == '__main__':

    class_name = 'cable'

    


    train_dat = MVTecTrainDataset_NSA(root_path='/root/test/wss/datasets/', class_name=class_name)

    
    
    for i in range(100):
        data = train_dat.__getitem__(i)

    quit()
    loader_train = DataLoader(train_dat, 1, shuffle=True, num_workers=os.cpu_count(),
                              worker_init_fn=lambda _: np.random.seed(torch.utils.data.get_worker_info().seed % 2**32))

    for item in loader_train:
        pass
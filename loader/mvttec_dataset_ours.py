import os
import tarfile
from PIL import Image
from tqdm import tqdm
import urllib.request
import numpy as np
import imgaug.augmenters as iaa


import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from torch.utils.data import DataLoader
import cv2

from .self_sup_tasks import patch_aug



CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
OBJECTS = ['bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 
            'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
TEXTURES = ['carpet', 'grid', 'leather', 'tile', 'wood']


class  MVTecTestDataset_ours(Dataset):
    def __init__(self, pic_size=256, category='bottle') -> None:
        super(MVTecTestDataset_ours, self).__init__()
        self.pic_shape = (pic_size, pic_size)
        self.category = category
        self.data_dir = '/root/test/wss/datasets/mvtec_anomaly_detection'
        self.data_list, self.mask_list = self.get_data_list()

        positive_count, negative_count, defect_type, count_perType = self.get_statistics()
        print("datasets info:")
        print(f"category:{self.category}, positive count:{positive_count}, negative count:{negative_count}")
        for i in range(len(defect_type)):
            print(f"{defect_type[i]}:{count_perType[i]} ", end='')
        print()


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
        test_dir = os.path.join(self.data_dir, self.category, 'test')
        groundtruth_dir = os.path.join(self.data_dir, self.category, 'ground_truth')
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
        ''' 测试时,返回样本， 像素标签， 图片标签， 图片路径 '''
        image_path = self.data_list[idx]
        mask_path = self.mask_list[idx]

        image = self.resize_transform(Image.open(image_path).convert('RGB'))
        image = np.asarray(image)[...,::-1].copy()  #BGR
        image = self.norm_transform(image)

        defecttype = str.split(mask_path, '/')[-2]
        if defecttype == 'good':
            mask = torch.zeros([1,self.pic_shape[0],self.pic_shape[1]])
            label = 0
        else:
            # 转换为二值图像
            mask = self.resize_transform(Image.open(mask_path).convert('1'))
            mask = np.asarray(mask)
            mask = np.where(mask < 0.5, 0., 1.)
            mask = torch.tensor(mask[np.newaxis, ...])
            label = 1
        return image, mask, label, image_path


class MVTecTrainDataset_ours(Dataset):
    def __init__(self, pic_size=256, category='bottle', positive_aug_ratio=2,negative_aug_ratio=2):
        super(MVTecTrainDataset_ours, self).__init__()
        assert category in CLASS_NAMES, 'class_name: {}, should be in {}'.format(category, CLASS_NAMES)
        self.data_dir = '/root/test/wss/datasets/mvtec_anomaly_detection'
        self.class_name = category
        self.positive_aug_ratio = positive_aug_ratio
        self.negative_aug_ratio = negative_aug_ratio
        self.low_res = pic_size

        # set transforms
        # load data

        self.resize_transform = T.Resize(self.low_res)

        self.norm_transform = T.Compose([
                T.ToTensor(), 
                T.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])])  #ImageNet的均值和标准差, BGR


        # load dataset
        self.x, self.paths = self.load_dataset_folder()

        self.prev_idx = np.random.randint(len(self.x))

        self.negativa_augmenters = iaa.SomeOf(
            (0,None), [
                iaa.Affine(rotate=(-10,10), mode='edge'),
                iaa.Affine(scale=(0.95,1.05), mode='edge'),
                iaa.Affine(translate_percent={'x':(-0.05,0.05),'y':(-0.05,0.05)}, mode='edge')
            ], random_order=True
        )

    
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
        image = np.asarray(image)[...,::-1].copy()    #转换成BGR
        image = self.negativa_augmenters(image = image)
        image_defect = np.array(image, dtype=np.uint8)
        mask = np.zeros([1,self.low_res, self.low_res], dtype=np.float32)
        label = 0

        return self.norm_transform(image), self.norm_transform(image_defect), torch.tensor(mask), label
    
    def get_augmented_positive(self, image, defect_source):
        image = np.asarray(image)
        defect_source = np.asarray(defect_source)
        image = image[...,::-1].copy()  #BGR
        defect_source = defect_source[...,::-1].copy()  #BGR
        
        image_defect, mask, label = patch_aug(image, defect_source)


        mask = torch.tensor(mask).float().permute(2,0,1)
        
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
    from utils import imshow

    def TensortoImg(norm_img:torch.Tensor):
        # height, width, channel   BGR
        norm_img = norm_img.permute(1,2,0)
        mean=[0.406, 0.456, 0.485]
        std=[0.225, 0.224, 0.229]
        for i in range(3):
            norm_img[:,:,i] = (norm_img[:,:,i]*std[i]+mean[i]) * 255
        norm_img = norm_img.numpy().astype(np.uint8)
        return norm_img
    
    class_name = 'cable'

    test_data = MVTecTrainDataset_NSA(category='cable', negative_aug_ratio=0, positive_aug_ratio=1)
    image, image_defect, mask, label, imgPath = test_data.__getitem__(0)
    image_defect = TensortoImg(image_defect)
    mask = mask.permute(1,2,0).numpy()*255
    mask = mask.astype(np.uint8)
    
    # imshow([image_defect, mask])
    



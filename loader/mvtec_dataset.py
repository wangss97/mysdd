import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import os
import cv2
import numpy as np
import imgaug.augmenters as iaa
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import utils
from perlin import *


class MVTecTestDataset(Dataset):
    def __init__(self, pic_size, categroy) -> None:
        super(MVTecTestDataset, self).__init__()
        self.pic_shape = (pic_size, pic_size)
        self.categroy = categroy
        self.data_dir = '/root/test/wss/datasets/mvtec_anomaly_detection'
        self.data_list, self.mask_list = self.get_data_list()

        positive_count, negative_count, defect_type, count_perType = self.get_statistics()
        print("datasets info:")
        print(f"categroy:{self.categroy}, positive count:{positive_count}, negative count:{negative_count}")
        for i in range(len(defect_type)):
            print(f"{defect_type[i]}:{count_perType[i]} ", end='')
        print()

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
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, self.pic_shape)
        image = image.astype(np.uint8)
        image = T.ToTensor()(image)

        mask_path = self.mask_list[idx]
        defecttype = str.split(mask_path, '/')[-2]
        if defecttype == 'good':
            mask = torch.zeros([1,self.pic_shape[0],self.pic_shape[1]])
            label = 0
        else:
            # 转换为二值图像
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.pic_shape)
            mask = np.where(mask>128, 255, 0).astype(np.uint8)
            mask = T.ToTensor()(mask)
            label = 1
        
        return image, mask, label, image_path

class MVTecTrainDataset(Dataset):
    def __init__(self, pic_size, categroy, positive_aug_ratio, negative_aug_ratio) -> None:
        super(MVTecTrainDataset, self).__init__()
        self.positive_aug_ratio = positive_aug_ratio
        self.negative_aug_ratio = negative_aug_ratio
        self.pic_shape = (pic_size, pic_size)
        self.categroy = categroy
        self.data_dir = '/root/test/wss/datasets/mvtec_anomaly_detection'
        self.data_list = self.get_train_data_list()
        self.defect_source_list = self.get_defect_source_list()

        # 应用于缺陷源图片dtd的图像增广方式
        self.positive_augmenters = iaa.SomeOf(3, [
                    iaa.GammaContrast((0.5,2.0),per_channel=True),
                    iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                    iaa.pillike.EnhanceSharpness(),
                    iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                    iaa.Solarize(0.5, threshold=(32,128)),
                    iaa.Posterize(),
                    iaa.Invert(),
                    iaa.pillike.Autocontrast(),
                    iaa.pillike.Equalize(),
                    iaa.Affine(rotate=(-45, 45))
                ], random_order=True
        ) 
        # 应用于正常样本的图像增广方式,增广后的图片依然是正常样本
        self.negativa_augmenters = iaa.SomeOf(
            (0,None), [
                iaa.Affine(rotate=(-15,15), mode='edge'),
                iaa.Affine(scale=(0.9,1.2), mode='edge'),
                iaa.Affine(translate_percent={'x':(-0.1,0.1),'y':(-0.1,0.1)}, mode='edge')
            ], random_order=True
        )

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90,90))])

        positive_count, negative_count, defect_type, count_perType = self.get_statistics()
        print("datasets info:")
        print(f"categroy:{self.categroy}, positive count:{positive_count}, negative count:{negative_count}")
        for i in range(len(defect_type)):
            print(f"{defect_type[i]}:{count_perType[i]} ", end='')
        print()

    def __len__(self):
        # return len(self.data_list)
        return len(self.data_list) * (self.positive_aug_ratio +self.negative_aug_ratio)

    def get_statistics(self):
        positive_count = 0
        negative_count = len(self.data_list)
        defect_type = np.unique([str.split(item,'/')[-2] for item in self.data_list])
        count_perType = []
        for type in defect_type:
            count_perType.append(len([item for item in self.data_list if str.split(item,'/')[-2]==type]))
        return positive_count, negative_count, defect_type, count_perType

    def get_train_data_list(self):
        fdir = os.path.join(self.data_dir, self.categroy, 'train/good')
        data_list = os.listdir(fdir)
        data_list = [os.path.join(fdir, item) for item in data_list]
        return data_list
    
    def get_defect_source_list(self):
        fdir = '/root/test/wss/datasets/dtd/images'
        return glob.glob(fdir+'/*/*.jpg')

    def get_augmented_positive(self, image):
        image = self.negativa_augmenters(image = image)

        # 获得缺陷来源图, 随机选择一个
        defect_source = cv2.imread(np.random.choice(self.defect_source_list), cv2.IMREAD_COLOR)
        defect_source = cv2.resize(defect_source, self.pic_shape)
        defect_source = self.positive_augmenters(image = defect_source)

        # 获得柏林噪声, 柏林噪声用于指定一幅图中哪些区域是正常区域, 哪些是缺陷区域
        perlin_scale = 6
        min_perlin_scale = 0
        perlin_scalex = 2 ** np.random.randint(min_perlin_scale, perlin_scale)
        perlin_scaley = 2 ** np.random.randint(min_perlin_scale, perlin_scale)
        perlin_noise = rand_perlin_2d_np(self.pic_shape, (perlin_scalex, perlin_scaley))  #噪声元素值范围是[-1,1], 噪声形状是self.pic_shape,二维
        perlin_noise = self.rot(image = perlin_noise)
        # 噪声二值化
        perlin_noise_bin = np.where(perlin_noise > 0.5, 1., 0.)
        perlin_noise_bin = perlin_noise_bin[:,:,np.newaxis].astype(np.float32)  #[h, w, 1]

        # 原图正常部分``````````````````````
        image_norm_part = np.array(image,dtype=np.float32) * (1-perlin_noise_bin)

        # 原图缺陷部分
        image_defect_part = np.array(image, dtype=np.float32) * perlin_noise_bin

        # 透光率
        beta = np.random.rand() * 0.8  # 保证合成时缺陷更明显一点

        # 缺陷来源图的缺陷部分
        defect_source_defect_part = defect_source * perlin_noise_bin

        defect_image = image_norm_part + beta * image_defect_part + (1-beta) * defect_source_defect_part
        defect_image = np.array(defect_image, dtype=np.uint8)

        if 1. in perlin_noise_bin:
            label = 1
        else:
            label = 0

        transformer = T.ToTensor()

        return transformer(image), transformer(defect_image), torch.tensor(perlin_noise_bin).permute((2,0,1)), label

    def get_augmented_negative(self, image):
        image = self.negativa_augmenters(image = image)
        defect_image = np.array(image, dtype=np.uint8)
        mask = np.zeros([1,self.pic_shape[0], self.pic_shape[1]], dtype=np.float32)
        label = 0

        transformer = T.ToTensor()

        return transformer(image), transformer(defect_image), torch.tensor(mask), label

    
    def __getitem__(self, idx):
        # data_list的长度是length，idx的范围是[0,negative_aug_ratio*length + positive_aug_ratio*length),
        # 当idx在[0,negative_aug_ratio*length)内时，返回正常样本；当idx在[negative_aug_ratio*length, negative_aug_ratio*length + positive_aug_ratio*length)内时，返回缺陷样本

        length = len(self.data_list)
        negative_length = length * self.negative_aug_ratio
        sample_idx = idx % length
        image_path = self.data_list[sample_idx]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, self.pic_shape)
        
        if idx < negative_length:
            image, defect_image, mask, label = self.get_augmented_negative(image)
        else:
            image, defect_image, mask, label = self.get_augmented_positive(image)
     
        return image, defect_image, mask, label, image_path



if __name__ == '__main__':
    # aug = iaa.Affine(rotate=(-180,180), mode='constant')
    # aug = iaa.Affine(scale=(0.9,1.2), mode='edge')
    # aug = iaa.Affine(translate_percent={'x':(-0.1,0.1),'y':(-0.1,0.1)}, mode='edge')

    # pic = cv2.imread('/root/test/wss/datasets/mvtec_anomaly_detection/bottle/train/good/100.png', 1)
    # for i in range(10):
    #     utils.visualize([pic, aug(image=pic)], './tmp', f'tmp{i}.jpg')

    perlin_scale = 6
    min_perlin_scale = 0
    perlin_scalex = 2 ** np.random.randint(min_perlin_scale, perlin_scale)
    perlin_scaley = 2 ** np.random.randint(min_perlin_scale, perlin_scale)
    perlin_noise = rand_perlin_2d_np([128,128], (perlin_scalex, perlin_scaley))
    perlin_noise = perlin_noise*128 + 128
    cv2.imwrite('./tmp/tmp.jpg', perlin_noise)


    defect_source_list = glob.glob('/root/test/wss/datasets/dtd/images/*/*.jpg')
    pic_shape = [512,512]

    # 应用于缺陷源图片dtd的图像增广方式
    positive_augmenters = iaa.SomeOf(3, [
                iaa.GammaContrast((0.5,2.0),per_channel=True),
                iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                iaa.pillike.EnhanceSharpness(),
                iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                iaa.Solarize(0.5, threshold=(32,128)),
                iaa.Posterize(),
                iaa.Invert(),
                iaa.pillike.Autocontrast(),
                iaa.pillike.Equalize(),
                iaa.Affine(rotate=(-45, 45))
            ], random_order=True
    ) 
    # 应用于正常样本的图像增广方式,增广后的图片依然是正常样本
    negativa_augmenters = iaa.SomeOf(
        (0,None), [
            iaa.Affine(rotate=(-180,180), mode='edge'),
            iaa.Affine(scale=(0.9,1.2), mode='edge'),
            iaa.Affine(translate_percent={'x':(-0.1,0.1),'y':(-0.1,0.1)}, mode='edge')
        ], random_order=True
    )

    rot = iaa.Sequential([iaa.Affine(rotate=(-90,90))])

    def get_augmented_positive( image):
        image = negativa_augmenters(image = image)

        # 获得缺陷来源图, 随机选择一个
        defect_source = cv2.imread(np.random.choice(defect_source_list), cv2.IMREAD_COLOR)
        defect_source = cv2.resize(defect_source, pic_shape)
        defect_source = positive_augmenters(image = defect_source)

        # 获得柏林噪声, 柏林噪声用于指定一幅图中哪些区域是正常区域, 哪些是缺陷区域
        perlin_scale = 6
        min_perlin_scale = 0
        perlin_scalex = 2 ** np.random.randint(min_perlin_scale, perlin_scale)
        perlin_scaley = 2 ** np.random.randint(min_perlin_scale, perlin_scale)
        perlin_noise = rand_perlin_2d_np(pic_shape, (perlin_scalex, perlin_scaley))  #噪声元素值范围是[-1,1], 噪声形状是pic_shape,二维
        perlin_noise = rot(image = perlin_noise)
        # 噪声二值化
        perlin_noise_bin = np.where(perlin_noise > 0.5, 1., 0.)
        perlin_noise_bin = perlin_noise_bin[:,:,np.newaxis].astype(np.float32)  #[h, w, 1]

        # 原图正常部分``````````````````````
        image_norm_part = np.array(image,dtype=np.float32) * (1-perlin_noise_bin)

        # 原图缺陷部分
        image_defect_part = np.array(image, dtype=np.float32) * perlin_noise_bin

        # 透光率
        beta = np.random.rand() * 0.8  # 保证合成时缺陷更明显一点

        # 缺陷来源图的缺陷部分
        defect_source_defect_part = defect_source * perlin_noise_bin

        defect_image = image_norm_part + beta * image_defect_part + (1-beta) * defect_source_defect_part
        defect_image = np.array(defect_image, dtype=np.uint8)

        if 1. in perlin_noise_bin:
            label = 1
        else:
            label = 0

        transformer = T.ToTensor()

        return image, defect_image, perlin_noise_bin, label

    pic = cv2.imread('/root/test/wss/datasets/mvtec_anomaly_detection/bottle/train/good/100.png', 1)
    pic = cv2.resize(pic, (512,512))
    for i in range(10):
        image, defect, mask, label = get_augmented_positive(pic)
        utils.visualize([image, defect, mask*255], './tmp', f'tmp{i}.jpg')

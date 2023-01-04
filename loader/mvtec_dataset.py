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
from loader.mvtec_dataset_NSA import BACKGROUND
import utils
from perlin import *


class MVTecDataset_clf(Dataset):
    def __init__(self, pic_size, category, train_flag) -> None:
        super(MVTecDataset_clf, self).__init__()
        self.train_flag = train_flag
        self.pic_shape = (pic_size, pic_size)
        self.category = category
        self.data_dir = '/root/test/wss/datasets/mvtec_anomaly_detection'
        self.train_data_list, self.test_data_list = self.get_data_list()
        self.train_neg = [item for item in self.train_data_list if 'good' in item]
        self.train_pos = [item for item in self.train_data_list if 'good' not in item]

        positive_count, negative_count, defect_type, count_perType = self.get_statistics()
        print("datasets info:")
        print(f"category:{self.category}, positive count:{positive_count}, negative count:{negative_count}")
        for i in range(len(defect_type)):
            print(f"{defect_type[i]}:{count_perType[i]} ", end='')
        print()

        self.negativa_augmenters = iaa.SomeOf(
            (0,None), [
                iaa.Affine(rotate=(-4,4), mode='edge'),
                iaa.Affine(scale=(0.98,1.02), mode='edge'),
                iaa.Affine(translate_percent={'x':(-0.02,0.02),'y':(-0.02,0.02)}, mode='edge')
            ], random_order=True
        )

    def __len__(self):
        if self.train_flag:
            return len(self.train_neg) + len(self.train_pos) * 2
        else:
            return len(self.test_data_list)

    def get_statistics(self):
        ''' 获得数据集数量信息 '''
        if self.train_flag:
            data_list = self.train_data_list
        else:
            data_list = self.test_data_list
        positive_count = len([item for item in data_list if 'good' in item])
        negative_count = len(data_list) - positive_count
        defect_type = np.unique([str.split(item,'/')[-2] for item in data_list])
        count_perType = []
        for type in defect_type:
            count_perType.append(len([item for item in data_list if str.split(item,'/')[-2]==type]))
        return positive_count, negative_count, defect_type, count_perType


    def get_data_list(self):
        ''' 获得数据路径列表 和 像素标签路径列表 '''
        train_data_list = []
        test_data_list = []

        test_dir = os.path.join(self.data_dir, self.category, 'test')
        defecttype_dirs = os.listdir(test_dir)
        for tdir in defecttype_dirs:
            data_fdir = os.path.join(test_dir, tdir)
            type_data_list = os.listdir(data_fdir)
            type_data_list.sort()
            train_data_list += [os.path.join(data_fdir, item) for item in type_data_list[:len(type_data_list)//2]]
            test_data_list += [os.path.join(data_fdir, item) for item in type_data_list[len(type_data_list)//2:]]

        
        norm_dir = os.path.join(self.data_dir, self.category, 'train','good')
        norm_data_list = os.listdir(norm_dir)
        norm_data_list.sort()
        train_data_list += [os.path.join(norm_dir, item) for item in norm_data_list[:len(norm_data_list)//2]]
        test_data_list += [os.path.join(norm_dir, item) for item in norm_data_list[len(norm_data_list)//2:]]
        return train_data_list, test_data_list
    
    def __getitem__(self, idx):
        ''' 测试时,返回样本， 像素标签， 图片标签， 图片路径， 像素标签路径 '''
        transformer = T.Compose([
                T.ToTensor(), 
                T.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])])   #BGR

        if self.train_flag:
            if idx < len(self.train_neg):
                image_path = self.train_neg[idx]
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                image = cv2.resize(image, self.pic_shape)
                label = 0
            else:
                idx = idx - len(self.train_neg)
                nidx = idx % len(self.train_pos)
                image_path = self.train_pos[nidx]
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                image = cv2.resize(image, self.pic_shape)
                label = 1
                if idx > len(self.train_pos):
                    image = self.negativa_augmenters(image = image)
            image = transformer(image)
            return image, label, image_path
        else:
            image_path = self.test_data_list[idx]
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, self.pic_shape)
            image = transformer(image)
            label = 0 if 'good' in image_path else 1
            return image, label, image_path

class MVTecDataset_tsne(Dataset):
    def __init__(self, pic_size, category, method) -> None:
        super(MVTecDataset_tsne, self).__init__()
        self.pic_shape = (pic_size, pic_size)
        self.category = category
        self.method = method
        if method == 'ori':
            self.data_dir = '/root/test/wss/datasets/mvtec_anomaly_detection'
        elif method == 'ours':
            self.data_dir = '/root/test/wss/mysdd/vis/mvtec'
        elif method == 'draem':
            self.data_dir = '/root/test/wss/workspace/DRAEM/vis'
        elif method == 'riad':
            self.data_dir = '/root/test/wss/workspace/RIAD/vis'
        self.data_list = self.get_data_list()
        self.neg = [item for item in self.data_list if 'good' in item]
        self.pos = [item for item in self.data_list if 'good' not in item]

        positive_count, negative_count, defect_type, count_perType = self.get_statistics()
        print("datasets info:")
        print(f"category:{self.category}, positive count:{positive_count}, negative count:{negative_count}")
        for i in range(len(defect_type)):
            print(f"{defect_type[i]}:{count_perType[i]} ", end='')
        print()

    def __len__(self):
            return len(self.data_list)

    def get_statistics(self):
        ''' 获得数据集数量信息 '''
        data_list = self.data_list
        positive_count = len([item for item in data_list if 'good' not in item])
        negative_count = len(data_list) - positive_count
        defect_type = np.unique([str.split(item,'/')[-2] for item in data_list])
        count_perType = []
        for type in defect_type:
            count_perType.append(len([item for item in data_list if str.split(item,'/')[-2]==type]))
        return positive_count, negative_count, defect_type, count_perType


    def get_data_list(self):
        ''' 获得数据路径列表 和 像素标签路径列表 '''
        data_list = []
        if self.method == 'ori':
            test_dir = os.path.join(self.data_dir, self.category, 'test')
            defecttype_dirs = os.listdir(test_dir)
            for tdir in defecttype_dirs:
                data_fdir = os.path.join(test_dir, tdir)
                type_data_list = os.listdir(data_fdir)
                type_data_list.sort()
                data_list += [os.path.join(data_fdir, item) for item in type_data_list]
        elif self.method == 'ours':
            data_fdir = os.path.join(self.data_dir, self.category, 'test', 'epoch_best_seg')
            data_list = [os.path.join(data_fdir, item) for item in os.listdir(data_fdir)]
        elif self.method == 'draem':
            data_fdir = os.path.join(self.data_dir, self.category, 'test')
            data_list = [os.path.join(data_fdir, item) for item in os.listdir(data_fdir)]
        elif self.method == 'riad':
            data_fdir = os.path.join(self.data_dir, self.category)
            data_list = [os.path.join(data_fdir, item) for item in os.listdir(data_fdir)]

        return data_list
    
    def __getitem__(self, idx):
        ''' 测试时,返回样本， 像素标签， 图片标签， 图片路径， 像素标签路径 '''
        transformer = T.Compose([
                T.ToTensor(), 
                T.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])])   #BGR
        image_path = self.data_list[idx]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        ''' 裁出重构图 '''
        if self.method == 'ori':
            pass
        elif self.method == 'ours':
            width = image.shape[1]//4
            image = image[:,width:2*width]
        elif self.method == 'draem':
            width = image.shape[1]//4
            image = image[:,width:2*width]        
        elif self.method == 'riad':
            width = image.shape[1]//2
            image = image[:,width:2*width]
        image = cv2.resize(image, self.pic_shape)
        image = transformer(image)
        label = 0 if 'good' in image_path else 1
        return image, label, image_path

''' 未实现 '''
class MVTecPatch_tsne(Dataset):
    def __init__(self, pic_size, category, method) -> None:
        super(MVTecPatch_tsne, self).__init__()
        self.pic_shape = (pic_size, pic_size)
        self.category = category
        self.method = method
        if method == 'ori':
            self.data_dir = '/root/test/wss/datasets/mvtec_anomaly_detection'
        elif method == 'ours':
            self.data_dir = '/root/test/wss/mysdd/vis/mvtec'
        elif method == 'draem':
            self.data_dir = '/root/test/wss/workspace/DRAEM/vis'
        elif method == 'riad':
            self.data_dir = '/root/test/wss/workspace/RIAD/vis'
        self.data_list = self.get_data_list()
        self.neg = [item for item in self.data_list if 'good' in item]
        self.pos = [item for item in self.data_list if 'good' not in item]

        positive_count, negative_count, defect_type, count_perType = self.get_statistics()
        print("datasets info:")
        print(f"category:{self.category}, positive count:{positive_count}, negative count:{negative_count}")
        for i in range(len(defect_type)):
            print(f"{defect_type[i]}:{count_perType[i]} ", end='')
        print()

    def __len__(self):
            return len(self.data_list)

    def get_statistics(self):
        ''' 获得数据集数量信息 '''
        data_list = self.data_list
        positive_count = len([item for item in data_list if 'good' not in item])
        negative_count = len(data_list) - positive_count
        defect_type = np.unique([str.split(item,'/')[-2] for item in data_list])
        count_perType = []
        for type in defect_type:
            count_perType.append(len([item for item in data_list if str.split(item,'/')[-2]==type]))
        return positive_count, negative_count, defect_type, count_perType


    def get_data_list(self):
        ''' 获得数据路径列表 和 像素标签路径列表 '''
        data_list = []
        if self.method == 'ori':
            test_dir = os.path.join(self.data_dir, self.category, 'test')
            defecttype_dirs = os.listdir(test_dir)
            for tdir in defecttype_dirs:
                data_fdir = os.path.join(test_dir, tdir)
                type_data_list = os.listdir(data_fdir)
                type_data_list.sort()
                data_list += [os.path.join(data_fdir, item) for item in type_data_list]
        elif self.method == 'ours':
            data_fdir = os.path.join(self.data_dir, self.category, 'test', 'epoch_best_seg')
            data_list = [os.path.join(data_fdir, item) for item in os.listdir(data_fdir)]
        elif self.method == 'draem':
            data_fdir = os.path.join(self.data_dir, self.category, 'test')
            data_list = [os.path.join(data_fdir, item) for item in os.listdir(data_fdir)]
        elif self.method == 'riad':
            data_fdir = os.path.join(self.data_dir, self.category)
            data_list = [os.path.join(data_fdir, item) for item in os.listdir(data_fdir)]

        return data_list
    
    def __getitem__(self, idx):
        ''' 测试时,返回样本， 像素标签， 图片标签， 图片路径， 像素标签路径 '''
        transformer = T.Compose([
                T.ToTensor(), 
                T.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])])   #BGR
        image_path = self.data_list[idx]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        ''' 裁出重构图 '''
        if self.method == 'ori':
            pass
        elif self.method == 'ours':
            width = image.shape[1]//4
            image = image[:,width:2*width]
        elif self.method == 'draem':
            width = image.shape[1]//4
            image = image[:,width:2*width]        
        elif self.method == 'riad':
            width = image.shape[1]//2
            image = image[:,width:2*width]
        image = cv2.resize(image, self.pic_shape)
        image = transformer(image)
        label = 0 if 'good' in image_path else 1
        return image, label, image_path


class MVTecTestDataset_Draem(Dataset):
    def __init__(self, pic_size, category, sigma=None) -> None:
        super(MVTecTestDataset_Draem, self).__init__()
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

        self.sigma = sigma

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
        ''' 测试时,返回样本， 像素标签， 图片标签， 图片路径， 像素标签路径 '''
        image_path = self.data_list[idx]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, self.pic_shape)
        image = image.astype(np.uint8)
        # ''' 添加高斯模糊 '''
        # image = cv2.GaussianBlur(image, (25, 25), self.sigma)

        transformer = T.Compose([
                T.ToTensor(), 
                T.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])])   #BGR

        image = transformer(image)

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

class MVTecTrainDataset_Draem(Dataset):
    def __init__(self, pic_size, category, positive_aug_ratio, negative_aug_ratio) -> None:
        super(MVTecTrainDataset_Draem, self).__init__()
        self.positive_aug_ratio = positive_aug_ratio
        self.negative_aug_ratio = negative_aug_ratio
        self.pic_shape = (pic_size, pic_size)
        self.category = category
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
                iaa.Affine(rotate=(-4,4), mode='edge'),
                iaa.Affine(scale=(0.98,1.02), mode='edge'),
                iaa.Affine(translate_percent={'x':(-0.02,0.02),'y':(-0.02,0.02)}, mode='edge')
            ], random_order=True
        )

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90,90))])

        positive_count, negative_count, defect_type, count_perType = self.get_statistics()
        print("datasets info:")
        print(f"category:{self.category}, positive count:{positive_count}, negative count:{negative_count}")
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
        fdir = os.path.join(self.data_dir, self.category, 'train/good')
        data_list = os.listdir(fdir)
        # num = 50
        # data_list = data_list[:num]
        data_list = [os.path.join(fdir, item) for item in data_list]
        return data_list
    
    def get_defect_source_list(self):
        fdir = '/root/test/wss/datasets/dtd/images'
        return glob.glob(fdir+'/*/*.jpg')
        # fdir = f'{self.data_dir}/{self.category}/train'
        # return glob.glob(fdir+'/good/*.png')

    def get_augmented_positive(self, image):
        image = self.negativa_augmenters(image = image)

        # 获得缺陷来源图, 随机选择一个
        # defect_source = cv2.imread(np.random.choice(self.defect_source_list), cv2.IMREAD_COLOR)
        # defect_source = cv2.resize(defect_source, self.pic_shape)
        # defect_source = self.positive_augmenters(image = defect_source)
        # 使用随机噪声替换DTD缺陷源
        # defect_source = np.random.uniform(0, 256, image.shape)
        # 使用原图当作缺陷源
        defect_source = np.array(image)
        defect_source = iaa.Affine(rotate=(-90,90), mode='edge')(image = defect_source)
        defect_source = self.positive_augmenters(image = defect_source)



        # 获得柏林噪声, 柏林噪声用于指定一幅图中哪些区域是正常区域, 哪些是缺陷区域
        perlin_scale = 6
        min_perlin_scale = 3
        perlin_scalex = 2 ** np.random.randint(min_perlin_scale, perlin_scale)
        perlin_scaley = 2 ** np.random.randint(min_perlin_scale, perlin_scale)
        perlin_noise = rand_perlin_2d_np(self.pic_shape, (perlin_scalex, perlin_scaley))  #噪声元素值范围是[-1,1], 噪声形状是self.pic_shape,二维
        perlin_noise = self.rot(image = perlin_noise)
        ''' 透明度拼接 '''
        # # 噪声二值化
        # perlin_noise_bin = np.where(perlin_noise > 0.5, 1., 0.)
        # perlin_noise_bin = perlin_noise_bin[:,:,np.newaxis].astype(np.float32)  #[h, w, 1]

        # # 原图正常部分
        # image_norm_part = np.array(image,dtype=np.float32) * (1-perlin_noise_bin)

        # # 原图缺陷部分
        # image_defect_part = np.array(image, dtype=np.float32) * perlin_noise_bin

        # # 透光率
        # beta = np.random.uniform(0.2,0.7)

        # # 缺陷来源图的缺陷部分
        # defect_source_defect_part = defect_source * perlin_noise_bin

        # defect_image = image_norm_part + beta * image_defect_part + (1-beta) * defect_source_defect_part
        # defect_image = np.array(defect_image, dtype=np.uint8)

        ''' 按seamlessclone拼接 '''
        defect_image = image.copy()
        perlin_noise_bin = np.where(perlin_noise > 0.5, 255, 0)
        perlin_noise_bin = perlin_noise_bin[:,:,np.newaxis].astype(np.uint8)  #[h, w, 1]

        num_labels, labels = cv2.connectedComponents(perlin_noise_bin, connectivity=8)
        for idx in range(1, num_labels):
            mask_roi= np.zeros_like(perlin_noise_bin)
            k = labels == idx
            mask_roi[k] = 255         # 单个联通分量
            cnts = cv2.findContours(mask_roi, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]   # 只有一个轮廓
            x, y, w, h = cv2.boundingRect(cnts[0])
            if x <= 3 and y <= 3:
                perlin_noise_bin[k] = 0
                continue
            if (w <= 20 and h <= 20 )or  cv2.contourArea(cnts[0]) < 800:
                perlin_noise_bin[k] = 0
                continue
                # 使用透光率拼接
                mask_roi = mask_roi.astype(np.float32) / 255
                # 原图正常部分
                image_norm_part = np.array(defect_image,dtype=np.float32) * (1-mask_roi)

                # 原图缺陷部分
                image_defect_part = np.array(defect_image, dtype=np.float32) * mask_roi

                # 透光率
                beta = np.random.rand() * 0.8  # 保证合成时缺陷更明显一点

                # 缺陷来源图的缺陷部分
                defect_source_defect_part = defect_source * mask_roi

                defect_image = image_norm_part + beta * image_defect_part + (1-beta) * defect_source_defect_part
                defect_image = defect_image.astype(np.uint8)
                continue
            # 使用seamless拼接
            try:
                defect_image = cv2.seamlessClone(src=defect_source[y:y+h,x:x+w], dst=defect_image, mask=mask_roi[y:y+h,x:x+w], p=(x+w//2,y+h//2), flags=cv2.NORMAL_CLONE)
            except:
                perlin_noise_bin[k] = 0
        perlin_noise_bin = perlin_noise_bin.astype(np.float32)/255
        
        # utils.visualize([image, defect_source, np.array(perlin_noise_bin,dtype=np.uint8), defect_image], './tmp', 'tmp.jpg')

        if 1. in perlin_noise_bin:
            label = 1
        else:
            label = 0
        transformer = T.Compose([
                T.ToTensor(), 
                T.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])])   #BGR

        return transformer(image), transformer(defect_image), torch.tensor(perlin_noise_bin).permute((2,0,1)), label

    def get_augmented_negative(self, image):
        image = self.negativa_augmenters(image = image)
        defect_image = np.array(image, dtype=np.uint8)
        mask = np.zeros([1,self.pic_shape[0], self.pic_shape[1]], dtype=np.float32)
        label = 0

        transformer = T.Compose([
                T.ToTensor(), 
                T.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])])   #BGR

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

    dataset = MVTecTrainDataset_Draem(256, 'bottle', 1, 0)
    dataset.__getitem__(10)
    quit()

    background, threshold = BACKGROUND["bottle"]
    img_src = '/root/test/wss/datasets/mvtec_anomaly_detection/bottle/train/good/001.png'
    img_src = cv2.imread(img_src, 1)
    img_src = cv2.resize(img_src, (256,256))
    img = np.abs(img_src.mean(axis=-1, keepdims=True) - background)
    img = np.where(img > threshold, 255, 0).astype(np.uint8)
    mask = cv2.medianBlur(img[...,0], 15)

    utils.visualize([img_src, img, mask], './tmp','img.jpg')

    # src_object_mask = np.ones_like(img_src[...,0:1])
    # dest_object_mask = np.ones_like(img_dest[...,0:1])

    # src_object_mask &= np.uint8(np.abs(img_src.mean(axis=-1, keepdims=True) - background) > threshold)
    # dest_object_mask &= np.uint8(np.abs(img_dest.mean(axis=-1, keepdims=True) - background) > threshold)


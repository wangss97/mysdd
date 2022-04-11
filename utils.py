import os
import logging
import math
import re
import time
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

def get_logger(file):
    fdir = os.path.split(file)[0]
    if not os.path.exists(fdir):
        print(f"不存在日志文件夹:{fdir}, 创建新的日志文件夹")
        os.makedirs(fdir)

    if not os.path.exists(file):
        print(f'日志文件不存在，创建新的日志文件:{file}')
        open(file, 'w')

    logger = logging.getLogger(__name__)
    for handler in logger.handlers:
        logger.removeHandler(handler)

    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(file)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(levelname)s]   %(asctime)s    %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

def psnr(mse):
    ''' 计算峰值信噪比 '''
    return 10*math.log10(1./mse)

def path_to_name(path):
    path_split = str.split(path, '/')
    name = path_split[-2]+'-'+path_split[-1]
    return name

def loss_figure(log_path, figure_path = None):
    with open(log_path, 'r') as log:
        lines = log.readlines()
        lines = [item for item in lines if 'loss' in item]
        epochs = []
        losses = []
        for line in lines:
            res_epoch = re.findall(r'epoch:\[\d+\]', line)

            res_loss = re.findall(r'loss: [\+|-]?\d+\.\d*[e]?[\+|-]?\d+', line)            

            if len(res_epoch)>0 and len(res_loss)>0:
                epochs.append(int(res_epoch[0][7:-1]))
                losses.append(float(res_loss[0][5:]))

        plt.plot(epochs, losses)
        
        if figure_path is not None:
            if not os.path.exists(os.path.split(figure_path)[0]):
                os.makedirs(os.path.split(figure_path)[0])
            plt.savefig(figure_path)
        else:
            if not os.path.exists('./tmp'):
                os.makedirs('./tmp')
            plt.savefig(f'./tmp/lose_figure_{time.time()}.jpg')
        plt.close()
        return epochs, losses

def precision_figure(log_path, figure_path = None):
    with open(log_path, 'r') as log:
        lines = log.readlines()
        lines = [item for item in lines if 'loss' in item]
        epochs = []
        losses = []
        for line in lines:
            res_epoch = re.findall(r'epoch:\[\d+\]', line)

            res_loss = re.findall(r'loss: [\+|-]?\d+\.\d*[e]?[\+|-]?\d+', line)            

            if len(res_epoch)>0 and len(res_loss)>0:
                epochs.append(int(res_epoch[0][7:-1]))
                losses.append(float(res_loss[0][5:]))

        plt.plot(epochs, losses)
        
        if figure_path is not None:
            if not os.path.exists(os.path.split(figure_path)[0]):
                os.makedirs(os.path.split(figure_path)[0])
            plt.savefig(figure_path)
        else:
            if not os.path.exists('./tmp'):
                os.makedirs('./tmp')
            plt.savefig(f'./tmp/lose_figure_{time.time()}.jpg')
        plt.close()
        return epochs, losses

def visualize(img_list, save_dir, img_name):
    img_show = None
    for img in img_list:
        img = np.array(img, dtype=np.uint8)
        if len(img.shape) == 2:
            img = img[:,:,np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
        elif len(img.shape) == 3 and img.shape[2]==1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if img_show is None:
            img_show = img
        else:
            img_show = np.concatenate([img_show, img], axis=1)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cv2.imwrite(save_dir+'/'+img_name, img_show)

def toImg(dataBatch):
    mean=[0.406, 0.456, 0.485]
    std=[0.225, 0.224, 0.229]

    for i in range(3):
        dataBatch[...,i] = (dataBatch[...,i] * std[i] + mean[i])*255
    dataBatch = np.clip(dataBatch, 0, 255)
    
    return np.array(dataBatch, dtype=np.uint8)

def patch_split(image:torch.Tensor, patch_size = 16, stride=16):
    ''' 将image，分成patch
        输入[b,c,h,w]
        输出[b*patch_num, c, patch_size,patch_size]
     '''
    b, c, h, w = image.shape
    # [b, c*kernel_size*kernel_size, patch_num]  其中的c*kernel_size*kernel_size即为切出的每一个patch块
    image_patch = torch.nn.Unfold(kernel_size=patch_size,padding=0,stride=stride)(image)
    b, K, patch_n = image_patch.shape
    image_patch = torch.transpose(image_patch, 1, 2)    # [b, patch_n, K]
    image_patch = torch.reshape(image_patch, (b*patch_n, c, patch_size, patch_size ))
    return image_patch

def mask_to_patchLabel(mask, patch_size=16, stride=16):
    ''' mask [b,c,h,w] '''

    mask_patch = patch_split(mask, patch_size, stride)
    mask_patch = torch.sum(mask_patch,dim=[1,2,3])
    label = mask_patch > 0
    label = label.to(dtype=torch.long)
    return label

def patchLabel_to_mask(label, batchsize, pic_size, patch_size):
    ''' label [batchsize * patch_num_h * patch_num_w]， 返回resize成原图大小的mask [b,c,h,w] '''
    b, c = label.shape
    patch_num = pic_size // patch_size
    label = torch.reshape(label,(batchsize, patch_num, patch_num, c))
    label = torch.permute(label, (0,3,1,2))

    label = torch.nn.functional.interpolate(input=label, size=(pic_size,pic_size), mode='bilinear')
    return label

def mask_resize(mask, ratio, int_flag = False):
    ''' 输入mask [b,c,h,w]， 按照ratio进行resize '''
    size = mask.shape[2]
    size_hat = int(size*ratio)
    mask = torch.nn.functional.interpolate(input=mask, size=(size_hat, size_hat), mode='bilinear')
    if int_flag:
        mask = torch.ceil(mask)
    return mask


if __name__ == '__main__':
    patch_split(None,None,2)
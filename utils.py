import os
import logging
import math
import re
import time
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from skimage.measure import label, regionprops
from sklearn.metrics import auc

ckpInfo = {
    'bottle':{
        'detect':{
            'epoch':'301',
            'auc':'0.9992',
            'detach':'no',
            'pos':'yes'
        },
        'segment':{
            'epoch':'587',
            'auc':'0.9819',
            'pro':'0.9432',
            'detach':'no',
            'pos':'yes'
        }
    },
    'cable':{
        'detect':{
            'epoch':'302',
            'auc':'0.9771',
            'detach':'yes',
            'pos':'yes'
        },
        'segment':{
            'epoch':'302',
            'auc':'0.9753',
            'pro':'0.9361',
            'detach':'yes',
            'pos':'yes'
        }
    },
    'screw':{
        'detect':{
            'epoch':'51',
            'auc':'1',
            'detach':'yes',
            'pos':'yes'
        },
        'segment':{
            'epoch':'568',
            'auc':'0.9843',
            'pro':'0.9287',
            'detach':'yes',
            'pos':'yes'
        }
    },
    'transistor':{
        'detect':{
            'epoch':'346',
            'auc':'0.9950',
            'detach':'yes',
            'pos':'yes'
        },
        'segment':{
            'epoch':'106',
            'auc':'0.9771',
            'pro':'0.9301',
            'detach':'yes',
            'pos':'yes'
        }
    }
}



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

def toImg(dataBatch, in_channel=3):
    if in_channel == 3:
        mean=[0.406, 0.456, 0.485]
        std=[0.225, 0.224, 0.229]

        for i in range(3):
            dataBatch[...,i] = (dataBatch[...,i] * std[i] + mean[i])*255
        dataBatch = np.clip(dataBatch, 0, 255)
    if in_channel == 1:
        dataBatch = dataBatch*255
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

def rescale(x):
    return (x - x.min()) / (x.max() - x.min())

def AUPRO(gt_mask, super_mask):
    ''' super_mask [sample_count, height, width] '''
    max_step = 100
    expect_fpr = 0.3  # default 30%
    max_th = super_mask.max()
    min_th = super_mask.min()
    delta = (max_th - min_th) / max_step
    ious_mean = []
    ious_std = []
    pros_mean = []
    pros_std = []
    threds = []
    fprs = []
    binary_score_maps = np.zeros_like(super_mask, dtype=np.bool)
    
    for step in range(max_step):
        thred = max_th - step * delta
    # for thred in np.unique(super_mask):
        # segmentation
        thred += 0.00001
        binary_score_maps[super_mask <= thred] = 0
        binary_score_maps[super_mask >  thred] = 1
        pro = []  # per region overlap
        iou = []  # per image iou
        # pro: find each connected gt region, compute the overlapped pixels between the gt region and predicted region
        # iou: for each image, compute the ratio, i.e. intersection/union between the gt and predicted binary map 
        for i in range(len(binary_score_maps)):    # for i th image
            # pro (per region level)
            label_map = label(gt_mask[i], connectivity=2)
            props = regionprops(label_map)
            for prop in props:
                x_min, y_min, x_max, y_max = prop.bbox    # find the bounding box of an anomaly region 
                cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                # cropped_mask = gt_mask[i][x_min:x_max, y_min:y_max]   # bug!
                cropped_mask = prop.filled_image    # corrected!
                intersection = np.logical_and(cropped_pred_label, cropped_mask).astype(np.float32).sum()
                pro.append(intersection / prop.area)
            # iou (per image level)
            intersection = np.logical_and(binary_score_maps[i], gt_mask[i]).astype(np.float32).sum()
            union = np.logical_or(binary_score_maps[i], gt_mask[i]).astype(np.float32).sum()
            if gt_mask[i].any() > 0:    # when the gt have no anomaly pixels, skip it
                iou.append(intersection / union)
        # against steps and average metrics on the testing data
        ious_mean.append(np.array(iou).mean())
        #print("per image mean iou:", np.array(iou).mean())
        ious_std.append(np.array(iou).std())
        pros_mean.append(np.array(pro).mean())
        pros_std.append(np.array(pro).std())
        # fpr for pro-auc
        gt_masks_neg = ~(gt_mask==1)
        fpr = np.logical_and(gt_masks_neg, binary_score_maps).sum() / gt_masks_neg.sum()
        fprs.append(fpr)
        threds.append(thred)
    # as array
    threds = np.array(threds)
    pros_mean = np.array(pros_mean)
    pros_std = np.array(pros_std)
    fprs = np.array(fprs)
    ious_mean = np.array(ious_mean)
    ious_std = np.array(ious_std)
    # best per image iou
    best_miou = ious_mean.max()
    #print(f"Best IOU: {best_miou:.4f}")
    # default 30% fpr vs pro, pro_auc
    idx = fprs <= expect_fpr  # find the indexs of fprs that is less than expect_fpr (default 0.3)
    fprs_selected = fprs[idx]
    fprs_selected = rescale(fprs_selected)  # rescale fpr [0,0.3] -> [0, 1]
    pros_mean_selected = pros_mean[idx]
    try:
        seg_pro_auc = auc(fprs_selected, pros_mean_selected)
    except:
        seg_pro_auc = 0
    return seg_pro_auc

if __name__ == '__main__':
    patch_split(None,None,2)
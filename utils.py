import os
import logging
import math
import re
import time
import matplotlib.pyplot as plt
import numpy as np
import cv2

def get_logger(file):
    fdir = os.path.split(file)[0]
    if not os.path.exists(fdir):
        print(f"不存在日志文件夹:{fdir}, 创建新的日志文件夹")
        os.makedirs(fdir)

    if not os.path.exists(file):
        print(f'日志文件不存在，创建新的日志文件:{file}')
        open(file, 'w')

    logging.shutdown()
    logger = logging.getLogger(__name__)
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


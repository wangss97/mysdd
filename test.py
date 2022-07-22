import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import re
import shutil
import torchvision
import numpy as np
import cv2

from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib.pyplot as plt

''' 验证梯度 '''

# a = torch.tensor([1.1])
# weight = Variable(torch.tensor([2.2]),requires_grad=True)
# loss1 = a * weight

# b = loss1.detach()
# weight2 = Variable(torch.tensor([0.5]),requires_grad=True)
# loss2 = b*weight2
# opt = torch.optim.SGD([weight,weight2],1)

# print(a,weight,loss1, weight2,loss2)

# opt.zero_grad()
# loss1.backward()
# loss2.backward()
# opt.step()
# print(a,weight,loss1, weight2,loss2)


''' 查看cpt '''
# cpt = torch.load('./checkpoints/segNet_conMem3/cable/epoch_10.pth')
# # cpt = torch.load('./checkpoints/segNet_conMem2/cable/epoch_10.pth')

# for key, value in cpt.items():
#     if 'pos_embeddings.1' in key:
#         print(value.shape)
#         print(torch.mean(value))
#         print(torch.sum(value))
#         print(torch.std(value))

# # cpt = torch.load('./checkpoints/segNet_conMem2/cable/epoch_20.pth')

# cpt = torch.load('./checkpoints/segNet_conMem3/cable/epoch_20.pth')
# for key, value in cpt.items():
#     if 'pos_embeddings.1' in key:
#         print(value.shape)
#         print(torch.mean(value))
#         print(torch.sum(value))
#         print(torch.std(value))

''' 删除低epoch的checkpoints 和 vis '''
# for root, dirs, files in os.walk('./checkpoints'):
#     for file in files:
#         res_epoch = re.findall(r'ch_\d+\.pth', file)
#         if len(res_epoch)>0:
#             num = int(res_epoch[0][3:-4])
#             if num <= 100:
#                 os.remove(root+'/'+file)

# for root, dirs, files in os.walk('./vis'):
#     for dir in dirs:
#         res_epoch = re.findall(r'ch_\d+', dir)
#         if len(res_epoch)>0:
#             num = int(res_epoch[0][3:])
#             if num <= 150:
#                 shutil.rmtree(root+'/'+dir)


''' 将torchvision中的fashionMNIST数据集保存到本地 '''
# dataset = torchvision.datasets.FashionMNIST(root='../datasets',train=True, download=True)

# num = np.zeros(10,dtype=int)

# for i in range(dataset.__len__()):
#     image, target = dataset.__getitem__(i)
    
#     image.save(f'../datasets/FashionMNIST/train/{target}/{num[target]}.png')
#     num[target] += 1

# for i in range(10):
#     imgs = os.listdir(f'../datasets/FashionMNIST/train/{i}')
#     for img in imgs:
#         os.remove(f'../datasets/FashionMNIST/train/{i}/{img}')


OBJECTS = ['bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 
            'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
TEXTURES = ['carpet', 'grid', 'leather', 'tile', 'wood']

''' 按异常分数写测试图片 '''
# tt='best_pro'
# for category in OBJECTS + TEXTURES:
#     category = 'transistor'
#     if not os.path.exists(f'./tmp/{category}/{tt}/'):
#         os.makedirs(f'./tmp/{category}/{tt}/')
#     groundtruth = np.load(f'./tmp/{category}_groundtruth_{tt}.npy')
#     score = np.load(f'./tmp/{category}_score_{tt}.npy')
#     name = np.load(f'./tmp/{category}_name_{tt}.npy')
    
#     idxs = np.argsort(score)
#     score = score[idxs]
#     name = name[idxs]

#     for i, sc in enumerate(score):
#         shutil.copy(f'./vis/mvtec/{category}/test/epoch_{tt}/{name[i]}', f'./tmp/{category}/{tt}/{int(sc*1000)}_{name[i]}')
#     quit()
# quit()

''' 画异常分数分布图 '''
# for category in OBJECTS + TEXTURES:

#     category = 'carpet'
#     groundtruth = np.load(f'./tmp/{category}_groundtruth_best_det996_noPos.npy')
#     score = np.load(f'./tmp/{category}_score_best_det996_noPos.npy')

#     fpr, tpr, thres = roc_curve(groundtruth, score, pos_label=1)
#     roc = roc_auc_score(groundtruth, score)
#     print(f'{category}:{roc}')
#     diff = tpr-fpr
#     idxs = np.argsort(diff)
#     # best_thres = (thres[idxs[-1]] + thres[idxs[-2]])/2
#     print(idxs[-1])
#     best_thres = thres[idxs[-1]]

#     idxs = np.argsort(groundtruth)
#     score = score[idxs]
#     groundtruth = groundtruth[idxs]

#     x_axis = np.arange(len(groundtruth))

#     pos_idxs = groundtruth == 1
#     neg_idxs = groundtruth == 0

#     normal_score = score[neg_idxs]

#     # plt.vlines(np.sum(groundtruth == 0), 0, 1)
#     plt.hlines(best_thres, -1, len(groundtruth)+5, linestyles=':', linewidth=5, color='black')
#     plt.scatter(x_axis[neg_idxs], score[neg_idxs], color='green')
#     plt.scatter(x_axis[pos_idxs] + 1, score[pos_idxs], color='red')

#     # plt.ylim(0,1)
#     plt.xlim(-1,len(groundtruth)+5)
#     # plt.yticks([0,0.25,0.50,0.75,1.00], [0,0.25,0.50,0.75,1.00])
#     plt.xticks([])
#     plt.title(f'{category}', size=30)

#     # ax.spines['top'].set_visible(False)
#     # ax.spines['right'].set_visible(False)
#     fig = plt.gcf()
#     fig.subplots_adjust(top=0.9, bottom=0.1, right=0.9, left=0.1)
#     plt.savefig(f'./tmp/{category}_fig.png')
#     plt.close()
#     quit()

''' 修改DAGM的组织结构 像MVTec一样 '''

print('1')

# path = '/root/test/wss/datasets/DAGM_mvteclike'

# CLASS_NAME = ['Class1','Class2','Class3','Class4','Class5','Class6','Class7','Class8','Class9','Class10']

# for class_name in CLASS_NAME:

#     if os.path.exists(os.path.join(path, class_name, 'test', 'Thumbs.db')):
#         os.remove(os.path.join(path, class_name, 'test', 'Thumbs.db'))

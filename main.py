from cProfile import label
from distutils.command.config import config
from tkinter.messagebox import NO
from unicodedata import category
from matplotlib import tri
import torch
import math
import cv2
import os
import numpy as np
import argparse
from loader.mvtec_dataset_NSA import MVTecTrainDataset_NSA, MVTecTestDataset_NSA, OBJECTS, TEXTURES
from loader.mvtec_dataset_GOAD import MVTecTrainDataset_GOAD

import utils
from model.loss import SSIM, FocalLoss, ssim, EntropyLoss
from model.model import model
from model.segnet import DiscriminativeSubNetwork
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from skimage.metrics import structural_similarity as compare_ssim

from tqdm import tqdm

from loader.mvtec_dataset import MVTecTestDataset_Draem, MVTecTrainDataset_Draem
import torch.nn.functional as F


class Agent():
    def __init__(self, config) -> None:
        self.config = config
        self.STEP = config['step']
        self.EPOCHS = config['epochs']
        self.CPT_DIR = f"./checkpoints/{self.config['tag']}/{self.config['category']}"
        self.LOG_PATH = f"./log/{self.config['tag']}/{self.config['category']}.txt"

        self.logger = utils.get_logger(self.LOG_PATH)
        self.get_model()
        self.logger.info(self.config)

    def get_model(self):
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.config['GPU_ID']}" if self.config['GPU_ID'] in [0,1,2,3] else 'cuda')
            # self.device = torch.device(f"cuda")
        else:
            self.device = torch.device("cpu")
        print('use device:',self.device)
        # self.model = torch.nn.parallel.DataParallel(model())
        self.model = model(mem_block_list=config['mblock_size'], device=self.device).to(self.device)
        # self.segnet = DiscriminativeSubNetwork(in_channels=6, out_channels=2).to(self.device)
        

        if not os.path.exists(self.CPT_DIR):
            os.makedirs(self.CPT_DIR)

        if self.config['step'] == -1:
            cpt_files = os.listdir(self.CPT_DIR)
            cpt_files = [item for item in cpt_files if 'mem' not in item and 'seg' not in item]

            if len(cpt_files) == 0:
                cpt_path = f"{self.CPT_DIR}/epoch_-1.pth"
                seg_cpt_path = f"{self.CPT_DIR}/seg_epoch_-1.pth"
                # mem_path = f"{self.CPT_DIR}/mem_epoch_-1.pth"
            else:
                cpt_files.sort(key=lambda x: os.path.getctime(self.CPT_DIR+'/'+x))
                cpt_path = f"{self.CPT_DIR}/{cpt_files[-1]}"
                seg_cpt_path = f"{self.CPT_DIR}/seg_{cpt_files[-1]}"
                # mem_path = f"{self.CPT_DIR}/mem_{cpt_files[-1]}"
        else:
            cpt_path =f"{self.CPT_DIR}/epoch_{self.config['step']}.pth"
            seg_cpt_path = f"{self.CPT_DIR}/seg_epoch_{self.config['step']}.pth"
            # mem_path = f"{self.CPT_DIR}/mem_{self.config['step']}.pth"
        
        if os.path.exists(cpt_path):
            self.model.load_state_dict(torch.load(cpt_path, map_location=self.device))
            print(f'model load weight from {cpt_path}')
        else:
            print(f'checkpoint file:{cpt_path} is not existed, train a new model.')

        # if os.path.exists(seg_cpt_path):
        #     self.segnet.load_state_dict(torch.load(seg_cpt_path, map_location=self.device))
        #     print(f'segnet load weight from {seg_cpt_path}')
        # else:
        #     print(f'checkpoint file:{seg_cpt_path} is not existed, train a new model.')

    def get_loss_weights(self, epoch, total_epoch):

        if self.config['DYN_BALANCED_LOSS']:
            rec_loss_weight = 1 - (1.0*epoch / total_epoch)
            seg_loss_weight = self.config['seg_loss_weight'] * (epoch / total_epoch)
        else:
            rec_loss_weight = 1
            seg_loss_weight = self.config['seg_loss_weight']

        return rec_loss_weight, seg_loss_weight


    def train(self):        
        print('start trainning..')
        self.logger.info('start trainning..')
        # optimizer = torch.optim.Adam([
        #                               {"params": self.model.parameters(), "lr": self.config['lr']},
        #                               {"params": self.segnet.parameters(), "lr": self.config['lr']}], weight_decay=1e-5)
        optimizer = torch.optim.Adam(self.model.parameters(),lr=self.config['lr'], weight_decay=1e-5)
        warmup = 10
        def adjust_lr(epoch):
            t = warmup
            n_t = 0.5
            T = self.STEP + self.EPOCHS
            if epoch < t:
                return (0.9*epoch / t+0.1)
            else:
                if n_t * (1+math.cos(math.pi*(epoch - t)/(T-t))) < 0.1:
                    return  0.1 
                else:
                    return n_t * (1+math.cos(math.pi*(epoch - t)/(T-t)))
            # lr = return_value * default_lr

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=adjust_lr, last_epoch=-1)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.STEP+self.EPOCHS+1, eta_min=1e-4)

        # for epoch in range(0, self.STEP + 1):
        #     scheduler.step()
        
        dataset = MVTecTrainDataset_Draem(pic_size=self.config['pic_size'],
                        category=self.config['category'],
                        positive_aug_ratio = self.config['positive_aug_ratio'],
                        negative_aug_ratio = self.config['negative_aug_ratio'])
        # dataset = MVTecTrainDataset_NSA(pic_size=self.config['pic_size'], 
        #                 category = self.config['category'],
        #                 positive_aug_ratio = self.config['positive_aug_ratio'],
        #                 negative_aug_ratio = self.config['negative_aug_ratio'])
        # dataset = MVTecTrainDataset_GOAD(pic_size=self.config['pic_size'], 
        #                 category = self.config['category'],
        #                 positive_aug_ratio = self.config['positive_aug_ratio'],
        #                 negative_aug_ratio = self.config['negative_aug_ratio'])
        loss_l2 = torch.nn.MSELoss()
        loss_ssim = SSIM(device=self.device)
        loss_focal = FocalLoss()
        loss_cnt = torch.nn.CrossEntropyLoss()

        self.model.freeze_resnet()
        for epoch in range(self.STEP + 1, self.STEP + self.EPOCHS + 1):
            self.model.train()
            if epoch == 20:
                self.model.unfreeze()
            print(f'**** epoch {epoch} / {self.STEP + self.EPOCHS}     {self.config["tag"]}')

            dataloader = DataLoader(dataset, batch_size=self.config['batchsize'], shuffle=True)
            rec_loss_weight, seg_loss_weight = self.get_loss_weights(epoch-self.STEP-1, self.EPOCHS)

            loss_epoch = 0.
            closs_epoch = 0.
            disloss_epoch = 0.
            for image_batch, defect_image_batch, mask_batch, label_batch, image_path_batch in tqdm(dataloader):
                image_batch = image_batch.to(self.device)
                defect_image_batch = defect_image_batch.to(self.device)
                mask_batch = mask_batch.to(self.device)
                label_batch = label_batch.to(dtype=torch.long, device=self.device)
                
                image_hat_batch, mask_pred_batch, label_pred_batch, entropy_loss_value, triplet_loss_value,\
                 norm_loss_value, compact_loss_value, distance_loss_value, l1_loss_value =\
                      self.model(defect_image_batch, image_batch, label_batch)
                
                ''' 重构损失 '''
                score_map_batch = torch.sqrt(torch.mean(torch.square(image_hat_batch-defect_image_batch), dim=1)).detach().cpu().tolist()  # [b,h,w]
                loss_l2_value = loss_l2(image_batch, image_hat_batch)
                loss_ssim_value = loss_ssim(image_batch, image_hat_batch)
                
                '''segnet的mask选择'''
                mask_pred_batch = mask_pred_batch[-1]

                ''' segNet损失'''
                # label_patch = utils.mask_to_patchLabel(mask_batch, patch_size=8, stride=8)
                # loss_cnt_value = loss_cnt(mask_pred_batch, label_patch)


                # loss_focal_value = loss_focal(torch.softmax(mask_pred_batch[-1], dim=1), mask_batch) + \
                #     loss_focal(torch.softmax(mask_pred_batch[-2], dim=1), utils.mask_resize(mask_batch,0.5,True))
                loss_focal_value = loss_focal(torch.softmax(mask_pred_batch,dim=1), mask_batch)
                # mask_pred_batch = (mask_pred_batch[-1] + utils.mask_resize(mask_pred_batch[-2],2))/2
                mask_pred_batch = torch.softmax(mask_pred_batch, dim=1)
                ''' clsNet损失 '''
                # loss_cls_value = loss_cnt(label_pred_batch, label_batch)

                # loss = rec_loss_weight*(rec_loss_value + compact_loss_value + 0.0001*distance_loss_value) +\
                #      seg_loss_weight*loss_focal_value
                
                # loss = loss_l2_value + compact_loss_value + 0.0001*distance_loss_value +\
                #     loss_focal_value

                loss = loss_l2_value + compact_loss_value + 0.0001 * distance_loss_value + loss_focal_value
                # loss = loss_ssim_value + loss_l2_value + loss_focal_value + loss_cls_value
                # loss = loss_l2_value + compact_loss_value + 0.05 * l1_loss_value + loss_focal_value
                # loss = loss_l2_value + loss_ssim_value +  compact_loss_value + loss_focal_value
                # loss = loss_l2_value + loss_ssim_value + entropy_loss_value
                
                # loss = rec_loss_weight* (loss_l2_value + loss_ssim_value) + seg_loss_weight * loss_focal_value

                loss_epoch += loss.item()
                closs_epoch += compact_loss_value.item()
                disloss_epoch += distance_loss_value.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 可视化
                if epoch % self.config['validation_period'] == 0:
                    image_batch = image_batch.permute((0,2,3,1)).detach().cpu().numpy()
                    image_batch = utils.toImg(image_batch)

                    image_hat_batch = image_hat_batch.permute((0,2,3,1)).detach().cpu().numpy()
                    image_hat_batch = utils.toImg(image_hat_batch)

                    defect_image_batch = defect_image_batch.permute((0,2,3,1)).detach().cpu().numpy()
                    defect_image_batch = utils.toImg(defect_image_batch)

                    mask_batch = (mask_batch.permute((0,2,3,1)) * 255).detach().cpu().numpy().astype(np.uint8)
                    
                    # mask_pred_batch = utils.patchLabel_to_mask(mask_pred_batch, image_batch.shape[0],self.config['pic_size'],8)
                    mask_pred_batch = (mask_pred_batch[:,1,:,:] * 255).detach().cpu().numpy().astype(np.uint8)
                    score_map_batch = (np.array(score_map_batch)*255).astype(np.uint8)
                    image_name_batch = np.array(list(map(utils.path_to_name, image_path_batch)))
                    for i in range(0, image_batch.shape[0], 2):
                        utils.visualize([image_batch[i], image_hat_batch[i],defect_image_batch[i],mask_batch[i],score_map_batch[i], mask_pred_batch[i]],
                        # utils.visualize([image_hat_batch[i]],
                                save_dir = f"./vis/{self.config['tag']}/{self.config['category']}/train/epoch_{epoch}", img_name = image_name_batch[i]) 

            scheduler.step()

            loss_epoch = loss_epoch / len(dataloader)
            utils.loss_figure(self.LOG_PATH, figure_path=f"./vis/{self.config['tag']}/{self.config['category']}/loss_figure.jpg")
            self.logger.info(f"category:{self.config['category']} epoch:[{epoch}], loss: {loss_epoch}, compact loss:{closs_epoch/len(dataloader)}, distance loss:{disloss_epoch/len(dataloader)}")

            print(f"category:{self.config['category']} epoch:[{epoch}], loss: {loss_epoch}")

            if epoch % self.config['validation_period'] == 0:
                self.save_model(save_step = epoch)
                # torch.save(self.memory, f"{self.CPT_DIR}/mem_epoch_{epoch}.pt")
                self.test(save_step = epoch)
            elif epoch % 2 == 0:
                self.test(save_step = epoch, visual=False)


    def test(self, save_step = None, visual=True):
        if save_step is None:
            save_step = self.STEP
        self.logger.info(f"start testing, epoch:[{save_step}]")
        print(f'start testing.. epoch:[{save_step}]')

        self.model.eval()
        with torch.no_grad():
            dataset = MVTecTestDataset_Draem(self.config['pic_size'], self.config['category'])
            # dataset = MVTecTestDataset_NSA(pic_size=self.config['pic_size'], category = self.config['category'])
            dataloader = DataLoader(dataset, batch_size=self.config['batchsize'], shuffle=True)
            grandTruth_list = []
            score_psnr_list = []
            score_ssim_list = []
            score_mask_list = []
            score_map_mse_list = []
            score_map_ssim_list = []
            score_map_mask_list = []
            mask_list = []

            for image_batch, mask_batch, label_batch, image_path_batch in tqdm(dataloader):
                image_batch = image_batch.to(self.device)
                label_batch = label_batch.to(dtype=torch.long, device=self.device)

                image_hat_batch, mask_pred_batch,label_pred_batch, entropy_loss_value, triplet_loss_list,\
                     norm_loss_value, compact_loss_value, distance_loss_value, l1_loss_value =\
                          self.model(image_batch, image_batch,label_batch)
                
                ''' Segnet '''
                # mask_pred_batch = (mask_pred_batch[-1] + utils.mask_resize(mask_pred_batch[-2],2))/2
                mask_pred_batch = mask_pred_batch[-1]
                mask_pred_batch = torch.softmax(mask_pred_batch, dim=1)
                # mask_pred_batch = utils.patchLabel_to_mask(mask_pred_batch,image_batch.shape[0],
                #     self.config['pic_size'], 8)

                grandTruth_list += label_batch.detach().cpu().tolist()

                ''' psnr当作异常分数 '''
                mse_batch = torch.mean(torch.square(image_hat_batch-image_batch), dim=(1,2,3)).detach().cpu().tolist()
                score_list_batch = np.array(list(map(utils.psnr, mse_batch)))
                score_psnr_list += score_list_batch.tolist()

                ''' ssim当作异常分数 '''
                ssim_batch, ssim_map_batch = ssim(image_batch.detach().cpu(), image_hat_batch.detach().cpu(),window_size=1, size_average=False)
                score_list_batch = ssim_batch.numpy()
                score_ssim_list += score_list_batch.tolist()

                ''' pred_mask当作异常分数 '''
                score_average = F.avg_pool2d(mask_pred_batch[:,1:,:,:], 21, stride=1, padding=21//2).detach().cpu().numpy()
                # score_average = mask_pred_batch[:,1:,:,:].detach().cpu().numpy()
                score_mask_list += np.max(score_average, axis=(1,2,3)).tolist()
                ''' cls网络输出标签当作异常分数 '''
                # score_mask_list += label_pred_batch[:,1].detach().cpu().tolist()

                ''' mse当作pixel异常分数 '''
                score_map_batch = torch.sqrt(torch.mean(torch.square(image_hat_batch-image_batch), dim=1)).detach().cpu().tolist()  # [b,h,w]
                score_map_mse_list += score_map_batch

                ''' ssim当作pixel异常分数 '''
                score_map_ssim_list += ssim_map_batch.mean(1).tolist()

                ''' pred_mask当作pixel异常分数 '''
                score_map_mask_list += mask_pred_batch[:,1,:,:].detach().cpu().tolist()

                mask_list += mask_batch.squeeze(1).detach().cpu().tolist()

                # 可视化  值域映射到0-255
                if visual:
                    image_batch = image_batch.permute((0,2,3,1)).detach().cpu().numpy()
                    image_batch = utils.toImg(image_batch)

                    image_hat_batch = image_hat_batch.permute((0,2,3,1)).detach().cpu().numpy()
                    image_hat_batch = utils.toImg(image_hat_batch)

                    mask_batch = (mask_batch.squeeze(1) * 255).detach().cpu().numpy().astype(np.uint8)
                    mask_pred_batch = (mask_pred_batch[:,1,:,:] * 255).detach().cpu().numpy().astype(np.uint8)
                    score_map_batch = (127 - ssim_map_batch.mean(1).numpy()*127).astype(np.uint8)
                    image_name_batch = np.array(list(map(utils.path_to_name, image_path_batch)))
                    for i in range(0, image_batch.shape[0]):
                        utils.visualize([image_batch[i], image_hat_batch[i], mask_batch[i], score_map_batch[i], mask_pred_batch[i]],
                        save_dir = f"./vis/{self.config['tag']}/{self.config['category']}/test/epoch_{save_step}", img_name = image_name_batch[i]) 

            score_psnr_list = np.max(score_psnr_list) - np.array(score_psnr_list)
            score_ssim_list = np.max(score_ssim_list) - np.array(score_ssim_list)
            score_map_ssim_list = 1 - np.array(score_map_ssim_list)

            detect_auc_psnr = roc_auc_score(grandTruth_list, score_psnr_list, labels=1)
            detect_auc_ssim = roc_auc_score(grandTruth_list, score_ssim_list, labels=1)
            detect_auc_mask = roc_auc_score(grandTruth_list, score_mask_list, labels = 1)
            mask_list_flatten = np.array(mask_list).flatten()
            segment_auc_mse = roc_auc_score(mask_list_flatten, np.array(score_map_mse_list).flatten())
            segment_auc_ssim = roc_auc_score(mask_list_flatten, np.array(score_map_ssim_list).flatten())
            segment_auc_mask = roc_auc_score(mask_list_flatten, np.array(score_map_mask_list).flatten())

            self.logger.info('##########  Test Metric: ')
            self.logger.info(f"detect_auc_psnr:{detect_auc_psnr:.4f}")
            self.logger.info(f"detect_auc_ssim:{detect_auc_ssim:.4f}")
            self.logger.info(f"detect_auc_mask:{detect_auc_mask:.4f}")
            self.logger.info(f"segment_auc_mse:{segment_auc_mse:.4f}")
            self.logger.info(f"segment_auc_ssim:{segment_auc_ssim:.4f}")
            self.logger.info(f"segment_auc_mask:{segment_auc_mask:.4f}")
            self.logger.info('')
            print('')
            print('##########  Test Metric: ')
            print(f"detect_auc_psnr:{detect_auc_psnr:.4f}")
            print(f"detect_auc_ssim:{detect_auc_ssim:.4f}")
            print(f"detect_auc_mask:{detect_auc_mask:.4f}")
            print(f"segment_auc_mse:{segment_auc_mse:.4f}")
            print(f"segment_auc_ssim:{segment_auc_ssim:.4f}")
            print(f"segment_auc_mask:{segment_auc_mask:.4f}")
            print('')

    def save_model(self, save_step):
        self.logger.info(f"model saved: {self.CPT_DIR}/epoch_{save_step}.pth  segnet saved:{self.CPT_DIR}/seg_epoch_{save_step}.pth")
        torch.save(self.model.state_dict(),
                            f"{self.CPT_DIR}/epoch_{save_step}.pth")

    def featuremap_vis(feature, sdir):
        feature = feature[0]
        for idx,img in enumerate(feature):
            img = (img - np.min(img))*255/(np.max(img) - np.min(img))
            img = img.astype(np.uint8)
            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
            img = img[:,:,::-1]
            cv2.imwrite(sdir+'/'+str(idx)+'.png', img)

if __name__ == "__main__":


    def parse_arguments():
        """
            Parse the command line arguments of the program.
        """
        parser = argparse.ArgumentParser(description='param of model')

        parser.add_argument("--description", type=str, default='')
        
        parser.add_argument("--train",action="store_true",help="train mode")
        parser.add_argument("--test",action="store_true",help="Define if we wanna test the model")
        parser.add_argument("--category",type=str,default='bottle')
        parser.add_argument("--load_weight",action="store_true")
        parser.add_argument("--tag",type=str,default= 'default')
        parser.add_argument("--step",type=int,default = -1)
        parser.add_argument("--epochs",type=int,default = 2)
        parser.add_argument("--validation_period",type=int,default = 5)
        parser.add_argument("--batchsize",type=int,default = 8)
        parser.add_argument("--lr",type=float,default = 1e-3)
        parser.add_argument("--GPU_ID",type=int, default=3)
        parser.add_argument("--pic_size",type=int, default=256)
        parser.add_argument("--msize",type=int, default=20)
        parser.add_argument("--mdim",type=int, default=2048)
        parser.add_argument("--mblock_size",type=str, default='1,2,4,8')
        parser.add_argument("--positive_aug_ratio",type=int, default=5)
        parser.add_argument("--negative_aug_ratio",type=int, default=5)

        parser.add_argument("--seg_loss_weight",type=float, default=1.)
        parser.add_argument("--DYN_BALANCED_LOSS", action="store_false", default=True)

        return parser.parse_args()
    
    categorys = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    # categorys = ['bottle', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    categorys = ['cable',  'carpet']
    # categorys = categorys[4:7]
    # categorys = categorys[12:]
    

    config = vars(parse_arguments())
    config['mblock_size'] = [int(item) for item in str.split(config['mblock_size'],',')]

    if config['category'] == 'all':
        for category in categorys:
            config['category'] = category

            agent = Agent(config=config)
            if config['train']:
                agent.train()
            elif config['test']:
                agent.test()
    else:
        agent = Agent(config=config)
        if config['train']:
            agent.train()
        elif config['test']:
            agent.test()
        
    ''' 
    resnet18 输出channel: 512
    wide_resnet50_2 输出channel:2048
     '''
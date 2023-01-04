from asyncio.log import logger
import torch
import math
import cv2
import os
import numpy as np
import argparse
from loader.mvtec_dataset_NSA import MVTecTrainDataset_NSA, MVTecTestDataset_NSA, OBJECTS, TEXTURES
from loader.mvtec_dataset_GOAD import MVTecTrainDataset_GOAD
from loader.mvttec_dataset_ours import MVTecTrainDataset_ours, MVTecTestDataset_ours
from loader.MNIST_dataset import MNISTTestDataset, MNISTTrainDataset
from loader.DAGM_dataset import DAGMTrainDataset, DAGMTestDataset
from loader.EL_dataset import ELTrainDataset, ELTestDataset
from model.resnet import wide_resnet50_2

import utils
from model.loss import SSIM, FocalLoss, ssim, EntropyLoss, SegmentationLosses
from model.model import model
from model.segnet import DiscriminativeSubNetwork
from model.discriminator import discriminator
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from skimage.metrics import structural_similarity as compare_ssim

from scipy import interpolate

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

        self.in_channel = 3
        self.out_channel = 3

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
        self.model = model(in_channel=self.in_channel,out_channel=self.out_channel,mem_block_list=config['mblock_size'], device=self.device, rec_depth=5,seg_depth=6).to(self.device)
        self.discriminator = None
        

        if not os.path.exists(self.CPT_DIR):
            os.makedirs(self.CPT_DIR)

        if self.config['step'] == -1:
            cpt_files = os.listdir(self.CPT_DIR)
            cpt_files = [item for item in cpt_files if 'mem' not in item and 'seg' not in item]

            if len(cpt_files) == 0:
                cpt_path = f"{self.CPT_DIR}/epoch_-1.pth"
            else:
                cpt_files.sort(key=lambda x: os.path.getctime(self.CPT_DIR+'/'+x))
                cpt_path = f"{self.CPT_DIR}/{cpt_files[-1]}"
        else:
            cpt_path =f"{self.CPT_DIR}/epoch_{self.config['step']}.pth"
        
        if self.config['cpt_path'] != '':
            cpt_path = self.config['cpt_path']
        
        if os.path.exists(cpt_path):
            self.load_state_dict(cpt_path)
            print(f'model load weight from {cpt_path}')
        else:
            print(f'checkpoint file:{cpt_path} is not existed, train a new model.')


    def load_state_dict(self, cpt_path):
        pretrained_dict = torch.load(cpt_path, map_location=self.device)
        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)



    def train(self):
        print('start trainning..')
        self.logger.info('start trainning..')
        optimizer = torch.optim.Adam(self.model.parameters(),lr=self.config['lr'], weight_decay=1e-5)
        warmup = 20
        def adjust_lr(epoch):
            t = warmup
            n_t = 0.5
            T = self.STEP + self.EPOCHS
            if epoch <= t:
                return (0.9*epoch / t+0.1)
            else:
                if n_t * (1+math.cos(math.pi*(epoch - t)/(T-t))) < 0.1:
                    return  0.1 
                else:
                    return n_t * (1+math.cos(math.pi*(epoch - t)/(T-t)))
            # lr = return_value * default_lr

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=adjust_lr, last_epoch=-1)

        for epoch in range(0, self.STEP + 1):
            scheduler.step()
        
        dataset = MVTecTrainDataset_Draem(pic_size=self.config['pic_size'],
                        category=self.config['category'],
                        positive_aug_ratio = self.config['positive_aug_ratio'],
                        negative_aug_ratio = self.config['negative_aug_ratio'])
        # dataset = MVTecTrainDataset_ours(pic_size=self.config['pic_size'],
        #                 category=self.config['category'],
        #                 positive_aug_ratio = self.config['positive_aug_ratio'],
        #                 negative_aug_ratio = self.config['negative_aug_ratio'])
        # dataset = MVTecTrainDataset_NSA(pic_size=self.config['pic_size'],
        #                 category=self.config['category'],
        #                 positive_aug_ratio = self.config['positive_aug_ratio'],
        #                 negative_aug_ratio = self.config['negative_aug_ratio'])
        # dataset = DAGMTrainDataset(pic_size=self.config['pic_size'],
        #                 category=self.config['category'],
        #                 positive_aug_ratio = self.config['positive_aug_ratio'],
        #                 negative_aug_ratio = self.config['negative_aug_ratio'])


        loss_l2 = torch.nn.MSELoss()
        loss_focal = FocalLoss()
        best_detect_auc, best_seg_auc, best_seg_pro = 0,0,0

        if self.config["detach"]:
            self.model.detach_on()
        else:
            self.model.detach_off()
        for epoch in range(self.STEP + 1, self.STEP + self.EPOCHS + 1):

            self.model.train()

            print(f'**** epoch {epoch} / {self.STEP + self.EPOCHS}     {self.config["tag"]}  {self.config["category"]}')

            dataloader = DataLoader(dataset, batch_size=self.config['batchsize'], shuffle=True)
           
            loss_epoch = 0.
            closs_epoch = 0.
            disloss_epoch = 0.
            rec_loss_epoch = 0.
            for image_batch, defect_image_batch, mask_batch, label_batch, image_path_batch in tqdm(dataloader):
                image_batch = image_batch.to(self.device)
                defect_image_batch = defect_image_batch.to(self.device)
                mask_batch = mask_batch.to(self.device)
                label_batch = label_batch.to(dtype=torch.long, device=self.device)
                batchsize = image_batch.shape[0]

                image_hat_batch, mask_pred_batch, compact_loss_value, distance_loss_value =\
                      self.model(defect_image_batch, image_batch)


                ''' 重构损失 '''
                loss_l2_value = loss_l2(image_batch, image_hat_batch)
                
                ''' segNet损失'''
                mask_pred_batch = torch.softmax(mask_pred_batch, dim=1)
                loss_focal_value = loss_focal(mask_pred_batch, mask_batch)
                mask_pred_batch = mask_pred_batch[:,1:]


                loss = loss_l2_value + compact_loss_value + 0.0001 * distance_loss_value  + loss_focal_value

                loss_epoch += loss.item()
                rec_loss_epoch += loss_l2_value.item()
                closs_epoch += compact_loss_value.item()
                disloss_epoch += distance_loss_value.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 可视化
                if epoch % self.config['validation_period'] == 0:
                    image_batch = image_batch.permute((0,2,3,1)).detach().cpu().numpy()
                    image_batch = utils.toImg(image_batch, in_channel=self.in_channel)

                    image_hat_batch = image_hat_batch.permute((0,2,3,1)).detach().cpu().numpy()
                    image_hat_batch = utils.toImg(image_hat_batch, in_channel=self.in_channel)

                    defect_image_batch = defect_image_batch.permute((0,2,3,1)).detach().cpu().numpy()
                    defect_image_batch = utils.toImg(defect_image_batch, in_channel=self.in_channel)

                    mask_batch = (mask_batch.squeeze(1) * 255).detach().cpu().numpy().astype(np.uint8)
                    # mask_batch = (mask_batch * 255).detach().cpu().numpy().astype(np.uint8)
                    
                    # mask_pred_batch = utils.patchLabel_to_mask(mask_pred_batch, image_batch.shape[0],self.config['pic_size'],8)
                    # mask_pred_batch = torch.sigmoid(mask_pred_batch)
                    mask_pred_batch = (mask_pred_batch.squeeze(1) * 255).detach().cpu().numpy().astype(np.uint8)
                    image_name_batch = np.array(list(map(utils.path_to_name, image_path_batch)))
                    for i in range(0, image_batch.shape[0], 2):
                        utils.visualize([image_batch[i], image_hat_batch[i],defect_image_batch[i],mask_batch[i],mask_pred_batch[i]],
                        # utils.visualize([image_hat_batch[i]],
                                save_dir = f"./vis/{self.config['tag']}/{self.config['category']}/train/epoch_{epoch}", img_name = image_name_batch[i]) 

            scheduler.step()

            loss_epoch = loss_epoch / len(dataloader)
            utils.loss_figure(self.LOG_PATH, figure_path=f"./vis/{self.config['tag']}/{self.config['category']}/loss_figure.jpg")
            self.logger.info(f"category:{self.config['category']} epoch:[{epoch}], loss: {loss_epoch},"+
                f" compact loss:{closs_epoch/len(dataloader)}, distance loss:{disloss_epoch/len(dataloader)}, rec loss:{rec_loss_epoch/len(dataloader)}")

            print(f"category:{self.config['category']} epoch:[{epoch}], loss: {loss_epoch}")

            detect_auc, seg_auc, seg_pro = 0, 0, 0
            if epoch % self.config['validation_period'] == 0:
                # torch.save(self.memory, f"{self.CPT_DIR}/mem_epoch_{epoch}.pt")
                detect_auc, seg_auc, seg_pro = self.test(save_step = epoch)
            elif epoch % 1 == 0:
                detect_auc, seg_auc, seg_pro = self.test(save_step = epoch, visual=False)

            if detect_auc > best_detect_auc:
                best_detect_auc = detect_auc
                self.save_model(f'best_detect_auc')

            if seg_auc > best_seg_auc:
                best_seg_auc = seg_auc
                self.save_model(f'best_seg_auc')

            if seg_pro > best_seg_pro:
                best_seg_pro = seg_pro
                self.save_model(f'best_seg_pro')      

            self.remove_model(f"epoch_{epoch-1}")
            self.save_model(f"epoch_{epoch}")
            print('')
        self.logger.info(f"best_detect_auc:{best_detect_auc}  best_seg_auc:{best_seg_auc}  best_seg_pro:{best_seg_pro}")


    def test(self, save_step = None, visual=True):
        if save_step is None:
            save_step = self.STEP
        self.logger.info(f"start testing, epoch:[{save_step}]")
        print(f'start testing.. epoch:[{save_step}]')

        self.model.eval()

        with torch.no_grad():
            dataset = MVTecTestDataset_Draem(self.config['pic_size'], self.config['category'])
            # dataset = MVTecTestDataset_ours(self.config['pic_size'], self.config['category'])
            # dataset = DAGMTestDataset(self.config['pic_size'], self.config['category'])
            # dataset = ELTestDataset(self.config['pic_size'], self.config['category'])

            dataloader = DataLoader(dataset, batch_size=self.config['batchsize'], shuffle=True)
            groundTruth_list = []
            score_map_list = []
            mask_list = []

            score_map_ssim = []
            score_ssim = []
            score_map_mse = []
            score_mse = []

            for image_batch, mask_batch, label_batch, image_path_batch in tqdm(dataloader):
                image_batch = image_batch.to(self.device)
                label_batch = label_batch.to(dtype=torch.long, device=self.device)
                mask_batch = mask_batch.to(device=self.device)
                batchsize = image_batch.shape[0]

                image_hat_batch, mask_pred_batch, compact_loss_value, distance_loss_value =\
                     self.model(image_batch, image_batch)
                                
                ''' Segnet '''
                mask_pred_batch = torch.softmax(mask_pred_batch,dim=1)[:,1:]

                label_batch = label_batch.detach().cpu().tolist()
                groundTruth_list += label_batch if isinstance(label_batch,list) else [label_batch]

                ''' ssim当作异常分数 '''
                ssim_batch, ssim_map_batch = ssim(image_batch.detach().cpu(), image_hat_batch.detach().cpu(),window_size=31, size_average=False)
                ssim_map_batch = (1 - ssim_map_batch.mean(1).numpy())/2
                score_map_ssim.append(ssim_map_batch)
                score_ssim += ssim_batch.tolist()
                ''' mse当作异常分数 '''
                mse_map_batch = torch.square(image_batch - image_hat_batch).mean(1).detach().cpu().numpy()
                score_map_mse.append(mse_map_batch)
                score_mse += np.mean(mse_map_batch, axis=(1,2)).tolist()

                score_map_list.append(mask_pred_batch)
                mask_list.append(mask_batch)


                # 可视化  值域映射到0-255
                if visual:
                    image_batch = image_batch.permute((0,2,3,1)).detach().cpu().numpy()
                    image_batch = utils.toImg(image_batch, in_channel=self.in_channel)

                    image_hat_batch = image_hat_batch.permute((0,2,3,1)).detach().cpu().numpy()
                    image_hat_batch = utils.toImg(image_hat_batch, in_channel=self.in_channel)

                    mask_batch = (mask_batch.squeeze(1) * 255).detach().cpu().numpy().astype(np.uint8)
                    mask_pred_batch = (mask_pred_batch.squeeze(1) * 255).detach().cpu().numpy().astype(np.uint8)

                    image_name_batch = np.array(list(map(utils.path_to_name, image_path_batch)))
                    for i in range(0, image_batch.shape[0]):
                        utils.visualize([image_batch[i], image_hat_batch[i], mask_batch[i], mask_pred_batch[i]],
                        save_dir = f"./vis/{self.config['tag']}/{self.config['category']}/test/epoch_{save_step}", img_name = image_name_batch[i]) 

            masks = torch.concat(mask_list, dim=0).squeeze(1).detach().cpu().numpy()
            score_maps = torch.concat(score_map_list, dim=0)

            score_average = score_maps
            score_average = F.avg_pool2d(score_average, 21, stride=1, padding=21//2)
            score_average = score_average.reshape(score_maps.shape[0], -1)

            topk_value, _ = torch.topk(score_average, 5, dim=1)
            scores = np.mean(topk_value.detach().cpu().numpy(), axis=1).tolist()


            scores = np.array(scores)
            detect_auc = roc_auc_score(groundTruth_list, scores)
            masks = np.array(masks)
            score_maps = score_maps.detach().squeeze(1).detach().cpu().numpy()
            segment_auc = roc_auc_score(masks.flatten(), score_maps.flatten())
            score_maps = np.around(score_maps, decimals=3)
            # segment_pro = utils.AUPRO(masks, score_maps)
            # detect_auc = 0.
            # segment_auc = 0.
            segment_pro = 0.

            self.logger.info('##########  Test Metric: ')
            self.logger.info(f"detect_auc:{detect_auc:.4f}")
            self.logger.info(f"segment_auc:{segment_auc:.4f}")
            self.logger.info(f"segment_aupro:{segment_pro:.4f}")
            self.logger.info('')
            print('')
            print('##########  Test Metric: ')
            print(f"detect_auc:{detect_auc:.4f}")
            print(f"segment_auc:{segment_auc:.4f}")
            print(f"segment_aupro:{segment_pro:.4f}")

            ''' 计算tpr,tnr '''
            # fpr, tpr, thres = roc_curve(groundTruth_list, scores, pos_label=1)
            # diff = tpr-fpr
            # idxs = np.argsort(diff)

            # best_thres = thres[idxs[-1]]-0.00001
            # pred = np.where(scores > best_thres, 1, 0).tolist()
            # matrix = confusion_matrix(groundTruth_list, pred)
            # TN, FP, FN, TP = matrix[0][0],matrix[0][1],matrix[1][0],matrix[1][1]
            # print(matrix)
            # TPR = TP*1.0/(TP+FN)
            # TNR = TN*1.0/(TN+FP)
            # print(f'TPR:{TPR}  TNR:{TNR}')
            # self.logger.info(f'TPR:{TPR}  TNR:{TNR}')

            ''' 计算TPR为95%时的FPR '''
            # fpr,tpr,thresh = roc_curve(groundTruth_list, scores, pos_label=1)
            # fpr95 = float(interpolate.interp1d(tpr, fpr)(0.95))
            # print(f'TPR:0.95  FPR:{fpr95}')
            # self.logger.info(f'TPR:0.95  FPR:{fpr95}')

            ''' 计算recall99时的pre '''
            # res = utils.rec99(groundTruth_list, scores)
            # print(res)
            # self.logger.info(res)

            return detect_auc, segment_auc, segment_pro

    def save_model(self, model_name='default'):
        self.logger.info(f"model saved: {self.CPT_DIR}/{model_name}.pth")
        torch.save(self.model.state_dict(),
                            f"{self.CPT_DIR}/{model_name}.pth")
    
    def remove_model(self, model_name='xxx'):
        if os.path.exists(f"{self.CPT_DIR}/{model_name}.pth"):
            os.remove(f"{self.CPT_DIR}/{model_name}.pth")

    def remove_model_paramater(self, model_name='default'):
        if os.path.exists(f"{self.CPT_DIR}/f{model_name}.pth"):
            os.remove(f"{self.CPT_DIR}/f{model_name}.pth")

    

if __name__ == "__main__":


    def parse_arguments():
        """
            Parse the command line arguments of the program.
        """
        parser = argparse.ArgumentParser(description='param of model')

        parser.add_argument("--description", type=str, default='')
        
        parser.add_argument("--train",action="store_true",help="train mode")
        parser.add_argument("--test",action="store_true",help="Define if we wanna test the model")
        parser.add_argument("--detach",action="store_true")
        parser.add_argument("--category",type=str,default='bottle')
        parser.add_argument("--load_weight",action="store_true")
        parser.add_argument("--tag",type=str,default= 'default')
        parser.add_argument("--step",type=int,default = -1)
        parser.add_argument("--epochs",type=int,default = 2)
        parser.add_argument("--validation_period",type=int,default = 50)
        parser.add_argument("--batchsize",type=int,default = 8)
        parser.add_argument("--lr",type=float,default = 1e-3)
        parser.add_argument("--GPU_ID",type=int, default=0)
        parser.add_argument("--pic_size",type=int, default=256)
        parser.add_argument("--msize",type=int, default=50)
        parser.add_argument("--mblock_size",type=str, default='1,2,4,8')
        parser.add_argument("--positive_aug_ratio",type=int, default=2)
        parser.add_argument("--negative_aug_ratio",type=int, default=2)
        parser.add_argument("--cpt_path", type=str, default="")

        return parser.parse_args()
    

    config = vars(parse_arguments())
    config['mblock_size'] = [int(item) for item in str.split(config['mblock_size'],',')]
    categorys = [item for item in str.split(config['category'],',')]

    if config['category'] == 'all':
        categorys = OBJECTS + TEXTURES
    elif config['category'] == 'textures':
        categorys = TEXTURES
    elif config['category'] == 'objects':
        categorys = OBJECTS
    else:
        categorys = [item for item in str.split(config['category'],',')]

    # categorys = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    # steps = [375, 200, 665, 200, 125, 685, 215, 480, 490, 340, 630, 340, 130, 500, 590]

    for category in categorys:
        config['category'] = category
        if category in ['cable','screw','transistor']:
            config['detach'] = True
        else:
            config['detach'] = False

        agent = Agent(config=config)
        if config['train']:
            agent.train()
        elif config['test']:
            agent.test()

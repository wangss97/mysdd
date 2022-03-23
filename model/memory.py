import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter

from model.loss import EntropyLoss


def positionalencoding2d(D, H, W):
    """
    :param D: dimension of the model
    :param H: H of the positions
    :param W: W of the positions
    :return: DxHxW position matrix
    """
    if D == 0:
        return torch.zeros(D, H, W)
    if D % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(D))
    P = torch.zeros(D, H, W)
    # Each dimension use half of D
    D = D // 2
    div_term = torch.exp(torch.arange(0.0, D, 2) * -(math.log(1e4) / D))
    pos_w = torch.arange(0.0, W).unsqueeze(1)   
    pos_h = torch.arange(0.0, H).unsqueeze(1)
    P[0:D:2, :, :]  = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[1:D:2, :, :]  = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[D::2,  :, :]  = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    P[D+1::2,:, :]  = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    return P


class BlockMemory_trainableFalse(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def read(self, score_of_mem:torch.Tensor, mem):
        ''' 从memBank中读出重组后的query_hat, 为了节省计算速度，score_of_mem在外部计算后传入该函数内\n
            query_hat的形状是[b*h*w, d]
         '''
        _, m = score_of_mem.size()
        weight_thres = 1./mem.size(0)

        # 相关度小于阈值的memory item不参与query的重组
        score_of_mem = torch.where(score_of_mem < weight_thres, torch.zeros_like(score_of_mem), score_of_mem)
        # 除以L1范数做归一化
        score_of_mem = F.normalize(score_of_mem, p=1, dim=1)

        # [b*h*w, m] * [m, d] = [b*h*w, d]
        query_hat = torch.matmul(score_of_mem, mem)

        return query_hat

    def update(self, query:torch.Tensor, score_of_mem:torch.Tensor, score_of_query:torch.Tensor, mem):
        ''' 更新memory bank '''
        b, d, h, w = query.size()
        _, m = score_of_mem.size()
        m, _ = score_of_query.size()

        query = query.permute((0, 2, 3, 1)).reshape((b*h*w, d))
        mem_update = torch.zeros_like(mem)

        nearest_mem_idx = torch.argmax(score_of_mem, dim=1)

        for i in range(m):
            nearest_query_idx = (nearest_mem_idx == i)
            if torch.sum(nearest_query_idx) == 0:
                continue
            query_select = query[nearest_query_idx]
            weight = score_of_query[i, nearest_query_idx]
            weight = weight / torch.max(weight)

            mem_update[i] = F.normalize(mem[i] + torch.matmul(weight.unsqueeze(0), query_select), p=2)
        return mem_update.detach()

    
    def similarity_score(self, query:torch.Tensor, mem):
        ''' 返回每个memory item 与query中每个像素的相似度分数\n
            score_of_mem，形状[b*h*w, m]，第i行为长度m的向量，表示第i个像素与所有m个memory item的相似度\n
            score_of_query，形状[m, b*h*w]，第i行为长度b*h*w的向量，表示第i个memory item与所有bb*h*w个像素的相似度\n
        '''
        m, d = mem.size()
        b, d, h, w = query.size()

        query = torch.permute(query, (0,2,3,1))  # [b, h, w, d]
        query_reshape = torch.reshape(query, (b*h*w, d))

        mem_trans = torch.transpose(mem, 0, 1)   #[d, m]

        # 点乘作为相似度
        # [b*h*w, d] * [d, m] = [b*h*w, m]
        score = torch.matmul(query_reshape, mem_trans)

        # 再除以模长，余弦夹角作为相似度
        # query_reshape_norm = torch.sqrt(torch.sum(torch.square(query_reshape), dim=1))  # [b*h*w]
        # mem_norm = torch.sqrt(torch.sum(torch.square(mem), dim=1)) # [m]

        # # [b*h*w,m] / ([b*h*w,1] pointwise mul [1,m])
        # score = score / (query_reshape_norm.unsqueeze(1) * mem_norm.unsqueeze(0))
        score_of_mem = torch.softmax(score, dim=1)  # [b*h*w, m]
        score_of_query = torch.softmax(score.transpose(0, 1), dim=1)   # [m, b*h*w]

        return score_of_mem, score_of_query

    def forward(self, query, mem):

        b, d, h, w = query.size()

        score_of_mem, score_of_query = self.similarity_score(query, mem)

        query_hat = self.read(score_of_mem, mem)
        query_hat = query_hat.reshape((b, h, w, d)).permute((0, 3, 1, 2))

        mem_update = self.update(query, score_of_mem, score_of_query, mem)

        return query_hat, mem_update
    

class MemoryUnit(nn.Module):
    def __init__(self, input_size, mem_size, fea_dim, shrink_thres=0.0025, device='cuda'):
        super(MemoryUnit, self).__init__()
        self.mem_size = mem_size
        self.fea_dim = fea_dim
        self.input_size = input_size
        self.weight = Parameter(torch.Tensor(self.mem_size, self.fea_dim))  # M x C

        # self.acti = torch.nn.Sigmoid()
        self.bias = None
        self.shrink_thres= shrink_thres
        # self.hard_sparse_shrink_opt = nn.Hardshrink(lambd=shrink_thres)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def triplet_margin_loss(self, input, att_weight):
        ''' input形状[b*h*w, c]， att_weight形状[b*h*w, m]
        让input中每一个像素与self.weight中最近的memory item更近，与第二近的memory item更远
         '''
        lossFunc = torch.nn.TripletMarginLoss()
        if self.mem_size >=2:
            _, indices = torch.topk(att_weight, 2, dim=1)   # indices的形状[b*h*w, 2]

            pos = self.weight[indices[:,0]]    # [b*h*w, c]， 与input中每个像素最近的memory item
            neg = self.weight[indices[:,1]]    # [b*h*w, c],  与input中每个像素第二近的memory item

            triplet_loss = lossFunc(input, pos, neg)
        else:
            _, indices = torch.topk(att_weight, 1, dim=1)   # indices的形状[b*h*w, 1]

            pos = self.weight[indices[:,0]]    # [b*h*w, c]， 与input中每个像素最近的memory item

            triplet_loss = torch.nn.MSELoss()(input, pos)

        return triplet_loss

    def compact_loss(self, input, att_weight, label):
        ''' input形状[b*h*w, c]， att_weight形状[b*h*w, m], label的形状是[b],对于label是1的样本不计算compact_loss

        让input中每一个像素与self.weight中最近的memory item更近
         '''
        
        b = label.shape[0]
        m, c = self.weight.shape

        _, indices = torch.topk(att_weight, 1, dim=1)  # indices的形状[b*h*w, 1]

        pos = self.weight[indices[:,0]]   # [b*h*w, c], 与input中每个像素最近的memory item

        compact_loss = torch.nn.MSELoss()(input, pos)  # [b*h*w, c]
        # compact_loss = torch.reshape(compact_loss, [b, -1, c])
        # compact_loss = compact_loss[label==0]
        # compact_loss = compact_loss.mean()
        return compact_loss

    def distance_loss(self):
        ''' 让self.weight中的memory item彼此都离得更远 '''
        m, c = self.weight.size()
        if m == 1:
            return 0

        margin = 1
        distance = margin - torch.square(self.weight.unsqueeze(0) - self.weight.unsqueeze(1)).sum(-1)   # distance形状[m, m]，表示m个memory item互相之间的距离
        mask = distance > 0

        # 只保留距离小于margin的， 距离大于margin的置为0，不需要优化（不产生损失）
        distance *= mask.float()

        # 只保留上三角矩阵（不包含主对角线）
        distance = torch.triu(distance, diagonal=1)

        distance_loss = distance.sum()*2 / (m*(m-1))

        return distance_loss


    def forward(self, input:torch.Tensor, label_batch):
        ''' 输入形状[b*h*w, c], self.weight形状[m, c] '''
        T,C = input.size()
        M,C = self.weight.size()
        # att_weight = -torch.norm(input.reshape(T, 1, C) - self.weight.reshape(1, M, C), dim=2)    # TxM

        att_weight = F.linear(input, self.weight)  # Fea x Mem^T, (TxC) x (CxM) = TxM

        att_weight = F.softmax(att_weight, dim=1)  # TxM
        
        # import numpy as np
        # torch.set_printoptions(threshold=np.inf)
        # print(torch.max(att_weight,dim=1))
        # print(torch.norm(self.weight, dim=1))
        # quit()
        triplet_loss = self.triplet_margin_loss(input, att_weight)
        compact_loss = self.compact_loss(input, att_weight, label_batch)
        distance_loss = self.distance_loss()

        # ReLU based shrinkage, hard shrinkage for positive value
        if(self.shrink_thres>0):
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            # att_weight = F.softshrink(att_weight, lambd=self.shrink_thres)
            # normalize???
            att_weight = F.normalize(att_weight, p=1, dim=1)

            # att_weight = F.softmax(att_weight, dim=1)
            # att_weight = self.hard_sparse_shrink_opt(att_weight)

        mem_trans = self.weight.permute(1, 0)  # Mem^T, MxC
        output = F.linear(att_weight, mem_trans)  # AttWeight x Mem^T^T = AW x Mem, (TxM) x (MxC) = TxC
        # output = F.normalize(output, dim=-1)

        norm_loss = torch.abs(1 - torch.norm(self.weight, 2, dim=1)).mean()
        entropy_loss = EntropyLoss()(att_weight)
        return output, entropy_loss, triplet_loss, norm_loss, compact_loss, distance_loss   # output, att_weight

    def extra_repr(self):
        return 'mem_size={}, fea_dim={}'.format(
            self.mem_size, self.fea_dim is not None
        )

# relu based hard shrinkage function, only works for positive values
def hard_shrink_relu(input, lambd=0, epsilon=1e-12):
    output = (F.relu(input-lambd) * input) / (torch.abs(input - lambd) + epsilon)
    return output

class BlockMemory(torch.nn.Module):
    def __init__(self, block_list, mem_size_list, fea_dim, pos_ebd_weight = 1., shrink_thres=0.0025, device='cuda') -> None:
        super().__init__()

        memorys = []
        pos_embeddings = []
        self.upsamplers = []
        self.block_list = block_list
        self.device = device
        self.pos_ebd_weight = pos_ebd_weight

        for i in range(len(block_list)):
            memorys.append(MemoryUnit(input_size = block_list[i], mem_size = mem_size_list[i], fea_dim = fea_dim, shrink_thres = shrink_thres, device = device))
            pos_embeddings.append(Parameter(torch.randn((1, fea_dim, block_list[i], block_list[i]))))

        self.pos_embeddings = nn.ParameterList(pos_embeddings)
        self.memorys = nn.ModuleList(memorys)

    def spatial_pyramid_pool(self, input, out_pool_size:list):
        '''
        previous_conv: a tensor vector of previous convolution layer
        num_sample: an int number of image in the batch
        previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
        out_pool_size: a int vector of expected output size of max pooling layer
        
        returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
        '''    
        # print(previous_conv.size())
        b, c, h, w = input.size()

        input_pool_list = []

        for i in range(len(out_pool_size)):
            if out_pool_size[i] == h and out_pool_size[i] == w:
                input_pool = input
            elif out_pool_size[i] == 1:
                maxpool = nn.MaxPool2d((h, w), padding=0)
                input_pool = maxpool(input)
            else:
                h_wid = int(math.ceil(h / out_pool_size[i]))
                w_wid = int(math.ceil(w / out_pool_size[i]))
                h_pad = math.ceil((h_wid*out_pool_size[i] - h )/2)
                w_pad = math.ceil((w_wid*out_pool_size[i] - w )/2)
                maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
                input_pool = maxpool(input)
            input_pool_list.append(input_pool)

        return input_pool_list

    def spatial_pyramid_upsample(self, input_list, dsize):
        input_upsample_concat = []
        for input in input_list:
            input_upsample = F.interpolate(input=input, size=dsize, mode='bilinear')
            input_upsample_concat.append(input_upsample)

        input_upsample_concat = torch.concat(input_upsample_concat, dim=1)
        
        return input_upsample_concat
    
    def forward(self, input:torch.Tensor, label_batch):
        ''' input shape [b, c, h, w] '''
        b, c, h, w = input.size()

        input_pool_list = self.spatial_pyramid_pool(input, self.block_list)

        memory_rec_list = []
        entropy_loss_list = []
        triplet_loss_list = []
        norm_loss_list = []
        compact_loss_list = []
        distance_loss_list = []
        
        for i, input_pool in enumerate(input_pool_list):
            b, c, h_i, w_i = input_pool.size()

            # 加入位置编码 相加
            # input_pool = input_pool + self.pos_ebd_weight * self.pos_embeddings[i]

            input_pool = input_pool.permute((0,2,3,1)).reshape((-1,c))

            memory_rec, entropy_loss, triplet_loss, norm_loss, compact_loss, distance_loss = self.memorys[i](input_pool,label_batch)

            memory_rec = memory_rec.reshape((b, h_i, w_i, c)).permute((0, 3, 1, 2))

            memory_rec_list.append(memory_rec)
            entropy_loss_list.append(entropy_loss)
            triplet_loss_list.append(triplet_loss)
            norm_loss_list.append(norm_loss)
            compact_loss_list.append(compact_loss)
            distance_loss_list.append(distance_loss)
        

        upsample_concat = self.spatial_pyramid_upsample(memory_rec_list, (h,w))

        return upsample_concat, sum(entropy_loss_list), sum(triplet_loss_list), sum(norm_loss_list), sum(compact_loss_list), sum(distance_loss_list)

if __name__ == '__main__':

    cpt_path = './checkpoints/skipMem_noPositive/cable/epoch_10.pth'

    checkpoint = torch.load(cpt_path)

    for key in checkpoint:
        print(key)


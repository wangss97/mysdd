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
    

class MemoryUnit_prototype(nn.Module):
    def __init__(self,fea_map_size, mem_size, fea_dim, pos=False, skip= False):
        super(MemoryUnit_prototype, self).__init__()
        # 有几个mem item
        self.mem_size = mem_size
        # mem item的维度， 同时也是像素的维度
        self.fea_dim = fea_dim

        self.pos_embedding = Parameter(torch.randn((1, fea_dim, fea_map_size, fea_map_size)))

        self.Mheads = nn.Linear(fea_dim, mem_size, bias=False)

        self.pos = pos
        self.skip = skip
        print('proto mem:',end='')
        if self.pos:
            print('使用位置编码')
        else:
            print('不使用位置编码')
        print('proto mem:',end='')
        if self.skip:
            print('mem跳步连接')
        else:
            print('mem没有跳步连接')

        # self.new_cpt = 'False'
        # print('proto mem: 使用旧的compact loss')
        self.new_cpt = 'True'
        print('proto mem: 使用新的compact loss')
        # self.new_cpt = 'mix'
        # print('proto mem: 使用混合compact loss')

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.mem_size)
        torch.nn.init.uniform_(self.Mheads.weight, -stdv, stdv)
   
    def compact_loss(self, proto:torch.Tensor, keys:torch.Tensor, score_of_proto:torch.Tensor):
        ''' proto形状[b, m, d]， keys形状[b, h*w, d], score_of_proto形状[b, h*w, m],

        让keys中每一个像素与距离它最近的proto向量更近
         '''
        B, M, D = proto.size()

        _, gathering_indices = torch.topk(score_of_proto, 2, dim=-1)  # gathering_indicces形状[b, h*w, 2]

        # 1st closest memories
        # proto的形状[b, m, c]
        # gathering_indices[:,:,:1].repeat((1,1,dims))形状[b, h*w, c]
        # pos 的形状[b, h*w, c] 
        if self.new_cpt == 'True':
            pos = torch.gather(proto,1,gathering_indices[:,:,:1].repeat((1,1,D)))
            compact_loss = torch.nn.MSELoss()(keys, pos)
        elif self.new_cpt == 'False':
            gathering_indices = gathering_indices[:,:,:1].repeat((1,1,D))
            gathering_indices = torch.clip(gathering_indices, min=0, max=keys.shape[1]-1)
            pos = torch.gather(keys,1,gathering_indices)
            compact_loss = torch.nn.MSELoss()(keys, pos)
        elif self.new_cpt == 'mix':
            pos = torch.gather(proto,1,gathering_indices[:,:,:1].repeat((1,1,D)))
            compact_loss1 = torch.nn.MSELoss()(keys, pos)

            gathering_indices = gathering_indices[:,:,:1].repeat((1,1,D))
            gathering_indices = torch.clip(gathering_indices, min=0, max=keys.shape[1]-1)
            pos = torch.gather(keys,1,gathering_indices)
            compact_loss2 = torch.nn.MSELoss()(keys, pos)

            compact_loss = compact_loss1 + compact_loss2

        return compact_loss

    def distance_loss(self, proto:torch.Tensor):
        ''' 让proto中的memory item彼此都离得更远, proto形状[b, m, fea_dim] '''

        # [b, 1, m, d] - [b, m, 1, d] = [b, m, m, d]
        # dis的形状是[b, m, m]，表示proto之间的相似度， 1是margin
        dis = 1 - torch.square(proto.unsqueeze(1) - proto.unsqueeze(2)).sum(-1)
        
        mask = dis>0
        # 此处dis只保留了距离在1以内的proto (目标是所有proto之间的距离都大于margin)
        dis *= mask.float()
        # 保留上三角矩阵,不保留主对角线
        dis = torch.triu(dis, diagonal=1)
        # 所有proto之间的距离相加,一共self.mem_size*(self.mem_size-1) / 2个距离, 期望 1-dis越来越小,即dis越来越大,即proto之间的距离都变大
        dis_loss = dis.sum(1).sum(1)*2/(self.mem_size*(self.mem_size-1))
        dis_loss = dis_loss.mean()

        return dis_loss

    def l1_loss(self, score_of_proto:torch.Tensor):
        ''' 让score_of_proto尽量稀疏的损失 '''
        return torch.norm(score_of_proto, p=1, dim=list(range(len(score_of_proto.shape))))

    def get_score(self, proto:torch.Tensor, query:torch.Tensor):
        ''' proto 形状 [bs, m, fea_dim]
            query 形状 [bs, h*w, fea_dim]
            该函数输出形状 [bs, h*w, m]，输出像素与每个原型的相关度
         '''
        b, m, d = proto.size()
        b, n, d = query.size()

        # [b, n, d] bmm [b, d, m] = [b, n, m]
        score = torch.bmm(query, proto.permute(0,2,1))

        score_of_proto = torch.softmax(score, dim=-1)

        return score_of_proto

    def forward(self, input:torch.Tensor, label_batch):
        ''' 输入形状[b, c, h, w]
        流程，1，用key和Mheads算出每个像素对原型的贡献度
        2，用key和贡献度算出原型
        3，用query和原型算出和原型的相关度
        4，用相关度和原型得到重构后的query作为输出
         '''
        B, C, H, W = input.size()
        # 加入位置编码 相加
        if self.pos:
            input = input + self.pos_embedding

        # key用于构造proto
        key = input
        key = key.permute(0,2,3,1).reshape(B, -1, C)
        # query用于构造输出
        query = input
        query = query.permute(0,2,3,1).reshape(B, -1, C)

        proto_weight = self.Mheads(key)  # 输出为[b, h*w, m]
        # 对proto_weight进行softmax，维度是h*w，表示一副图上所有像素对某一个原型的贡献度和是1
        proto_weight = F.softmax(proto_weight, dim = 1)

        # [b, m, h*w] 矩阵乘 [b, h*w, d] 得到 [b, m, d]
        proto = torch.matmul(proto_weight.permute(0,2,1),key)

        # 令每个proto向量是标准向量
        proto = F.normalize(proto, dim=-1)

        score_of_proto = self.get_score(proto, query)  # [b, h*w, m]

        # [b, h*w, m] 矩阵乘  [b, m, d] 得到 [b, h*w, d]，表示重组后的特征图  
        new_query = torch.matmul(score_of_proto, proto)
        # 令重组的特征图的每个像素是标准向量
        new_query = F.normalize(new_query, dim=-1)
 
        # 获得训练proto用的损失 
        compact_loss = self.compact_loss(proto, query, score_of_proto)
        distance_loss = self.distance_loss(proto)
        l1_loss = self.l1_loss(score_of_proto)

        # 跳步连接
        if self.skip:
            new_query = new_query + query

        # reshape
        new_query = new_query.permute(0,2,1).reshape(B, C, H, W)

        # return new_query, compact_loss, distance_loss, l1_loss
        return new_query, compact_loss, distance_loss


class MemoryUnit(nn.Module):
    def __init__(self,fea_map_size, mem_size, fea_dim, shrink_thres=0.0025,pos=False,skip=False):
        super(MemoryUnit, self).__init__()
        self.mem_size = mem_size
        self.fea_dim = fea_dim
        self.weight = Parameter(torch.Tensor(self.mem_size, self.fea_dim))  # M x C

        self.pos_embedding = Parameter(torch.randn((1, fea_dim, fea_map_size, fea_map_size)))

        # self.acti = torch.nn.Sigmoid()
        self.bias = None
        self.shrink_thres= shrink_thres
        # self.hard_sparse_shrink_opt = nn.Hardshrink(lambd=shrink_thres)

        self.pos = pos
        self.skip = skip
        print('mem:',end='')
        if self.pos:
            print('使用位置编码')
        else:
            print('不使用位置编码')
        if self.skip:
            print('mem跳步连接')
        else:
            print('mem没有跳步连接')

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        torch.nn.init.uniform_(self.weight, -stdv, stdv)
        # self.weight.data.uniform_(-stdv, stdv)
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
        ''' 输入形状[b,c,h,w], self.weight形状[m, c] '''
        b, c, h, w = input.size()
        
        # 位置编码
        if self.pos:
            input = input + self.pos_embedding

        input_reshape = input.permute((0,2,3,1)).reshape((-1,c))
        M,C = self.weight.size()
        # att_weight = -torch.norm(input.reshape(T, 1, C) - self.weight.reshape(1, M, C), dim=2)    # TxM

        att_weight = F.linear(input_reshape, self.weight)  # Fea x Mem^T, (TxC) x (CxM) = TxM

        att_weight = F.softmax(att_weight, dim=1)  # TxM
        
        # import numpy as np
        # torch.set_printoptions(threshold=np.inf)
        # print(torch.max(att_weight,dim=1))
        # print(torch.norm(self.weight, dim=1))
        # quit()
        triplet_loss = self.triplet_margin_loss(input_reshape, att_weight)
        compact_loss = self.compact_loss(input_reshape, att_weight, label_batch)
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
        output = output.reshape((b,h,w,c)).permute((0,3,1,2))

        # 跳步连接
        if self.skip:
            output = output + input

        norm_loss = torch.abs(1 - torch.norm(self.weight, 2, dim=1)).mean()
        entropy_loss = EntropyLoss()(att_weight)
        # return output, entropy_loss, triplet_loss, norm_loss, compact_loss, distance_loss   # output, att_weight
        return output, compact_loss, distance_loss   # output, att_weight

    def extra_repr(self):
        return 'mem_size={}, fea_dim={}'.format(
            self.mem_size, self.fea_dim is not None
        )

# relu based hard shrinkage function, only works for positive values
def hard_shrink_relu(input, lambd=0, epsilon=1e-12):
    output = (F.relu(input-lambd) * input) / (torch.abs(input - lambd) + epsilon) 
    return output

class BlockMemory(torch.nn.Module):
    def __init__(self, block_list=[16], mem_size_list=[50], fea_dim=1024, shrink_thres=0.0025,pos=False,skip=False, device='cuda') -> None:
        super().__init__()

        memorys = []
        self.block_list = block_list
        self.device = device

        for i in range(len(block_list)):
            memorys.append(MemoryUnit_prototype(fea_map_size=block_list[i], mem_size=mem_size_list[i], fea_dim=fea_dim, pos=pos, skip=skip))
            # memorys.append(MemoryUnit(fea_map_size=block_list[i], mem_size = mem_size_list[i], fea_dim = fea_dim, shrink_thres = shrink_thres,pos=pos,skip=skip))

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
        l1_loss_list = []
        
        for i, input_pool in enumerate(input_pool_list):

            entropy_loss, triplet_loss, norm_loss, l1_loss = 0, 0, 0, 0

            memory_rec, compact_loss, distance_loss = self.memorys[i](input_pool,label_batch)

            memory_rec_list.append(memory_rec)
            entropy_loss_list.append(entropy_loss)
            triplet_loss_list.append(triplet_loss)
            norm_loss_list.append(norm_loss)
            compact_loss_list.append(compact_loss)
            distance_loss_list.append(distance_loss)
            l1_loss_list.append(l1_loss)
        

        upsample_concat = self.spatial_pyramid_upsample(memory_rec_list, (h,w))

        return upsample_concat, sum(entropy_loss_list), sum(triplet_loss_list), \
            sum(norm_loss_list), sum(compact_loss_list), sum(distance_loss_list), sum(l1_loss_list)

if __name__ == '__main__':

    cpt_path = './checkpoints/skipMem_noPositive/cable/epoch_10.pth'

    checkpoint = torch.load(cpt_path)

    for key in checkpoint:
        print(key)


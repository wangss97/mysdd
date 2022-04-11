import torch
import torch.nn as nn
import utils
from torch.autograd import Variable


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
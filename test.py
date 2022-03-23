import torch
import torch.nn as nn

cpt = torch.load('./checkpoints/segNet_conMem3/cable/epoch_10.pth')
# cpt = torch.load('./checkpoints/segNet_conMem2/cable/epoch_10.pth')

for key, value in cpt.items():
    if 'pos_embeddings.1' in key:
        print(value.shape)
        print(torch.mean(value))
        print(torch.sum(value))
        print(torch.std(value))

# cpt = torch.load('./checkpoints/segNet_conMem2/cable/epoch_20.pth')

cpt = torch.load('./checkpoints/segNet_conMem3/cable/epoch_20.pth')
for key, value in cpt.items():
    if 'pos_embeddings.1' in key:
        print(value.shape)
        print(torch.mean(value))
        print(torch.sum(value))
        print(torch.std(value))
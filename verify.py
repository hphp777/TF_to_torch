from datasets import ChestXLoader, ChestXTestLoader
import torch
from matplotlib import pyplot as plt

data = ChestXTestLoader()
train_loader = torch.utils.data.DataLoader(data, batch_size = 4, shuffle = False)

for batch_idx, (img, target) in enumerate(train_loader):
    # img = img[0].permute(1,2,0)
    # plt.imshow(img.numpy())
    # plt.show()
    # print(target)
    pass
import torch
from models import *
import time
import numpy as np
from utils import Mydataset
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt


def inference(eval_loader, model, filePath, show=False):
    para_state = torch.load(filePath)
    model.load_state_dict(para_state['net'])
    start = time.time()
    for idx, (eval_x, eval_y) in enumerate(eval_loader):
        eval_x, eval_y = eval_x.cuda(), eval_y.cuda()
        out = model(eval_x)
        eval_x = torch.permute(eval_x, dims=[0, 2, 3, 1])
        for i in range(eval_x.shape[0]):
            if show:
                plt.imshow(eval_x[i].cpu().detach().numpy())
                plt.title(str(eval_y[i].cpu().detach().numpy()))
                plt.show()
    end  = time.time()
    endurance = end - start
    print(f"consumption:{endurance}秒")




if __name__ == "__main__":
    bs = 8
    n_w = 4
    filePath = "./assets/para.pth"
    eval_data = Mydataset(MNIST(root='./lib', train=False, download=True))
    eval_loader = DataLoader(
        dataset=eval_data,
        batch_size=bs,
        shuffle=False,
        num_workers=n_w
    )
    model = Net().cuda()
    inference(eval_loader, model, filePath)


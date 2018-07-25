# from sklearn.metrics import precision_recall_fscore_support
from Pytorch.Network import Unet
from Pytorch.Dataset import Custom_dataset
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

torch.set_printoptions(profile='full')
if __name__ == '__main__':
    device = torch.device("cuda")
    model_dir = 'models/07121619/25epo_210000step.ckpt'
    model = torch.load(model_dir).to(device)
    dataset = Custom_dataset(root_dir='../test')
    dataloader = DataLoader(dataset, 4, shuffle=False)
    writer = SummaryWriter('log/Image_test')
    model.eval()
    for i, batch in enumerate(dataloader):
        img = batch.to(device)
        # mask = batch['mask'].to(device)
        with torch.no_grad():
            pred, loss = model(img)
        pred = pred[5].data
        pred = pred.requires_grad_(False)
        writer.add_image(model_dir+', img', img, i)
        writer.add_image(model_dir+', mask', pred, i)
    writer.close()


# from sklearn.metrics import precision_recall_fscore_support
from network import Unet
from dataset import DUTSdataset
import torch
import os
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import argparse
from sklearn.metrics import precision_recall_curve

torch.set_printoptions(profile='full')
if __name__ == '__main__':

    models = sorted(os.listdir('models/state_dict/10151622'), key=lambda x: int(x.split('epo_')[1].split('step')[0]))
    duts_dataset = DUTSdataset(root_dir='../DUTS-TE', train=False)
    dataloader = DataLoader(duts_dataset, 4, shuffle=True)
    beta_square = 0.3
    device = torch.device("cuda")
    writer = SummaryWriter('log/F_Measure/10151622_20181119')
    model = Unet().to(device)
    for model_name in models:
        if int(model_name.split('epo_')[1].split('step')[0]) % 1000 != 0:
            continue
        # if int(model_name.split('epo_')[1].split('step')[0]) != 383000:
        #     continue
        state_dict = torch.load('models/state_dict/10151622/' + model_name)
        model.load_state_dict(state_dict)
        model.eval()
        mae = 0
        preds = []
        masks = []
        precs = []
        recalls = []
        for i, batch in enumerate(dataloader):
            img = batch['image'].to(device)
            mask = batch['mask'].to(device)
            with torch.no_grad():
                pred, loss = model(img, mask)
            pred = pred[5].data
            mae += torch.mean(torch.abs(pred - mask))
            pred = pred.requires_grad_(False)
            preds.append(pred)
            masks.append(mask)
            prec, recall = torch.zeros(mask.shape[0], 256), torch.zeros(mask.shape[0], 256)
            pred = pred.squeeze(dim=1).cpu()
            mask = mask.squeeze(dim=1).cpu()
            thlist = torch.linspace(0, 1 - 1e-10, 256)
            for j in range(256):
                y_temp = (pred >= thlist[j]).float()
                tp = (y_temp * mask).sum(dim=-1).sum(dim=-1)
                # avoid prec becomes 0
                prec[:, j], recall[:, j] = (tp + 1e-10) / (y_temp.sum(dim=-1).sum(dim=-1) + 1e-10), (tp + 1e-10) / (mask.sum(dim=-1).sum(dim=-1) + 1e-10)
            # (batch, threshold)
            precs.append(prec)
            recalls.append(recall)

        prec = torch.cat(precs, dim=0).mean(dim=0)
        recall = torch.cat(recalls, dim=0).mean(dim=0)
        f_score = (1 + beta_square) * prec * recall / (beta_square * prec + recall)
        thlist = torch.linspace(0, 1 - 1e-10, 256)
        writer.add_scalar("Max F_score", torch.max(f_score),
                          global_step=int(model_name.split('epo_')[1].split('step')[0]))
        writer.add_scalar("Max_F_threshold", thlist[torch.argmax(f_score)],
                          global_step=int(model_name.split('epo_')[1].split('step')[0]))
        pred = torch.cat(preds, 0)
        mask = torch.cat(masks, 0).round().float()
        writer.add_pr_curve('PR_curve', mask, pred, global_step=int(model_name.split('epo_')[1].split('step')[0]))
        writer.add_scalar('MAE', torch.mean(torch.abs(pred - mask)), global_step=int(model_name.split('epo_')[1].split('step')[0]))
        # Measure method from https://github.com/AceCoooool/DSS-pytorch solver.py
        pred = pred.cpu()
        mask = mask.cpu()
        print(model_name.split('epo_')[1].split('step')[0])

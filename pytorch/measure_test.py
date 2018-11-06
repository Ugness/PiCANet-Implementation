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
    dataloader = DataLoader(duts_dataset, 8, shuffle=True)
    beta_square = 0.3
    device = torch.device("cuda")
    writer = SummaryWriter('log/F_Measure/10151622_adjusted')
    model = Unet().to(device)
    for model_name in models:
        if int(model_name.split('epo_')[1].split('step')[0]) % 1000 != 0:
            continue

        state_dict = torch.load('models/state_dict/10151622/' + model_name)
        model.load_state_dict(state_dict)
        model.eval()
        mae = 0
        preds = []
        masks = []
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
        pred = torch.cat(preds, 0)
        mask = torch.cat(masks, 0)
        writer.add_pr_curve('PR_curve', mask, pred, global_step=int(model_name.split('epo_')[1].split('step')[0]))
        writer.add_scalar('MAE', torch.mean(torch.abs(pred - mask)), global_step=int(model_name.split('epo_')[1].split('step')[0]))
        # Measure method from https://github.com/AceCoooool/DSS-pytorch solver.py
        pred = pred.cpu()
        mask = mask.round().float().cpu()
        prec, recall = torch.zeros(256), torch.zeros(256)
        thlist = torch.linspace(0, 1 - 1e-10, 256)
        for i in range(256):
            y_temp = (pred >= thlist[i]).float()
            tp = (y_temp * mask).sum()
            # avoid prec becomes 0
            prec[i], recall[i] = (tp + 1e-10) / (y_temp.sum() + 1e-10), (tp + 1e-10) / (mask.sum() + 1e-10)
        f_score = (1 + beta_square) * prec * recall / (beta_square * prec + recall)
        print(torch.max(f_score))
        writer.add_scalar("Max F_score", torch.max(f_score), global_step=int(model_name.split('epo_')[1].split('step')[0]))
        writer.add_scalar("Max_F_threshold", thlist[torch.argmax(f_score)], global_step=int(model_name.split('epo_')[1].split('step')[0]))
        print(model_name.split('epo_')[1].split('step')[0])
        """
        for edge in range(100):
            threshold = edge/100.0
            avg_precision, avg_recall, avg_fscore = [], [], []
            tp, tn, fp, fn, = 0, 0, 0, 0
            for i, batch in enumerate(dataloader):
                img = batch['image'].to(device)
                mask = batch['mask'].to(device)
                with torch.no_grad():
                    pred, loss = model(img, mask)
                pred = pred[5].data
                writer.add_pr_curve('1234', mask, pred)
                mae += F.mse_loss(pred, mask)
                pred = pred.requires_grad_(False)
                pred = torch.round(pred + threshold - 0.5).data
                t = mask.type(torch.cuda.FloatTensor)
                p = pred.type(torch.cuda.FloatTensor)
                f = 1 - mask.type(torch.cuda.FloatTensor)
                n = 1 - pred.type(torch.cuda.FloatTensor)
                # based on http://blog.acronym.co.kr/556
                tp += float(torch.sum(t * p))
                tn += float(torch.sum(f * n))
                fp += float(torch.sum(f * p))
                fn += float(torch.sum(t * n))
                if i % 100 == 0 and i > 0:
                    print('Model: '+model_name)
                    print('i: ', i)
                    print('tp: '+str(tp))
                    print('tn: '+str(tn))
                    print('fp: '+str(fp))
                    print('fn: '+str(fn))
                    break
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            fscore = (1 + beta_square) * precision * recall / (beta_square * precision + recall)
            writer.add_scalar('precision', precision, global_step=int(model_name.split('epo_')[1].split('step')[0]))
            writer.add_scalar('recall', recall, global_step=int(model_name.split('epo_')[1].split('step')[0]))
            writer.add_scalar('F_score', fscore, global_step=int(model_name.split('epo_')[1].split('step')[0]))
            print('Model : ' + model_name)
            print('Threshold : '+str(threshold))
            print('Precision : ' + str(precision))
            print('Recall : ' + str(recall))
            print('F_score : ' + str(fscore))
        print('MAE:' + str(mae / 10000))
        writer.add_scalar('MAE', mae / 10000, global_step=int(model_name.split('epo_')[1].split('step')[0]))
        """
# from sklearn.metrics import precision_recall_fscore_support
from Pytorch.Network import Unet
from Pytorch.Dataset import DUTS_dataset
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

torch.set_printoptions(profile='full')
if __name__ == '__main__':
    models = sorted(os.listdir('models/07121619'), key=lambda x: int(x.split('epo_')[1].split('step')[0]))
    dataset = DUTS_dataset(root_dir='../DUTS-TE', train=False)
    dataloader = DataLoader(dataset, 4, shuffle=False)
    beta_square = 0.3
    device = torch.device("cuda")
    writer = SummaryWriter('log/test')
    for model_name in models:
        if int(model_name.split('epo_')[1].split('step')[0]) < 17000:
            continue
        if int(model_name.split('epo_')[1].split('step')[0]) % 10000 != 0:
            continue
        model = torch.load('models/07121619/' + model_name).to(device)
        model.eval()
        avg_precision, avg_recall, avg_fscore = [], [], []
        for i, batch in enumerate(dataloader):
            img = batch['image'].to(device)
            mask = batch['mask'].to(device)
            with torch.no_grad():
                pred, loss = model(img, mask)
            pred = pred[5].data
            pred = pred.requires_grad_(False)
            pred = torch.round(pred).data
            t = mask.type(torch.cuda.FloatTensor)
            p = pred.type(torch.cuda.FloatTensor)
            f = 1 - mask.type(torch.cuda.FloatTensor)
            n = 1 - pred.type(torch.cuda.FloatTensor)
            # based on http://blog.acronym.co.kr/556
            tp = torch.sum(t * p)
            tn = torch.sum(f * n)
            fp = torch.sum(f * p)
            fn = torch.sum(t * n)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            fscore = (1 + beta_square) * precision * recall / (beta_square * precision + recall)
            # precision, recall, fscore, _ = \
            #     precision_recall_fscore_support(mask.data.cpu().numpy(), pred.data.cpu().numpy(), average='binary')
            avg_precision.append(float(precision))
            avg_recall.append(float(recall))
            avg_fscore.append(float(fscore))
            if i % 100 == 0:
                print(model_name, i,
                      sum(avg_precision) / float(len(avg_precision)),
                      sum(avg_recall) / float(len(avg_recall)),
                      sum(avg_fscore) / float(len(avg_fscore)))
        avg_precision = sum(avg_precision) / float(len(avg_precision))
        avg_recall = sum(avg_recall) / float(len(avg_recall))
        avg_fscore = sum(avg_fscore) / float(len(avg_fscore))
        writer.add_scalar('precision', avg_precision, global_step=int(model_name.split('epo_')[1].split('step')[0]))
        writer.add_scalar('recall', avg_recall, global_step=int(model_name.split('epo_')[1].split('step')[0]))
        writer.add_scalar('F_score', avg_fscore, global_step=int(model_name.split('epo_')[1].split('step')[0]))
        print('Model : ' + model_name)
        print('Precision : ' + str(avg_precision))
        print('Recall : ' + str(avg_recall))
        print('F_score : ' + str(avg_fscore))
        print('F_score : ' + str(
            (1 + beta_square) * avg_precision * avg_recall / (beta_square * avg_precision + avg_recall)))
        avg_precision, avg_recall, avg_fscore = [], [], []

# from sklearn.metrics import precision_recall_fscore_support
from Network import Unet
from Dataset import DUTS_dataset
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

torch.set_printoptions(profile='full')
if __name__ == '__main__':
    models = sorted(os.listdir('models/state_dict/07121619'), key=lambda x: int(x.split('epo_')[1].split('step')[0]))
    dataset = DUTS_dataset(root_dir='../DUTS-TE', train=False)
    dataloader = DataLoader(dataset, 4, shuffle=False)
    beta_square = 0.3
    device = torch.device("cuda")
    writer = SummaryWriter('log/test2_X_round')
    model = Unet().to(device)
    for model_name in models:
        # if int(model_name.split('epo_')[1].split('step')[0]) < 259000:
        #     continue
        if int(model_name.split('epo_')[1].split('step')[0]) % 10000 != 0:
            continue
        state_dict = torch.load('models/state_dict/07121619/' + model_name)
        model.load_state_dict(state_dict)
        model.eval()
        avg_precision, avg_recall, avg_fscore = [], [], []
        tp, tn, fp, fn = 0, 0, 0, 0
        threshold = 0.5
        for i, batch in enumerate(dataloader):
            img = batch['image'].to(device)
            mask = batch['mask'].to(device)
            with torch.no_grad():
                pred, loss = model(img, mask)
            pred = pred[5].data
            pred = pred.requires_grad_(False) + (0.5-threshold)
            pred = torch.round(pred).data
            mask = torch.round(mask).data
            t = mask.type(torch.cuda.FloatTensor)
            p = pred.type(torch.cuda.FloatTensor)
            f = 1 - mask.type(torch.cuda.FloatTensor)
            n = 1 - pred.type(torch.cuda.FloatTensor)
            # based on http://blog.acronym.co.kr/556
            tp += float(torch.sum(t * p))
            tn += float(torch.sum(f * n))
            fp += float(torch.sum(f * p))
            fn += float(torch.sum(t * n))
            if i % 100 == 0:
                print('Model: '+model_name)
                print('i: ', i)
                print('tp: '+str(tp))
                print('tn: '+str(tn))
                print('fp: '+str(fp))
                print('fn: '+str(fn))
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fscore = (1 + beta_square) * precision * recall / (beta_square * precision + recall)
        writer.add_scalar('precision', precision, global_step=int(model_name.split('epo_')[1].split('step')[0]))
        writer.add_scalar('recall', recall, global_step=int(model_name.split('epo_')[1].split('step')[0]))
        writer.add_scalar('F_score', fscore, global_step=int(model_name.split('epo_')[1].split('step')[0]))
        print('Model : ' + model_name)
        print('Precision : ' + str(precision))
        print('Recall : ' + str(recall))
        print('F_score : ' + str(fscore))
from network import Unet
from dataset import PairDataset
import torch
import os
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import argparse
from tqdm import tqdm

torch.set_printoptions(profile='full')
if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    print("Default path : " + os.getcwd())
    parser.add_argument("--model_dir", required=True,
                        help="Directory of folder which contains pre-trained models, you can download at \n"
                             "https://drive.google.com/drive/folders/1s4M-_SnCPMj_2rsMkSy3pLnLQcgRakAe?usp=sharing")
    parser.add_argument('--dataset', help='Directory of your test_image ""folder""', required=True)
    parser.add_argument('--cuda', help="cuda for cuda, cpu for cpu, default = cuda", default='cuda')
    parser.add_argument('--batch_size', help="batchsize, default = 4", default=4, type=int)
    parser.add_argument('--logdir', help="logdir, log on tensorboard", default=None)
    parser.add_argument('--which_iter', help="Specific Iter to measure", default=-1, type=int)
    parser.add_argument('--cont', help="Measure scores from this iter", default=0, type=int)
    parser.add_argument('--step', help="Measure scores per this iter step", default=10000, type=int)

    args = parser.parse_args()

    models = sorted(os.listdir(args.model_dir), key=lambda x: int(x.split('epo_')[1].split('step')[0]))
    pairdataset = PairDataset(root_dir=args.dataset, train=False, data_augmentation=False)
    dataloader = DataLoader(pairdataset, 8, shuffle=True)
    beta_square = 0.3
    device = torch.device("cuda")
    if args.logdir is not None:
        writer = SummaryWriter(args.logdir)
    model = Unet().to(device)
    for model_name in models:
        model_iter = int(model_name.split('epo_')[1].split('step')[0])
        if model_iter % args.step != 0:
            continue
        if model_iter < args.cont:
            continue
        if args.which_iter > 0 and args.which_iter != model_iter:
            continue
        state_dict = torch.load(os.path.join(args.model_dir, model_name))
        model.load_state_dict(state_dict)
        model.eval()
        mae = 0
        preds = []
        masks = []
        precs = []
        recalls = []
        print('==============================')
        print("On iteration : " + str(model_iter))
        for i, batch in enumerate(dataloader):
            img = batch['image'].to(device)
            mask = batch['mask'].to(device)
            with torch.no_grad():
                pred, loss = model(img, mask)
            pred = pred[5].data
            mae += torch.mean(torch.abs(pred - mask))
            pred = pred.requires_grad_(False)
            preds.append(pred.cpu())
            masks.append(mask.cpu())
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
        print("Max F_score :", torch.max(f_score))
        print("Max_F_threshold :", thlist[torch.argmax(f_score)])
        if args.logdir is not None:
            writer.add_scalar("Max F_score", torch.max(f_score),
                              global_step=model_iter)
            writer.add_scalar("Max_F_threshold", thlist[torch.argmax(f_score)],
                              global_step=model_iter)
        pred = torch.cat(preds, 0)
        mask = torch.cat(masks, 0).round().float()
        if args.logdir is not None:
            writer.add_pr_curve('PR_curve', mask, pred, global_step=model_iter)
            writer.add_scalar('MAE', torch.mean(torch.abs(pred - mask)), global_step=model_iter)
        print("MAE :", torch.mean(torch.abs(pred-mask)))
        # Measure method from https://github.com/AceCoooool/DSS-pytorch solver.py

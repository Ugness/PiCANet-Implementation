# from sklearn.metrics import precision_recall_fscore_support
from Network import Unet
from Dataset import Custom_dataset
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import argparse

torch.set_printoptions(profile='full')
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    print("Default path : " + os.getcwd())
    parser.add_argument("--model_dir",
                        help="Directory of pre-trained model, you can download at "
                             "https://drive.google.com/drive/folders/1s4M-_SnCPMj_2rsMkSy3pLnLQcgRakAe?usp=sharing")
    parser.add_argument('-img', '--image_dir', help='Directory of your test_image ""folder""')
    parser.add_argument('--cuda', help="'cuda' for cuda, 'cpu' for cpu, default = cuda", default='cuda')
    parser.add_argument('--batch_size', help="batchsize, default = 4", default=4, type=int)
    args = parser.parse_args()

    print(args)
    print(os.getcwd())
    device = torch.device(args.cuda)
    # model_dir = 'models/07121619/25epo_210000step.ckpt'
    state_dict = torch.load(args.model_dir)
    model = Unet().to(device)
    model.load_state_dict(state_dict)
    dataset = Custom_dataset(root_dir=args.image_dir)
    dataloader = DataLoader(dataset, args.batch_size, shuffle=False)
    writer = SummaryWriter('log/Image_test')
    model.eval()
    for i, batch in enumerate(dataloader):
        img = batch.to(device)
        # mask = batch['mask'].to(device)
        with torch.no_grad():
            pred, loss = model(img)
        pred = pred[5].data
        pred = pred.requires_grad_(False)
        writer.add_image(args.model_dir + ', img', img, i)
        writer.add_image(args.model_dir + ', mask', pred, i)
    writer.close()

import datetime
import os
import numpy as np
import argparse

import torch
from torch.utils.data import DataLoader
import torchvision
from tensorboardX import SummaryWriter
from tqdm import tqdm

from network import Unet
from dataset import DUTSDataset, PairDataset

cfg = {'PicaNet': "GGLLL",
       'Size': [28, 28, 28, 56, 112, 224],
       'Channel': [1024, 512, 512, 256, 128, 64],
       'loss_ratio': [0.5, 0.5, 0.5, 0.8, 0.8, 1]}

if __name__ == '__main__':
    torch.cuda.manual_seed_all(1234)
    torch.manual_seed(1234)

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    print("Default path : " + os.getcwd())
    parser.add_argument("--load",
                        help="Directory of pre-trained model, you can download at \n"
                             "https://drive.google.com/file/d/109a0hLftRZ5at5hwpteRfO1A6xLzf8Na/view?usp=sharing\n"
                             "None --> Do not use pre-trained model. Training will start from random initialized model")
    parser.add_argument('--dataset', help='Directory of your Dataset', required=True, default=None)
    parser.add_argument('--cuda', help="'cuda' for cuda, 'cpu' for cpu, default = cuda",
                        default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--batch_size', help="batchsize, default = 1", default=1, type=int)
    parser.add_argument('--epoch', help='# of epochs. default = 20', default=20, type=int)
    parser.add_argument('-lr', '--learning_rate', help='learning_rate. default = 0.001', default=0.001, type=float)
    parser.add_argument('--lr_decay', help='Learning rate decrease by lr_decay time per decay_step, default = 0.1',
                        default=0.1, type=float)
    parser.add_argument('--decay_step', help='Learning rate decrease by lr_decay time per decay_step,  default = 7000',
                        default=7000, type=int)
    parser.add_argument('--display_freq', help='display_freq to display result image on Tensorboard',
                        default=1000, type=int)


    args = parser.parse_args()
    # TODO : Add multiGPU Model
    device = torch.device(args.cuda)
    batch_size = args.batch_size
    epoch = args.epoch
    duts_dataset = PairDataset(args.dataset)
    load = args.load
    start_iter = 0
    model = Unet(cfg).cuda()
    vgg = torchvision.models.vgg16(pretrained=True)
    model.encoder.seq.load_state_dict(vgg.features.state_dict())
    now = datetime.datetime.now()
    start_epo = 0
    del vgg

    if load is not None:
        state_dict = torch.load(load, map_location=args.cuda)

        start_iter = int(load.split('epo_')[1].strip('step.ckpt')) + 1
        start_epo = int(load.split('/')[3].split('epo')[0])
        now = datetime.datetime.strptime(load.split('/')[2], '%m%d%H%M')

        print("Loading Model from {}".format(load))
        print("Start_iter : {}".format(start_iter))
        print("now : {}".format(now.strftime('%m%d%H%M')))
        model.load_state_dict(state_dict)
        for cell in model.decoder:
            if cell.mode == 'G':
                cell.picanet.renet.vertical.flatten_parameters()
                cell.picanet.renet.horizontal.flatten_parameters()
        print('Loading_Complete')

    # Optimizer Setup
    learning_rate = args.learning_rate
    lr_decay = args.lr_decay
    decay_step = args.decay_step  # from 50000 step
    learning_rate = learning_rate * (lr_decay ** (start_iter // decay_step))
    opt_en = torch.optim.SGD(model.encoder.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    opt_dec = torch.optim.SGD(model.decoder.parameters(), lr=learning_rate * 10, momentum=0.9, weight_decay=0.0005)
    # Dataloader Setup
    dataloader = DataLoader(duts_dataset, batch_size, shuffle=True, num_workers=0)
    # Logger Setup
    os.makedirs(os.path.join('log', now.strftime('%m%d%H%M')), exist_ok=True)
    weight_save_dir = os.path.join('models', 'state_dict', now.strftime('%m%d%H%M'))
    os.makedirs(os.path.join(weight_save_dir), exist_ok=True)
    writer = SummaryWriter(os.path.join('log', now.strftime('%m%d%H%M')))
    iterate = start_iter
    for epo in range(start_epo, epoch):
        print("\nEpoch : {}".format(epo))
        for i, batch in enumerate(tqdm(dataloader)):
            if i > 10:
                break
            opt_dec.zero_grad()
            opt_en.zero_grad()
            img = batch['image'].to(device)
            mask = batch['mask'].to(device)
            pred, loss = model(img, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
            opt_dec.step()
            opt_en.step()
            writer.add_scalar('loss', float(loss), global_step=iterate)
            if iterate % args.display_freq == 0:
                for masked in pred:
                    writer.add_image('{}'.format(masked.size()[2]), masked, global_step=iterate)
                writer.add_image('GT', mask, iterate)
                writer.add_image('Image', img, iterate)

            if iterate % 200 == 0:
                if i != 0:
                    torch.save(model.state_dict(),
                               os.path.join(weight_save_dir, '{}epo_{}step.ckpt'.format(epo, iterate)))
            if iterate % 1000 == 0 and i != 0:
                for file in weight_save_dir:
                    if '00' in file and '000' not in file:
                        os.remove(os.path.join(weight_save_dir, file))
            if (i + epo * len(dataloader)) % decay_step == 0 and i != 0:
                learning_rate *= lr_decay
                opt_en = torch.optim.SGD(model.encoder.parameters(), lr=learning_rate, momentum=0.9,
                                         weight_decay=0.0005)
                opt_dec = torch.optim.SGD(model.decoder.parameters(), lr=learning_rate * 10, momentum=0.9,
                                          weight_decay=0.0005)
            iterate += args.batch_size
            del loss
        start_iter = 0

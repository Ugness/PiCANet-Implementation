import torch
from torch.utils.data import DataLoader
import torchvision
from Pytorch.Network import Unet
from Pytorch.Dataset import DUTS_dataset
from tensorboardX import SummaryWriter
import datetime
import os
from itertools import islice
import numpy as np

cfg = {'PicaNet': "GGLLL",
       'Size': [28, 28, 28, 56, 112, 224],
       'Channel': [1024, 512, 512, 256, 128, 64],
       'loss_ratio': [0.5, 0.5, 0.5, 0.8, 0.8, 1]}

if __name__ == '__main__':
    torch.cuda.manual_seed_all(1234)
    torch.manual_seed(1234)
    device = torch.device("cuda")
    batch_size = 1
    epoch = 200
    dataset = DUTS_dataset('../DUTS-TR')
    # load = None
    load = 'models/07121619/23epo_195600step.ckpt'
    start_iter = 0
    # noise = torch.randn((batch_size, 3, 224, 224)).type(torch.cuda.FloatTensor)
    # target = torch.randn((batch_size, 1, 224, 224)).type(torch.cuda.FloatTensor)
    # print(vgg.features(noise))
    # print(model(noise))
    # print(model.seq)
    # print(vgg.features)
    # print(F.mse_loss(model.seq[:8](noise), vgg.features[:8](noise)))
    if load is None:
        model = Unet(cfg).cuda()
        vgg = torchvision.models.vgg16(pretrained=True)
        model.encoder.seq.load_state_dict(vgg.features.state_dict())
        now = datetime.datetime.now()
        start_epo = 0
        del vgg
    else:
        model = torch.load(load)
        start_iter = int(load.split('_')[1].strip('step.ckpt')) + 1
        start_epo = int(load.split('/')[2].split('epo')[0])
        now = datetime.datetime.strptime(load.split('/')[1], '%m%d%H%M')
        print("Loading Model from {}".format(load))
        print("Start_iter : {}".format(start_iter))
        print("now : {}".format(now.strftime('%m%d%H%M')))
        for cell in model.decoder:
            if cell.mode == 'G':
                cell.picanet.renet.vertical.flatten_parameters()
                cell.picanet.renet.horizontal.flatten_parameters()
        print('Loading_Complete')
    learning_rate = 0.001
    lr_decay = 0.1
    # decay_step = 7000
    decay_step = 20000  # from 50000 step
    learning_rate = learning_rate * (lr_decay ** (start_iter // decay_step))
    opt_en = torch.optim.SGD(model.encoder.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    opt_dec = torch.optim.SGD(model.decoder.parameters(), lr=learning_rate * 10, momentum=0.9, weight_decay=0.0005)
    # opt_en = torch.optim.Adam(model.encoder.parameters(), lr=learning_rate, weight_decay=0.0005)
    # opt_dec = torch.optim.Adam(model.decoder.parameters(), lr=learning_rate * 10, weight_decay=0.0005)
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=0)
    os.makedirs('log/{}'.format(now.strftime('%m%d%H%M')), exist_ok=True)
    writer = SummaryWriter('log/{}'.format(now.strftime('%m%d%H%M')))
    # print(len(dataloader))
    iterate = start_iter
    for epo in range(start_epo, epoch):
        for i, batch in enumerate(islice(dataloader, start_iter, None)):
            # print(batch['image'].size())
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
            # print(iterate)
            if iterate % 10 == 0:
                for masked in pred:
                    writer.add_image('{}'.format(masked.size()[2]), masked, global_step=iterate)
                writer.add_image('GT', mask, iterate)
                writer.add_image('Image', img, iterate)

            if iterate % 200 == 0:
                if i != 0:
                    os.makedirs('models/{}'.format(now.strftime('%m%d%H%M')), exist_ok=True)
                    torch.save(model, 'models/{}/{}epo_{}step.ckpt'.format(now.strftime('%m%d%H%M'), epo, iterate))
                    print('models/{}/{}epo_{}step.ckpt'.format(now.strftime('%m%d%H%M'), epo, iterate))
                # for name, param in model.named_parameters():
                #     writer.add_histogram(name, param, iterate, bins=np.arange(-0.003, 0.003, 0.0001))
            if iterate % 1000 == 0 and i != 0:
                check = os.listdir('models/{}'.format(now.strftime('%m%d%H%M')))
                for file in check:
                    if '00' in file and '000' not in file:
                        os.remove('models/{}/{}'.format(now.strftime('%m%d%H%M'), file))
            if i + epo * len(dataloader) % decay_step == 0 and i != 0:
                learning_rate *= lr_decay
                opt_en = torch.optim.SGD(model.encoder.parameters(), lr=learning_rate, momentum=0.9,
                                         weight_decay=0.0005)
                opt_dec = torch.optim.SGD(model.decoder.parameters(), lr=learning_rate * 10, momentum=0.9,
                                          weight_decay=0.0005)
                # opt_en = torch.optim.Adam(model.encoder.parameters(), lr=learning_rate, weight_decay=0.0005)
                # opt_dec = torch.optim.Adam(model.decoder.parameters(), lr=learning_rate * 10, weight_decay=0.0005)
            iterate += 1
            del loss
        start_iter = 0

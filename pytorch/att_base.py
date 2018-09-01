from Network4Att_Test import Unet
from Dataset import CustomDataset
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import argparse

torch.set_printoptions(profile='full')
if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    print("Default path : " + os.getcwd())
    parser.add_argument("--model_dir",
                        help="Directory of pre-trained model, you can download at \n"
                             "https://drive.google.com/drive/folders/1s4M-_SnCPMj_2rsMkSy3pLnLQcgRakAe?usp=sharing",
                        default='models/state_dict/07121619/36epo_299000step.ckpt'
                        )
    parser.add_argument('-img', '--image_dir', help='Directory of your test_image ""folder""', default='Att_test')
    parser.add_argument('--cuda', help="'cuda' for cuda, 'cpu' for cpu, default = cuda", default='cuda')
    parser.add_argument('--batch_size', help="batchsize, default = 1", default=1, type=int)
    args = parser.parse_args()

    print(args)
    print(os.getcwd())
    device = torch.device(args.cuda)
    # model_dir = 'models/07121619/25epo_210000step.ckpt'
    state_dict = torch.load(args.model_dir)
    model = Unet().to(device)
    model.load_state_dict(state_dict)
    dataset = CustomDataset(root_dir=args.image_dir)
    dataloader = DataLoader(dataset, args.batch_size, shuffle=False)
    writer = SummaryWriter('log/Att_test')
    model.eval()
    for i, batch in enumerate(dataloader):
        img = batch.to(device)
        # mask = batch['mask'].to(device)
        with torch.no_grad():
            pred, loss, attention = model(img)
        for j, _attention in enumerate(attention):
            size = _attention.size()  # batch, (org_size, org_size), (224, ) or (13*224//org_size, )
            if size[3] != 224:
                patch = F.pad(img, (6 * 224 // size[2], 6 * 224 // size[2], 6 * 224 // size[2], 6 * 224 // size[2]))
                patch = patch.unfold(2, size[3], 224 // size[1]) \
                    .unfold(3, size[4], 224 // size[2]) \
                    .view(size[0], 3, size[1], size[2], size[3], size[4])  # batch, 3, H, W, H', W'
            else:
                patch = img.view(size[0], 3, 1, 1, size[3], size[4]).repeat(1, 1, size[1], size[2], 1, 1)
            _attention = _attention.unsqueeze(1).repeat(1, 3, 1, 1, 1, 1)
            att_map = (_attention * patch).view(-1, 3, size[3], size[4])
            # print(att_map)
            writer.add_image('attention_{}'.format(size[3]), _attention.view(-1, 1, size[3], size[4]),
                             i * len(attention) + j)
            writer.add_image('att_map_{}'.format(size[3]), att_map, i * len(attention) + j)
        pred = pred[5].data
        pred.requires_grad_(False)
        writer.add_image(args.model_dir + ', img', img, i)
        writer.add_image(args.model_dir + ', mask', pred, i)
    writer.close()

# A script to visualize the ERF.
# Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs (https://arxiv.org/abs/2203.06717)
# Github source: https://github.com/DingXiaoH/RepLKNet-pytorch
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

import os
import argparse
import numpy as np
import torch
from timm.utils import AverageMeter
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
from erf.resnet_for_erf import resnet101, resnet152
from erf.replknet_for_erf import RepLKNetForERF
from torch import optim as optim
from model.Fusion_Mamba import Net
from setting.options import opt
from setting.dataLoader import get_loader
from model.FusAtNet import FusAtNet
from model.ExVit import MViT


def parse_args():
    parser = argparse.ArgumentParser('Script for visualizing the ERF', add_help=False)
    parser.add_argument('--model', default='ExVit', type=str, help='model name')
    parser.add_argument('--weights', default=None, type=str, help='path to weights file. For resnet101/152, ignore this arg to download from torchvision')
    parser.add_argument('--data_path', default='path_to_imagenet', type=str, help='dataset path')
    parser.add_argument('--save_path', default='ExVit.npy', type=str, help='path to save the ERF matrix (.npy file)')
    parser.add_argument('--num_images', default=2, type=int, help='num of images to use')
    args = parser.parse_args()
    return args

# def get_input_grad(model, samples):
#     outputs = model(samples)
#     out_size = outputs.size()
#     central_point = torch.nn.functional.relu(outputs[:, :, out_size[2] // 2, out_size[3] // 2]).sum()
#     grad = torch.autograd.grad(central_point, samples)
#     grad = grad[0]
#     grad = torch.nn.functional.relu(grad)
#     aggregated = grad.sum((0, 1))
#     grad_map = aggregated.cpu().numpy()
#     return grad_map

def get_input_grad(model, sample1, sample2):
    outputs,_ = model(sample1, sample2)
    out_size = outputs.size()
    central_point = torch.nn.functional.relu(outputs[:, :, out_size[2] // 2, out_size[3] // 2]).sum()
    grad1 = torch.autograd.grad(central_point, sample1,retain_graph=True)
    grad2 = torch.autograd.grad(central_point, sample2)
    grad1 = grad1[0].squeeze()
    grad2 = grad2[0]
    grad1 = torch.nn.functional.relu(grad1)
    grad2 = torch.nn.functional.relu(grad2)
    aggregated1 = grad1.sum((0, 1))  #第0和第一个维度进行求和
    aggregated2 = grad2.sum((0, 1))
    grad_map1 = aggregated1.cpu().numpy()
    grad_map2 = aggregated2.cpu().numpy()
    return (grad_map1+grad_map2)/2


def main(args):
    #   ================================= transform: resize to 1024x1024
    # t = [
    #     transforms.Resize((1024, 1024), interpolation=Image.BICUBIC),
    #     transforms.ToTensor(),
    #     transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    # ]
    # transform = transforms.Compose(t)

    # print("reading from datapath", args.data_path)
    # root = os.path.join(args.data_path, 'val')
    # dataset = datasets.ImageFolder(root, transform=transform)
    # nori_root = os.path.join('/home/dingxiaohan/ndp/', 'imagenet.val.nori.list')
    # from nori_dataset import ImageNetNoriDataset      # Data source on our machines. You will never need it.
    # dataset = ImageNetNoriDataset(nori_root, transform=transform)
    train_loader,test_loader,trntst_loader,all_loader,train_num,val_num,trntst_num=get_loader(dataset=opt.dataset, 
        batchsize=opt.batchsize,num_workers=opt.num_work,useval=opt.useval, pin_memory=True)
    # sampler_val = torch.utils.data.SequentialSampler(dataset)
    # data_loader_val = torch.utils.data.DataLoader(dataset, sampler=sampler_val,
    #     batch_size=1, num_workers=1, pin_memory=True, drop_last=False)

    if args.model == 'resnet101':
        model = resnet101(pretrained=args.weights is None)
    elif args.model == 'MSFMamba': 
        model = Net('Berlin').cuda()   #不加cuda无法加载
        model.load_state_dict(torch.load('checkpoints/Berlin/weight/2024-08-30_19-41-29_77.88681402439023_Berlin_Net_epoch_4.pth'))
        # model.load_state_dict(torch.load('checkpoints/Berlin/weight/2024-08-30_21-12-12_78.07980114669762_Berlin_Net_epoch_1.pth'))
    elif args.model == 'FusAtNet':
        # model = resnet152(pretrained=args.weights is None)
        model = FusAtNet(input_channels=244, input_channels2=4, num_classes=8).cuda()
        model.load_state_dict(torch.load('checkpoints/BerlinFusAtNet_epoch_1.pth'))
    elif args.model == 'ExVit':
        model = MViT(patch_size = 11,num_patches = [244,4],num_classes = 8,dim = 64,
            depth = 6,heads = 4,mlp_dim = 32,dropout = 0.1,emb_dropout = 0.1,mode = 'MViT'
            ).cuda()
        model.load_state_dict(torch.load('checkpoints/ExVit/BerlinExVit_epoch_25.pth'))
    else:
        raise ValueError('Unsupported model. Please add it here.')

    # if args.weights is not None:
    #     print('load weights')
    #     weights = torch.load(args.weights, map_location='cpu')
    #     if 'model' in weights:
    #         weights = weights['model']
    #     if 'state_dict' in weights:
    #         weights = weights['state_dict']
    #     model.load_state_dict(weights)
    #     print('loaded')

    model.cuda()
    model.eval()    #   fix BN and droppath

    optimizer = optim.SGD(model.parameters(), lr=0, weight_decay=0)

    meter = AverageMeter()
    optimizer.zero_grad()
    # for _, (samples, _) in enumerate(test_loader):
    for hsi, x, hsi_pca, test_labels,h,w in test_loader:
        if meter.count == args.num_images-1:
            np.save(args.save_path, meter.avg)
            exit()
        hsi = hsi.cuda()
        hsi_pca = hsi_pca.unsqueeze(1).cuda()
        x = x.cuda()
        hsi.requires_grad = True
        hsi_pca.requires_grad = True
        x.requires_grad = True
        # samples = samples.cuda(non_blocking=True)
        # samples.requires_grad = True
        optimizer.zero_grad()
        # contribution_scores = get_input_grad(model, hsi_pca, x)
        contribution_scores = get_input_grad(model, hsi, x)

        if np.isnan(np.sum(contribution_scores)):
            print('got NAN, next image')
            continue
        else:
            print('accumulate')
            meter.update(contribution_scores)



if __name__ == '__main__':
    args = parse_args()
    main(args)
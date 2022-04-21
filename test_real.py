import os
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import argparse
import cv2
import h5py
from makedataset import Dataset

from model import GNet
from skimage.measure.simple_metrics import compare_psnr, compare_mse
from skimage.measure import compare_ssim
from torchvision.utils import save_image as imwrite

from loss import *
from torchvision.models import vgg16
import math
from PIL import Image

#调用GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    #开关定义
    parser = argparse.ArgumentParser(description = "network pytorch")
    #train
    parser.add_argument("--epoch", type=int, default = 1000, help = 'epoch number')
    parser.add_argument("--bs", type=str, default =16, help = 'batchsize')
    parser.add_argument("--lr", type=str, default = 1e-4, help = 'learning rate')
    parser.add_argument("--model", type=str, default = "./checkpoint/", help = 'checkpoint')
    #value
    parser.add_argument("--intest", type=str, default = "./input/", help = 'input syn path')
    parser.add_argument("--outest", type=str, default = "./result/", help = 'output syn path')
    argspar = parser.parse_args()
    
    print("\nnetwork pytorch")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')
    arg = parser.parse_args()
    
    #train
    print('> Loading dataset...')

    GNet, G_optimizer, cur_epoch = load_checkpoint(argspar.model, argspar.lr)
    
    test(argspar, GNet)


#加载模型
def load_checkpoint(checkpoint_dir, learnrate):
    Gmodel_name = 'GNet.tar'
    if os.path.exists(checkpoint_dir + Gmodel_name):
        #加载存在的模型
        Gmodel_info = torch.load(checkpoint_dir + Gmodel_name)
        print('==> loading existing model:', checkpoint_dir + Gmodel_name)
        #模型名称
        Model = GNet()
        #显卡使用
        device_ids = [0]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        G_optimizer = torch.optim.Adam(Model.parameters(), lr=learnrate)
        Model = torch.nn.DataParallel(Model, device_ids=device_ids).cuda()
        #将模型参数赋值进net
        Model.load_state_dict(Gmodel_info['state_dict'])
        G_optimizer = torch.optim.Adam(Model.parameters())
        G_optimizer.load_state_dict(Gmodel_info['optimizer'])
        cur_epoch = Gmodel_info['epoch']
            
    else:
        # 创建模型
        Model = GNet()
        #显卡使用
        device_ids = [0]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        G_optimizer = torch.optim.Adam(Model.parameters(), lr=learnrate)
        Model = torch.nn.DataParallel(Model, device_ids=device_ids).cuda()
        cur_epoch = 0
    return Model, G_optimizer, cur_epoch

def tensor_metric(img, imclean, model, data_range=1):#计算图像PSNR输入为Tensor

    img_cpu = img.data.cpu().numpy().astype(np.float32).transpose(0,2,3,1)
    imgclean = imclean.data.cpu().numpy().astype(np.float32).transpose(0,2,3,1)
    
    SUM = 0
    for i in range(img_cpu.shape[0]):
        if model == 'PSNR':
            SUM += compare_psnr(imgclean[i, :, :, :], img_cpu[i, :, :, :],data_range=data_range)
        elif model == 'MSE':
            SUM += compare_mse(imgclean[i, :, :, :], img_cpu[i, :, :, :])
        elif model == 'SSIM':
            SUM += compare_ssim(imgclean[i, :, :, :], img_cpu[i, :, :, :], data_range=data_range, multichannel = True)
        else:
            print('Model False!')
        
    return SUM/img_cpu.shape[0]

def upsample(x,y):
    _,_,H,W = y.size()
    return F.upsample(x,size=(H,W),mode='bilinear')
  
def test(argspar, model):
    files = os.listdir(argspar.intest) 
    a = []
    for i in range(len(files)):
        haze = np.array(Image.open(argspar.intest + files[i]))/255  
        model.eval()
        with torch.no_grad():
            haze = torch.Tensor(haze.transpose(2, 0, 1)[np.newaxis,:,:,:]).cuda()

            starttime = time.clock()
            T_out, out1, out2, out = model(haze)
            endtime1 = time.clock()
            #out1=upsample(out1,T_out)
            #out2=upsample(out2,T_out)
            
            result = out#torch.cat((haze,out), dim = 3)
            imwrite(result, argspar.outest+files[i], range=(0, 1))
            #imwrite(result, argspar.outest+files[i][:-4]+'_our.png', range=(0, 1))
            #imwrite(out1, argspar.outest+files[i][:-4]+'_our1.png', range=(0, 1))
            #imwrite(out2, argspar.outest+files[i][:-4]+'_our2.png', range=(0, 1))
            a.append(endtime1-starttime)

            print('The '+str(i)+' Time: %.4f.'%(endtime1-starttime))
    print(np.mean(np.array(a)))
            
    
if __name__ == '__main__':
    main()

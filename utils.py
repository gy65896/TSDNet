import numpy as np
import torch
from torch.autograd import Variable
from skimage.measure.simple_metrics import compare_psnr, compare_mse
from skimage.measure import compare_ssim
import pandas as pd

def adjust_learning_rate(optimizer, epoch, lr_update_freq):
    if not epoch % lr_update_freq and epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] /2#* 0.1
    return optimizer


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

def save_checkpoint(stateF,stateawl, checkpoint, epoch, psnr1, ssim1, filename='model.tar'):#保存学习率
    torch.save(stateF, checkpoint + 'Fmodel_%d_%.4f_%.4f.tar'%(epoch,psnr1, ssim1))
    torch.save(stateawl, checkpoint + 'awlmodel.tar')
    
def syn_haze(clear, depth, minA=0.7, maxA=1.0, mint=0.08, maxt=0.3):
    #图像导入
    clear = clear.numpy()
    depth = depth.numpy()
    haze = np.zeros(clear.shape)
    for nx in range(clear.shape[0]):
        #合成图像
        A = np.random.uniform(minA, maxA)
        k = np.random.uniform(mint, maxt)
        T = np.exp(-k*depth)
        haze = clear*T + A*(1-T)

    haze = torch.from_numpy(haze.copy()).type(torch.FloatTensor)
    clear = torch.from_numpy(clear.copy()).type(torch.FloatTensor)
    T = torch.from_numpy(T.copy()).type(torch.FloatTensor)
    with torch.no_grad():
        haze = Variable(haze.cuda(),requires_grad=True)
        clear = Variable(clear.cuda(),requires_grad=True)
        T = Variable(T.cuda(),requires_grad=True)
    return haze, clear, T #输出雾气、清晰图像

def load_excel(x):
    data1 = pd.DataFrame(x)

    writer = pd.ExcelWriter('./log/A.xlsx')		# 写入Excel文件
    data1.to_excel(writer, 'SOTS-PSNR', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
    writer.save()
    writer.close()
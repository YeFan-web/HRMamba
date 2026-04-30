import torch
from torch import nn
import torch.nn.functional as F
import cv2
import numpy as np
from net.ptcolor import rgb2lab
from net.Qnt import quantAB,quantL
from net import ptcolor as ptcolor
class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()

    def forward(self, x):  ## 灰度世界
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Dr = torch.pow(mr-0.5, 2)
        Dg = torch.pow(mg-0.5, 2)
        Db = torch.pow(mb-0.5, 2)
        k = torch.pow(torch.pow(Dr, 2) + torch.pow(Dg, 2) + torch.pow(Db, 2), 0.5)
        return k

class ColorLossImproved(nn.Module):
    def __init__(self):
        super(ColorLossImproved, self).__init__()

    def forward(self, x):
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Dr = torch.abs(mr-0.5)
        Dg = torch.abs(mg-0.5)
        Db = torch.abs(mb-0.5)
        k = torch.pow(Dr+Dg+Db, 2)
        return k

def histogram_spread(channel):
    hist, _ = np.histogram(channel, bins=256, range=(0, 1))
    return np.std(hist)

class ColorLoss1(nn.Module):
    def __init__(self):
        super(ColorLoss1, self).__init__()

    def forward(self, x):

        ## 数据预处理
        x_np = x.squeeze().permute(1, 2, 0).cpu().detach().numpy()
        # Convert from RGB to BGR if needed
        input_img = cv2.cvtColor(x_np, cv2.COLOR_RGB2BGR)

        ## zip [(img_mean, img)], it (b, g, r)
        small, medium, large = sorted(list(zip(cv2.mean(input_img), cv2.split(input_img), ['b', 'g', 'r'])))
        ## sorted by mean (small to large)
        small, medium, large = list(small), list(medium), list(large)

        if histogram_spread(medium[1]) < histogram_spread(large[1]) and (large[0] - medium[0]) < 0.07 and small[2] == 'r':  ### 同时满足三个条件
            large, medium = medium, large  ## 中等 和大 交换

        loss = np.sqrt((large[0] - cv2.mean(medium[1])[0])**2 + (large[0] - cv2.mean(small[1])[0])**2)

        return loss


def RecoverCLAHE(sceneRadiance):
    # clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(4, 4))
    clahe = cv2.createCLAHE(clipLimit=0.1, tileGridSize=(8, 8)) ## re-waternet中的设置
    # clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(4, 4))
    for i in range(3):

        # sceneRadiance[:, :, i] =  cv2.equalizeHist(sceneRadiance[:, :, i])
        sceneRadiance[:, :, i] = clahe.apply((sceneRadiance[:, :, i]))

    return sceneRadiance

def tensor_to_cv2_img(tensor_img):
    # 将 PyTorch 张量的形状转换为 (h, w, 3)
    # img_np = tensor_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_np = tensor_img.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

    # 转换数据类型为 uint8
    img_np = (img_np * 255).astype('uint8')
    return img_np


def cv2_img_to_tensor(cv2_img):
    # 将 cv2 格式的图像数据转换为 PyTorch 张量
    tensor_img = torch.tensor(cv2_img, dtype=torch.float32)  # 将数据类型转换为 float32
    # 将通道顺序从 BGR 转换为 RGB
    tensor_img = tensor_img.permute(2, 0, 1)

    # 将数据范围从 [0, 255] 转换为 [0, 1]
    # tensor_img /= 255.0
    # 添加批次维度
    tensor_img = tensor_img.unsqueeze(0)
    return tensor_img

def CLAHE_loss(img):  ## 损失不下降？？

    img_cv2 = tensor_to_cv2_img(img)
    CLAHE = RecoverCLAHE(img_cv2)
    CLAHE_tensor = cv2_img_to_tensor(CLAHE)

    mse_loss = nn.MSELoss()
    clahe_loss = mse_loss(img, CLAHE_tensor)
    return clahe_loss

def contrast_loss(image):
    # 计算图像梯度
    gradient_x = torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])
    gradient_y = torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :])

    # 对梯度进行平滑处理，以减少噪音
    gradient_x_smooth = F.avg_pool2d(gradient_x, kernel_size=3, stride=1, padding=(0, 1))
    gradient_y_smooth = F.avg_pool2d(gradient_y, kernel_size=3, stride=1, padding=(1, 0))

    # 计算梯度的均值，作为对比度损失
    contrast_loss = torch.mean(gradient_x_smooth) + torch.mean(gradient_y_smooth)

    return contrast_loss

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)  # filter
        down = filtered[:, :, ::2, ::2]  # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4  # upsample
        filtered = self.conv_gauss(new_filter)  # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss



class lab_Loss(nn.Module):
    def __init__(self, alpha=1,weight=1,levels=7,vmin=-80,vmax=80):
        super(lab_Loss, self).__init__()
        self.alpha=alpha
        self.weight=weight
        self.levels=levels
        self.vmin=vmin
        self.vmax=vmax

    def Hist_2_Dist_L(self,img, tab,alpha):
        img_dist=((img.unsqueeze(1)-tab)**2)
        p=F.softmax(-alpha*img_dist,dim=1)
        return p

    def Hist_2_Dist_AB(self,img,tab,alpha):
        img_dist=((img.unsqueeze(1)-tab)**2).sum(2)
        p = torch.nn.functional.softmax(-alpha*img_dist, dim=1)
        return p

    def loss_ab(self,img,gt,alpha,tab,levels):
        p= self.Hist_2_Dist_AB(img, tab,alpha).cuda()
        q= self.Hist_2_Dist_AB(gt,tab,alpha).cuda()
        p = torch.clamp(p, 0.001, 0.999)
        loss = -(q*torch.log(p)).sum([1,2,3]).mean()
        return loss




    def forward(self,gt,img):
	    tab=quantAB(self.levels,self.vmin,self.vmax).cuda()
	    lab_img=torch.clamp(rgb2lab(img),self.vmin,self.vmax)
	    lab_gt=torch.clamp(rgb2lab(gt),self.vmin,self.vmax)

	    loss_l=torch.abs(lab_img[:,0,:,:]-lab_gt[:,0,:,:]).mean()
	    loss_AB=self.loss_ab(lab_img[:,1:,:,:],lab_gt[:,1:,:,:],self.alpha,tab,self.levels)
	    loss=loss_l+self.weight*loss_AB
	    #return (loss,loss_l,loss_AB)
	    return loss



class lch_Loss(nn.Module):
    def __init__(self, weightC=1,weightH=1,levels=4,eps=0.01,weight=None):
        super(lch_Loss, self).__init__()
        self.weightC=weightC
        self.weightH=weightH
        self.levels=levels
        self.eps=eps
        self.weight=weight


    def hue_to_distribution(self,h, levels, eps=0.0):
        h = h * (levels / 360.0)
        a = torch.arange(levels).float().to(h.device)
        a = a.view(1, levels, 1, 1)
        h=h.unsqueeze(1)
        p = torch.relu(1 - torch.abs(h - a))
        p = p + (a == 0.0).float() * p[:, -1:, :, :]
        p = (p + torch.ones_like(p) * eps) / (1.0 + levels * eps)
        return p



    def forward(self,gt,img):
        img_lch= ptcolor.rgb2lch(img)
        gt_lch= ptcolor.rgb2lch(gt)
        loss_L=torch.mean(torch.abs(img_lch[:,0,:,:]-gt_lch[:,0,:,:]))
        loss_C=torch.mean(torch.abs(img_lch[:,1,:,:]-gt_lch[:,1,:,:]))
        img_H_Dist=torch.clamp(self.hue_to_distribution(img_lch[:,2,:,:],self.levels,self.eps),0.001, 0.999)
        gt_H_Dist =torch.clamp(self.hue_to_distribution(gt_lch[:, 2, :, :], self.levels),0.001, 0.999)
        if self.weight is None:
            loss_H = torch.mean(-torch.mul(gt_H_Dist, torch.log(img_H_Dist)))
        else:
            loss_H = -(gt_lch[:,1,:,:]*(gt_H_Dist*torch.log(img_H_Dist)).sum(1,keepdim=True)).mean()
        loss=loss_L+self.weightC*loss_C+self.weightH*loss_H
        #return(loss,loss_L,loss_C,loss_H)
        return loss

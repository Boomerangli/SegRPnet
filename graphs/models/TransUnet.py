import torch
import torch.nn as nn
from einops import rearrange
from torch.autograd import Variable
from utils.vit import ViT
import torch
from torch import nn
from torch.autograd import Variable
from graphs.models.custom_layers.resnet_zzq import resnet50
from utils.help_funcs import *

# 先写卷积函数和上采样函数，
# 其中卷积是双重卷积，包括bathnoarm2d和relu算一次卷积，池化不用自己调用就行
# !!!!!卷积只改变通道数，上采样池化等只改变特征图！！！
class Double_conv(nn.Module):
    # 输入输出待定，根据网络写
    def __init__(self,input,output):
        super(Double_conv, self).__init__()

        self.conv=nn.Sequential(
            nn.Conv2d(input,output,kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(output,output),
            nn.ReLU(inplace=False),
            nn.Conv2d(output,output,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(output),
            nn.ReLU(inplace=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(output, output, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(output),
            nn.ReLU(inplace=False)
        )

    def forward(self,x):
        x=self.conv(x)
        return x

#上采样
class Up_conv(nn.Module):
    def __init__(self,input,output):
        super(Up_conv, self).__init__()

        self.up_conv=nn.Sequential(
            nn.Upsample(scale_factor=2),#代表扩大两倍，这里就一个参数
            #卷积三步曲
            nn.Conv2d(input,output,kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(output), #batchnorm也只有一个参数
            nn.ReLU(inplace=True))

    def forward(self,x):
        x=self.up_conv(x)
        return x


class UnetDecoder(nn.Module):
    def __init__(self, f0_dim, f1_dim, f2_dim, f3_dim, f4_dim, f5_dim,numclass):
        super(UnetDecoder, self).__init__()
        # U-Net的上采样部分
        self.up_conv4 = Up_conv(f5_dim, f4_dim)  #f5-->f4
        self.up_conv3 = Up_conv(f4_dim, f3_dim)   #f4-->f3
        self.up_conv2 = Up_conv(f3_dim, f2_dim)   #f3-->f2
        self.up_conv1 = Up_conv(f2_dim, f1_dim)   #f2-->f1
        #self.up_conv0 = Up_conv(f1_dim, f1_dim)   # f1-->x

        self.num_classes = numclass

        # 后续的卷积层，用于进一步处理特征
        self.conv4 = Double_conv(f4_dim*2, f4_dim)
        self.conv3 = Double_conv(f3_dim*2, f3_dim)
        self.conv2 = Double_conv(f2_dim*2, f2_dim)
        self.conv1 = Double_conv(f1_dim*2, f1_dim)
        #self.conv0 = Double_conv(f1_dim+f0_dim, 32)

        # 分割头
        self.final_super = nn.Sequential(
            nn.BatchNorm2d(f1_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(f1_dim, self.num_classes, kernel_size=4, stride=2, padding=1)
            # nn.Conv2d(64, self.num_classes, 3, padding=1)
        )

        # 最终的卷积层，用于生成分割图
        self.classfy = nn.Conv2d(numclass+3, self.num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, f0, f1, f2, f3, f4, f5):
        f4_up = self.up_conv4(f5)
        f4_up = self.conv4(torch.cat([f4, f4_up],dim=1))

        f3_up = self.up_conv3(f4_up)
        f3_up = self.conv3(torch.cat([f3, f3_up], dim=1))

        f2_up = self.up_conv2(f3_up)
        f2_up = self.conv2(torch.cat([f2, f2_up], dim=1))

        f1_up = self.up_conv1(f2_up)
        f1_up = self.conv1(torch.cat([f1, f1_up], dim=1))

        out = self.final_super(f1_up)
        out = self.classfy(torch.cat([out,f0],dim=1))

        return out

class TransUNet(nn.Module):

    def __init__(self, input=3, output=1):
        super(TransUNet, self).__init__()

        self.resnet50 = resnet50(pretrained=True)
        self.unet_decoder = UnetDecoder(f0_dim=3, f1_dim=64, f2_dim=256, f3_dim=512, f4_dim=1024, f5_dim=2048,numclass=output)
        self.transformer_encoder = Transformer_NoFeedForward(dim=2048, depth=6, heads=8, dim_head=64)

    def forward(self, x):
        f1, f2, f3, f4, f5 = self.resnet50(x, is_fpn=True)
        batch_size, channels, height, width = f5.size()
        f5 = f5.view(batch_size, -1, channels)
        f5 = self.transformer_encoder(f5)
        f5 = f5.permute(0, 2, 1).view(batch_size, channels, height, width)
        out = self.unet_decoder(x, f1, f2, f3, f4, f5)
        return out

if __name__ == '__main__':
    x1 = Variable(torch.randn(2, 3, 512, 512))
    model = TransUNet()
    y = model(x1)

    get_thop_params_flops_Seg(model, x1)
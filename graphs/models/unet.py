import torch
from torch import nn
from torch.autograd import Variable

from utils.help_funcs import get_thop_params_flops_Seg


# 先写卷积函数和上采样函数，
# 其中卷积是双重卷积，包括bathnoarm2d和relu算一次卷积，池化不用自己调用就行
# !!!!!卷积只改变通道数，上采样池化等只改变特征图！！！
class Double_conv(nn.Module):
    # 输入输出待定，根据网络写
    def __init__(self,input,output):
        super(Double_conv, self).__init__()

        self.conv=nn.Sequential(
            nn.Conv2d(input,output,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(output,output),
            nn.ReLU(inplace=True),
            nn.Conv2d(output,output,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(output),
            nn.ReLU(inplace=True))

    def forward(self,x):
        x=self.conv(x)
        return x


class Up_conv(nn.Module):
    def __init__(self,input,output):
        super(Up_conv, self).__init__()

        self.up_conv=nn.Sequential(
            nn.Upsample(scale_factor=2),#代表扩大两倍，这里就一个参数
            #卷积三步曲
            nn.Conv2d(input,output,kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(output), #batchnorm也只有一个参数
            nn.ReLU(inplace=True))

    def forward(self,x):
        x=self.up_conv(x)
        return x


# 网络部分按照模型图指定输入输出
class Unet(nn.Module):
    def __init__(self,input=3,output=1):
        super(Unet, self).__init__()

        self.conv1=Double_conv(input,64)
        self.conv2=Double_conv(64,128)
        self.conv3 =Double_conv(128, 256)
        self.conv4= Double_conv(256, 512)
        self.conv5= Double_conv(512, 1024)
        self.maxpool1=nn.MaxPool2d(kernel_size=2,stride=2)#2的核与2的步长正好保证特征图减半
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool3=nn.MaxPool2d(kernel_size=2,stride=2)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up_conv1=Up_conv(1024,512)
        self.up_conv2=Up_conv(512,256)
        self.up_conv3=Up_conv(256,128)
        self.up_conv4=Up_conv(128,64)
        #下面卷积通道数基于拼接之后的
        self.conv6=Double_conv(1024,512)
        self.conv7=Double_conv(512,256)
        self.conv8=Double_conv(256,128)
        self.conv9=Double_conv(128,64)
        #单次卷积
        self.conv10=nn.Conv2d(64,output,kernel_size=1,stride=1,padding=0)#步长为1不需要补充padding

    def forward(self, x):
        # encoder
        # 第一层
        u1=self.conv1(x)
        u2=self.maxpool1(u1)
        # 第二层
        u3=self.conv2(u2)
        u4=self.maxpool2(u3)
        # 第三层
        u5=self.conv3(u4)
        u6=self.maxpool3(u5)
        # 第四层
        u7=self.conv4(u6)
        u8=self.maxpool4(u7)
        # 第五层
        u9=self.conv5(u8)

        # decoder
        # 第四层
        u10=self.up_conv1(u9)
        u10=torch.cat((u10, u7), dim=1)
        u11=self.conv6(u10)
        # 第三层
        u12=self.up_conv2(u11)
        u12=torch.cat((u12, u5), dim=1)
        u13=self.conv7(u12)
        # 第二层
        u14=self.up_conv3(u13)
        u14=torch.cat((u14, u3), dim=1)
        u15=self.conv8(u14)
        # 第一层
        u16 = self.up_conv4(u15)
        u16 = torch.cat((u16, u1), dim=1)
        u17 = self.conv9(u16)
        fin_out=self.conv10(u17)

        return fin_out


if __name__ == '__main__':
    x1 = Variable(torch.randn(2, 3, 512, 512))
    model = Unet()
    y = model(x1)

    get_thop_params_flops_Seg(model, x1)

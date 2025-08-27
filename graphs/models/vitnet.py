import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from transformers import ViTFeatureExtractor
from graphs.models.unet import Unet
from utils.help_funcs import get_thop_params_flops_Seg


class Vitnet(nn.Module):
    def __init__(self):
        super(Vitnet, self).__init__()
        self.feature_extractor = ViTFeatureExtractor(do_resize=True, size=224, do_normalize=True)
        self.unet = Unet()

    def forward(self, x):
        x = torch.tensor(self.feature_extractor(x).pixel_values)
        x = self.unet(x)

        return x

if __name__ == '__main__':
    x1 = Variable(torch.randn(2, 3, 224, 224))
    x1 = (x1 - x1.min()) / (x1.max() - x1.min())
    model = Vitnet()
    y = model(x1)
    print(y.shape)
    get_thop_params_flops_Seg(model, x1)
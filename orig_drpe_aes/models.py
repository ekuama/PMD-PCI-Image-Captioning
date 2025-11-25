import torch
from torch import nn
from torchvision.models import ResNet101_Weights, resnet101

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN_Encoder(nn.Module):
    def __init__(self, encoded_image_size=14):
        super(CNN_Encoder, self).__init__()
        self.enc_image_size = encoded_image_size
        resnet = resnet101(weights=ResNet101_Weights.DEFAULT)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fine_tune()
        self.fine_tune_all()

    def forward(self, images):
        out = self.resnet(images)
        out = self.adaptive_pool(out)
        out = out.permute(0, 2, 3, 1)
        return out

    def fine_tune_all(self, fine_tune=True):
        for p in self.resnet.parameters():
            p.requires_grad = True


class CNN_Encoder2(nn.Module):
    def __init__(self, encoded_image_size=14):
        super(CNN_Encoder2, self).__init__()
        self.enc_image_size = encoded_image_size
        resnet = resnet101(weights=ResNet101_Weights.DEFAULT)
        self.conv = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        modules = list(resnet.children())[1:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fine_tune_all()

    def forward(self, real, imag):
        images = torch.cat((real, imag), dim=1)
        out = self.conv(images)
        out = self.resnet(out)
        out = self.relu(self.bn1(self.cnn1(out)))
        out = self.adaptive_pool(out)
        out = out.permute(0, 2, 3, 1)
        return out

    def fine_tune_all(self, fine_tune=True):
        for p in self.resnet.parameters():
            p.requires_grad = True

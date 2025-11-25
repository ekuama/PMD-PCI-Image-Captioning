from torch import nn
from torchvision.models import ResNet101_Weights, resnet101


class CNN_Encoder(nn.Module):
    def __init__(self, cnn_name, encoded_image_size=14):
        super(CNN_Encoder, self).__init__()
        self.enc_image_size = encoded_image_size
        self.name = cnn_name
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

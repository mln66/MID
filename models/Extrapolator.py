import torch.nn as nn
from torch.nn import init
from torchvision import models


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(7 * 7 * 64, 200)
        self.relu3 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(200, 10)

    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        features = out
        out = out.view(out.size(0), -1)
        out = self.relu3(self.linear1(out))
        out = self.linear2(out)
        return out





class LeNet5_autoencoder(nn.Module):
    def __init__(self):
        super(LeNet5_autoencoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, return_indices=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, return_indices=True)

        self.dec1 = nn.MaxUnpool2d(2)
        self.dec2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.dec3 = nn.MaxUnpool2d(2)
        self.dec4 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.linear1 = nn.Linear(7 * 7 * 64, 200)
        self.relu3 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(200, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out, indice_1 = self.maxpool1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        features, indice_2 = self.maxpool2(out)

        out_rec = self.dec1(features, indice_2)
        out_rec = self.dec2(out_rec)
        out_rec = self.dec3(out_rec, indice_1)
        out_rec = self.dec4(out_rec)

        out_cls = features.view(features.size(0), -1)
        out_cls = self.relu3(self.linear1(out_cls))
        out_cls = self.linear2(out_cls)

        return out_cls, out_rec







class Extrapolator(nn.Module):  # CIFAR10的外插器
    def __init__(self):
        super(Extrapolator, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, return_indices=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, return_indices=True)

        self.dec1 = nn.MaxUnpool2d(2)
        self.dec2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.dec3 = nn.MaxUnpool2d(2)
        self.dec4 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out, indice_1 = self.maxpool1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        features, indice_2 = self.maxpool2(out)
        out_rec = self.dec1(features, indice_2)
        out_rec = self.dec2(out_rec)
        out_rec = self.dec3(out_rec, indice_1)
        out_rec = self.dec4(out_rec)

        return out_rec






###################################################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
   # if hasattr(m.bias, 'data'):
        #init.constant(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        if hasattr(m.bias, 'data'):
            init.constant(m.bias.data, 0.0)


class dense_Extrapolator(nn.Module):

    def __init__(self, class_num):
        super(dense_Extrapolator,self).__init__()
        model_ft = models.densenet121(pretrained=True)
        # add pooling to the model
        # in the originial version, pooling is written in the forward function
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1,1))

        add_block = []
        # num_bottleneck = 1024

        # num_bottleneck = 1664  # for densenet169
        num_bottleneck = 1024  # for densenet121

        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        model_ft.bn = add_block
        self.model = model_ft


        self.fc0 = nn.Linear(num_bottleneck, class_num, bias = True)
        init.normal(self.fc0.weight.data, std=0.001)
        if hasattr(self.fc0.bias, 'data'):
            init.constant(self.fc0.bias.data, 0.0)


        self.dec1 = nn.ConvTranspose2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1)
        self.dec2 = nn.ConvTranspose2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.dec3 = nn.ConvTranspose2d(in_channels=20, out_channels=10, kernel_size=3, stride=1, padding=1)
        self.dec4 = nn.ConvTranspose2d(in_channels=10, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(self, x, is_train=True):
        feature = self.model.features(x)
        # x = feature.view(feature.size(0), -1)
        # x = self.model.bn(x)
        # predict = self.fc0(x)

        img_rec = self.dec4(self.dec3(self.dec2(self.dec1(feature.view(feature.shape[0],1,32,32)))))

        return img_rec




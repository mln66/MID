import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F


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

class ft_net(nn.Module):

    def __init__(self, class_num):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)

        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))

        add_block = []
        num_bottleneck = 2048

        add_block += [nn.BatchNorm1d(num_bottleneck)]

        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        model_ft.fc = add_block
        self.model = model_ft

        self.fc0 = nn.Linear(num_bottleneck, class_num, bias = True)
        init.normal(self.fc0.weight.data, std=0.001)
        if hasattr(self.fc0.bias, 'data'):
            init.constant(self.fc0.bias.data, 0.0)

    def forward(self, x, if_train = True):
        x = self.model(x)

        if if_train == False:  # if False, return features
            return x
        x = self.fc0(x)
        return x


class ft_net_dense(nn.Module):
    def __init__(self, class_num):
        super(ft_net_dense, self).__init__()
        model_ft = models.densenet121(pretrained=True)
        # model_ft = models.densenet201(pretrained=True)
        # add pooling to the model
        # in the originial version, pooling is written in the forward function 
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1,1))

        add_block = []
        # num_bottleneck = 1024

        # num_bottleneck = 1664  # for densenet169
        num_bottleneck = 1024  # for densenet121
        # num_bottleneck = 1920  # for densenet201

        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        model_ft.bn = add_block
        self.model = model_ft


        self.fc0 = nn.Linear(num_bottleneck, class_num, bias = True)
        init.normal(self.fc0.weight.data, std=0.001)
        if hasattr(self.fc0.bias, 'data'):
            init.constant(self.fc0.bias.data, 0.0)

    def forward(self, x, is_train=True):
        # print(self.model.features)
        x = self.model.features(x)
        x = x.view(x.size(0), -1)
        x = self.model.bn(x)

        # print('is_train=', is_train)

        if is_train == False:
            print('testing!')
            print(x.shape)
            return x
        # print(x.shape)
        x = self.fc0(x)
        return x

# debug model structure
#net = ft_net(751)
#net = ft_net_dense(751)
#print(net)
#input = Variable(torch.FloatTensor(8, 3, 224, 224))
#output = net(input)
#print('net output size:')
#print(output.shape)













#################################################################################################




class ft_net_dense_40x40(nn.Module):

    def __init__(self, class_num ):
        super(ft_net_dense_40x40,self).__init__()
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

    def forward(self, x, is_train=True):
        # print(self.model.features)
        x = self.model.features(x)
        x = x.view(x.size(0), -1)
        x = self.model.bn(x)

        # print('is_train=', is_train)

        if is_train == False:
            print('testing!')
            print(x.shape)
            return x
        # print(x.shape)
        x = self.fc0(x)
        return x




class dense_encoder(nn.Module):

    def __init__(self, class_num ):
        super(dense_encoder,self).__init__()
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

    def forward(self, x, is_train=True):
        x = self.model.features(x)
        return x




class dense_classifier(nn.Module):

    def __init__(self, class_num ):
        super(dense_classifier,self).__init__()
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

    def forward(self, x, is_train=True):
        x = x.view(x.size(0), -1)
        x = self.model.bn(x)
        if is_train == False:
            return x
        x = self.fc0(x)
        return x




class dense_decoder(nn.Module):

    def __init__(self):
        super(dense_decoder,self).__init__()
        self.dec1 = nn.ConvTranspose2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1)
        self.dec2 = nn.ConvTranspose2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.dec3 = nn.ConvTranspose2d(in_channels=20, out_channels=10, kernel_size=3, stride=1, padding=1)
        self.dec4 = nn.ConvTranspose2d(in_channels=10, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(self, x, is_train=True):
        x = x.view(x.shape[0],1,32,32)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        return x




class dense_autoencoder(nn.Module):

    def __init__(self, class_num ):
        super(dense_autoencoder,self).__init__()
        model_ft = models.densenet121(pretrained=True)
        # add pooling to the model
        # in the originial version, pooling is written in the forward function
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))

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
        # self.dec1 = nn.ConvTranspose2d(in_channels=1, out_channels=1024, kernel_size=3, stride=1, padding=1)
        # self.dec2 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1)
        # self.dec3 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=1)
        # self.dec4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        # self.dec5 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)

        self.unpool = nn.MaxUnpool2d(2)
    def forward(self, x, is_train=True):
        feature = self.model.features(x)

        x = feature.view(feature.size(0), -1)
        x = self.model.bn(x)
        predict = self.fc0(x)
        # print('特征size', feature.shape)
        # feature = feature.view(feature.shape[0], 1, 32, 32)
        # # print('特征size', feature.shape)
        # out_rec = self.dec1(feature)
        # # print('dec1:', out_rec.shape)
        # out_rec = self.dec2(out_rec)
        # # print('dec2:', out_rec.shape)
        # out_rec = self.dec3(out_rec)
        # # print('dec3:', out_rec.shape)
        # out_rec = self.dec4(out_rec)
        # # print('dec4"', out_rec.shape)
        # out_rec = self.dec5(out_rec)
        # print('dec5', out_rec.shape)
        # print('##########################################')

        img_rec = self.dec4(self.dec3(self.dec2(self.dec1(feature.view(feature.shape[0],1,32,32)))))

        return predict, img_rec
        # return out_rec









class ft_net_dense_mnist(nn.Module):
    def __init__(self, class_num):
        super(ft_net_dense_mnist, self).__init__()
        model_ft = models.densenet169(pretrained=True)
        # add pooling to the model
        # in the originial version, pooling is written in the forward function
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1,1))

        add_block = []
        # num_bottleneck = 1024

        num_bottleneck = 1664  # for densenet169
        # num_bottleneck = 1024  # for densenet121

        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        model_ft.bn = add_block
        self.model = model_ft


        self.fc0 = nn.Linear(num_bottleneck, class_num, bias = True)
        init.normal(self.fc0.weight.data, std=0.001)
        if hasattr(self.fc0.bias, 'data'):
            init.constant(self.fc0.bias.data, 0.0)

    def forward(self, x, is_train=True):
        # print(self.model.features)
        x = self.model.features(x)
        x = x.view(x.size(0), -1)
        x = self.model.bn(x)

        # print('is_train=', is_train)

        if is_train == False:
            print('testing!')
            print(x.shape)
            return x
        # print(x.shape)
        x = self.fc0(x)
        return x






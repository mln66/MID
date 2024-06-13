import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.nn.functional as F

__all__ = ['ResNet50', 'ResNet101', 'ResNet152']


def Conv1(in_planes, places, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    )


class Bottleneck(nn.Module):
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=4):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places * self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(places * self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, blocks, num_classes=10, expansion=4):
        super(ResNet, self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes=3, places=64)

        self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places=256, places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512, places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024, places=512, block=blocks[3], stride=2)

        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places, stride, downsampling=True))
        for i in range(1, block):
            layers.append(Bottleneck(places * self.expansion, places))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def ResNet50(num_cls=10):
    return ResNet([3, 4, 6, 3], num_cls)


def ResNet101(num_cls=10):
    return ResNet([3, 4, 23, 3],num_cls)


def ResNet152(num_cls=10):
    return ResNet([3, 8, 36, 3], num_cls)







class ResNet_encoder_SVHN(nn.Module):
    def __init__(self, blocks, num_classes=10, expansion=4):
        super(ResNet_encoder_SVHN, self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes=3, places=64)

        self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places=256, places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512, places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024, places=512, block=blocks[3], stride=2)



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places, stride, downsampling=True))
        for i in range(1, block):
            layers.append(Bottleneck(places * self.expansion, places))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = F.avg_pool2d(x, 4)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return x

def ResNet50_encoer_svhn():
    return ResNet([3, 4, 6, 3])



class svhn_classifier_head(nn.Module):
    def __init__(self):
        super(svhn_classifier_head, self).__init__()
        self.fc = nn.Linear(2048, 10)

    def forward(self, x):
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


















# ResNet 18
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()

        # 这里的即为两个3*3 conv
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            # bias为偏置，False表示不添加偏置
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()  # shortcut connections
        if stride != 1 or inchannel != outchannel:  # 判断入通道和出通道是否一样，不一样的话进行卷积操作
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, ResidualBlock, class_num=10):
        super(ResNet18, self).__init__()

        # 图片处理，也就是白色方框内的3*3 conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # 中间的残差网络部分，与图上的结构一一对应
        self.layer1 = self.make_layer(ResidualBlock, 64, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 64, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 128, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 256, 512, 2, stride=2)
        self.avg_pool2d = nn.AvgPool2d(4)
        self.fc = nn.Linear(512, class_num)

    # 相当于看处理几次，18的是每个处理两次
    def make_layer(self, block, inchannel, outchannel, num_blocks, stride):
        layers = []
        for i in range(1, num_blocks):
            layers.append(block(inchannel, outchannel, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool2d(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ResNet18_STA(nn.Module):
    def __init__(self, ResidualBlock):
        super(ResNet18_STA, self).__init__()

        # 图片处理，也就是白色方框内的3*3 conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # 中间的残差网络部分，与图上的结构一一对应
        self.layer1 = self.make_layer(ResidualBlock, 64, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 64, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 128, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 256, 512, 2, stride=2)
        self.avg_pool2d = nn.AvgPool2d(4)
        self.fc = nn.Linear(512, 10)

    # 相当于看处理几次，18的是每个处理两次
    def make_layer(self, block, inchannel, outchannel, num_blocks, stride):
        layers = []
        for i in range(1, num_blocks):
            layers.append(block(inchannel, outchannel, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool2d(out)
        out = out.view(out.size(0), -1)
        feature = out
        out = self.fc(out)
        return feature, out

class ResNet18_40x40(nn.Module):
    def __init__(self, ResidualBlock):
        super(ResNet18_40x40, self).__init__()

        # 图片处理，也就是白色方框内的3*3 conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # 中间的残差网络部分，与图上的结构一一对应
        self.layer1 = self.make_layer(ResidualBlock, 64, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 64, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 128, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 256, 512, 2, stride=2)
        self.avg_pool2d = nn.AvgPool2d(4)
        self.fc = nn.Linear(512, 10)

    # 相当于看处理几次，18的是每个处理两次
    def make_layer(self, block, inchannel, outchannel, num_blocks, stride):
        layers = []
        for i in range(1, num_blocks):
            layers.append(block(inchannel, outchannel, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool2d(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def ResNet18_cifar(class_num = 10):
    return ResNet18(ResidualBlock, class_num=class_num)

def ResNet18_cifar_STA():
    return ResNet18_STA(ResidualBlock)

def ResNet18_cifar_40x40():
    return ResNet18_40x40(ResidualBlock)



class ResNet_autoencoder(nn.Module):
    def __init__(self, ResidualBlock):
        super(ResNet_autoencoder, self).__init__()

        # 图片处理，也就是白色方框内的3*3 conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # 中间的残差网络部分，与图上的结构一一对应
        self.layer1 = self.make_layer(ResidualBlock, 64, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 64, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 128, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 256, 512, 2, stride=2)

        self.dec1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.dec3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.dec4 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dec5 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)

        self.avg_pool2d = nn.AvgPool2d(4)
        self.fc = nn.Linear(512, 10)

    # 相当于看处理几次，18的是每个处理两次
    def make_layer(self, block, inchannel, outchannel, num_blocks, stride):
        layers = []
        for i in range(1, num_blocks):
            layers.append(block(inchannel, outchannel, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        # print('初始形状', x.shape)
        out = self.conv1(x)
        # print('第一层卷积', out.shape)
        out = self.layer1(out)
        # print('layer1输出', out.shape)
        out = self.layer2(out)
        # print('layer2输出', out.shape)
        out = self.layer3(out)
        # print('layer3输出', out.shape)
        out = self.layer4(out)
        # print('layer4输出', out.shape, '分界线')
        features = out

        out = self.dec1(features)
        # print('dec1输出', out.shape)
        out = self.dec2(out)
        # print('dec2输出', out.shape)
        out = self.dec3(out)
        # print('dec3输出', out.shape)
        out = self.dec4(out)
        # print('dec4输出', out.shape)
        out = self.dec5(out)
        img_rec = out

        out2 = self.avg_pool2d(features)
        out2 = out2.view(out2.size(0), -1)
        predict = self.fc(out2)

        return predict, img_rec
        # return img_rec


class ResNet_Encoder(nn.Module):
    def __init__(self, ResidualBlock):
        super(ResNet_Encoder, self).__init__()

        # 图片处理，也就是白色方框内的3*3 conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(), )
        # 中间的残差网络部分，与图上的结构一一对应
        self.layer1 = self.make_layer(ResidualBlock, 64, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 64, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 128, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 256, 512, 2, stride=2)

    # 相当于看处理几次，18的是每个处理两次
    def make_layer(self, block, inchannel, outchannel, num_blocks, stride):
        layers = []
        for i in range(1, num_blocks):
            layers.append(block(inchannel, outchannel, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        features = out

        return features


class Decoder(nn.Module):
    def __init__(self, ResidualBlock):
        super(Decoder, self).__init__()

        # 图片处理，也就是白色方框内的3*3 conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # 中间的残差网络部分，与图上的结构一一对应
        self.dec1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.dec3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.dec4 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dec5 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)

    def make_layer(self, block, inchannel, outchannel, num_blocks, stride):
        layers = []
        for i in range(1, num_blocks):
            layers.append(block(inchannel, outchannel, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.dec1(x)
        # print('dec1输出', out.shape)
        out = self.dec2(out)
        # print('dec2输出', out.shape)
        out = self.dec3(out)
        # print('dec3输出', out.shape)
        out = self.dec4(out)
        # print('dec4输出', out.shape)
        out = self.dec5(out)
        img_rec = out
        return img_rec


class Classifier_head(nn.Module):
    def __init__(self):
        super(Classifier_head, self).__init__()
        # 中间的残差网络部分，与图上的结构一一对应
        self.avg_pool2d = nn.AvgPool2d(4)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        out = self.avg_pool2d(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out




###ResNet for MNIST
class ResNet18_MNIST(nn.Module):
    def __init__(self, ResidualBlock):
        super(ResNet18_MNIST, self).__init__()

        # 图片处理，也就是白色方框内的3*3 conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # 中间的残差网络部分，与图上的结构一一对应
        self.layer1 = self.make_layer(ResidualBlock, 64, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 64, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 128, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 256, 512, 2, stride=2)
        self.avg_pool2d = nn.AvgPool2d(4)
        self.fc = nn.Linear(512, 10)

    # 相当于看处理几次，18的是每个处理两次
    def make_layer(self, block, inchannel, outchannel, num_blocks, stride):
        layers = []
        for i in range(1, num_blocks):
            layers.append(block(inchannel, outchannel, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool2d(out)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.fc(out)
        return out








class ResNet18_tsne(nn.Module):
    def __init__(self, ResidualBlock):
        super(ResNet18_tsne, self).__init__()

        # 图片处理，也就是白色方框内的3*3 conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # 中间的残差网络部分，与图上的结构一一对应
        self.layer1 = self.make_layer(ResidualBlock, 64, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 64, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 128, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 256, 512, 2, stride=2)
        self.avg_pool2d = nn.AvgPool2d(4)
        self.fc = nn.Linear(512, 10)

    # 相当于看处理几次，18的是每个处理两次
    def make_layer(self, block, inchannel, outchannel, num_blocks, stride):
        layers = []
        for i in range(1, num_blocks):
            layers.append(block(inchannel, outchannel, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool2d(out)
        out = out.view(out.size(0), -1)
        feature = out
        out = self.fc(out)
        return out, feature








class ResNet_Encoder_tsne(nn.Module):
    def __init__(self, ResidualBlock):
        super(ResNet_Encoder_tsne, self).__init__()

        # 图片处理，也就是白色方框内的3*3 conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(), )
        # 中间的残差网络部分，与图上的结构一一对应
        self.layer1 = self.make_layer(ResidualBlock, 64, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 64, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 128, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 256, 512, 2, stride=2)
        self.avg_pool2d = nn.AvgPool2d(4)

    # 相当于看处理几次，18的是每个处理两次
    def make_layer(self, block, inchannel, outchannel, num_blocks, stride):
        layers = []
        for i in range(1, num_blocks):
            layers.append(block(inchannel, outchannel, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool2d(out)
        features = out
        return features





class ResNet_Denoiser(nn.Module):
    def __init__(self, ResidualBlock):
        super(ResNet_Denoiser, self).__init__()

        # 图片处理，也就是白色方框内的3*3 conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # 中间的残差网络部分，与图上的结构一一对应
        self.layer1 = self.make_layer(ResidualBlock, 64, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 64, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 128, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 256, 512, 2, stride=2)

        self.dec1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.dec3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.dec4 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dec5 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)

        self.avg_pool2d = nn.AvgPool2d(4)
        self.fc = nn.Linear(512, 10)

    # 相当于看处理几次，18的是每个处理两次
    def make_layer(self, block, inchannel, outchannel, num_blocks, stride):
        layers = []
        for i in range(1, num_blocks):
            layers.append(block(inchannel, outchannel, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        # print('初始形状', x.shape)
        out = self.conv1(x)
        # print('第一层卷积', out.shape)
        out = self.layer1(out)
        # print('layer1输出', out.shape)
        out = self.layer2(out)
        # print('layer2输出', out.shape)
        out = self.layer3(out)
        # print('layer3输出', out.shape)
        out = self.layer4(out)
        # print('layer4输出', out.shape, '分界线')
        features = out

        out = self.dec1(features)
        # print('dec1输出', out.shape)
        out = self.dec2(out)
        # print('dec2输出', out.shape)
        out = self.dec3(out)
        # print('dec3输出', out.shape)
        out = self.dec4(out)
        # print('dec4输出', out.shape)
        out = self.dec5(out)
        img_rec = out

        return img_rec






class ResNet18_Dnet(nn.Module):
    def __init__(self, ResidualBlock, class_num=1):
        super(ResNet18_Dnet, self).__init__()

        # 图片处理，也就是白色方框内的3*3 conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # 中间的残差网络部分，与图上的结构一一对应
        self.layer1 = self.make_layer(ResidualBlock, 64, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 64, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 128, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 256, 512, 2, stride=2)
        self.avg_pool2d = nn.AvgPool2d(4)
        self.fc = nn.Linear(512, class_num)

    # 相当于看处理几次，18的是每个处理两次
    def make_layer(self, block, inchannel, outchannel, num_blocks, stride):
        layers = []
        for i in range(1, num_blocks):
            layers.append(block(inchannel, outchannel, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool2d(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out




class ResNet_MNIST(nn.Module):
    def __init__(self, blocks, num_classes=10, expansion=4):
        super(ResNet_MNIST, self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes=1, places=64)

        self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places=256, places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512, places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024, places=512, block=blocks[3], stride=2)

        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places, stride, downsampling=True))
        for i in range(1, block):
            layers.append(Bottleneck(places * self.expansion, places))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def ResNet50_MNIST():
    return ResNet_MNIST([3, 4, 6, 3])


def ResNet101_MNIST():
    return ResNet_MNIST([3, 4, 23, 3])


def ResNet152_MNIST():
    return ResNet_MNIST([3, 8, 36, 3])


















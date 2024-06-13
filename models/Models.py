import torch.nn as nn

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



class LeNet5_35x35(nn.Module):

    def __init__(self):
        super(LeNet5_35x35, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(4096, 200)
        self.relu3 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(200, 10)

    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
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



class LeNet5_encoder(nn.Module):
    def __init__(self):
        super(LeNet5_encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, return_indices=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, return_indices=True)

    def forward(self, x):
        # out = self.maxpool1(self.relu1(self.conv1(x)))
        # out = self.maxpool2(self.relu2(self.conv2(out)))
        # print('输入之前', x.shape)
        out = self.conv1(x)
        # print('卷积1', out.shape)
        out = self.relu1(out)
        # print('RELU1', out.shape)
        out, indice_1 = self.maxpool1(out)
        # print('池化1', out.shape)
        out = self.conv2(out)
        # print('卷积2', out.shape)
        out = self.relu2(out)
        # print('RELU2', out.shape)
        out, indice_2 = self.maxpool2(out)
        # print('池化2', out.shape)
        return out, indice_1, indice_2


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dec1 = nn.MaxUnpool2d(2)
        self.dec2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.dec3 = nn.MaxUnpool2d(2)
        self.dec4 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x, indice_1, indice_2):
        out = self.dec1(x, indice_2)
        out = self.dec2(out)
        out = self.dec3(out, indice_1)
        out = self.dec4(out)
        return out

class Classifier_head(nn.Module):
    def __init__(self):
        super(Classifier_head, self).__init__()
        self.linear1 = nn.Linear(7 * 7 * 64, 200)
        self.relu3 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(200, 10)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.relu3(self.linear1(out))
        out = self.linear2(out)
        return out


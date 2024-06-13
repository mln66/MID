import torch
import torch.nn as nn
from collections import OrderedDict


def getgrad(net):
    g = OrderedDict()
    for name, param in net.named_parameters():
        if param.grad is not None:
            # print(name)
            # print(type(torch.tensor(param.grad).view(-1)))
            g[name] = param.grad.clone()
            # param.grad.data.zero_()
    # print(type(g))
    return g


def cloned_state_dict(Model):
    cloned_state_dict = {
        key: val.clone()
        for key, val in Model.state_dict().items()
    }
    return cloned_state_dict


def resize_features(features):
    features = nn.AvgPool2d(4)(features)
    features = features.view(features.size(0), -1)
    return features


def get_cossimi(feature1, feature2):
    # 先预处理，池化+拉平
    feature1 = resize_features(feature1)
    feature2 = resize_features(feature2)
    distance = torch.cosine_similarity(feature1, feature2)
    return distance.sum()
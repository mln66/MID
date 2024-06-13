# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 10:39:25 2021

@author: admin
"""

import os
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
# from Tools import filters, JSMA
from torchvision import utils as vutils
import numpy as np
from collections import OrderedDict
import random
# from frank_wolfe import FrankWolfe
# from autoattack import AutoAttack
import advertorch
from advertorch.attacks import LinfPGDAttack, CarliniWagnerL2Attack, DDNL2Attack, SinglePixelAttack, LocalSearchAttack, SpatialTransformAttack,L1PGDAttack
from autoattack import AutoAttack
from models.LeNet import LeNet5_autoencoder, LeNet5_encoder, Decoder, Classifier_head
import torchvision.transforms as transforms


# basic settings
seed = 0
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现。
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
NUM_EPOCHS = 90
LEARNING_RATE = 1e-3
BATCH_SIZE = 256


# def save_decoded_image(img, name):
#     img = img.view(img.size(0), 3, 32, 32)
#     save_image(img, name)
#

###preprocess###
transform = transforms.Compose(
    [transforms.ToTensor()
     ]
)

trainset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
testset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)
trainloader = DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True
)
testloader = DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


model = torch.load('./saving_models/LeNet5_MNIST.pkl')
model.eval()
FGSM_N = advertorch.attacks.GradientSignAttack(predict=model, loss_fn=nn.CrossEntropyLoss(reduction="sum"))

FGSM_T = advertorch.attacks.GradientSignAttack(predict=model, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
targeted=True)

PGD_N = LinfPGDAttack(
            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.30,
            nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)

PGD_T = LinfPGDAttack(
            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.30,
            nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=True)


CW_N = CarliniWagnerL2Attack(
    model, 10, clip_min=0.0, clip_max=1.0, max_iterations=500, confidence=1, initial_const=1, learning_rate=1e-2,
    binary_search_steps=4, targeted=False)

CW_T = CarliniWagnerL2Attack(
    model, 10, clip_min=0.0, clip_max=1.0, max_iterations=500, confidence=1, initial_const=1, learning_rate=1e-2,
    binary_search_steps=4, targeted=True)

DDN = DDNL2Attack(model, nb_iter=100, gamma=0.05, init_norm=1.0, quantize=True, levels=256, clip_min=0.0,
                        clip_max=1.0, targeted=False, loss_fn=None)

STA = SpatialTransformAttack(
    model, 10, clip_min=0.0, clip_max=1.0, max_iterations=5000, search_steps=20, targeted=False)

AA_N = AutoAttack(model, norm='Linf', eps=0.3, version='standard')

JSMA_T = advertorch.attacks.JacobianSaliencyMapAttack(predict=model, num_classes=10, clip_min=0.0, clip_max=1.0,
                                                    loss_fn=None, theta=1.0, gamma=1.0,
                                                    comply_cleverhans=False)


#APE-gan





















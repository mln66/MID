# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 10:39:25 2021

@author: admin
"""
from sklearn.manifold import TSNE
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
from models.LeNet import LeNet5_autoencoder, LeNet5_encoder, Decoder, Classifier_head, LeNet5, LeNet5_STA, LeNet5_tsne, LeNet5_dnet
from models.ResNet import ResNet18_MNIST, ResidualBlock
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
NUM_EPOCHS = 200
LEARNING_RATE = 1e-3
BATCH_SIZE = 1


# def save_decoded_image(img, name):
#     img = img.view(img.size(0), 3, 32, 32)
#     save_image(img, name)
#

###preprocess###
transform = transforms.Compose(
    [transforms.ToTensor()
     ]
)

# trainset = datasets.CIFAR100(
#     root='./datas/CIFAR100',
#     train=True,
#     download=True,
#     transform=transform
# )
# testset = datasets.CIFAR100(
#     root='./datas/CIFAR100',
#     train=False,
#     download=True,
#     transform=transform
# )

trainset = datasets.SVHN(
    root='./datas/SVHN',
    split='train',
    download=True,
    transform=transform
)
testset = datasets.SVHN(
    root='./datas/SVHN',
    split='test',
    download=True,
    transform=transform
)



trainloader = DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=False
)
testloader = DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

#
# model = torch.load('./saving_models/LeNet5_MNIST.pkl')
# model.eval()
#
# # pixel constrain
# FGSM_N = advertorch.attacks.GradientSignAttack(predict=model, loss_fn=nn.CrossEntropyLoss(reduction="sum"))
#
# FGSM_T = advertorch.attacks.GradientSignAttack(predict=model, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
# targeted=True)
#
# MMT_N = advertorch.attacks.MomentumIterativeAttack(predict=model, loss_fn=nn.CrossEntropyLoss(), eps=0.3, nb_iter=40,
#                                                    decay_factor=1.0, eps_iter=0.01,
#                                                    clip_min=0.0, clip_max=1.0, targeted=False)
# MMT_T = advertorch.attacks.MomentumIterativeAttack(predict=model, loss_fn=nn.CrossEntropyLoss(), eps=0.3, nb_iter=40,
#                                                    decay_factor=1.0, eps_iter=0.01,
#                                                    clip_min=0.0, clip_max=1.0, targeted=True)
# BIM_N = advertorch.attacks.LinfBasicIterativeAttack(predict=model, loss_fn=nn.CrossEntropyLoss(), eps=0.3, nb_iter=40,
#                                                     eps_iter=0.05,
#                                                     clip_min=0.0, clip_max=1.0, targeted=False)
# BIM_T = advertorch.attacks.LinfBasicIterativeAttack(predict=model, loss_fn=nn.CrossEntropyLoss(), eps=0.5, nb_iter=40,
#                                                     eps_iter=0.05,
#                                                     clip_min=0.0, clip_max=1.0, targeted=True)
#
# PGD_N = LinfPGDAttack(
#             model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.30,
#             nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
#
# PGD_T = LinfPGDAttack(
#             model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.50,
#             nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=True)
#
#
# CW_N = CarliniWagnerL2Attack(
#     model, 10, clip_min=0.0, clip_max=1.0, max_iterations=500, confidence=1, initial_const=1, learning_rate=1e-2,
#     binary_search_steps=4, targeted=False)
#
# CW_T = CarliniWagnerL2Attack(
#     model, 10, clip_min=0.0, clip_max=1.0, max_iterations=500, confidence=1, initial_const=1, learning_rate=1e-2,
#     binary_search_steps=4, targeted=True)
#
# DDN = DDNL2Attack(model, nb_iter=100, gamma=0.05, init_norm=1.0, quantize=True, levels=256, clip_min=0.0,
#                         clip_max=1.0, targeted=False, loss_fn=None)
#
# STA = SpatialTransformAttack(
#     model, 10, clip_min=0.0, clip_max=1.0, max_iterations=5000, search_steps=20, targeted=False)
#
# AA_N = AutoAttack(model, norm='Linf', eps=0.3, version='standard')
#
# JSMA_T = advertorch.attacks.JacobianSaliencyMapAttack(predict=model, num_classes=10, clip_min=0.0, clip_max=1.0,
#                                                     loss_fn=None, theta=1.0, gamma=1.0,
#                                                     comply_cleverhans=False)
#
# # spatial constrain
# # DDN
# DDN_N = DDNL2Attack(model, nb_iter=100, gamma=0.05, init_norm=1.0, quantize=True, levels=256, clip_min=0.0,
#                         clip_max=1.0, targeted=False, loss_fn=None)
#
#
# # STA
# STA_N = SpatialTransformAttack(
#     model, 10, clip_min=0.0, clip_max=1.0, max_iterations=5000, search_steps=20, targeted=False)





# lenet5_mnist = torch.load('./saving_models/LeNet5_MNIST_DST.pkl').cuda()
# lenet5_mnist.eval()
# total = 0
# correct = 0
# temp = 0
# targetlabel = torch.zeros(BATCH_SIZE).cuda()
# targetlabel = targetlabel.to('cuda', dtype=torch.int64)
# for img, label in testloader:
#     targetlabel_temp = targetlabel[0:img.shape[0]]
#     img, label = img.cuda(), label.cuda()
#
#     img_adv = CW_N.perturb(img)
#
#     x = lenet5_mnist(img_adv)
#     _, prediction = torch.max(x, 1)
#     total += label.size(0)
#     correct += (prediction == label).sum()
#     # vutils.save_image(img_adv, './saving_samples/AA_N/img_adv_{}.jpg'.format(temp))
#     print('当前temp:', temp, '当前batch正确的个数:', (prediction == label).sum())
#     temp += 1
# print('There are ' + str(correct.item()) + ' correct pictures.')
# print('Accuracy=%.2f' % (100.00 * correct.item() / total))
#







# normal train
# model = LeNet5().cuda()
# model.train()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))
# criterion_cls = nn.CrossEntropyLoss()
# train_loss = []
# for epoch in range(NUM_EPOCHS):
#     running_loss = 0.0
#     for img, label in trainloader:
#         img, label = img.cuda(), label.cuda()
#         optimizer.zero_grad()
#
#         predict = model(img)
#         loss = criterion_cls(predict, label)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#     loss = running_loss / len(trainloader)
#     train_loss.append(loss)
#     print('Epoch {} of {}, Train Loss: {:.6f}, '.format(epoch + 1, NUM_EPOCHS, loss))
# torch.save(model, './saving_models/LeNet5_MNIST_2.pkl')


# # test
# resnet18_cifar10 = torch.load('./saving_models/LeNet5_MNIST_AE.pkl').cuda()
# resnet18_cifar10.eval()
# total = 0
# correct = 0
# temp = 0
# targetlabel = torch.zeros(BATCH_SIZE).cuda()
# targetlabel = targetlabel.to('cuda', dtype=torch.int64)
# for img, label in testloader:
#     targetlabel_temp = targetlabel[0:img.shape[0]]
#     img, label = img.cuda(), label.cuda()
#
#     x,_ = resnet18_cifar10(img)
#     _, prediction = torch.max(x, 1)
#     total += label.size(0)
#     correct += (prediction == label).sum()
#     # vutils.save_image(img_adv, './saving_samples/AA_N/img_adv_{}.jpg'.format(temp))
#     print('当前temp:', temp, '当前batch正确的个数:', (prediction == label).sum())
#     temp += 1
# print('There are ' + str(correct.item()) + ' correct pictures.')
# print('Accuracy=%.2f' % (100.00 * correct.item() / total))

# class_all = 64
# class_temp = 0
# img_save = 0
# for img, label in testloader:
#     img, label = img.cuda(), label.cuda()
#     if label[0] == class_temp:
#         if label[0] == 0:
#             img_save = img
#
#         else:
#             img_save = torch.cat((img_save, img), 0)
#         class_temp += 1
#     if class_temp==class_all:
#         break
# vutils.save_image(img_save, '/media/dl/25a5b9b0-9c59-446d-a02f-81759393ee65/zyhhh/project/Meta_defense/saving_samples/show/img.jpg')

print(torchvision.__version__)

from timm_local.models import create_model

'vit_base_patch32_224'

vit_tiny_patch16_224 = create_model(
    'vit_base_patch32_224',
    pretrained=False,
    num_classes=100,
    drop_path_rate=0,
).cuda()


# optimizer = torch.optim.Adam(vit_tiny_patch16_224.parameters(), lr=0.001, betas=(0.9, 0.99))
# criterion_cls = nn.CrossEntropyLoss()
# criterion_rec = nn.L1Loss()
# train_loss = []
# for epoch in range(NUM_EPOCHS):
#     running_loss = 0.0
#     for img, label in trainloader:
#         img, label = img.cuda(), label.cuda()
#         optimizer.zero_grad()
#         predict = vit_tiny_patch16_224(img)
#         cls_loss = criterion_cls(predict, label)
#         loss = cls_loss
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#     loss = running_loss / len(trainloader)
#     train_loss.append(loss)
#     print('Epoch {} of {}, Train Loss: {:.6f}, Cls Loss: {:.6f}'.format(epoch + 1, NUM_EPOCHS, loss, cls_loss))
#     torch.save(vit_tiny_patch16_224, './saving_models/vit_base_ImageNet100.pkl')

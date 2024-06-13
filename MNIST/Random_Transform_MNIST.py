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
from advertorch.attacks import LinfPGDAttack, CarliniWagnerL2Attack, DDNL2Attack, SinglePixelAttack, LocalSearchAttack, \
    SpatialTransformAttack, L1PGDAttack
from autoattack import AutoAttack
from models.LeNet import LeNet5_autoencoder, LeNet5_encoder, Decoder, Classifier_head
import torchvision.transforms as transforms


# print(np.__version__)

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
BATCH_SIZE = 1




###preprocess###
transform = transforms.Compose(
    [
        transforms.ToTensor(),
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






# training LeNet 35x35
# model = LeNet5_35x35().cuda()
# model.train()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.99))
# criterion = nn.CrossEntropyLoss()
# train_loss = []
# for epoch in range(NUM_EPOCHS):
#     running_loss = 0.0
#     for img, label in trainloader:
#         img, label = img.cuda(), label.cuda()
#         optimizer.zero_grad()
#         predict = model(img)
#         loss = criterion(predict, label)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#     loss = running_loss / len(trainloader)
#     train_loss.append(loss)
#     print('Epoch {} of {}, Train Loss: {:.6f}, '.format(epoch + 1, NUM_EPOCHS, loss))
#     torch.save(model, './saving_models/LeNet5_MNIST_35X35.pkl')
#









# # 随机化缓解对抗效应
# def padding_layer(img, width=35, height=35):
#     # 随机调整大小层
#     targetsize =35
#     vutils.save_image(img, './saving_samples/random_transform/img_adv.jpg')
#     img = img.cpu()
#     img_pil = transforms.ToPILImage()(img)
#     img_pil.save('/media/dl/25a5b9b0-9c59-446d-a02f-81759393ee65/zyhhh/project/Meta_defense/saving_samples/random_transform/img_copy.jpg')
#     # 随机调整大小
#     a = random.choice(range(30, 34))
#     b = random.choice(range(30, 34))
#     img_resize = transforms.Resize([a, b])(img_pil)
#     img_resize = transforms.ToTensor()(img_resize)
#     # print('img.shape=', img_resize.shape)
#     c = random.choice(range(0, targetsize - a))
#     d = random.choice(range(0, targetsize - b))
#     # print('a=',a,'b=',b,'c=',c,'d=',d)
#     # print('左补=', c)
#     # print('右补=', targetsize-a-c)
#     # print('上补=', d)
#     # print('下补=', targetsize-b-d)
#     padding = nn.ZeroPad2d(padding=( d, targetsize-b-d, c, targetsize-a-c))  # 左，右，上，下
#     img_resize_padding = padding(img_resize)
#     # print('不起之后', img_resize_padding.shape)
#     vutils.save_image(img_resize_padding, './saving_samples/random_transform/img_resize_padding.jpg')
#     return img_resize_padding.unsqueeze(0).cuda()
#
#     # 填充
# lenet_minist_35X35 = torch.load('./saving_models/LeNet5_MNIST_35X35.pkl')
# total = 0
# temp = 0
# correct = 0
#
# targetlabel = torch.zeros(BATCH_SIZE).cuda()
# targetlabel = targetlabel.to('cuda', dtype=torch.int64)
# for img, label in testloader:
#     img, label = img.cuda(), label.cuda()
#     targetlabel_temp = targetlabel[0:img.shape[0]]
#     img_adv = FGSM_N(img, targetlabel_temp)
#     img_RT = padding_layer(img_adv[0])
#     x = lenet_minist_35X35(img_RT)
#     _, prediction = torch.max(x, 1)
#     total += label.size(0)
#     correct += (prediction == label).sum()
#     # vutils.save_image(img_adv, './saving_samples/AA_N/img_adv_{}.jpg'.format(temp))
#     print('当前temp:', temp, '当前batch正确的个数:', (prediction == label).sum())
#     temp += 1
# print('There are ' + str(correct.item()) + ' correct pictures.')
# print('Accuracy=%.2f' % (100.00 * correct.item() / total))
























# 随机化缓解对抗效应
def padding_layer(img, width=35, height=35):
    # 随机调整大小层
    targetsize =28
    vutils.save_image(img, './saving_samples/random_transform/img_adv.jpg')
    img = img.cpu()
    img_pil = transforms.ToPILImage()(img)
    img_pil.save('/media/dl/25a5b9b0-9c59-446d-a02f-81759393ee65/zyhhh/project/Meta_defense/saving_samples/random_transform/img_copy.jpg')
    # 随机调整大小
    a = random.choice(range(20, 28))
    b = random.choice(range(20, 28))
    img_resize = transforms.Resize([a, b])(img_pil)
    img_resize = transforms.ToTensor()(img_resize)
    # print('img.shape=', img_resize.shape)
    c = random.choice(range(0, targetsize - a))
    d = random.choice(range(0, targetsize - b))
    # print('a=',a,'b=',b,'c=',c,'d=',d)
    # print('左补=', c)
    # print('右补=', targetsize-a-c)
    # print('上补=', d)
    # print('下补=', targetsize-b-d)
    padding = nn.ZeroPad2d(padding=( d, targetsize-b-d, c, targetsize-a-c))  # 左，右，上，下
    img_resize_padding = padding(img_resize)
    # print('不起之后', img_resize_padding.shape)
    vutils.save_image(img_resize_padding, './saving_samples/random_transform/img_resize_padding.jpg')
    return img_resize_padding.unsqueeze(0).cuda()

    # 填充
lenet_minist = torch.load('./saving_models/LeNet5_MNIST.pkl')
total = 0
temp = 0
correct = 0

targetlabel = torch.zeros(BATCH_SIZE).cuda()
targetlabel = targetlabel.to('cuda', dtype=torch.int64)
for img, label in testloader:
    img, label = img.cuda(), label.cuda()
    targetlabel_temp = targetlabel[0:img.shape[0]]

    # img_adv = PGD_N(img)
    img_adv = PGD_T.perturb(img, targetlabel_temp)
    # img_adv = AA_N.run_standard_evaluation(img, label, bs=BATCH_SIZE)

    img_RT = padding_layer(img_adv[0])
    x = lenet_minist(img_RT)
    _, prediction = torch.max(x, 1)
    total += label.size(0)
    correct += (prediction == label).sum()
    # vutils.save_image(img_adv, './saving_samples/AA_N/img_adv_{}.jpg'.format(temp))
    print('当前temp:', temp, '当前batch正确的个数:', (prediction == label).sum())
    temp += 1
print('There are ' + str(correct.item()) + ' correct pictures.')
print('Accuracy=%.2f' % (100.00 * correct.item() / total))
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 10:39:25 2021

@author: admin
"""
import os
import sys
sys.path.append(os.path.abspath('../XXX'))
from autoattack import *

# from ..autoattack import AutoAttack

# import os
# import torch
# import torchvision
# import torch.nn as nn
# import torchvision.transforms as transforms
# import torch.optim as optim
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# from torchvision import datasets
# from torch.utils.data import DataLoader
# from torchvision.utils import save_image
# # from Tools import filters, JSMA
# from torchvision import utils as vutils
# import numpy as np
# from collections import OrderedDict
# import random
# # from frank_wolfe import FrankWolfe
# # from autoattack import AutoAttack
# import advertorch
# from advertorch.attacks import LinfPGDAttack, CarliniWagnerL2Attack, DDNL2Attack, SinglePixelAttack, LocalSearchAttack, SpatialTransformAttack,L1PGDAttack
# from autoattack import AutoAttack
# from models.LeNet import LeNet5_autoencoder, LeNet5_encoder, Decoder, Classifier_head, LeNet5
# import torchvision.transforms as transforms

#
# # basic settings
# seed = 0
# torch.manual_seed(seed)  # 为CPU设置随机种子
# torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
# torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
# random.seed(seed)
# np.random.seed(seed)
# os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现。
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# NUM_EPOCHS = 90
# LEARNING_RATE = 1e-3
# BATCH_SIZE = 256
#
#
# # def save_decoded_image(img, name):
# #     img = img.view(img.size(0), 3, 32, 32)
# #     save_image(img, name)
# #
#
# ###preprocess###
# transform = transforms.Compose(
#     [transforms.ToTensor()
#      ]
# )
#
# trainset = datasets.MNIST(
#     root='./data',
#     train=True,
#     download=True,
#     transform=transform
# )
# testset = datasets.MNIST(
#     root='./data',
#     train=False,
#     download=True,
#     transform=transform
# )
# trainloader = DataLoader(
#     trainset,
#     batch_size=BATCH_SIZE,
#     shuffle=True
# )
# testloader = DataLoader(
#     testset,
#     batch_size=BATCH_SIZE,
#     shuffle=False
# )
#
#
# model = torch.load('./saving_models/LeNet5_MNIST.pkl')
# model.eval()
# FGSM_N = advertorch.attacks.GradientSignAttack(predict=model, loss_fn=nn.CrossEntropyLoss(reduction="sum"))
#
# FGSM_T = advertorch.attacks.GradientSignAttack(predict=model, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
# targeted=True)
#
# PGD_N = LinfPGDAttack(
#             model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.30,
#             nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
#
# PGD_T = LinfPGDAttack(
#             model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.30,
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
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # adversarial training
# # model = LeNet5().cuda()
# # model.train()
# # PGD_N = LinfPGDAttack(
# #             model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.30,
# #             nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
# # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.99))
# # criterion = nn.CrossEntropyLoss()
# # train_loss = []
# # for epoch in range(NUM_EPOCHS):
# #     running_loss = 0.0
# #     for img, label in trainloader:
# #         img, label = img.cuda(), label.cuda()
# #         img_adv = PGD_N.perturb(img)
# #         optimizer.zero_grad()
# #         predict = model(img)
# #         predict_adv = model(img_adv)
# #         loss = criterion(predict, label) + criterion(predict_adv, label)
# #         loss.backward()
# #         optimizer.step()
# #         running_loss += loss.item()
# #     loss = running_loss / len(trainloader)
# #     train_loss.append(loss)
# #     print('Epoch {} of {}, Train Loss: {:.6f}, '.format(epoch + 1, NUM_EPOCHS, loss))
# # torch.save(model, './saving_models/LeNet5_MNIST_AT.pkl')
#
# # test
# lenet_minist_at = torch.load('./saving_models/LeNet5_MNIST_AT.pkl')
# lenet_minist_at.eval()
# total = 0
# correct = 0
# temp = 0
# targetlabel = torch.zeros(BATCH_SIZE).cuda()
# targetlabel = targetlabel.to('cuda', dtype=torch.int64)
# for img, label in testloader:
#     targetlabel_temp = targetlabel[0:img.shape[0]]
#     img, label = img.cuda(), label.cuda()
#
#     # img_adv = FGSM_N.perturb(img)
#     # img_adv = JSMA_T.perturb(img, targetlabel_temp)
#
#
#     x = lenet_minist_at(img)
#     _, prediction = torch.max(x, 1)
#     total += label.size(0)
#     correct += (prediction == label).sum()
#     # vutils.save_image(img_adv, './saving_samples/AA_N/img_adv_{}.jpg'.format(temp))
#     print('当前temp:', temp, '当前batch正确的个数:', (prediction == label).sum())
#     temp += 1
# print('There are ' + str(correct.item()) + ' correct pictures.')
# print('Accuracy=%.2f' % (100.00 * correct.item() / total))
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # distillation train
# # lenet_mnist_teacher = torch.load('./saving_models/LeNet5_MNIST.pkl')
# # lenet_mnist_student = LeNet5().cuda()
# # optimizer = torch.optim.Adam(lenet_mnist_student.parameters(), lr=0.0001, betas=(0.9, 0.99))
# # criterion_cls = nn.CrossEntropyLoss()
# # criterion_dst = nn.KLDivLoss()
# # alpha = .5
# # for epoch in range(NUM_EPOCHS):
# #     running_loss = 0.0
# #     for img, label in trainloader:
# #         img, label = img.cuda(), label.cuda()
# #         optimizer.zero_grad()
# #         outputs, _ = lenet_mnist_student(img)
# #         loss_cls = criterion_cls(outputs, label)
# #
# #         predict_teacher = F.log_softmax(lenet_mnist_teacher(img), )
# #         predict_student = F.log_softmax(outputs)
# #         loss_dst = criterion_dst(predict_student, predict_teacher.detach())
# #         loss = loss_cls * (1-alpha) + loss_dst * alpha
# #         loss.backward()
# #         optimizer.step()
# #         running_loss += loss.item()
# #     loss = running_loss / len(trainloader)
# #     print('Epoch {} of {}, Train Loss: {:.6f}, '.format(epoch + 1, NUM_EPOCHS, loss))
# #     torch.save(lenet_mnist_student, './saving_models/LeNet5_MNIST_DST.pkl')
#
# # test
# # total = 0
# # correct = 0
# # temp = 0
# # model_dst = torch.load('./saving_models/LeNet5_MNIST_DST.pkl')
# # model_dst.eval()
# # targetlabel = torch.zeros(BATCH_SIZE).cuda()
# # targetlabel = targetlabel.to('cuda', dtype=torch.int64)
# # for img, label in testloader:
# #     length = img.shape[0]
# #     targetlabel_temp = targetlabel[0:length]
# #     img, label = img.cuda(), label.cuda()
# #
# #     # img_adv = PGD_N.perturb(img)
# #     img_adv = CW_T.perturb(img, targetlabel_temp)
# #     # img_adv = AA_N.run_standard_evaluation(img, label, bs=BATCH_SIZE)
# #
# #     x = model_dst(img_adv)
# #     _, prediction = torch.max(x, 1)
# #     total += label.size(0)
# #     correct += (prediction == label).sum()
# #     # vutils.save_image(img_adv, './saving_samples/AA_N/img_adv_{}.jpg'.format(temp))
# #     print('当前temp:', temp, '当前batch正确的个数:', (prediction == label).sum())
# #     temp += 1
# # print('There are ' + str(correct.item()) + ' correct pictures.')
# # print('Accuracy=%.2f' % (100.00 * correct.item() / total))
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # train the encoder by loss_ce and loss_rec
# # lenet_mnist_aotoencoder = LeNet5_autoencoder().cuda()
# # lenet_mnist_aotoencoder.train()
# # optimizer = torch.optim.Adam(lenet_mnist_aotoencoder.parameters(), lr=0.0001, betas=(0.9, 0.99))
# # criterion_cls = nn.CrossEntropyLoss()
# # criterion_rec = nn.MSELoss()
# # train_loss = []
# # for epoch in range(200):
# #     running_loss = 0.0
# #     for img, label in trainloader:
# #         img, label = img.cuda(), label.cuda()
# #         optimizer.zero_grad()
# #         predict, img_rec = lenet_mnist_aotoencoder(img)
# #         cls_loss, rec_loss = criterion_cls(predict, label), criterion_rec(img_rec, img)
# #         loss = cls_loss + rec_loss
# #         loss.backward()
# #         optimizer.step()
# #         running_loss += loss.item()
# #     loss = running_loss / len(trainloader)
# #     train_loss.append(loss)
# #     print('Epoch {} of {}, Train Loss: {:.6f}, Cls Loss: {}, Rec_loss: {}'.format(epoch + 1, NUM_EPOCHS, loss, cls_loss, rec_loss ))
# #     vutils.save_image(img_rec, '/media/dl/25a5b9b0-9c59-446d-a02f-81759393ee65/zyhhh/project/Meta_defense/saving_samples/autoencoder/decoded{}.png'.format(epoch))
# #     vutils.save_image(img, '/media/dl/25a5b9b0-9c59-446d-a02f-81759393ee65/zyhhh/project/Meta_defense/saving_samples/autoencoder/original{}.png'.format(epoch))
# # torch.save(lenet_mnist_aotoencoder, '/media/dl/25a5b9b0-9c59-446d-a02f-81759393ee65/zyhhh/project/Meta_defense/saving_models/lenet_mnist_aotoencoder.pkl')
#
# # # test the auto encoder
# # lenet_mnist_aotoencoder = torch.load('/media/dl/25a5b9b0-9c59-446d-a02f-81759393ee65/zyhhh/project/Meta_defense/saving_models/LeNet5_MNIST_AE.pkl')
# # lenet_mnist_aotoencoder.eval()
# # total = 0
# # correct = 0
# # temp = 0
# # targetlabel = torch.zeros(BATCH_SIZE).cuda()
# # targetlabel = targetlabel.to('cuda', dtype=torch.int64)
# # for img, label in testloader:
# #     img, label = img.cuda(), label.cuda()
# #     targetlabel_temp = targetlabel[0:img.shape[0]]
# #
# #     # img_adv = PGD_N.perturb(img)
# #     img_adv = CW_T.perturb(img, targetlabel_temp)
# #     # img_adv = AA_N.run_standard_evaluation(img, label, bs=BATCH_SIZE)
# #
# #     x, _ = lenet_mnist_aotoencoder(img_adv)
# #     _, prediction = torch.max(x, 1)
# #     total += label.size(0)
# #     correct += (prediction == label).sum()
# #     print('当前temp:', temp, '当前batch正确的个数:', (prediction == label).sum())
# #     temp += 1
# # print('There are ' + str(correct.item()) + ' correct pictures.')
# # print('Accuracy=%.2f' % (100.00 * correct.item() / total))
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#

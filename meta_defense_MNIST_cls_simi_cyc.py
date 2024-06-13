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
from torchvision import utils as vutils
import numpy as np
from collections import OrderedDict
import random
# from frank_wolfe import FrankWolfe
# from autoattack import AutoAttack
import advertorch
from advertorch.test_utils import LeNet5
from advertorch.attacks import LinfPGDAttack, CarliniWagnerL2Attack, DDNL2Attack, SinglePixelAttack, LocalSearchAttack, \
    SpatialTransformAttack, L1PGDAttack
from models.LeNet import LeNet5_autoencoder, LeNet5_encoder, Decoder, Classifier_head
from autoattack import AutoAttack
from tools.Tools import getgrad, cloned_state_dict, resize_features, get_cossimi

# basic settings
seed = 0
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现。
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
NUM_EPOCHS = 120
LEARNING_RATE = 1e-4
BATCH_SIZE = 256


def save_decoded_image(img, name):
    img = img.view(img.size(0), 3, 32, 32)
    save_image(img, name)


###preprocess###
transform = transforms.Compose(
    [transforms.ToTensor()
     ]
)

trainset = datasets.MNIST(
    root='./MNIST_data',
    train=True,
    download=True,
    transform=transform
)
testset = datasets.MNIST(
    root='./MNIST_data',
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
    shuffle=True
)

'''
MID
'''



encoder_clean = LeNet5_encoder().cuda()
decoder = Decoder().cuda()
classifier_head = Classifier_head().cuda()
encoder_clean.eval()
encoder_adv = LeNet5_encoder().cuda().train()
classifier_head.eval()
decoder.eval()
optimizer = optim.Adam(encoder_adv.parameters(), lr=LEARNING_RATE)


LeNet5_autoencoder_ae = torch.load('./saving_models/LeNet5_MNIST_AE.pkl')
LeNet5_autoencoder_dict = LeNet5_autoencoder_ae.state_dict()
encoder_clean_dict = encoder_clean.state_dict()
classifier_head_dict = classifier_head.state_dict()
decoder_dict = decoder.state_dict()





# 装载干净编码器参数
pretrained_dict_encoder = {k: v for k, v in LeNet5_autoencoder_dict.items() if k in encoder_clean_dict}
encoder_clean_dict.update(pretrained_dict_encoder)
encoder_clean.load_state_dict(encoder_clean_dict)

# 装载分类头参数
pretrained_dict_classifier = {k: v for k, v in LeNet5_autoencoder_dict.items() if k in classifier_head_dict}
classifier_head_dict.update(pretrained_dict_classifier)
classifier_head.load_state_dict(classifier_head_dict)

# 装载解码器参数
pretrained_dict_decoder = {k: v for k, v in LeNet5_autoencoder_dict.items() if k in decoder_dict}
decoder_dict.update(pretrained_dict_decoder)
decoder.load_state_dict(decoder_dict)



lenet_mnist = torch.load('./saving_models/Revision/MNIST_LeNet5.pkl')
# LeNet5_dict = lenet_mnist.state_dict()
# for k, v in LeNet5_dict.items():
#     print(k, end='\t')

FGSM_N = advertorch.attacks.GradientSignAttack(predict=lenet_mnist, loss_fn=nn.CrossEntropyLoss(reduction="sum"))

FGSM_T = advertorch.attacks.GradientSignAttack(predict=lenet_mnist, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                                               targeted=True)
FFA = advertorch.attacks.FastFeatureAttack(predict=lenet_mnist, loss_fn=nn.CrossEntropyLoss(), eps=0.3, eps_iter=0.05,
                                           nb_iter=10,
                                           rand_init=True, clip_min=0.0, clip_max=1.0)
BIM_N = advertorch.attacks.LinfBasicIterativeAttack(predict=lenet_mnist, loss_fn=nn.CrossEntropyLoss(), eps=0.3,
                                                    nb_iter=40,
                                                    eps_iter=0.05,
                                                    clip_min=0.0, clip_max=1.0, targeted=False)
BIM_T = advertorch.attacks.LinfBasicIterativeAttack(predict=lenet_mnist, loss_fn=nn.CrossEntropyLoss(), eps=0.3,
                                                    nb_iter=40,
                                                    eps_iter=0.05,
                                                    clip_min=0.0, clip_max=1.0, targeted=True)
MMT_N = advertorch.attacks.MomentumIterativeAttack(predict=lenet_mnist, loss_fn=nn.CrossEntropyLoss(), eps=0.3,
                                                   nb_iter=40,
                                                   decay_factor=1.0, eps_iter=0.01,
                                                   clip_min=0.0, clip_max=1.0, targeted=False)
MMT_T = advertorch.attacks.MomentumIterativeAttack(predict=lenet_mnist, loss_fn=nn.CrossEntropyLoss(), eps=0.3,
                                                   nb_iter=40,
                                                   decay_factor=1.0, eps_iter=0.01,
                                                   clip_min=0.0, clip_max=1.0, targeted=True)

PGD_N = LinfPGDAttack(
    lenet_mnist, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.30,
    nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)

PGD_T = LinfPGDAttack(
    lenet_mnist, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.30,
    nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=True)

CW_T = CarliniWagnerL2Attack(
    lenet_mnist, 10, clip_min=0.0, clip_max=1.0, max_iterations=500, confidence=1, initial_const=1, learning_rate=1e-2,
    binary_search_steps=4, targeted=False)

DDN = DDNL2Attack(lenet_mnist, nb_iter=100, gamma=0.05, init_norm=1.0, quantize=True, levels=256, clip_min=0.0,
                  clip_max=1.0, targeted=False, loss_fn=None)

STA = SpatialTransformAttack(
    lenet_mnist, 10, clip_min=0.0, clip_max=1.0, max_iterations=5000, search_steps=20, targeted=False)

AA_N = AutoAttack(lenet_mnist, norm='Linf', eps=0.3, version='standard')

JSMA_T = advertorch.attacks.JacobianSaliencyMapAttack(predict=lenet_mnist, num_classes=10, clip_min=0.0, clip_max=1.0,
                                                      loss_fn=None, theta=1.0, gamma=1.0,
                                                      comply_cleverhans=False)

# DDN
DDN_N = DDNL2Attack(lenet_mnist, nb_iter=100, gamma=0.05, init_norm=1.0, quantize=True, levels=256, clip_min=0.0,
                        clip_max=1.0, targeted=False, loss_fn=None)


# STA
STA_N = SpatialTransformAttack(
    lenet_mnist, 10, clip_min=0.0, clip_max=1.0, max_iterations=5000, search_steps=20, targeted=False)





targetlabel = torch.zeros(BATCH_SIZE).cuda()
targetlabel = targetlabel.to('cuda', dtype=torch.int64)

#  start train
# batch_idx = 0
# for epoch in range(NUM_EPOCHS):
#     for img, label in trainloader:
#         targetlabel_temp = targetlabel[0:img.shape[0]]
#         meta_train_loss = 0
#         img, label = img.cuda(), label.cuda()
#         img_pgdn = PGD_N(img)
#         img_pgdt = PGD_T(img, targetlabel_temp)
#         img_mmtn = MMT_N(img)
#         img_mmtt = MMT_T(img, targetlabel_temp)
#         feature_clean, indice_1_clean, indice_2_clean = encoder_clean(img)
#         data_all = [img, img_pgdn, img_pgdt, img_mmtn, img_mmtt]
#         # data_all = [img, img_pgdn, img_pgdt]
#         index_val = np.random.choice(a=np.arange(0, len(data_all)), size=1)[0]
#
#         # meta train
#         for j in range(len(data_all)):
#             if j == index_val:
#                 continue
#             feature_train1, indice_1_meta1, indice_2_meta1 = encoder_adv(data_all[j])
#             outputs_train1 = classifier_head(feature_train1)
#             img_rec1 = decoder(feature_train1, indice_1_meta1, indice_2_meta1)
#             classifier_loss1 = nn.CrossEntropyLoss()(outputs_train1, label)  # 分类损失
#             simi_loss1 = nn.KLDivLoss()(nn.LogSoftmax()(feature_train1), nn.Softmax()(feature_clean))  # 特征相似形损失
#             rec_loss1 = nn.L1Loss()(img_rec1, img)  # 重构损失
#             cyc_loss1 = nn.KLDivLoss()(nn.LogSoftmax()(encoder_clean(img_rec1)[0]), nn.LogSoftmax()(feature_clean))
#
#             meta_train_loss += classifier_loss1 + 1.5 * simi_loss1 + 0.8 * cyc_loss1
#
#         optimizer.zero_grad()
#         meta_train_loss.backward(retain_graph=True)
#
#         grads_encoder_adv = getgrad(encoder_adv)
#         encoder_adv_temp = LeNet5_encoder().cuda().train()
#         adapted_params_encoder_adv = OrderedDict()
#         fast_weights_encoder_adv = cloned_state_dict(encoder_adv)
#         for key, val in zip(grads_encoder_adv.keys(), grads_encoder_adv.values()):
#             adapted_params_encoder_adv[key] = fast_weights_encoder_adv[key] - LEARNING_RATE * val
#             fast_weights_encoder_adv[key] = adapted_params_encoder_adv[key]
#         encoder_adv_temp = encoder_adv_temp.cuda()
#         encoder_adv_temp.load_state_dict(fast_weights_encoder_adv)
#
#         # grads_classifier_head = getgrad(classifier_head)
#         # classifier_head_temp = Classifier_head().cuda()
#         # adapted_params_classifier_head = OrderedDict()
#         # fast_weights_classifier_head = cloned_state_dict(classifier_head)
#         # for key, val in zip(grads_classifier_head.keys(), grads_classifier_head.values()):
#         #     adapted_params_classifier_head[key] = fast_weights_classifier_head[key] - LEARNING_RATE * val
#         #     fast_weights_classifier_head[key] = adapted_params_classifier_head[key]
#         # classifier_head_temp = classifier_head_temp.cuda()
#         # classifier_head_temp.load_state_dict(fast_weights_classifier_head)
#
#         # meta test
#         feature_train2, indice_1_meta2, indice_2_meta2 = encoder_adv_temp(data_all[j])
#         # outputs_train2 = classifier_head_temp(feature_train2)
#         outputs_train2 = classifier_head(feature_train2)
#
#         img_rec2 = decoder(feature_train2, indice_1_meta2, indice_2_meta2)
#         classifier_loss2 = nn.CrossEntropyLoss()(outputs_train2, label)
#         simi_loss2 = nn.KLDivLoss()(nn.LogSoftmax()(feature_train2), nn.Softmax()(feature_clean))
#         # rec_loss2 = nn.MSELoss()(img_rec2, img)  # 重构损失
#         cyc_loss2 = nn.KLDivLoss()(nn.LogSoftmax()(encoder_clean(img_rec2)[0]), nn.LogSoftmax()(feature_clean))
#         meta_test_loss = classifier_loss2 + simi_loss2 + cyc_loss2
#
#         # final update
#         total_loss = meta_train_loss + meta_test_loss
#         optimizer.zero_grad()
#         total_loss.backward()
#         optimizer.step()
#
#
#         print('epoch=', epoch, 'batch_idx=', batch_idx, 'total_loss=',
#               total_loss, 'meta_train_loss=', meta_train_loss, 'meta_test_loss=', meta_test_loss)
#         batch_idx += 1
#         if batch_idx == 235:
#             batch_idx = 0
#
# torch.save(encoder_adv, './saving_models/final_models/encoder_adv_addclean_MNIST_cls_simi_cyc217.pkl')



'''
test
'''
encoder_adv = torch.load('./saving_models/final_models/encoder_adv_addclean_MNIST_cls_simi_cyc217.pkl').cuda().eval()
encoder_clean.eval()
classifier_head.eval()
total = 0
correct = 0
temp = 0
for img, label in testloader:
    img, label = img.cuda(), label.cuda()
    targetlabel_temp = targetlabel[0:img.shape[0]]

    img_adv = img
    # img_adv = BIM_T(img,targetlabel_temp)
    # img_adv = BIM_N(img)
    # img_adv = FGSM_N(img)
    # img_adv = FGSM_T(img,targetlabel_temp)

    # img_adv = MMT_N(img)
    # img_adv = MMT_T(img,targetlabel_temp)
    # img_adv = PGD_T.perturb(img, targetlabel_temp)
    # img_adv = CW_T.perturb(img)
    # img_adv = CW_T.perturb(img, targetlabel_temp)
    # img_adv = AA_N.run_standard_evaluation(img, label, bs=BATCH_SIZE)
    # img_adv = PGD_N.perturb(img)
    # img_adv = JSMA_T.perturb(img,targetlabel_temp)
    feature, _, _ = encoder_adv(img_adv)
    x = classifier_head(feature)
    _, prediction = torch.max(x, 1)
    total += label.size(0)
    correct += (prediction == label).sum()
    print('当前temp:', temp, '当前batch正确的个数:', (prediction == label).sum())
    temp += 1
print("meta_train network clean test")
print('There are ' + str(correct.item()) + ' correct pictures.')
print('Accuracy=%.2f' % (100.00 * correct.item() / total))

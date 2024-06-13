# encoding: utf-8
from __future__ import absolute_import

from torchvision.transforms import *
from PIL import Image
import random
import math


class RectScale(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        if h == self.height and w == self.width:
            return img
        return img.resize((self.width, self.height), self.interpolation)


class RandomSizedRectCrop(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.64, 1.0) * area
            aspect_ratio = random.uniform(2, 3)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))

                return img.resize((self.width, self.height), self.interpolation)

        # Fallback
        scale = RectScale(self.height, self.width,
                          interpolation=self.interpolation)
        return scale(img)


class RandomErasing(object):
    def __init__(self, EPSILON=0.5, mean=[0.485, 0.456, 0.406]):
        self.EPSILON = EPSILON
        self.mean = mean

    def __call__(self, img):

        # if random.uniform(0, 1) > self.EPSILON:
        #     return img

        for attempt in range(100):

            area = img.size()[1] * img.size()[2]
            print('area', area)

            target_area = random.uniform(0.02, 0.2) * area
            print('target_area=', target_area)  # 擦除的面积（之前随机的小数*总的面积，现在改写成固定的大小）

            # aspect_ratio = random.uniform(0.3, 3)
            aspect_ratio = 1
            print('aspect_ratio', aspect_ratio)  # 宽高比，之前的随机的宽高比， 现在改成固定的宽高比


            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            print('h', h, 'w', w)


            if w <= img.size()[2] and h <= img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)

                # img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                # img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                # img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]

                img[0, x1:x1 + h, y1:y1 + w] = 0
                img[1, x1:x1 + h, y1:y1 + w] = 0
                img[2, x1:x1 + h, y1:y1 + w] = 0



                return img

        return img





class RandomErasing_adv(object):
    def __init__(self, EPSILON=0.5, mean=[0.485, 0.456, 0.406]):
        self.EPSILON = EPSILON
        self.mean = mean

    def __call__(self, img, num=8):

        # if random.uniform(0, 1) > self.EPSILON:
        #     return img

        for attempt in range(num):  # 加的噪块的个数
            area = img.size()[2] * img.size()[3]
            # print('area', area)

            # target_area = random.uniform(0.02, 0.2) * area
            target_area = 16
            # print('target_area=', target_area)  # 擦除的面积（之前随机的小数*总的面积，现在改写成固定的大小4*4）

            # aspect_ratio = random.uniform(0.3, 3)
            aspect_ratio = 1
            # print('aspect_ratio', aspect_ratio)  # 宽高比，之前的随机的宽高比， 现在改成固定的宽高比


            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            # print('h', h, 'w', w)



            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))


            # if w <= img.size()[3] and h <= img.size()[2]:
            #     x1 = random.randint(0, img.size()[2] - h)  # 随机选取x轴初始点
            #     y1 = random.randint(0, img.size()[3] - w)  # 随机选取y轴初始点
            #
            #     # img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
            #     # img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
            #     # img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
            #
            #     img[:, 0, x1:x1 + h, y1:y1 + w] = 0
            #     img[:, 1, x1:x1 + h, y1:y1 + w] = 0
            #     img[:, 2, x1:x1 + h, y1:y1 + w] = 0
            #     return img

            for idx in range(img.shape[0]):
                if w <= img.size()[3] and h <= img.size()[2]:
                    x1 = random.randint(0, img.size()[2] - h)
                    y1 = random.randint(0, img.size()[3] - w)

                    # img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    # img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    # img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]

                    img[idx, 0, x1:x1 + h, y1:y1 + w] = 0
                    img[idx, 1, x1:x1 + h, y1:y1 + w] = 0
                    img[idx, 2, x1:x1 + h, y1:y1 + w] = 0

        return img










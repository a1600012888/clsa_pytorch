# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import ImageFilter
import random
import torchvision.transforms as transforms


class CALSMultiResolutionTransform(object):
    def __init__(self, base_transform, stronger_transfrom, num_res=5):
        '''
        Note: RandomResizedCrop should be includeed in stronger_transfrom
        '''
        resolutions = [96, 128, 160, 192, 224]
        
        self.res = resolutions[:num_res]
        self.resize_crop_ops = [transforms.RandomResizedCrop(res, scale=(0.2, 1.)) for res in self.res]
        self.num_res = num_res

        self.base_transform = base_transform
        self.stronger_transfrom = stronger_transfrom

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)

        q_stronger_augs = []
        for resize_crop_op in self.resize_crop_ops:
            q_s = self.stronger_transfrom(resize_crop_op(x))
            q_stronger_augs.append(q_s)
        
        return [q, k, q_stronger_augs]
        


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

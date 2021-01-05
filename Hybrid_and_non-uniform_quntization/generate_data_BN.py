import pytorchcv
from pytorchcv.model_provider import get_model as ptcv_get_model
import torch
import torchvision.models as models
from __future__ import print_function
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import pytorchcv
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import os
import json
import copy
import torchvision
import numpy as np
import cv2

class UniformDataset(Dataset):
    """
    get random uniform samples with mean 0 and variance 1
    """
    def __init__(self, length, size, transform):
        self.length = length
        self.transform = transform
        self.size = size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # var[U(-128, 127)] = (127 - (-128))**2 / 12 = 5418.75
        sample = (torch.randint(high=255, size=self.size).float() -
                  127.5) / 5418.75
        return sample


def getRandomData(dataset='cifar10', batch_size=512, for_inception=False):
    """
    get random sample dataloader
    dataset: name of the dataset
    batch_size: the batch size of random data
    for_inception: whether the data is for Inception because inception has input size 299 rather than 224
    """
    if dataset == 'cifar10':
        size = (3, 32, 32)
        num_data = 100
    elif dataset == 'imagenet':
        num_data = 10
        if not for_inception:
            size = (3, 224, 224)
        else:
            size = (3, 299, 299)
    else:
        raise NotImplementedError
    dataset = UniformDataset(length=10, size=size, transform=None)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=32)
    return data_loader

def own_loss(A, B):
    """
	L-2 loss between A and B normalized by length.
    Shape of A should be (features_num, ), shape of B should be (batch_size, features_num)
	"""
    return (A - B).norm()**2 / B.size(0)
    #return ((A - B)**2) / (B.size(0) * (A+B))
    #return torch.sum(((A-B)**2) /(A+B))/B.size(0)

def KLD_ND(m1,m2,sd1,sd2):
  ret = torch.sum(torch.log2(sd1/sd2) + ((torch.square(sd2) + torch.square(m1-m2))/(2*torch.square(sd1))) - 0.5)/m2.size(0)
  return ret

class output_hook(object):
    """
	Forward_hook used to get the output of the intermediate layer.
	"""
    def __init__(self):
        super(output_hook, self).__init__()
        self.outputs = None

    def hook(self, module, input, output):
        self.outputs = output

    def clear(self):
        self.outputs = None

def MSE(A, B):
  A = A.cuda()
  B = B.cuda()
  return torch.sum(torch.square(A-B))


def getDistilData(teacher_model,
                  dataset,
                  batch_size,
                  num_batch=1,
                  for_inception=False):
    """
	Generate distilled data according to the BatchNorm statistics in the pretrained single-precision model.
	Currently only support a single GPU.

	teacher_model: pretrained single-precision model
	dataset: the name of the dataset
	batch_size: the batch size of generated distilled data
	num_batch: the number of batch of generated distilled data
	for_inception: whether the data is for Inception because inception has input size 299 rather than 224
	"""

    eps = 1e-6
    # initialize hooks and single-precision model
    hooks, hook_handles, bn_stats, refined_gaussian = [], [], [], []
    teacher_model = teacher_model.cuda()
    teacher_model = teacher_model.eval()

    # get number of BatchNorm layers in the model
    layers = sum([
        1 if isinstance(layer, nn.BatchNorm2d) else 0
        for layer in teacher_model.modules()
    ])

    for n, m in teacher_model.named_modules():
        if isinstance(m, nn.Conv2d) and len(hook_handles) < layers:
            # register hooks on the convolutional layers to get the intermediate output after convolution and before BatchNorm.
            hook = output_hook()
            hooks.append(hook)
            hook_handles.append(m.register_forward_hook(hook.hook))
        if isinstance(m, nn.BatchNorm2d):
            # get the statistics in the BatchNorm layers
            bn_stats.append(
                (m.running_mean.detach().clone().flatten().cuda(),
                 torch.sqrt(m.running_var +
                            eps).detach().clone().flatten().cuda()))
    assert len(hooks) == len(bn_stats)

    for i in range(1000):
      dataloader = getRandomData(dataset=dataset,
                               batch_size=10,
                               for_inception=for_inception)
      for j, gaussian_data in enumerate(dataloader):
        '''if i == num_batch:
            break'''
        # initialize the criterion, optimizer, and scheduler
        gaussian_data = gaussian_data.cuda()
        gaussian_data.requires_grad = True
        crit = nn.CrossEntropyLoss().cuda()
        optimizer = optim.Adam([gaussian_data], lr=0.5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         min_lr=1e-4,
                                                         verbose=False,
                                                         patience=100)

        input_mean = torch.zeros(1, 3).cuda()
        input_std = torch.ones(1, 3).cuda()
        target = [0.00001]*1000
        #target[0] = 0
        target = torch.tensor(target)
        target[i] = 0.99001
        #print(target)
        print('---------------------------------------')
        print(i)
        print('---------------------------------------')
        for it in range(700):
            teacher_model.zero_grad()
            optimizer.zero_grad()
            for hook in hooks:
                hook.clear()
            output = teacher_model(gaussian_data)
            class_loss = MSE(target, F.softmax(output, dim=1))
            #class_loss = torch.square()
            mean_loss = 0
            std_loss = 0
            curr_loss = 0
            # compute the loss according to the BatchNorm statistics and the statistics of intermediate output
            for cnt, (bn_stat, hook) in enumerate(zip(bn_stats, hooks)):
                tmp_output = hook.outputs
                bn_mean, bn_std = bn_stat[0], bn_stat[1]
                #print('bn_state 0')
                #print(bn_stat[0])
                #print('bn_state 1')
                #print(bn_stat[1])
                tmp_mean = torch.mean(tmp_output.view(tmp_output.size(0),
                                                      tmp_output.size(1), -1),
                                      dim=2)
                tmp_std = torch.sqrt(
                    torch.var(tmp_output.view(tmp_output.size(0),
                                              tmp_output.size(1), -1),
                              dim=2) + eps)
                mean_loss += own_loss(bn_mean, tmp_mean)
                std_loss += own_loss(bn_std, tmp_std)

                #print('mean_loss')
                #print(mean_loss)
                #print('std_loss')
                #print(std_loss)

                #curr_loss += KLD_ND(bn_mean,tmp_mean,bn_std,tmp_std)
            #new_data = np.array(dataloader[i][j].cpu()).transpose(2, 1, 0)
            tmp_mean = torch.mean(gaussian_data.view(gaussian_data.size(0), 3,
                                                     -1),
                                  dim=2)
            tmp_std = torch.sqrt(
                torch.var(gaussian_data.view(gaussian_data.size(0), 3, -1),
                          dim=2) + eps)
            mean_loss += own_loss(input_mean, tmp_mean)
            std_loss += own_loss(input_std, tmp_std)
            total_loss = mean_loss + std_loss
            total_loss += class_loss
            total_loss.backward()
            optimizer.step()
            scheduler.step(total_loss.item())

        for k in range(10):
          new_data = np.array(gaussian_data.detach().clone()[k].cpu()).transpose(2, 1, 0)
          x = new_data * 255;
          path = '/content/drive/My Drive/resnet56_cifar10_data/' + 'image'+ str(i) + str(k) + '.jpg'
          cv2.imwrite(path, x)
          im = cv2.imread(path)
          im_bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
          cv2.imwrite(path, im_bgr)
        refined_gaussian.append(gaussian_data.detach().clone())

    for handle in hook_handles:
        handle.remove()
    return refined_gaussian

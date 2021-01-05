#import urllib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.nn import Module, Parameter
from PIL import Image
from torchvision import transforms
import torch
import torchvision.models as models
#from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import copy
from pytorchcv.models.common import ConvBlock
from pytorchcv.models.shufflenetv2 import ShuffleUnit, ShuffleInitBlock
from .hybrid_quant_util import *
from .quant_utils import *




url, filename = ("https://github.com/pytorch/hub/raw/master/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

# sample execution (requires torchvision)

input_image = Image.open('dog.jpg')
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(input_image)

input_batch = input_tensor.unsqueeze(0)
input_batch = input_batch.cuda()


kld_meter_pc = []
kld_meter_pt = []

def hybrid_quant(nn_model):

  '''model_for_pc = models.resnet18(pretrained=True, progress=True)
  model_original = models.resnet18(pretrained=True, progress=True)
  model_for_pt = models.resnet18(pretrained=True, progress=True)'''

  model_for_pc = copy.deepcopy(nn_model)
  model_original = copy.deepcopy(nn_model)
  model_for_pt = copy.deepcopy(nn_model)
  model = copy.deepcopy(nn_model)
  model_original.eval()

  k = 8
  flag = 0
  for n, m in model_for_pc.named_modules():
    if isinstance(m, nn.Conv2d):
      #m.weight = Parameter(torch.zeros(m.weight.shape))
      #w = self.weight
      original_weight = m.weight
      x_transform = m.weight.data.contiguous().view(m.weight.shape[0], -1)
      w_min = x_transform.min(dim=1).values
      w_max = x_transform.max(dim=1).values
      scale, zero_point = asymmetric_linear_quantization_params(
              k, w_min, w_max)
      new_quant_x = linear_quantize_pc(m.weight, scale, zero_point, inplace=False)
      n = 2**(k - 1)
      new_quant_x = torch.clamp(new_quant_x, -n, n - 1)
      quant_x = linear_dequantize_pc(new_quant_x,
                                  scale,
                                  zero_point,
                                  inplace=False)
      m.weight = Parameter(torch.autograd.Variable(quant_x))
      flag = 1
    if isinstance(m, nn.Linear):
      #w = self.weight
      original_weight = m.weight
      x_transform = m.weight.data.detach()
      #print("x_transform")
      #print(x_transform)

      w_min = x_transform.min(dim=1).values
      w_max = x_transform.max(dim=1).values

      scale, zero_point = asymmetric_linear_quantization_params(
              k, w_min, w_max)
      new_quant_x = linear_quantize_pc(m.weight, scale, zero_point, inplace=False)
      n = 2**(k - 1)
      new_quant_x = torch.clamp(new_quant_x, -n, n - 1)
      quant_x = linear_dequantize_pc(new_quant_x,
                                  scale,
                                  zero_point,
                                  inplace=False)
      m.weight = Parameter(torch.autograd.Variable(quant_x))
      flag = 1

    if flag != 0:
      model_for_pc.eval()
      output_pc = model_for_pc(input_batch)
      output_pc1 = F.softmax(output_pc, dim=1)
      ref_output = model_original(input_batch)
      ref_output1 = F.softmax(ref_output, dim=1)

      curr_kld = kl_div(ref_output1, output_pc1)
      kld_meter_pc.append(curr_kld.detach())

      m.weight = original_weight
      flag = 0




  k = 8
  flag = 0
  for n, m in model_for_pt.named_modules():
    if isinstance(m, nn.Conv2d):
      #print(n)
      original_weight = m.weight
      #m.weight = Parameter(torch.zeros(m.weight.shape))
      #w = self.weight
      #w = self.weight
      #print(m.weight.shape)
      x_transform = m.weight.data.contiguous().view(m.weight.shape[0], -1)
      w_min = x_transform.min(dim=1).values
      w_max = x_transform.max(dim=1).values
      w_min_pt = torch.min(w_min)
      w_max_pt = torch.max(w_max)

      scale, zero_point = asymmetric_linear_quantization_params(
              k, w_min_pt, w_max_pt)
      new_quant_x = linear_quantize_pt(m.weight, scale, zero_point, inplace=False)
      n = 2**(k - 1)
      new_quant_x = torch.clamp(new_quant_x, -n, n - 1)
      quant_x = linear_dequantize_pt(new_quant_x,
                                  scale,
                                  zero_point,
                                  inplace=False)
      #print(quant_x.shape)
      #print(m.weight.shape)
      m.weight = Parameter(torch.autograd.Variable(quant_x.reshape(m.weight.shape)))
      #print(m.weight.shape)

      flag = 1

    if isinstance(m, nn.Linear):
      #w = self.weight
      original_weight = m.weight
      x_transform = m.weight.data.detach()
      #print("x_transform")
      #print(x_transform)

      w_min = x_transform.min(dim=1).values
      w_max = x_transform.max(dim=1).values
      w_min_pt = torch.min(w_min)
      w_max_pt = torch.max(w_max)

      scale, zero_point = asymmetric_linear_quantization_params(
              k, w_min_pt, w_max_pt)
      new_quant_x = linear_quantize_pt(m.weight, scale, zero_point, inplace=False)
      n = 2**(k - 1)
      new_quant_x = torch.clamp(new_quant_x, -n, n - 1)
      quant_x = linear_dequantize_pt(new_quant_x,
                                  scale,
                                  zero_point,
                                  inplace=False)
      #print(quant_x.shape)
      #print(m.weight.shape)
      m.weight = Parameter(torch.autograd.Variable(quant_x.reshape(m.weight.shape)))
      #print(m.weight.shape)

      flag = 1

    if flag != 0:
      model_for_pt.eval()
      #print(n)
      #print(m.weight.shape)
      output_pt = model_for_pt(input_batch)
      output_pt1 = F.softmax(output_pt, dim=1)
      ref_output = model_original(input_batch)
      ref_output1 = F.softmax(ref_output, dim=1)

      curr_kld = kl_div(ref_output1, output_pt1)
      kld_meter_pt.append(curr_kld.detach())

      m.weight = original_weight
      #print()
      flag = 0

  mixed_flag = []

  for i in range(len(kld_meter_pc)):
    if(kld_meter_pt[i] - kld_meter_pc[i] <= 0):
      mixed_flag.append(1)
    else:
      mixed_flag.append(0)
    #mixed_flag.append(0)
    #mixed_flag.append(1)
  plt1, = plt.plot(kld_meter_pc, 'b*-')
  plt2, = plt.plot(kld_meter_pt, 'r*-')
  plt.legend(['per channel' , 'per tensor'])
  plt.grid()
  plt.savefig('pt_vs_pc_sensitivity comparison')
  print('***** mixed flag *****')
  print(mixed_flag)
  #model = models.resnet18(pretrained=True, progress=True)
  count = 0
  for n,m in model.named_modules():
    if isinstance(m, nn.Conv2d):
      if mixed_flag[count] == 0:
        #print('inside conv pc')
        #print(count)
        original_weight = m.weight
        x_transform = m.weight.data.contiguous().view(m.weight.shape[0], -1)
        w_min = x_transform.min(dim=1).values
        w_max = x_transform.max(dim=1).values
        scale, zero_point = asymmetric_linear_quantization_params(
                k, w_min, w_max)
        new_quant_x = linear_quantize_pc(m.weight, scale, zero_point, inplace=False)
        n = 2**(k - 1)
        new_quant_x = torch.clamp(new_quant_x, -n, n - 1)
        quant_x = linear_dequantize_pc(new_quant_x,
                                    scale,
                                    zero_point,
                                    inplace=False)
        m.weight = Parameter(torch.autograd.Variable(quant_x))

      if mixed_flag[count] == 1:
        #print('inside conv pt')
        #print(count)
        original_weight = m.weight
        x_transform = m.weight.data.contiguous().view(m.weight.shape[0], -1)
        w_min = x_transform.min(dim=1).values
        w_max = x_transform.max(dim=1).values
        w_min_pt = torch.min(w_min)
        w_max_pt = torch.max(w_max)

        scale, zero_point = asymmetric_linear_quantization_params(
                k, w_min_pt, w_max_pt)
        new_quant_x = linear_quantize_pt(m.weight, scale, zero_point, inplace=False)
        n = 2**(k - 1)
        new_quant_x = torch.clamp(new_quant_x, -n, n - 1)
        quant_x = linear_dequantize_pt(new_quant_x,
                                    scale,
                                    zero_point,
                                    inplace=False)
        #print(quant_x.shape)
        #print(m.weight.shape)
        m.weight = Parameter(torch.autograd.Variable(quant_x))

      count = count + 1

    if isinstance(m, nn.Linear):
      if mixed_flag[count] == 0:
        #print('inside linear pc')
        #print(count)
        original_weight = m.weight
        x_transform = m.weight.data.detach()
        #print("x_transform")
        #print(x_transform)

        w_min = x_transform.min(dim=1).values
        w_max = x_transform.max(dim=1).values

        scale, zero_point = asymmetric_linear_quantization_params(
                k, w_min, w_max)
        new_quant_x = linear_quantize_pc(m.weight, scale, zero_point, inplace=False)
        n = 2**(k - 1)
        new_quant_x = torch.clamp(new_quant_x, -n, n - 1)
        quant_x = linear_dequantize_pc(new_quant_x,
                                    scale,
                                    zero_point,
                                    inplace=False)
        m.weight = Parameter(torch.autograd.Variable(quant_x))

      if mixed_flag[count] == 1:
        #print('inside linear pt')
        #print(count)
        original_weight = m.weight
        x_transform = m.weight.data.detach()
        #print("x_transform")
        #print(x_transform)

        w_min = x_transform.min(dim=1).values
        w_max = x_transform.max(dim=1).values
        w_min_pt = torch.min(w_min)
        w_max_pt = torch.max(w_max)

        scale, zero_point = asymmetric_linear_quantization_params(
                k, w_min_pt, w_max_pt)
        new_quant_x = linear_quantize_pt(m.weight, scale, zero_point, inplace=False)
        n = 2**(k - 1)
        new_quant_x = torch.clamp(new_quant_x, -n, n - 1)
        quant_x = linear_dequantize_pt(new_quant_x,
                                    scale,
                                    zero_point,
                                    inplace=False)
        #print(quant_x.shape)
        #print(m.weight.shape)
        m.weight = Parameter(torch.autograd.Variable(quant_x.reshape(m.weight.shape)))

      count = count + 1

  return model

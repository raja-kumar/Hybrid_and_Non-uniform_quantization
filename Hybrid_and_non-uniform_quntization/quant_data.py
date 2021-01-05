from torch.utils.data import Dataset, DataLoader

from torchvision import datasets, transforms

import torch





def getQuantData():

  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                         std=[0.229, 0.224, 0.225])

  quant_dataset = datasets.ImageFolder(
      '/home/exx/Tejpratap/research/datageneration/quantization_images_resnet_torchvision_class_bn_loss',
      transforms.Compose([

          transforms.Resize(int(224 / 0.875)),

          transforms.CenterCrop(224),

          transforms.ToTensor(),

          normalize,

      ]))

  quant_loader = DataLoader(quant_dataset,

                            batch_size=32,

                            shuffle=False,

                            num_workers=32)



  return quant_loader

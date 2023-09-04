#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import pandas as pd
from torch.utils import data
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def custom_transform(mode):
#     normalize = transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     )
    
    # Transform
    if mode == 'train':
        transformer = transforms.Compose([
            transforms.RandomRotation(degrees = 20),
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.RandomVerticalFlip(p = 0.5),
            transforms.Resize([350,350]),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
#             normalize
        ])
    else:
        transformer = transforms.Compose([
            transforms.Resize([350,350]),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
#             normalize
        ])

    return transformer

def getData(mode):
    if mode == 'train':
        df = pd.read_csv('train.csv')
        path = df['Path'].tolist()
        label = df['label'].tolist()
        return path, label
    
    elif mode == "valid":
        df = pd.read_csv('valid.csv')
        path = df['Path'].tolist()
        label = df['label'].tolist()
        return path, label
    
    elif mode == "resnet_18":
        df = pd.read_csv('resnet_18_test.csv')
        path = df['Path'].tolist()
        return path
    
    elif mode == "resnet_50":
        df = pd.read_csv('resnet_50_test.csv')
        path = df['Path'].tolist()
        return path
    
    elif mode == "resnet_152":
        df = pd.read_csv('resnet_152_test.csv')
        path = df['Path'].tolist()
        return path

class LeukemiaLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.mode = mode
        if mode == 'train' or mode == 'valid':
            self.img_name, self.label = getData(mode)
        else:
            self.img_name = getData(mode)
            
        self.transform = custom_transform(mode = mode)    
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        path = os.path.join(self.root, f'{self.img_name[index]}')
        img = self.transform(Image.open(path))
        label = []
        if self.mode == 'train' or self.mode == 'valid':
            label = self.label[index]
        
        return img, label


# In[ ]:





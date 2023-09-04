#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import json
import pandas as pd
import torch
from torch.utils import data
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def custom_transform(): 
    # Transform
    transformer = transforms.Compose([
            transforms.Resize([64,64]),
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.ToTensor(),
            transforms.Normalize(std=[0.5, 0.5, 0.5], mean=[0.5, 0.5, 0.5])
        ])
        
    return transformer


def one_hot_encoding(objects):
    object_mapping = get_ObjectFile()
    encoding = [0] * len(object_mapping)  # 創建一個全為0的列表，與對象文件的長度相同
    for obj in objects:
        index = object_mapping.get(obj)  # 獲取對象的索引
        if index is not None:
            encoding[index] = 1  # 將對應索引的位置設為1，表示出現了
    return encoding

def get_ObjectFile():
    with open('\\objects.json', 'r') as json_file:
        object_mapping = json.load(json_file)
    
    return object_mapping

def getData(root, mode):
    if mode == 'train':
        Str = os.path.join(root, 'train.json')
        with open(Str, 'r') as json_file:
            data = json.load(json_file)
            
        # Convert dictionary to a list of tuples (image name, objects list)
        data_tuples = [(key, value) for key, value in data.items()]

        # Convert the list of tuples to a DataFrame
        df = pd.DataFrame(data_tuples, columns=['Image', 'objects'])
        
        path =  df['Image'].tolist()
        label = df['objects'].apply(one_hot_encoding).tolist()
        labels = np.asarray(label)
        labels = torch.from_numpy(labels.astype('long'))
        return path, labels
    
    elif mode == 'valid':
        Str = os.path.join(root, 'test.json')
        with open(Str, 'r') as json_file:
            data = json.load(json_file)
            
        one_hot_encoded_label = [one_hot_encoding(objects) for objects in data]
        labels = np.asarray(one_hot_encoded_label)
        labels = torch.from_numpy(labels.astype('long'))
        return labels
    
    elif mode == 'test':
        Str = os.path.join(root, 'new_test.json')
        with open(Str, 'r') as json_file:
            data = json.load(json_file)
            
        one_hot_encoded_label = [one_hot_encoding(objects) for objects in data]
        labels = np.asarray(one_hot_encoded_label)
        labels = torch.from_numpy(labels.astype('long'))
        return labels

class iclevrLoader(data.Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.mode = mode
        if mode == 'train':
            self.img_name, self.label = getData(root, mode)
            print("> Found %d images..." % (len(self.img_name)))
        else:
            self.label = getData(root,mode)
            
        self.transform = custom_transform()            

    def __len__(self):
        """'return the size of dataset"""
        return len(self.label)

    def __getitem__(self, index):
        if self.mode == 'train':
            Str = os.path.join(self.root, 'iclevr\\')
            path = os.path.join(Str, f'{self.img_name[index]}')
            img = self.transform(Image.open(path).convert('RGB'))
            label = self.label[index]
            return img, label
        else:
            return self.label[index]


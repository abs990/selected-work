from email.mime import base
from matplotlib import image
import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np

import os
import random

from PIL import Image, ImageOps

#import any other libraries you need below this line
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

import cv2

class Cell_data(Dataset):
    def __init__(self, data_dir, size, train='True', train_test_split=0.8, augment_data=True):
        ##########################inputs##################################
        # data_dir(string) - directory of the data#########################
        # size(int) - size of the images you want to use###################
        # train(boolean) - train data or test data#########################
        # train_test_split(float) - the portion of the data for training###
        # augment_data(boolean) - use data augmentation or not#############
        super(Cell_data, self).__init__()
        transformations = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor()
        ])

        def gray_reader(image_path):
            im = Image.open(image_path)
            im2 = ImageOps.grayscale(im)
            im.close()
            return im2

        base_data_points = datasets.ImageFolder(root=data_dir, transform=transformations, loader=gray_reader)
        N_images = int(len(base_data_points)/2)
        self.data = []
        for i in range(N_images):
            image_loc = i + N_images
            image = base_data_points[image_loc][0]
            mean, std = image.mean([1,2]), image.std([1,2])
            transform_norm = transforms.Normalize(mean=mean,std=std)
            image = transform_norm(image)
            mask = base_data_points[i][0]
            point = (image, mask)
            self.data.append(point)
        train_set_size = int(train_test_split * N_images)    
        if train:
            self.set_size = train_set_size
            self.data = self.data[0:self.set_size]
        else:
            self.set_size = N_images - train_set_size
            self.data = self.data[-1*self.set_size:]  

        #set data augnmentation related variables
        self.augment_data = augment_data
        if self.augment_data:
            self.vertical_flip = transforms.RandomVerticalFlip(p=1)
            self.horizontal_flip = transforms.RandomHorizontalFlip(p=1)
            self.zoom_factor = 1.2
            assert self.zoom_factor >= 1
            self.zoom_transform = transforms.Compose([
                transforms.CenterCrop(int(size/self.zoom_factor))
                ,transforms.Resize(size)
            ])
            
    def __getitem__(self, idx):
        # load image and mask from index idx of your data
        (image, mask) = self.data[idx]
        # data augmentation part
        if self.augment_data:
            augment_mode = np.random.randint(0, 5)
            if augment_mode == 0:
                #vertical flip
                image = self.vertical_flip(image)
                mask = self.vertical_flip(mask)
            elif augment_mode == 1:
                #horizontal flip
                image = self.horizontal_flip(image)
                mask = self.horizontal_flip(mask)   
            elif augment_mode == 2:
                #zoom transformation
                image = self.zoom_transform(image)
                mask = self.zoom_transform(mask)      
            elif augment_mode == 3:
                #left 90 degree rotation
                image = torch.rot90(image,1,[1,2])
                mask = torch.rot90(mask,1,[1,2])
            """    
            elif augment_mode == 4:
                image = F.adjust_gamma(img = image, gamma=1.01)
            """       
        return image[0], mask[0]

    def __len__(self):
        return len(self.data)


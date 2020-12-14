#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# authors: Neelabh Pareek (np2647@columbia.edu)
#          Nicholas Christman (nc2677@columbia.edu)
# organization: Columbia University
# status: <course work>
# description: library of helper functions specifically used for one-shot
#              learning. Refer to accompanying jupyter notebook.
# references: 
#    [1] https://gist.github.com/ttchengab/ad136f0af59c6e1362f16a9f557f3166
# Changelog: 
#    - 12-13-2020, christnp: initial

import sys
import time
import os
import copy
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Subset

from scipy.optimize import curve_fit
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split



################################
####
#### Begin one-shot helpers



# training helper for one-shot, siamese network
# source: https://gist.github.com/ttchengab/38309e1474e5837d92256673c6aa3bac#file-train-py
def train(model, train_loader, val_loader, num_epochs, criterion):
    train_losses = []
    val_losses = []
    cur_step = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        print("Starting epoch " + str(epoch+1))
        for img1, img2, labels in train_loader:
            
            # Forward
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.to(device)
            outputs = model(img1, img2)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        val_running_loss = 0.0
        
        #check validation loss after every epoch
        with torch.no_grad():
            model.eval()
            for img1, img2, labels in val_loader:
                img1 = img1.to(device)
                img2 = img2.to(device)
                labels = labels.to(device)
                outputs = model(img1, img2)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
        avg_val_loss = val_running_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print('Epoch [{}/{}],Train Loss: {:.4f}, Valid Loss: {:.8f}'
            .format(epoch+1, num_epochs, avg_train_loss, avg_val_loss))
    print("Finished Training")  
    return train_losses, val_losses  

# evaluation helper for one-shot learning
# srouce: https://gist.github.com/ttchengab/cb7377108368cca87551b51aa11cf053#file-eval-py
def eval(model, test_loader):
    with torch.no_grad():
        model.eval()
        correct = 0
        count = 0
        for mainImg, imgSets, label in test_loader:
            mainImg = mainImg.to(device)
            predVal = 0
            pred = -1
            
            # determine which category an image belongs to
            for i, testImg in enumerate(imgSets):
                testImg = testImg.to(device)
                output = model(mainImg, testImg)
                if output > predVal:
                    pred = i
                    predVal = output
            label = label.to(device)
            if pred == label:
                correct += 1
            count += 1
            if count % 20 == 0:
                print("Current Count is: {}".format(count))
                print('Accuracy on n way: {}'.format(correct/count))

#### End one-shot helpers
################################



################################
####
#### Begin classes

# OmniglotDataset
#    This class works under the directory structure of the Omniglot Dataset
#    It creates the pairs of images for inputs, same character label = 1, vice versa
class OmniglotDataset(Dataset):
    '''
        categories is the list of different alphabets (folders)
        root_dir is the root directory leading to the alphabet files, could be /images_background or /images_evaluation
        setSize is the size of the train set and the validation set combined
        transform is any image transformations
    '''
    def __init__(self, categories, root_dir, setSize, transform=None):
        self.categories = categories
        self.root_dir = root_dir
        self.transform = transform
        self.setSize = setSize
    def __len__(self):
        return self.setSize
    def __getitem__(self, idx):
        img1 = None
        img2 = None
        label = None
        if idx % 2 == 0: # select the same character for both images
            category = random.choice(categories)
            character = random.choice(category[1])
            imgDir = root_dir + category[0] + '/' + character
            img1Name = random.choice(os.listdir(imgDir))
            img2Name = random.choice(os.listdir(imgDir))
            img1 = Image.open(imgDir + '/' + img1Name)
            img2 = Image.open(imgDir + '/' + img2Name)
            label = 1.0
        else: # select a different character for both images
            category1, category2 = random.choice(categories), random.choice(categories)
            category1, category2 = random.choice(categories), random.choice(categories)
            character1, character2 = random.choice(category1[1]), random.choice(category2[1])
            imgDir1, imgDir2 = root_dir + category1[0] + '/' + character1, root_dir + category2[0] + '/' + character2
            img1Name = random.choice(os.listdir(imgDir1))
            img2Name = random.choice(os.listdir(imgDir2))
            while img1Name == img2Name:
                img2Name = random.choice(os.listdir(imgDir2))
            label = 0.0
            img1 = Image.open(imgDir1 + '/' + img1Name)
            img2 = Image.open(imgDir2 + '/' + img2Name)
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, torch.from_numpy(np.array([label], dtype=np.float32))  
    
# NWayShotSet function for creating dataset (included for completeness)
#    This class works under the directory structure of the Omniglot Dataset
#    It creates the pairs of images for inputs, same character label = 1, vice versa
# source: https://gist.github.com/ttchengab/5010a1da5b99166bb6fba9fce47a6cfe#file-nwayoneshotset-py
class NWayOneShotEvalSet(Dataset):
    '''
        categories is the list of different alphabets (folders)
        root_dir is the root directory leading to the alphabet files, could be 
        /images_background or /images_evaluation
        setSize is the size of the train set and the validation set combined
        numWay is the number of images (classes) you want to test for evaluation
        transform is any image transformations
    '''
    def __init__(self, categories, root_dir, setSize, numWay, transform=None):
        self.categories = categories
        self.root_dir = root_dir
        self.setSize = setSize
        self.numWay = numWay
        self.transform = transform
    def __len__(self):
        return self.setSize
    def __getitem__(self, idx):
        # find one main image
        category = random.choice(categories)
        character = random.choice(category[1])
        imgDir = root_dir + category[0] + '/' + character
        imgName = random.choice(os.listdir(imgDir))
        mainImg = Image.open(imgDir + '/' + imgName)
        if self.transform:
            mainImg = self.transform(mainImg)
        
        # find n numbers of distinct images, 1 in the same set as the main
        testSet = []
        label = np.random.randint(self.numWay)
        for i in range(self.numWay):
            testImgDir = imgDir
            testImgName = ''
            if i == label:
                testImgName = random.choice(os.listdir(imgDir))
            else:
                testCategory = random.choice(categories)
                testCharacter = random.choice(testCategory[1])
                testImgDir = root_dir + testCategory[0] + '/' + testCharacter
                while testImgDir == imgDir:
                    testImgDir = root_dir + testCategory[0] + '/' + testCharacter
                testImgName = random.choice(os.listdir(testImgDir))
            testImg = Image.open(testImgDir + '/' + testImgName)
            if self.transform:
                testImg = self.transform(testImg)
            testSet.append(testImg)
        return mainImg, testSet, torch.from_numpy(np.array([label], dtype = int))


# Basic ConvNet Siamese network
# source: https://gist.github.com/ttchengab/ad136f0af59c6e1362f16a9f557f3166
class SiameseNet(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Conv2d(input_channels, output_channels, kernel_size)
        self.conv1 = nn.Conv2d(1, 64, 10) 
        self.conv2 = nn.Conv2d(64, 128, 7)  
        self.conv3 = nn.Conv2d(128, 128, 4)
        self.conv4 = nn.Conv2d(128, 256, 4)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fcOut = nn.Linear(4096, 1)
        self.sigmoid = nn.Sigmoid()
    
    def convs(self, x):
        # out_dim = in_dim - kernel_size + 1  
        #1, 105, 105
        x = F.relu(self.bn1(self.conv1(x)))
        # 64, 96, 96
        x = F.max_pool2d(x, (2,2))
        # 64, 48, 48
        x = F.relu(self.bn2(self.conv2(x)))
        # 128, 42, 42
        x = F.max_pool2d(x, (2,2))
        # 128, 21, 21
        x = F.relu(self.bn3(self.conv3(x)))
        # 128, 18, 18
        x = F.max_pool2d(x, (2,2))
        # 128, 9, 9
        x = F.relu(self.bn4(self.conv4(x)))
        # 256, 6, 6
        return x

    def forward(self, x1, x2):
        x1 = self.convs(x1)
        x1 = x1.view(-1, 256 * 6 * 6)
        x1 = self.sigmoid(self.fc1(x1))
        x2 = self.convs(x2)
        x2 = x2.view(-1, 256 * 6 * 6)
        x2 = self.sigmoid(self.fc1(x2))
        x = torch.abs(x1 - x2)
        x = self.fcOut(x)
        return x

# More Complex VGG16 Siamese network
# source: https://gist.github.com/ttchengab/b4d8f5d2fe43a3bb17201ab5400ab458#file-siamesevgg-py
class VGGSiameseNet(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv11 = nn.Conv2d(1, 64, 3) 
        self.conv12 = nn.Conv2d(64, 64, 3)  
        self.conv21 = nn.Conv2d(64, 128, 3)
        self.conv22 = nn.Conv2d(128, 128, 3)
        self.conv31 = nn.Conv2d(128, 256, 3) 
        self.conv32 = nn.Conv2d(256, 256, 3)  
        self.conv33 = nn.Conv2d(256, 256, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 8 * 8, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fcOut = nn.Linear(4096, 1)
        self.sigmoid = nn.Sigmoid()
    
    def convs(self, x):
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.max_pool2d(x, (2,2))
        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))
        x = F.max_pool2d(x, (2,2))
        x = F.relu(self.conv31(x))
        x = F.relu(self.conv32(x))
        x = F.relu(self.conv33(x))
        x = F.max_pool2d(x, (2,2))
        return x

    def forward(self, x1, x2):
        x1 = self.convs(x1)
        x1 = x1.view(-1, 256 * 8 * 8)
        x1 = self.fc1(x1)
        x1 = self.sigmoid(self.fc2(x1))
        x2 = self.convs(x2)
        x2 = x2.view(-1, 256 * 8 * 8)
        x2 = self.fc1(x2)
        x2 = self.sigmoid(self.fc2(x2))
        x = torch.abs(x1 - x2)
        x = self.fcOut(x)
        return x

####
#### End classes
################################

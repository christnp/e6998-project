#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# authors: Neelabh Pareek (np2647@columbia.edu)
#          Nicholas Christman (nc2677@columbia.edu)
# organization: Columbia University
# status: <course work>
# description:  library of helper functions specifically used for one-shot
#               learning. Refer to accompanying jupyter notebook. A lot of this
#               code came from reference [1] below, but there appears to have 
#               been a lot of mistakes (e.g., not using self.variable in classes
#               was causing undefined varialbe errors, etc.); as such, some of
#               the code has been modified to work in our environment.
# references: 
#    [1] https://gist.github.com/ttchengab/ad136f0af59c6e1362f16a9f557f3166
# Changelog: 
#    - 12-13-2020, christnp: initial

import sys
import errno
import time
import os
import warnings
import random
import requests
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.optim import lr_scheduler,Adam

import PIL

from torch_utils import *
from general_utils import *


warnings.formatwarning = warning_on_one_line


################################
####
#### Begin one-shot helpers

def oneshot_dataset_preview(dataloader,title=''):
    # inputs are list(torch(img1),torch(img2),torch(labels))
    inputs = next(iter(dataloader))
    # we want [(img1,img2,label),(img1,img2,label),...]
    sets = list(zip(*inputs))
     # show 4 sample images, else as many as we have <4
    sel = 4 if len(sets)>=4 else len(sets)
    sets = random.sample(list(zip(*inputs)), sel)
    # set up the figure
    if not title:
        title = f'One-Shot Dataset Samples'
    title = f'{title}  (0=neg, 1=pos)'

    fig = plt.figure(figsize=(12,4))
    fig.suptitle(title, fontsize=16)
    for i,(img1,img2,label) in enumerate(sets):
        axn1 = fig.add_subplot(2, 4, i+1)
        axn1.set_title(f'Label: \'{label.item()}\'')
        axn1.axis('off')
        matplotlib_imshow(img1,one_channel=True) # <-- img1
        axn2 = fig.add_subplot(2, 4, (i+1)+sel)
        axn2.set_title(f'Label: \'{label.item()}\'')
        axn2.axis('off')
        matplotlib_imshow(img2,one_channel=True) # <-- img2
    
    plt.tight_layout(pad=1.0)
    plt.show()

def train_peak(img1s,img2s,labels,outputs):
    
    fig = plt.figure(figsize=(12,6))
    fig.suptitle('Comparing', fontsize=16)
    
    for i in range(len(img1s)):
        if i >= 4: break
        output = outputs[i].cpu().data.numpy()[0]
        axn1 = fig.add_subplot(2, 4, i+1)
        axn1.set_title(f'output: {output:0.4f},\nlabel:{labels[i].item()}', fontsize=10)
        if i == 0: axn1.set_ylabel(f'Image 1')
        else: axn1.set_ylabel('')
        axn1.set_yticklabels([])
        axn1.set_xticklabels([])
        matplotlib_imshow(img1s[i],one_channel=True) # <-- img1

        axn2 = fig.add_subplot(2, 4, i+5)
        if i == 0: axn2.set_ylabel(f'Image 2')
        else: axn2.set_ylabel('')        
        axn2.set_yticklabels([])
        axn2.set_xticklabels([])
        matplotlib_imshow(img2s[i],one_channel=True) # <-- img2
    plt.tight_layout(pad=1.0)
    plt.show()
    
def train_oneshot(device, model, dataloaders, dataset_sizes, 
                criterion=None, optimizer=None, scheduler=None, 
                num_epochs=100, checkpoints=10, output_dir='output', 
                status=10, train_acc=0, track_steps=False,
                seed=414921):
    ''' Helper function to train One-shot model based on parameters '''
    # pylint: disable=no-member 
    # # <-- VC code pylint complains about torch.sum() and .max()

    # TODO: make a for loop similar to other training function
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    # create the model directory if it doesn't exist
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # configure the training if it was not specified by user
    # ref 1: https://towardsdatascience.com/building-a-one-shot- \
    #       learning-network-with-pytorch-d1c3a5fafa4a#bc
    # ref 2: https://becominghuman.ai/siamese-networks-algorithm- \
    #       applications-and-pytorch-implementation-4ffa3304c18
    if not criterion:
        criterion = nn.BCEWithLogitsLoss()
    if not optimizer:
        optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    if not scheduler:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    # send the model to the device
    model = model.to(device)
    
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    metrics = []
    step_metrics = [] # if track_steps=True
    training_step = 0
    acc_reached = False
    for epoch in range(num_epochs):
        training_start_time = time.time()
        if (epoch) % status == 0 or epoch == num_epochs-1:
            print()
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)
        
        model.train()
        # for phase in ['train', 'val']:
        #     if phase == 'train':
        #         model.train()
        #     else:
        #         model.eval()

        epoch_phase_start_time = time.time()
        train_running_loss = 0.0
        train_running_corrects = 0.0
        # for oneshot, we have two input images and a label
        # where label:0=neg and label:1=pos 
        for img1, img2, labels in train_loader:
            step_start_time = time.time()
            # Forward
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.to(device)
            outputs = model(img1, img2)            
            # check if model predicted the match (1=correct)
            pred = labels[torch.argmax(outputs)]
            loss = criterion(outputs, labels)
            
            # for visualizing every so many epochs
#             if (epoch) % status == 0:
#                 train_peak(img1,img2,labels,outputs)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()
            train_running_corrects += pred.item() #torch.sum(preds == labels.data)

            if track_steps:
                # store per step metrics (WARNING! lots of data)
                step_metrics.append({
                    'device': str(device),
                    'epoch': epoch,
                    'training_step': training_step,
                    'training_step_loss': loss.item(),
                    'training_step_time': time.time() - step_start_time
                })
            training_step += 1

        # TODO: implement scheduling
        # if phase == 'train':
        #     scheduler.step()
        avg_train_loss = train_running_loss / len(train_loader)
        avg_train_acc = train_running_corrects / len(train_loader)
        
        # train_losses.append(avg_train_loss)
        val_running_loss = 0.0
        val_running_corrects = 0.0

        training_end_time = time.time()
        
        #check validation loss after every epoch
        with torch.no_grad():
            model.eval()
            for img1, img2, labels in val_loader:
                img1 = img1.to(device)
                img2 = img2.to(device)
                labels = labels.to(device)
                outputs = model(img1, img2)
                # check if model predicted the match (1=correct)
                pred = labels[torch.argmax(outputs)]
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item()
                val_running_corrects += pred.item() #torch.sum(preds == labels.data)
        avg_val_loss = val_running_loss / len(val_loader)
        avg_val_acc = val_running_corrects / len(val_loader)

        if avg_val_acc > best_acc:
            best_acc = avg_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        # val_losses.append(avg_val_loss)

        # print('Epoch [{}/{}],Train Loss: {:.4f}, Valid Loss: {:.8f}'
        #     .format(epoch+1, num_epochs, avg_train_loss, avg_val_loss))
        if (epoch) % status == 0 or epoch == num_epochs-1 or acc_reached:
            # print(f'{phase} Loss: {round(epoch_loss, 4)} Acc: {round(epoch_acc.item(), 4)}')
            print(f'Train Loss: {round(avg_train_loss, 4)} Train Acc: {round(avg_train_acc, 4)}')
            print(f'Val Loss: {round(avg_val_loss, 4)} Val Acc: {round(avg_val_acc, 4)}')

        else:
            prog = '-' * int(((epoch) % status))
            print('\r{}|{}'.format(prog,epoch),end='')
        # store per epoch metrics
        metrics.append({
                        'device': str(device),
                        'epoch': epoch,
                        'average_training_loss': avg_train_loss, 
                        'average_validation_loss': avg_val_loss,
                        'training_acc': avg_train_acc,
                        'validaton_acc': avg_val_acc,
                        'training_time': training_end_time - training_start_time,
                        'validation_time': time.time() - training_end_time
                    })
    print("Finished Training")  

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60}m {time_elapsed % 60}s')
    print(f'Best val Acc: {round(best_acc, 4)}')
    # load best model weights
    model.load_state_dict(best_model_wts)
    # set up return structures
    metrics_df = pd.DataFrame(data=metrics)
    step_metrics_df = pd.DataFrame(data=step_metrics) if step_metrics else None

    return model, metrics_df, step_metrics_df  

# evaluation helper for one-shot learning
# srouce: https://gist.github.com/ttchengab/cb7377108368cca87551b51aa11cf053#file-eval-py
def eval(device, model, test_loader):
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
class OmniglotDataset(torch.utils.data.Dataset):
# pylint: disable=no-member
    '''
        categories is the list of different alphabets (folders) root is the root 
        directory leading to the alphabet files, could be /images_background 
        or /images_evaluationm, and set_size is the size of the train set and the 
        validation set combined and transform is any image transformations
        
        Reference: https://ramsane.gitbook.io/deep-learning/few-shot-learning/omniglot-dataset
    '''
    def __init__(self, categories, root, set_size, train=True, download=False, transform=None):
        self.categories = categories
        self.root = os.path.join(root,'Omniglot')
        self.train = train
        self.transform = transform
        self.set_size = set_size
        self.download = download
        # target data location
        self.training_file = 'images_background'
        self.test_file = 'images_evaluation'
        # source data location
        self.training_url = 'https://raw.github.com/brendenlake/omniglot/master/python/images_background.zip'
        self.test_url = 'https://raw.github.com/brendenlake/omniglot/master/python/images_evaluation.zip'
        
        if self.download:
            self._download()
            
        if not self._check_exists():
            warnings.warn(f'Dataset not found. You can use download=True to download it',RuntimeWarning)
            return
        if self.train:
            self.classes = os.listdir(os.path.join(self.root,self.training_file))
        else:
            self.classes = os.listdir(os.path.join(self.root,self.test_file))
            
        
    def __len__(self):
        return self.set_size
    def __getitem__(self, idx):
        img1 = None
        img2 = None
        label = None
        if idx % 2 == 0: # select the same character for both images
            category = random.choice(self.categories)
            if self.train:
                catDir = os.path.join(self.root,self.training_file,category)
            else:
                catDir = os.path.join(self.root,self.test_file,category)
            character = random.choice(os.listdir(catDir))
            imgDir = os.path.join(catDir,character) #self.root + category[0] + '/' + character
            img1Name = random.choice(os.listdir(imgDir))
            img2Name = random.choice(os.listdir(imgDir))
            img1 = PIL.Image.open(os.path.join(imgDir, img1Name))
            img2 = PIL.Image.open(os.path.join(imgDir, img2Name))
            label = 1.0
        else: # select a different character for both images
            category1 = random.choice(self.categories)
            category2 = random.choice(self.categories)
            if self.train:
                catDir1 = os.path.join(self.root,self.training_file,category1)
                catDir2 = os.path.join(self.root,self.training_file,category2)
            else:
                catDir1 = os.path.join(self.root,self.test_file,category1)
                catDir2 = os.path.join(self.root,self.test_file,category2)
                
            character1 = random.choice(os.listdir(catDir1))
            character2 = random.choice(os.listdir(catDir2))
            
            imgDir1 = os.path.join(catDir1,character1)
            imgDir2 = os.path.join(catDir2,character2)
            img1Name = random.choice(os.listdir(imgDir1))
            img2Name = random.choice(os.listdir(imgDir2))
            while img1Name == img2Name:
                img2Name = random.choice(os.listdir(imgDir2))
            label = 0.0
            img1 = PIL.Image.open(os.path.join(imgDir1, img1Name))
            img2 = PIL.Image.open(os.path.join(imgDir2, img2Name))
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        # vv VC code pylint compplains about from_numpy, it's fine
        return img1, img2, torch.from_numpy(np.array([label], dtype=np.float32))  
    
    def _download(self):
        """
        Download the Omniglot data if it doesn't exist already.
        Source: https://gist.github.com/branislav1991/f1a16e3d87389091d85699dbb1ba857d#file-siamese-py
        Update: christnp,2020-12-14: changed download and unzip mechanisms
        """
        import requests, zipfile, io

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.training_file))
            os.makedirs(os.path.join(self.root, self.test_file))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        
        subdirs = [self.training_file,self.test_file]
        urls = [self.training_url, self.test_url]
        for url,subdir in zip(urls,subdirs):
            print('Downloading ' + url)
            data = requests.get(url)
            if data.status_code != 200:
                warnings.warn(f'Bad URL or website is down, dataset not ' \
                                    'downloaded! src: {url}',RuntimeWarning)
            else:
                z = zipfile.ZipFile(io.BytesIO(data.content))
                z.extractall(self.root)                
        
        print('Done!')


    def _check_exists(self):
        training_files = os.path.join(self.root,self.training_file)
        test_files = os.path.join(self.root, self.test_file)
        return os.path.exists(training_files) and os.path.exists(test_files) \
            and os.listdir(training_files) and os.listdir(test_files)
             
        
    
# NWayShotSet function for creating dataset (included for completeness)
#    This class works under the directory structure of the Omniglot Dataset
#    It creates the pairs of images for inputs, same character label = 1, vice versa
# source: https://gist.github.com/ttchengab/5010a1da5b99166bb6fba9fce47a6cfe#file-nwayoneshotset-py
class NWayOneShotEvalSet(torch.utils.data.Dataset):
# pylint: disable=no-member
    '''
        categories is the list of different alphabets (folders)
        root_dir is the root directory leading to the alphabet files, could be 
        /images_background or /images_evaluation
        set_size is the size of the train set and the validation set combined
        numWay is the number of images (classes) you want to test for evaluation
        transform is any image transformations
    '''
    def __init__(self, categories, root_dir, set_size, numWay, transform=None):
        self.categories = categories
        self.root_dir = root_dir
        self.set_size = set_size
        self.numWay = numWay
        self.transform = transform
    def __len__(self):
        return self.set_size
    def __getitem__(self, idx):
        # find one main image
        category = random.choice(self.categories)
        character = random.choice(category[1])
        imgDir = self.root_dir + category[0] + '/' + character
        imgName = random.choice(os.listdir(imgDir))
        mainImg = PIL.Image.open(imgDir + '/' + imgName)
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
                testCategory = random.choice(self.categories)
                testCharacter = random.choice(testCategory[1])
                testImgDir = self.root_dir + testCategory[0] + '/' + testCharacter
                while testImgDir == imgDir:
                    testImgDir = self.root_dir + testCategory[0] + '/' + testCharacter
                testImgName = random.choice(os.listdir(testImgDir))
            testImg = PIL.Image.open(testImgDir + '/' + testImgName)
            if self.transform:
                testImg = self.transform(testImg)
            testSet.append(testImg)
        # vv VC code pylint compplains about from_numpy, it's fine
        return mainImg, testSet, torch.from_numpy(np.array([label], dtype = int))


# Simple ConvNet Siamese nsetwork
# source: https://gist.github.com/branislav1991/f1a16e3d87389091d85699dbb1ba857d#file-siamese-py
class SimpleSiameseNet(nn.Module):
# pylint: disable=no-member
    def __init__(self):
        super(SimpleSiameseNet,self).__init__()

        self.conv1 = nn.Conv2d(1, 64, 7)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.conv3 = nn.Conv2d(128, 256, 5)
        self.linear1 = nn.Linear(2304, 512)

        self.linear2 = nn.Linear(512, 2)
      
    def forward(self, data):
        res = []
        for i in range(2): # Siamese nets; sharing weights
            x = data[i]
            x = self.conv1(x)
            x = F.relu(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = self.conv3(x)
            x = F.relu(x)

            x = x.view(x.shape[0], -1)
            x = self.linear1(x)
            res.append(F.relu(x))

        res = torch.abs(res[1] - res[0]) # <-- VC code pylint compplains, it's fine
        res = self.linear2(res)
        return res

# Basic ConvNet Siamese network
# source: https://gist.github.com/ttchengab/ad136f0af59c6e1362f16a9f557f3166
class SiameseNet(nn.Module):
# pylint: disable=no-member
    def __init__(self):
        super(SiameseNet, self).__init__()
        
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
        x = torch.abs(x1 - x2) # <-- VC code pylint compplains, it's fine
        x = self.fcOut(x)
        return x

# More Complex VGG16 Siamese network
# source: https://gist.github.com/ttchengab/b4d8f5d2fe43a3bb17201ab5400ab458#file-siamesevgg-py
class VGGSiameseNet(nn.Module):
# pylint: disable=no-member
    def __init__(self):
        super(VGGSiameseNet, self).__init__()
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
        x = torch.abs(x1 - x2) # <-- VC code pylint compplains, it's fine
        x = self.fcOut(x)
        return x

####
#### End classes
################################


#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# authors: Neelabh Pareek (np2647@columbia.edu)
#          Nicholas Christman (nc2677@columbia.edu)
# organization: Columbia University
# status: <course work>
# description: library of helper functions used for PyTorch applications.
#              Refer to accompanying Jupyter notebook for details.
# Changelog: 
#    - 12-12-2020, christnp: initial
#    - 12-14-2020, christnp: implemented a custom MNIST Subloader
#      that replaced torchvision.datasets.MNIST in the mnist_dataloader()
#      function. Updated some helper functions to work with limited data.

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

from general_utils import *

warnings.formatwarning = warning_on_one_line

################################
####
#### Begin general PyTorch helpers
def mnist_dataset(data_transforms, batch_size, 
                        pred_size=0.05, include_labels=None,
                        exclude_labels=None, sample_size=0,
                        data_dir='../../data', download=True):
    ''' 
    This function returns dataloaders and dataset info for limited 
    MNIST datasets that use the Subloader class in this library of
    functions. This function adds labels to include or exclude
    as well as optional limited dataset size (sample_size).
    '''
    # create the dir for data
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    # standard MNIST training to testing split (used when sample_size>0)
    val_ratio = 10000/60000

    print(f'Data will be located in \'{data_dir}\'')

    # get the training data
    train_set = SubLoader(root=data_dir, train=True, 
                           download=download, 
                           transform=data_transforms['train'],
                           include_labels=include_labels,
                           exclude_labels=exclude_labels,
                           sample_size=sample_size) 

    # adjust the sample size based on ratio or class size
    sample_size = sample_size*val_ratio
    if(sample_size > 0 and sample_size < len(train_set.classes)):
        sample_size = len(train_set.classes)

    # get the validation data
    val_set = SubLoader(root=data_dir, train=False, 
                           download=download, 
                           transform=data_transforms['val'],
                           include_labels=include_labels,
                           exclude_labels=exclude_labels,
                           sample_size=sample_size) 
    # << trying to keep the test to training size equivalent

    # split off a small chunk for predicting
    val_idx, pred_idx = train_test_split(list(range(len(val_set))), 
                                          test_size=pred_size)
    val_set = Subset(val_set, val_idx)
    pred_set = Subset(val_set, pred_idx)

    # collect the dataset, size, and class names
    image_datasets = {'train':train_set, 'val':val_set, 'pred':pred_set}
    dataset_sizes = {x: len(image_datasets[x]) for x in image_datasets}
    class_names = image_datasets['train'].classes

    return image_datasets, dataset_sizes, class_names

def create_dataloader(image_datasets, batch_size):
    # make the dataloader
    dataloaders = {}
    for x in image_datasets:
        dataloaders[x] = torch.utils.data.DataLoader(image_datasets[x], 
                                                      batch_size=batch_size,
                                                      shuffle=True, num_workers=4)
    return dataloaders


####
#### End PyTocrch Tutorial Helpers
################################


################################
#### Begin PyTocrch Tutorial Helpers
# The following helper functions come from a PyTorch tutorial 
# ref: https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
#### 
def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    # pylint: disable=no-member 
    # # <-- VC code pylint complains about torch.max()
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().data.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels, classes):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # christnp,2020-12-14: updated to be dynamic with image size
    in_size = len(images)
    plot_size = 4 if in_size >=4 else in_size

    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 12*plot_size))
    for idx in np.arange(plot_size):
        ax = fig.add_subplot(1, plot_size, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig

# helper function to show an image
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().data.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
####
#### End PyTocrch Tutorial Helpers
################################


################################
####
#### Begin classes
class SubLoader(torchvision.datasets.MNIST):
    '''
    This MNIST subloader class extends the torchvision MNIST
    dataset class and was created to accomplish several 
    preprocessing manipulation actions. First, it allows for
    exlusion (i.e. removal) of specific classes by label; this 
    is done by specifying labels in the 'exclude_labels'. Second,
    it allows for the inverse; by specifying labels in the 
    'include_labels' one can get a smaller dataset more easily. 
    Finally, one can subsample the dataset with or without
    exclusions/inclusions. If no options are selected then this
    subloader is identical to the torchvision dataset MNIST class.
    Common usuage is as follows:
    # creates training data with only 100 samples of label 1
    train_data = SubLoader(root='./data', train=True, download=True, 
                           transform=transform, include_labels=[1],
                           sample_size=100) 
    # creates test data for all data with labels 1 and 5                       
    val_data = SubLoader(root='./data', train=False, download=True, 
                          transform=transform, exclude_labels=[1,5]) 
    '''
    def __init__(self, *args, include_labels=None, exclude_labels=None, 
                 sample_size=0, **kwargs):
        
        super(SubLoader, self).__init__(*args, **kwargs)
        
        # future use
        #if self.train:
            # can add training specifics
        #else:
             # can add training specifics
        if exclude_labels == None and include_labels == None and sample_size == 0:
            return
        
        if exclude_labels and include_labels:
            warnings.warn("Cannot both include and exclude classes." \
                           "Include labels has priority.")

        labels = np.array(self.targets)
        classes = self.classes
        if include_labels:
            include = np.array(include_labels).reshape(1, -1)
            mask = (labels.reshape(-1, 1) == include).any(axis=1)
            classes = [classes[i] for i in include_labels]
            # update the data, classes, and targets/labels
            self.data = self.data[mask]
            self.classes = classes #[self.classes[i] for i in class_list]
            self.targets = labels[mask].tolist()
        elif exclude_labels:
            exclude = np.array(exclude_labels).reshape(1, -1)
            mask = ~(labels.reshape(-1, 1) == exclude).any(axis=1)
            classes = [classes[i] for i,_ in enumerate(classes) 
                       if i not in exclude_labels]
            # update the data, classes, and targets/labels
            self.data = self.data[mask]
            self.classes = classes #[self.classes[i] for i in class_list]
            self.targets = labels[mask].tolist()

        if sample_size > 0:
            self.data,self.targets = self.limited_(self.data, self.targets, sample_size)
            
    def limited_(self,data,labels,size):
        '''
        This function takes a larger dataset (data), 
        a target size (size), and original labels as 
        variables to create a subsampled (limite) 
        dataset of length size. Sratify is used to 
        evenly split the classes.
        '''
        labels = np.array(labels)
        all_samples = len(data) # <-- 100%
        ratio = size/all_samples
        # use the sklearn function to effectively split the data
        _, lim_idx = train_test_split(list(range(all_samples)), 
                                      test_size=ratio,
                                      stratify = labels)
        return [data[lim_idx], list(labels[lim_idx])]
    
####
#### End classes
################################
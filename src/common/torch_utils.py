#
#
#
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



# def mnist_dataloader(data_transforms, batch_size, 
#                      pred_size=0.05, data_dir='../../data', 
#                      download=True):
#     ''' 
#     This function returns dataloaders and dataset info for MNIST
#     '''
#     # create the dir for data
#     if not os.path.isdir(data_dir):
#         os.mkdir(data_dir)
        
#     print(f'Data will be located in \'{data_dir}\'')

#     # get the training data
#     train_set = torchvision.datasets.MNIST(root=data_dir, 
#                                         train=True,
#                                         download=download, 
#                                         transform=data_transforms['train'])
    
#     # get the validation data
#     val_set = torchvision.datasets.MNIST(root=data_dir, 
#                                         train=False,
#                                         download=download, 
#                                         transform=data_transforms['val'])
    
#     # split off a small chunk for predicting
#     val_idx, pred_idx = train_test_split(list(range(len(val_set))), 
#                                           test_size=pred_size)
#     val_set = Subset(val_set, val_idx)
#     pred_set = Subset(val_set, pred_idx)
    
#     # make the dataloader
#     dataloaders = {}
#     image_datasets = {'train':train_set, 'val':val_set, 'pred':pred_set}
#     for x in image_datasets:
#         dataloaders[x] = torch.utils.data.DataLoader(image_datasets[x], 
#                                                       batch_size=batch_size,
#                                                       shuffle=True, num_workers=4)
#     # collect the dataset size and class names
#     dataset_sizes = {x: len(image_datasets[x]) for x in image_datasets}
#     class_names = image_datasets['train'].classes
    
#     return dataloaders, dataset_sizes, class_names

def mnist_dataloader(data_transforms, batch_size, 
                             pred_size=0.05, include_labels=[],
                             exclude_labels=[], sample_size=0,
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
    
    # make the dataloader
    dataloaders = {}
    image_datasets = {'train':train_set, 'val':val_set, 'pred':pred_set}
    for x in image_datasets:
        dataloaders[x] = torch.utils.data.DataLoader(image_datasets[x], 
                                                      batch_size=batch_size,
                                                      shuffle=True, num_workers=4)
    # collect the dataset size and class names
    dataset_sizes = {x: len(image_datasets[x]) for x in image_datasets}
    class_names = image_datasets['train'].classes
    
    return dataloaders, dataset_sizes, class_names

# replaced with a Subloader that extends torchvision MNIST dataset
# def mnist_limited_dataset(dataset,transform,tgt_label,class_names):
#     ''' 
#     this function splits all data associated with tgt_label from
#     the dataset into a new dataset, and removes it from the original
#     dataset. This function requires a all inputs.
    
#     Typical usage:
#         # define original dataset
#         transform = transforms.Compose([...])
#         original = torchvision.datasets.MNIST(...)
#         class_names = original.classes
#         # loop through the labels you want to remove
#         tgt_labels = [1,2,3]
#         datasets = []
#         for label in labels:
#             limited,original = mnist_limited_dataset(original,transform,label,class_names)
#             datasets.append(limited)
#         # create the final limited dataset
#         limited_dataset = torch.utils.data.ConcatDataset(datasets)
#         # orignal now has only the labels that are not in limited_dataset!
#     '''
#     # we have to create a new MNIST dataset in memory
#     limited = torchvision.datasets.MNIST(root='../../data', 
#                                     train=True,
#                                     download=False, 
#                                     transform=transform)
#     # select the indices to be split off in the new limited subset
#     rem_idxs = np.equal(dataset.targets,tgt_label).nonzero()
#     mask = np.zeros(len(dataset.data), dtype=bool)
#     mask[rem_idxs] = True
#     limited.data = dataset.data[mask]
#     limited.targets = dataset.targets[mask]
#     limited.classes = class_names[tgt_label]
        
#     # select the indices will be kept with the original dataset 
#     mask = np.logical_not(mask) # keep the ones that are not going
#     dataset.data = dataset.data[mask]
#     dataset.targets = dataset.targets[mask]
#     dataset.classes = [i for i in class_names if i != class_names[tgt_label]]
    
#     # return the new limited subset and original dataset
#     return limited,dataset


def dataset_preview(dataloader,title=''):
    '''
    This function provides a fairly naive approach to showing
    some of the data passed into the dataloader.
    '''
    # Get a batch of training data
    inputs, classes = next(iter(dataloader))
    # select 4 unique classes to demo (if they exist)
    uni, ind = np.unique(classes, return_index=True)
    sel = 4 if len(ind)>=4 else len(ind)
    use = np.random.choice(ind, sel, replace=False)
    # display samles
    # title = f'Samples of \'{}\' Dataset'
    if not title:
        title = f'Samples of Dataset'

    fig = plt.figure(figsize=(10,4))
    fig.suptitle(title, fontsize=16)
    for i,u in enumerate(use):
        class_name = classes[u].item()
        axn = fig.add_subplot(1, 4, i+1)
        axn.set_title(f'Class: \'{class_name}\'')
        axn.axis('off')
        matplotlib_imshow(inputs[u],one_channel=True) # <-- PyTorch Tutorial Helper
    plt.tight_layout(pad=1.0)
    plt.show()

    
def train_model(device, model, dataloaders, dataset_sizes, 
                criterion=None, optimizer=None, scheduler=None, 
                num_epochs=100, checkpoints=10, output_dir='output', 
                status=1, train_acc=0, track_steps=False,
                seed=414921):
    ''' Helper function to train PyTorch model based on parameters '''
    # create the model directory if it doesn't exist
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # configure the training if it was not specified by user
    if not criterion:
        criterion = nn.CrossEntropyLoss()
    if not optimizer:
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
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
        epoch_start_time = time.time()
        if (epoch) % status == 0 or epoch == num_epochs-1:
            print()
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            epoch_phase_start_time = time.time()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                step_start_time = time.time()
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
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
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_phase_end_time = time.time()
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc.item()
                best_model_wts = copy.deepcopy(model.state_dict())
            
            # check if training accuracy has met target, if so signal exit
            if (train_acc > 0) and (epoch_acc.item() >= train_acc) and phase == 'train':
                acc_reached = True
                print()
                print(f'Epoch {epoch}/{num_epochs - 1}')
                print('-' * 10)
                
            if (epoch) % status == 0 or epoch == num_epochs-1 or acc_reached:
                print(f'{phase} Loss: {round(epoch_loss, 4)} Acc: {round(epoch_acc.item(), 4)}')
            else:
                prog = '-' * int(((epoch) % status))
                print('\r{}|{}'.format(prog,epoch),end='')
                
            # store per epoch metrics
            metrics.append({
                            'device': str(device),
                            'epoch': epoch,
                            'training_epoch_loss': loss.item(),
                            'training_epoch_acc': epoch_acc.item(),
                            'training_epoch_time': time.time() - epoch_start_time
                        })

        ####### save checkpoint after epoch
        if (epoch > 0 and epoch != num_epochs-1) and \
            ((epoch+1) % checkpoints == 0 and os.path.isdir(output_dir)):
            checkpoint=os.path.join(output_dir,
                                f'epoch{epoch+1}_checkpoint_model.th')
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
            }, checkpoint)
            # dump the data for later
            json_file = os.path.join(output_dir,
                                    f'epoch{epoch+1}_checkpoint_metrics.json')
            with open(json_file, 'w') as fp:
                json.dump(metrics, fp)
        #######
        
        # if the target accuracy was reached during this epoch, it is time to exit
        if acc_reached: 
            break
    
    ####### save final checkpoint
    if os.path.isdir(output_dir):
        timestamp = time.strftime("%Y-%m-%dT%H%M%S")
        checkpoint= os.path.join(output_dir, f'final_model_{timestamp}.th')
        # save the model
        torch.save({
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
        }, checkpoint)
        # dump the data for later
        metric_path = os.path.join(output_dir,f'final_metrics_{timestamp}.json')
        with open(metric_path, 'w') as fp:
            json.dump(metrics, fp)
    #######
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60}m {time_elapsed % 60}s')
    print(f'Best val Acc: {round(best_acc, 4)}')
    # load best model weights
    model.load_state_dict(best_model_wts)
    # set up return structures
    metrics_df = pd.DataFrame(data=metrics)
    step_metrics_df = pd.DataFrame(data=step_metrics) if step_metrics else None
        
    return model, metrics_df, step_metrics_df


def get_device(verbose=True):
    # use a GPU if there is one available
    cuda_availability = torch.cuda.is_available()
    if cuda_availability:
        device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
    else:
        device = 'cpu'
    if verbose:
        device_name = torch.cuda.get_device_name()
        print('\n***********************************')
        print(f'GPU Available:  {cuda_availability}')
        print(f'Current Device: {device} ({device_name})')
        print('***********************************\n')
    
    return device


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
#### Begin ZSL specific helpers

def zsl_data_formatter(original,labels=None):
    ''' 
    For zero-shot learning we need to separate a couple of classes
    from the dataset. This function will output two datasets: 
        (1) N class labels with applicable images from dataset of size S
        (2) S - N class labels with applicable images (whatever is left)
    Note: N must be less than (or equal to) S
    
    input: PyTorch dataloader
    output: 
    '''
    original_size = list(range(len(original)))
    if not labels:
        labels = original.targets
        labels = labels[0] # get the first one?
    
    big_idx, lim_idx = train_test_split(original_size, 
                                       test_size=0.2, 
                                       stratify=labels)

    big_set = Subset(original, big_idx)
    lim_set = Subset(original, lim_idx)
    
    return big_set,lim_set

####
#### End ZSL specific helpers
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
    def __init__(self, *args, include_labels=[], exclude_labels=[], 
                 sample_size=0, **kwargs):
        
        super(SubLoader, self).__init__(*args, **kwargs)
        
        # future use
        #if self.train:
            # can add training specifics
        #else:
             # can add training specifics

        if exclude_labels == [] and include_labels == [] and sample_size == 0:
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
        else:
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

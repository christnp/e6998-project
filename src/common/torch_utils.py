
import sys
import time
import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pprint import pprint

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms

from scipy.optimize import curve_fit
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


def mnist_dataloader(data_transform, batch_size, 
                     data_dir='../../data', download=True,
                     train=True):
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    # get the data
    dataset = torchvision.datasets.MNIST(root=data_dir, 
                                        train=train,
                                        download=download, 
                                        transform=data_transform)
    print(f'Data is located in \'{data_dir}\'')

    # make the dataloader
    dataloader = torch.utils.data.DataLoader(dataset=dataset, 
                                             batch_size=batch_size,
                                             shuffle=True, 
                                             num_workers=4)
    return {'dataset':dataset, 'dataloader':dataloader}

def dataset_preview(dataloader,title=''):
    # Get a batch of training data
    inputs, classes = next(iter(dataloader))
    # select 4 unique classes to demo
    uni, ind = np.unique(classes, return_index=True)
    use = np.random.choice(ind, 4, replace=False)
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
        axn.imshow(inputs[u].permute(1, 2, 0))     
    plt.tight_layout(pad=1.0)
    plt.show()

def train_model(device, model, dataloaders, criterion=None, 
                optimizer=None, scheduler=None, num_epochs=100, 
                checkpoints=10, output_dir='output', 
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
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

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
                                'device': device,
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
                            'device': device,
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
        checkpoint= os.path.join(output_dir, 'final_model.th')
        # save the model
        torch.save({
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
        }, checkpoint)
        # dump the data for later
        metric_path = os.path.join(output_dir,'final_metrics.json')
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


    def get_device(verbose=False):
        # use a GPU if there is one available
        cuda_availability = torch.cuda.is_available()
        if cuda_availability:
            device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
        else:
            device = 'cpu'
        device_name = torch.cuda.get_device_name()
        print('\n***********************************')
        print(f'GPU Available:  {cuda_availability}')
        print(f'Current Device: {device} ({device_name})')
        print('***********************************\n')


import sys
import time
import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pprint import pprint

# use a GPU if there is one available
cuda_availability = torch.cuda.is_available()
if cuda_availability:
    device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
else:
    device = 'cpu'
print('\n*************************')
print('GPU Available: {}'.format(cuda_availability))
print('Current Device: {}'.format(device))
print('*************************\n')
# display the GPU info
if cuda_availability:
    !nvidia-smi

import sys
import time
import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pprint import pprint




   
# helper function to get GPU name
# tested on Tesla V100, P100, K80
def get_gpu_name(gpu_class='Tesla',verbose=0):
    import subprocess
    import re
    nvidia_smi = subprocess.check_output(['nvidia-smi']).decode('utf8')
    match = re.search(r'{} (\S+)'.format(gpu_class), nvidia_smi)
    gpu = match.group(1).split('-')[0]
    if verbose > 0:
        print('GPU: {}'.format(gpu))
    if verbose == 2:
        print(nvidia_smi)
    return str(gpu).upper()
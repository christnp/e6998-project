
import sys
import time
import os
import subprocess
import re
import numpy as np

# https://stackoverflow.com/questions/26430861/make-pythons-warnings-warn-not-mention-itself
def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return '%s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)
   
# helper function to get GPU name
# NOTE: this has been replaced by the torch.cuda.get_device_name()
# tested on Tesla V100, P100, K80
def get_gpu_name(gpu_class='Tesla',verbose=0):
    nvidia_smi = subprocess.check_output(['nvidia-smi']).decode('utf8')
    match = re.search(r'{} (\S+)'.format(gpu_class), nvidia_smi)
    gpu = match.group(1).split('-')[0]
    if verbose > 0:
        print('GPU: {}'.format(gpu))
    if verbose == 2:
        print(nvidia_smi)
    return str(gpu).upper()
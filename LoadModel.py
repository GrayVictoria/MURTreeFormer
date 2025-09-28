import logging,os,sys,random,copy
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time
from Parameters import *
from torch.autograd import Variable
import models

from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from tqdm import tqdm
from torch.cuda import amp


def main():
    model_name_all = 'models.' + model_name
    model = eval(model_name_all)(num_classes=num_classes)
    if gpu >= 0:
        model.cuda(gpu)
    
    if continue_checkpoint_model_file_name != '':
        print('| loading checkpoint file %s... ' %
              continue_checkpoint_model_file_name, end='')
        model.load_state_dict(torch.load(
            continue_checkpoint_model_file_name, map_location='cuda:0'))
        print('done!')

    

if __name__ == "__main__":
    main()
import os
from datetime import datetime
import math

dataset_name = 'MNIST' # MNIST, FashionMNIST, SVHN

if dataset_name == 'MNIST' or dataset_name == 'FashionMNIST':
    num_output_classes = 10
    num_input_channels = 1
    mean = 0.1307
    std = 0.3081
    set_size = 50000

elif dataset_name == 'SVHN':
    num_output_classes = 10
    num_input_channels = 3
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    set_size = 100000

#directory to save weights file
CHECKPOINT_PATH = 'checkpoint'

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
#time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

# device
device = 'cuda'

# runs
EPOCH = 510
SAVE_EPOCH = 10
shuffle = True

# early stop at EARLY_STOP_EPOCH or beyond if train acc falls below EARLY_STOP_ACC for the past 3 checkpoints.
EARLY_STOP_EPOCH = 51
EARLY_STOP_ACC = 0.15

# hyperparameters
MILESTONES = [60, 100, 300]
batch_size = 32
test_batch_size = 32
lr = 0.01
momentum = 0
warm = 1
tolerance = 0.001
weight_decay = 8e-4
rnd_aug = False
loss = 'MSE' # 'CE', 'MSE'

# parameters for computing theoretical bound
bound_batch_size = 256
bound_num_batches = 100
rank_bound_version = "BC0"

# architecture
net = 'convnet_wn_3'

# mlp, convnet, convnet_custom parameters 203
fc_width = 600
width = 600
alpha = 1
activation = 'relu'
bias = False
bn = False
out = 3

# mlp_custom, convnet_custom parameters
custom_layer = 5
custom_width = 100

# convnet_custom parameters
kernel_dim = 3
stride_length = 1
padding = 1

# resnet parameters
resnet_version = 18

# saving params
directory = './results/'
resume = False
normalize_dist = True
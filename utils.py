import math
import os
import sys
import re
import datetime
import operator
import numpy as np
import math
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchvision import datasets
import torchvision.transforms as transforms
from torch.nn.utils import spectral_norm

class NormalizeByNorm(object):
    def __init__(self):
        pass

    def __call__(self, image):
        norm = torch.norm(image)
        return image / norm
        
def get_network(settings):
    """ return given network
    """

    # Initializes activation function
    if settings.activation == 'relu':
        settings.activation = nn.ReLU()
    # Initializes the networks
    if settings.net == 'convnet_wn':
        from models.convnet_wn import convnet_wn
        net = convnet_wn(settings)
    elif settings.net == 'convnet':
        from models.convnet import convnet
        net = convnet(settings)
    elif settings.net == 'convnet_wn_4':
        from models.convnet_wn_4 import convnet_wn_4
        net = convnet_wn_4(settings)
    elif settings.net == 'convnet_wn_3':
        from models.convnet_wn_3 import convnet_wn_3
        net = convnet_wn_3(settings)
    elif settings.net == 'convnet_wn_5':
        from models.convnet_wn_5 import convnet_wn_5
        net = convnet_wn_5(settings)
    elif settings.net == 'convnet_wn_6':
        from models.convnet_wn_6 import convnet_wn_6
        net = convnet_wn_6(settings)
    elif settings.net == 'convnet_wn_7':
        from models.convnet_wn_7 import convnet_wn_7
        net = convnet_wn_7(settings)
    elif settings.net == 'convnet_wn_8':
        from models.convnet_wn_8 import convnet_wn_8
        net = convnet_wn_8(settings)
    elif settings.net == 'convnet_wn_deep':
        from models.convnet_wn_deep import convnet_wn_deep
        net = convnet_wn_deep(settings)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    net = net.to(settings.device)

    return net


def get_training_dataloader(dataset_name, mean, std, settings, batch_size=16, num_workers=2,
                            shuffle=True, rnd_aug=True, num_classes=10,
                            bound_num_batches=None):
    """ return training dataloader
    """
    dataset = operator.attrgetter(dataset_name)(torchvision.datasets)
    
    if dataset_name == 'MNIST' or dataset_name == 'FashionMNIST':
        im_size = 28
        #padded_im_size = 32
        
        #transform_train = transforms.Compose([transforms.Pad((padded_im_size - im_size) // 2),
        #                                      transforms.ToTensor(), 
        #                                      transforms.Normalize(mean, std)]
        #                                     )
        #training_set = dataset(root='./data', train=True, download=True, transform=transform_train)        
        transform_train = transforms.Compose([transforms.ToTensor(), 
                                          transforms.Normalize(mean, std)])
        training_set = dataset(root='./data', train=True, download=True, transform=transform_train)


    elif dataset_name == 'SVHN':
        from torch.utils.data import Subset

        # Define the data transformation
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        # Load the train and extra sets
        train_set = datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)
        extra_set = datasets.SVHN(root='./data', split='extra', download=True, transform=transform_train)
        extra_set = Subset(extra_set, range(settings.set_size))

        # Concatenate the train and extra sets
        training_set = torch.utils.data.ConcatDataset([train_set, extra_set])

    if bound_num_batches is not None:  # sampling for the bound
        sampler = RandomSampler(training_set,
                                replacement=True,
                                num_samples=bound_num_batches)
        training_loader = DataLoader(training_set,
                                     sampler=sampler,
                                     num_workers=num_workers,
                                     batch_size=batch_size)
    else:
        training_loader = DataLoader(training_set,
                                     shuffle=shuffle,
                                     num_workers=num_workers,
                                     batch_size=batch_size)

    return training_loader


def get_test_dataloader(dataset_name, mean, std,settings, batch_size=16, num_workers=2,
                        shuffle=True, num_classes=10):
    """ return test dataloader
    """
    dataset = operator.attrgetter(dataset_name)(torchvision.datasets)
    if dataset_name == 'MNIST' or dataset_name == 'FashionMNIST':
        im_size = 28
        #padded_im_size = 32
        #transform_test = transforms.Compose([transforms.Pad((padded_im_size - im_size) // 2),
        #                                     transforms.ToTensor(),
        #                                     transforms.Normalize(mean, std)]
        #                                    )
        # Define a function to compute the norm of an image
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean, std)])
        test_set = dataset(root='./data', train=False, download=True, transform=transform_test)

    elif dataset_name == 'SVHN':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_set = dataset(root='./data', split='test', download=True, transform=transform_test)

    test_loader = DataLoader(test_set, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return test_loader


def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data
    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]


def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]


def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
        raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch


def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]


def save_data(dir_name, graphs):
    ## save the results

    attrbts = [attr for attr in dir(graphs) if not \
        callable(getattr(graphs, attr)) and not attr.startswith("__")]

    for name in attrbts:

        _ = open(dir_name + '/' + name + ".txt", "w+")
        _.write(str(operator.attrgetter(name)(graphs)))
        _.close()


def get_dir_name(directory, prespecified=False, resume=False):
    if not prespecified:
        if not os.path.isdir(directory):
            os.mkdir(directory)

        # results directories
        sub_dirs_ids = [int(dir) for dir in os.listdir(directory)
                        if os.path.isdir(directory + '/' + dir)]

        # experiment id
        xid = max(sub_dirs_ids)
        dir_name = directory + '/' + str(xid)
    else:
        dir_name = directory

    # sweeps the INNER directories
    sub_dirs_ids = [int(dir) for dir in os.listdir(dir_name)
                    if os.path.isdir(dir_name + '/' + dir)]

    # current sweep
    if len(sub_dirs_ids) == 0:
        pid = 0
    else:
        pid = max(sub_dirs_ids)
        if not resume:
            pid += 1
    dir_name += '/' + str(pid)
    if not resume: os.mkdir(dir_name)

    return dir_name

def max_pixel_sums(train_loader, settings):
    dataset = operator.attrgetter(settings.dataset_name)(torchvision.datasets)
    # Load the images one by one using torchvision.datasets.ImageFolder
    dataset = dataset(root='./data', train=True, download=True)
    # Convert the first image to a Tensor and get its size
    image, _ = dataset[0]
    image_tensor = torchvision.transforms.functional.to_tensor(image)
    image_size = image_tensor.size()

    # Initialize a tensor to hold the pixel sums
    pixel_sums = torch.zeros(image_size)

    # Loop over all images in the dataset and add their pixels to the sums
    for i in range(len(dataset)):
        image, _ = dataset[i]
        image_tensor = torchvision.transforms.functional.to_tensor(image)
        pixel_sums += torch.pow(image_tensor, 2)

    # Return the tensor of pixel sums
    return pixel_sums.max().item()

def eval_rho(net):
    rho = 1
    for layer in net.all_layers:
        rho = rho * torch.norm(layer.weight).item()

    return rho

def our_total_bound(net, training_loader, settings):
    rho = eval_rho(net)
    #print("or", rho)
    n = settings.set_size
    k = settings.num_output_classes
    depth = net.depth
    delta = 0.001
    degs = net.degs
    max_deg = max(degs)**2
    deg_prod = np.prod(degs)
    mult1 = (rho + 1) / n
    mult2 = 2 ** 1.5 * (1 + math.sqrt(2 * (depth * np.log(2 * max_deg) + np.log(k))))
    max_sum_sqrt = math.sqrt(max_pixel_sums(training_loader, settings))
    mult3 = max_sum_sqrt * deg_prod
    add1 = 3 * math.sqrt(np.log((2 * (rho + 2) ** 2) / delta) / (2 * n))
    #print("om", mult3)
    bound = mult1 * mult2 * mult3 + add1

    return bound

def norm_2_1(K):
    norm_2 = torch.norm(K, p=2, dim=1, keepdim=True)
    norm_2_1 = torch.norm(norm_2, p=1)
    return norm_2_1

def lip_constant(layer):
    if isinstance(layer, nn.Conv2d):
        weight = layer.weight.data
        reshaped_weight = weight.view(weight.shape[0], -1)
    elif isinstance(layer, nn.Linear):
        reshaped_weight = layer.weight.data
    else:
        raise ValueError('layer type not recognized')

    if reshaped_weight.shape[1] > 1:
        _, s, _ = torch.svd(reshaped_weight)
        return s[0]
    else:
        return torch.abs(reshaped_weight[0])


def sum_norm(training_loader):
    total_norm = 0
    for i, (data, targets) in enumerate(training_loader):
        # Compute the squared norm of the batch samples
        norm = torch.norm(data, p=2, dim=[1, 2, 3]) ** 2
        # Add the sum of squared norms in the batch to total_norm
        total_norm += norm.sum().item()

    return math.sqrt(total_norm)


def max_norm(training_loader):
    max_norm = 0
    for i, (data, targets) in enumerate(training_loader):
        # Compute the squared norm of each sample in the batch
        norm = torch.norm(data.view(data.size(0), -1), p=2, dim=1) ** 2
        # Take the maximum value
        max_norm = max(norm.max().item(), max_norm)

    return math.sqrt(max_norm)


# Golowich bound

def eval_rho_golowich(net, settings): # TODO: Fix this bound
    rho = 1
    input_shape = (1, 28, 28)
    test_tensor = torch.zeros(1, *input_shape, device=settings.device)

    shapes = net(test_tensor)[1]
    
    for i, layer in enumerate(net.all_layers):
        rho = rho * torch.norm(layer.weight).item()
        if isinstance(layer, nn.Conv2d):
            rho = rho * math.sqrt(shapes[i + 1][2] * shapes[i + 1][3])

    #print("gr",rho)
    return rho

def golowich_total_bound(net, settings, training_loader):
    delta = 0.001
    n = settings.set_size
    k = settings.num_output_classes
    rho = eval_rho_golowich(net, settings)
    mult1 = (rho + 1) / n
    mult2 = 2 ** 1.5 * (1 + math.sqrt(2 * (net.depth * np.log(2) + np.log(k))))
    mult3 = sum_norm(training_loader)
    #print("gm",mult3)
    norm_ratio = mult3/math.sqrt(max_pixel_sums(training_loader, settings))
    add1 = 3 * math.sqrt(np.log((2 * (rho + 2) ** 2) / delta) / (2 * n))
    bound = mult1 * mult2 * mult3 + add1
    return bound, norm_ratio

# Ledent bound

def ledent_total_bound(net, settings, training_loader):
    input_shape = (1, 28, 28)
    test_tensor = torch.zeros(1, *input_shape, device=settings.device)
    n = settings.set_size
    L = len(net.all_layers)
    delta = 0.001
    bound = 8 / n + 3 * math.sqrt(np.log(2 / delta) / n) + math.sqrt(np.log(2) / n)
    R = 0
    Gamma = 0
    W = 0

    max_n = max_norm(training_loader)
    log_list = []

    shapes = net(test_tensor)[1]
    for shape in shapes:
        W = max(W, np.prod(shape))

    # Calculate Gamma
    for i, layer in enumerate(net.all_layers):
        if isinstance(layer, nn.Conv2d):
            O = shapes[i][1] * shapes[i][2]
            m = shapes[i][0]
        else:
            O = 1
            m = shapes[i][0]
        norm_2_1_v = norm_2_1(layer.weight)
        norm_prod = O * m * (norm_2_1_v + 1 / L)
        log_list += [norm_2_1_v / L + 2]
        for j, layer in enumerate(net.all_layers):
            if j != i:
                filter_v = layer.weight
                reshaped_filter = filter_v.view(filter_v.shape[0], -1)
                # reshaped_filter = filter.view(layer.weight.shapes[0], -1)
                spectral_norm = torch.linalg.svdvals(reshaped_filter).max()
                norm_prod *= spectral_norm + 1 / L

            Gamma = max(Gamma, norm_prod)
    Gamma = Gamma * (max_n + 1)

    # Calculate R
    for i, layer in enumerate(net.all_layers):
        norm_prod = 1
        for j, layer in enumerate(net.all_layers):
            if j != i:
                filter_v = layer.weight
                reshaped_filter = filter_v.view(filter_v.shape[0], -1)
                spectral_norm = torch.linalg.svdvals(reshaped_filter).max()
                norm_prod *= spectral_norm + 1 / L

        if isinstance(layer, nn.Conv2d):
            R += ((norm_2_1(layer.weight) + 1 / L) * norm_prod * layer.kernel_size) ** (2 / 3)
        else:
            R += ((norm_2_1(layer.weight) + 1 / L) * norm_prod * 1) ** (2 / 3)
    R = R ** (3 / 2) * (max_n + 1) * L

    R = R.item()
    Gamma = Gamma.item()
    bound += (1536 / math.sqrt(n)) * R * np.log2(32 * Gamma * n ** 2 + 7 * W * n) ** 0.5 * np.log(n)
    bound += 3 * math.sqrt(1 / n * np.log(1 + max_n) + sum(log_list))

    return bound


# Sedghi bound

def sedghi_total_bound(net, settings, normalize=True):
    n = settings.set_size
    delta = 0.001
    W = sum(p.numel() for p in net.parameters() if p.requires_grad)
    sum_norms = 0
    L = 0
    # claculate the norm
    if normalize: # if we normalize the weight matrices
        prod = 1
        for i, layer in enumerate(net.all_layers):
            prod *= lip_constant(layer)
            L += 1
        sum_norms = L*prod**(1/L)

    else:
        for i, layer in enumerate(net.all_layers):
            #matrix = layer.weight
            #norm = torch.norm(matrix, p=1, dim=0).sum()
            norm = lip_constant(layer)
            sum_norms += norm
    #print(W)
    #print(sum_norms)
    #if sum_norms >= 5:
    #    bound = np.sqrt((W * sum_norms.item() + np.log(1 / delta)) / n)
    #else:
    #    bound = sum_norms.item() * np.sqrt(W / n) + np.sqrt(np.log(1 / delta) / n)
    bound = sum_norms.item() * np.sqrt(W / n) + np.sqrt(np.log(1 / delta) / n)
    return bound


# Graf bound

def graf_total_bound(net, training_loader, settings):
    n = settings.set_size
    delta = 0.001
    bound = 3 * math.sqrt(np.log(2 / delta) / (2 * n))
    max_n = max_norm(training_loader)
    lip_consts = []
    lip_product = 1
    W = 0

    for layer in net.all_layers:
        lip_const = lip_constant(layer)
        lip_consts += [lip_const]
        lip_product *= lip_const
        W = max(W, sum(p.numel() for p in layer.parameters() if p.requires_grad))

    sum_C = 0
    for layer in net.all_layers:
        sum_C += (4 * max_n * lip_product * norm_2_1(layer.weight) / lip_constant(layer)) ** (2 / 3)

    sum_C = sum_C ** (3 / 2)
    bound += 8 / n + 24 * np.log(n) * np.log(2 * W) * sum_C / math.sqrt(n)
    prod_ratio = lip_product/eval_rho(net)
    return bound, prod_ratio.item()
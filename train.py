import time
from shutil import copyfile
import os

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import importlib
from tqdm import tqdm

import utils
from utils import get_network, get_training_dataloader, \
    get_test_dataloader, save_data, our_total_bound, golowich_total_bound, ledent_total_bound, sedghi_total_bound, graf_total_bound
import numpy as np
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Graphs:
    def __init__(self):
        self.train_accuracy = []
        self.test_accuracy = []

        self.train_loss = []
        self.test_loss = []

        self.our_bound = []
        self.golowich_bound = []
        self.ledent_bound = []
        self.sedghi_bound = []
        self.graf_bound = []
        self.prod_ratio = []
        self.norm_ratio = []

    def add_data(self, train_acc, test_acc, train_loss, test_loss, our_bound, golowich_bound, ledent_bound, sedghi_bound, graf_bound, prod_ratio, norm_ratio):

        if train_acc != None: self.train_accuracy += [train_acc]
        if test_acc != None: self.test_accuracy += [test_acc]
        if train_loss != None: self.train_loss += [train_loss]
        if test_loss != None: self.test_loss += [test_loss]
        if our_bound != None: self.our_bound += [our_bound]
        if golowich_bound != None: self.golowich_bound += [golowich_bound]
        if ledent_bound != None: self.ledent_bound += [ledent_bound]
        if sedghi_bound != None: self.sedghi_bound += [sedghi_bound]
        if graf_bound != None: self.graf_bound += [graf_bound]
        if prod_ratio != None: self.prod_ratio += [prod_ratio]
        if norm_ratio != None: self.norm_ratio += [norm_ratio]


def train_epoch(epoch, net, optimizer, train_loader, settings):
    start = time.time()
    if settings.loss == 'CE':
        loss_function = nn.CrossEntropyLoss()
    elif settings.loss == 'MSE':
        loss_function = nn.MSELoss()
    elif settings.loss == 'l1':
        loss_function = nn.L1Loss()
    net.train()
    for batch_index, (images, labels) in enumerate(tqdm(train_loader, leave=False, desc="  ")):

        if settings.device == 'cuda':
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)[0]
        if settings.loss == 'CE':
            loss = loss_function(outputs, labels)
        elif settings.loss == 'MSE':
            loss = loss_function(outputs, F.one_hot(labels, settings.num_output_classes).float())
        elif settings.loss == 'l1':
            loss = loss_function(outputs, F.one_hot(labels, settings.num_output_classes).float())

        loss.backward()
        optimizer.step()

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s\tLoss: {:0.4f}'.format(epoch, finish - start, loss.item()))


@torch.no_grad()
def eval_training(epoch, net, test_loader, settings):
    start = time.time()
    if settings.loss == 'CE':
        loss_function = nn.CrossEntropyLoss(reduction='sum')
    elif settings.loss == 'MSE':
        loss_function = nn.MSELoss(reduction='sum')
    elif settings.loss == 'l1':
        loss_function = nn.L1Loss(reduction='sum')
    net.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0

    for (images, labels) in test_loader:

        if settings.device == 'cuda':
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)[0]
        if settings.loss == 'CE':
            loss = loss_function(outputs, labels)
        elif settings.loss == 'MSE':
            loss = loss_function(outputs, F.one_hot(labels, settings.num_output_classes).float())
        elif settings.loss == 'l1':
            loss = loss_function(outputs, F.one_hot(labels, settings.num_output_classes).float())

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()

    dataset_size = len(test_loader.dataset)
    acc = correct / dataset_size
    test_loss = test_loss / dataset_size

    finish = time.time()
    if settings.device == 'cuda':
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss,
        acc,
        finish - start
    ))
    print()

    return acc, test_loss


@torch.no_grad()
def save_checkpoint(path, net, graphs, optimizer, epoch):
    path += '/model.pt'
    torch.save({'model_state_dict': net.state_dict(),
                'graphs': graphs,
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch}, path)


def main(results_path, resume):
    ## get directory name
    if results_path is None:
        directory = './results'
        dir_name = utils.get_dir_name(directory, prespecified=False)

        copyfile('./conf/global_settings.py', dir_name + '/global_settings.py')
    else:
        dir_name = utils.get_dir_name(results_path, prespecified=True, resume=resume)

        copyfile(os.path.join(results_path, 'global_settings.py'), dir_name + '/global_settings.py')

    spec = importlib.util.spec_from_file_location("module", dir_name + '/global_settings.py')
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)

    net = get_network(settings)
    graphs = Graphs()

    # data preprocessing:
    train_loader = get_training_dataloader(
            settings.dataset_name,
            settings.mean,
            settings.std,
            settings,
            num_workers=2,
            batch_size=settings.batch_size,
            shuffle=settings.shuffle,
            rnd_aug=settings.rnd_aug,
            num_classes=settings.num_output_classes
    )

    test_loader = get_test_dataloader(
            settings.dataset_name,
            settings.mean,
            settings.std,
            settings,
            num_workers=2,
            batch_size=settings.test_batch_size,
            shuffle=True,
            num_classes=settings.num_output_classes
    )


    optimizer = optim.SGD(net.parameters(), lr=settings.lr, momentum=settings.momentum,
                          weight_decay=settings.weight_decay)

    # weight_decay = list()
    # no_weight_decay = list()
    # if settings.net == 'conv_wn':
    #     for layer in net.layers:
    #         if isinstance(layer, nn.Conv2d):
    #             weight_decay.append(layer.weight_g)
    #             no_weight_decay.append(layer.weight_v)
    # weight_decay.append(net.fc.weight)
    #
    # optimizer = optim.SGD([{'params': weight_decay},
    #                        {'params': no_weight_decay, 'weight_decay': 0}],
    #                       lr=settings.lr,
    #                       momentum=settings.momentum,
    #                       weight_decay=settings.weight_decay)

    # only apply weight decay on last layer
    """

    """

    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES,
                                                     gamma=0.2)  # learning rate decay

    dist = 0
    start_epoch = 1
    skip_save = False

    # start from checkpoint
    if resume:
        checkpoint = torch.load(dir_name + '/model.pt')
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        graphs = checkpoint['graphs']
        start_epoch = checkpoint['epoch']
        skip_save = True  # prevents the results from getting double-saved when resuming from checkpoint

    for epoch in tqdm(range(start_epoch, settings.EPOCH + 1)):
        train_scheduler.step(epoch)

        if (settings.SAVE_EPOCH == 1 or epoch % settings.SAVE_EPOCH == 1) and not skip_save:

            our_bound = our_total_bound(net, train_loader, settings)
            golowich_bound, norm_ratio = golowich_total_bound(net, settings, train_loader)
            ledent_bound = ledent_total_bound(net, settings, train_loader)
            sedghi_bound = sedghi_total_bound(net, settings)
            graf_bound, prod_ratio = graf_total_bound(net, train_loader, settings)
            train_acc, train_loss = eval_training(epoch, net, train_loader, settings)
            test_acc, test_loss = eval_training(epoch, net, test_loader, settings)
            graf_bound = graf_bound.item()
            
            graphs.add_data(train_acc, test_acc, train_loss, test_loss, our_bound, golowich_bound, ledent_bound, sedghi_bound, graf_bound, prod_ratio, norm_ratio)

            save_data(dir_name, graphs)
            save_checkpoint(dir_name, net, graphs, optimizer, epoch)

        # evaluate dist and convergance
        if ('normalize_dist' in vars(settings)):
            normalize_dist = settings.normalize_dist
        else:
            normalize_dist = False
        #list_weights_old = analysis_convergence.get_weights(net)
        train_epoch(epoch, net, optimizer, train_loader, settings)
        #list_weights_new = analysis_convergence.get_weights(net)
        #dist = analysis_convergence.eval_dist(list_weights_old, list_weights_new, normalize_dist)

        skip_save = False

        # early stop
        if ('EARLY_STOP_EPOCH' in vars(settings)):
            if epoch >= settings.EARLY_STOP_EPOCH and graphs.train_accuracy[-1] <= settings.EARLY_STOP_ACC and \
                    graphs.train_accuracy[-2] <= settings.EARLY_STOP_ACC and graphs.train_accuracy[
                -3] <= settings.EARLY_STOP_ACC:
                print("=== Initiating Early Stop ===")
                _ = open(dir_name + '/EARLY_STOP', 'w')
                _.close()
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-path', action='store', default=None,
                        help='Specifies path of directory where results should be stored (including weights) and '
                             'where the global_settings.py file is located. Default behavior is to use '
                             'conf/global_settings.py and create a new directory in results.')
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction, default=False)
    _args = parser.parse_args()

    if _args.resume and _args.results_path is None:
        raise Exception("Cannot resume training without specifying results path.")

    main(_args.results_path, _args.resume)

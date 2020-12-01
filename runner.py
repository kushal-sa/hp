import numpy as np
import argparse
from time import time
import psutil

import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.models as models

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from model import HyperBandScheduler
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.suggest.bayesopt import BayesOptSearch

import config as cf
import util

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TRAIN_SET = datasets.CIFAR100(cf.data_paths['train'],train=True,download=True,transform=transforms.Compose([transforms.ToTensor()]))

VAL_SET = datasets.CIFAR100(cf.data_paths['val'],train=False,download=True,transform=transforms.Compose([transforms.ToTensor()]))

def categorical_to_uniform(cat: tune.sample.Categorical):
    return tune.uniform(min(cat), max(cat))

def get_data_loaders(batch_size):
    train_loader = DataLoader(TRAIN_SET, batch_size=batch_size, shuffle=True, num_workers=cf.NUM_CPU_PER_TRIAL-1)
    val_loader = DataLoader(VAL_SET, batch_size=batch_size, shuffle=True, num_workers=cf.NUM_CPU_PER_TRIAL-1)
    
    return train_loader, val_loader

def fill_config(config):
    return {**cf.param_defaults, **config}

def train(model, optimizer, criterion, train_loader):
    
    model.train()
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        predicted = torch.argmax(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    return correct / total

def test(model, data_loader):
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = model(data)
            predicted = torch.argmax(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return correct / total

# trainable object
def train_cifar_100(config):

    # Data Setup
    train_loader, val_loader = get_data_loaders(round(config['batch_size']))
    
    config = fill_config(config)

    # Model Setup
    model = models.resnet18()
    model.fc = nn.Linear(512,100,bias=True)
    model = model.to(DEVICE)

    # Optimizer
    optimizer = optim.SGD(
        model.parameters(), 
        lr=config["lr"], 
        momentum=config["momentum"],
        weight_decay=config["weight_decay"]
    )
    
    # LR Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer,round(config['step']),gamma=cf.SCHEDULER_GAMMA)

    # Loss Criterion
    criterion = nn.CrossEntropyLoss()
    
    while True:
        train_acc = train(model, optimizer, criterion, train_loader)
        val_acc = test(model, val_loader)
        scheduler.step()

        # Send the current training result back to Tune
        print('[log] time: ', time() - START_TIME)
        print('[log] ram: ', psutil.virtual_memory().used / (1024 ** 3) - START_RAM)
        print('[log] val_acc: ', val_acc)
        print('[log] train_acc: ', train_acc)
        tune.report(mean_accuracy=val_acc,train_acc=train_acc)

def get_tuner(exp,alg,param_n):
    
    param_space = cf.param_space
    if exp == 'EXP1':
        ### Experiment 1 ###
        max_t = 256
        reduction_factor = 4
        time_attr = 'time_total_s'
    else:
        ### Experiment 2 and 3 ###
        max_t = 27
        reduction_factor = 3
        time_attr = 'training_iteration'
    
    if exp == 'EXP3':
        ### Experiment 3 ###
        param_space = { k:v for k, v in cf.param_space.items() if k in cf.param_priority[:param_n]}

    num_samples = int(util.calculate_total_iters_hyperband(reduction_factor,max_t)[0] / max_t)
    
    search_alg = None
    scheduler = None

    stop = {time_attr: max_t}
    
    if alg == 'BayesOpt' or alg == 'Hybrid' :
        param_space['step'] = categorical_to_uniform(param_space['step'])
        param_space['batch_size'] = categorical_to_uniform(param_space['batch_size'])
        search_alg = BayesOptSearch(metric = 'mean_accuracy', mode='max')
    
    if alg == 'HyperBand' or alg == 'Hybrid':
        scheduler = HyperBandScheduler(
                time_attr = time_attr,
                reduction_factor = reduction_factor,
                max_t = max_t)
    
        num_samples = int(util.calculate_total_iters_hyperband(reduction_factor,max_t)[1])
    
    return param_space, num_samples, stop, scheduler, search_alg

def parse_arguments():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('exp', type = str, choices=['EXP1', 'EXP2', 'EXP3'])
    parser.add_argument('alg', type =str, choices=['Random', 'HyperBand', 'Hybrid', 'BayesOpt'])
    
    parser.add_argument('-n','--params_n',type=int,default=2)

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_arguments()
    name = args.exp + '_' + args.alg

    param_space, num_samples, stop, scheduler, search_alg = get_tuner(args.exp,args.alg,args.params_n)
    
    global START_TIME
    START_TIME = time()
    global START_RAM
    START_RAM = psutil.virtual_memory().used / (1024 ** 3)

    analysis = tune.run(train_cifar_100,
                      name = name,
                      metric = 'mean_accuracy',
                      mode = 'max',
                      config = param_space,
                      resources_per_trial = cf.resources_per_trial,
                      num_samples = num_samples,
                      local_dir = cf.result_paths['raw_logs'],
                      stop = stop,
                      scheduler = scheduler,
                      search_alg = search_alg)

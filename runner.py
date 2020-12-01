import numpy as np
import argparse

import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.models as models

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.schedulers import HyperBandScheduler
from ray.tune.schedulers import PopulationBasedTraining

from config import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TRAIN_SET = datasets.CIFAR100(data_paths['train'],train=True,download=True,transform=transforms.Compose([transforms.ToTensor()]))

VAL_SET = datasets.CIFAR100(data_paths['val'],train=False,download=True,transform=transforms.Compose([transforms.ToTensor()]))

def get_data_loaders(batch_size):
    train_loader = DataLoader(TRAIN_SET, batch_size=batch_size, shuffle=True, num_workers=NUM_CPU_PER_TRIAL-1)
    val_loader = DataLoader(VAL_SET, batch_size=batch_size, shuffle=True, num_workers=NUM_CPU_PER_TRIAL-1)
    
    return train_loader, val_loader

def fill_config(config):
    return {**param_default,**config}

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
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            predicted = torch.argmax(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return correct / total

# trainable object
def train_cifar_100(config):
    
    # Data Setup
    train_loader, val_loader = get_data_loaders(config['batch_size'])
    
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
    scheduler = optim.lr_scheduler.StepLR(optimizer,config['step'],gamma=SCHEDULER_GAMMA)

    # Loss Criterion
    criterion = nn.CrossEntropyLoss()
    
    while True:
        train_acc = train(model, optimizer, criterion, train_loader)
        val_acc = test(model, val_loader)
        scheduler.step()

        # Send the current training result back to Tune
        tune.report(mean_accuracy=val_acc,train_acc=train_acc)

def get_tuner(exp,algo,param_n):
    
    param_space = orig_param_space
    if exp == 'EXP1':
        ### Experiment 1 ###
        max_t = 300
        reduction_factor = 4
        time_attr = 'time_total_s'
    else:
        ### Experiment 2 and 3 ###
        max_t = 15
        reduction_factor = 3
        time_attr = 'training_iteration'
    
    if exp == 'EXP3':
        ### Experiment 3 ###
        param_space = { k:v for k, v in orig_param_space.items() if k in param_priority[:param_n]}

    num_samples = calculate_total_iters_hyperband(reduction_factor,max_t)[0]/max_t
    
    search_algo = None
    scheduler = None

    stop = {time_attr: max_t}
    
    if algo == 'BayOpt' or algo == 'Hybrid' :
        search_algo = BayesOptSearch(param_space, metric = 'mean_accuracy', mode='max')
    
    if algo == 'HyperBand' or algo == 'Hybrid':
        scheduler = HyperBandScheduler(
                time_attr = time_attr,
                metric = 'mean_accuracy',
                mode = 'max',
                reduction_factor = reduction_factor,
                max_t = max_t)
    
    return param_space, num_samples, stop, scheduler, search_algo

def parse_arguments():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('exp', type = str )
    parser.add_argument('algo', type =str )
    
    parser.add_argument('-n','--params_n',type=int,default=2)

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parseArguments()
    name = args.exp + '_' + args.algo

    param_space, num_samples, stop, scheduler, search_algo = get_tuner(args.exp,args.algo,args.params_n)
    
    analysis = tune.run(train_cifar_100,
                      name = name,
                      metric = 'mean_accuracy',
                      mode = 'max',
                      config = param_space,
                      resource_per_trial = resource_per_trial,
                      num_samples = num_samples,
                      local_dir = result_paths['logs'],
                      stop = stop,
                      scheduler = scheduler,
                      search_algo = search_algo)

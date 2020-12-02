from ray import tune
import os

base_dir = './results/'
result_paths = {
    'raw_logs':'logs/',
    'raw_data':'data/',
    'results':'results/',
}

result_paths = { k:base_dir+v for k,v in result_paths.items() }

data_paths = {
    'train': './data/train/',
    'val': './data/val/'
}

NUM_CPU_PER_TRIAL = os.cpu_count()
NUM_GPU_PER_TRIAL = 1 
resources_per_trial = {"gpu": NUM_GPU_PER_TRIAL, "cpu": NUM_CPU_PER_TRIAL}

MAX_TRAINING_EPOCH_PER_TRIAL = 15

SCHEDULER_GAMMA = 0.3

param_priority = ['lr','step','momentum','weight_decay','batch_size']

param_space = {
    'lr': tune.loguniform(1e-5,1e-1),
    'momentum': tune.uniform(0.5,0.99),
    'step': tune.choice([1,2,3]),
    'weight_decay': tune.loguniform(1e-8,1e-5),
    'batch_size': tune.choice([2**k for k in range(7,10)])
}

param_defaults = {
    'lr': 1e-2,
    'momentum': 0.9,
    'step': 2,
    'weight_decay': 1e-7,
    'batch_size': 128
}

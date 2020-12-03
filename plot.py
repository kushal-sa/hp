import matplotlib.pyplot as plt 
import numpy as np
import os 
import sys 
import argparse
import seaborn as sns

sns.set_theme()

name2color = {
    'Hyperband': 'b',
    'BayesOpt': 'r',
    'Random': 'g'
}

def plot_val_time(file_paths):
    for file_path in file_paths:
        setting = os.path.basename(file_path).split('_')[1]
        time_l = []
        val_acc_l = []
        curr_max_val_acc = float('-inf')
        with open(file_path, 'r') as f:
            for line in f:
                if '[log] time: ' in line:
                    time_l.append(float(line.split('[log] time: ')[-1]))
                elif '[log] val_acc: ' in line:
                    try:
                        val_acc = float(line.split('[log] val_acc: ')[-1]\
                            .replace('[', '(')\
                            .replace('\x1b', '(')\
                            .split('(')[0])
                    except:
                        print(line.split('[log] val_acc: ')[-1].replace('[', '(').split('(')[0])
                        print(line)
                        print(file_path)
                    curr_max_val_acc = max(val_acc, curr_max_val_acc)
                    val_acc_l.append(curr_max_val_acc)
        plt.plot(time_l[:101], val_acc_l[:101], label=setting)
    plt.title('Validation Accuracy vs. Wall Time (# of hyperparam: {})'.format(num_hyperparam))
    plt.xlabel('Time (s)')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.savefig('val_time.png', dpi=500)

def plot_val_itr(file_paths):
    for file_path in file_paths:
        setting = os.path.basename(file_path).split('_')[1]
        val_acc_l = []
        curr_max_val_acc = float('-inf')
        with open(file_path, 'r') as f:
            for line in f:
                if '[log] val_acc: ' in line:
                    try:
                        val_acc = float(line.split('[log] val_acc: ')[-1]\
                            .replace('[', '(')\
                            .replace('\x1b', '(')\
                            .split('(')[0])
                    except:
                        print(line.split('[log] val_acc: ')[-1].replace('[', '(').split('(')[0])
                        print(line)
                        print(file_path)
                    curr_max_val_acc = max(val_acc, curr_max_val_acc)
                    val_acc_l.append(curr_max_val_acc)
        plt.plot(val_acc_l[:101], label=setting)
    plt.title('Validation Accuracy vs. Training Epochs (# of hyperparam: {})'.format(num_hyperparam))
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.savefig('val_itr.png', dpi=500)

def plot_ram_time(file_paths):
    for file_path in file_paths:
        setting = os.path.basename(file_path).split('_')[1]
        time_l = []
        ram_l = []
        curr_max_ram = float('-inf')
        with open(file_path, 'r') as f:
            for line in f:
                if '[log] time: ' in line:
                    time_l.append(float(line.split('[log] time: ')[-1]))
                elif '[log] ram: ' in line:
                    try:
                        ram = float(line.split('[log] ram: ')[-1]\
                            .replace('[', '(')\
                            .replace('\x1b', '(')\
                            .split('(')[0])
                    except:
                        print(line.split('[log] ram: ')[-1].replace('[', '(').split('(')[0])
                        print(line)
                        print(file_path)
                    ram_l.append(ram)
        plt.plot(time_l, ram_l, label=setting)
    plt.title('RAM Usage vs. Wall Time (# of hyperparam: {})'.format(num_hyperparam))
    plt.xlabel('Time (s)')
    plt.ylabel('RAM Usage (GB)')
    plt.legend()
    plt.savefig('ram_time.png', dpi=500) 

def pp_val_time():
    hyperband_paths = [
        '/Users/zipengfu/Downloads/final2/manual_logs/EXP2_HyperBand_5',
        '/Users/zipengfu/Downloads/final3/manual_logs/EXP2_HyperBand_5',
        '/Users/zipengfu/Downloads/final4/manual_logs/EXP2_HyperBand_5'
    ]
    bayesopt_paths = [
        '/Users/zipengfu/Downloads/final2/manual_logs/EXP2_BayesOpt_5',
        '/Users/zipengfu/Downloads/final3/manual_logs/EXP2_BayesOpt_5',
        '/Users/zipengfu/Downloads/final4/manual_logs/EXP2_BayesOpt_5'
    ]
    random_paths = [
        '/Users/zipengfu/Downloads/final2/manual_logs/EXP2_Random_5',
        '/Users/zipengfu/Downloads/final3/manual_logs/EXP2_Random_5',
        '/Users/zipengfu/Downloads/final4/manual_logs/EXP2_Random_5'
    ]
    name2paths = {
        'Hyperband': hyperband_paths,
        'BayesOpt': bayesopt_paths,
        'Random': random_paths
    }

    for name, paths in name2paths.items():
        for i, file_path in enumerate(paths, start=1):
            setting = '{}_{}'.format(name, i)
            color = name2color[name]
            time_l = []
            val_acc_l = []
            curr_max_val_acc = float('-inf')
            with open(file_path, 'r') as f:
                for line in f:
                    if '[log] time: ' in line:
                        time_l.append(float(line.split('[log] time: ')[-1][:7]))
                    elif '[log] val_acc: ' in line:
                        # try:
                        #     val_acc = float(line.split('[log] val_acc: ')[-1]\
                        #         .replace('[', '(')\
                        #         .replace('\x1b', '(')\
                        #         .split('(')[0])
                        # except:
                        #     print(line.split('[log] val_acc: ')[-1].replace('[', '(').split('(')[0])
                        #     print(line)
                        #     print(file_path)
                        val_acc = float(line.split('[log] val_acc: ')[-1][:7])
                        curr_max_val_acc = max(val_acc, curr_max_val_acc)
                        val_acc_l.append(curr_max_val_acc)
            plt.plot(time_l[:101], val_acc_l[:101], color, label=setting)
    plt.title('Validation Accuracy vs. Wall Time (5 hyperparams, 100 epochs)')
    plt.xlabel('Time (s)')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.savefig('val_time.png', dpi=500)

def pp_ram_itr():
    hyperband_paths = [
        '/Users/zipengfu/Downloads/final2/manual_logs/EXP2_HyperBand_5',
        '/Users/zipengfu/Downloads/final3/manual_logs/EXP2_HyperBand_5',
        '/Users/zipengfu/Downloads/final4/manual_logs/EXP2_HyperBand_5'
    ]
    bayesopt_paths = [
        '/Users/zipengfu/Downloads/final2/manual_logs/EXP2_BayesOpt_5',
        '/Users/zipengfu/Downloads/final3/manual_logs/EXP2_BayesOpt_5',
        '/Users/zipengfu/Downloads/final4/manual_logs/EXP2_BayesOpt_5'
    ]
    random_paths = [
        '/Users/zipengfu/Downloads/final2/manual_logs/EXP2_Random_5',
        '/Users/zipengfu/Downloads/final3/manual_logs/EXP2_Random_5',
        '/Users/zipengfu/Downloads/final4/manual_logs/EXP2_Random_5'
    ]
    name2paths = {
        'Hyperband': hyperband_paths,
        'BayesOpt': bayesopt_paths,
        'Random': random_paths
    }

    for name, paths in name2paths.items():
        ram_ll = []
        for i, file_path in enumerate(paths, start=1):
            setting = '{}_{}'.format(name, i)
            color = name2color[name]
            time_l = []
            ram_l = []
            curr_max_val_acc = float('-inf')
            with open(file_path, 'r') as f:
                for line in f:
                    if '[log] ram: ' in line:
                        # try:
                        #     val_acc = float(line.split('[log] val_acc: ')[-1]\
                        #         .replace('[', '(')\
                        #         .replace('\x1b', '(')\
                        #         .split('(')[0])
                        # except:
                        #     print(line.split('[log] val_acc: ')[-1].replace('[', '(').split('(')[0])
                        #     print(line)
                        #     print(file_path)
                        ram_l.append(float(line.split('[log] ram: ')[-1][:7]))
            ram_ll.append(ram_l[:101])
        plt.plot(np.mean(ram_ll, axis=0), color, label=name)
        
    plt.title('Mean RAM Usage vs. Training Epochs (5 hyperparams, 100 epochs)')
    plt.xlabel('Training Epochs')
    plt.ylabel('Mean RAM Usage (GB)')
    plt.legend()
    plt.savefig('ram_time.png', dpi=500)

def pp_val_num_param():
    hyperband_vals = [
        [0.4209, 0.4065, 0.4235, 0.4152],
        [0.423, 0.4136, 0.3724, 0.4239]
    ]
    bayesopt_vals = [
        [0.4153, 0.4184, 0.4136, 0.4047],
        [0.4184, 0.4114, 0.4109, 0.3974]
    ]
    random_vals = [
        [0.4151, 0.3947, 0.3567, 0.3962],
        [0.4103, 0.3755, 0.4139, 0.32]
    ]
    plt.plot(range(2, 6), np.mean(hyperband_vals, axis=0), 'b', label='Hyperband')
    plt.plot(range(2, 6), np.mean(bayesopt_vals, axis=0), 'r', label='BayesOpt')
    plt.plot(range(2, 6), np.mean(random_vals, axis=0), 'g', label='Random')
    plt.title('Mean Terminal Validation Accuracy vs. Number of Hyperparams Tuned (100 epochs)')
    plt.xlabel('Number of Hyperparams')
    plt.ylabel('Mean Terminal Validation Accuacy')
    plt.xticks([2,3,4,5])
    plt.ylim(0.3, 0.45)
    plt.legend()
    plt.savefig('val_num_param.png', dpi=500)


def plot_val_num_param(file_paths):
    pass 

if __name__ == '__main__':
    pp_val_num_param()
    # parser = argparse.ArgumentParser()

    # parser.add_argument('what_to_plot', type = str, choices=['val_time', 'val_itr', 'ram_time', 'val_num_param'])
    # parser.add_argument('-f', '--file_paths', nargs='+', help='files to be read', required=True)
    # parser.add_argument('-s', '--suffix', type = str, choices=['5', '4', '3', '2', '1'])
    # args = parser.parse_args()

    # file_paths = args.file_paths
    # what_to_plot = args.what_to_plot
    # global num_hyperparam
    # num_hyperparam = args.suffix
    # print('what_to_plot: ', what_to_plot)
    # print('file_paths: ', file_paths)
    # print('num_hyperparam', num_hyperparam)

    # if what_to_plot == 'val_time':
    #     plot_val_time(file_paths)
    # elif what_to_plot == 'val_itr':
    #     plot_val_itr(file_paths)
    # elif what_to_plot == 'ram_time':
    #     plot_ram_time(file_paths)
    # elif what_to_plot == 'val_num_param':
    #     plot_val_num_param(file_paths)


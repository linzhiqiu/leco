import os
import argparse
import random
import numpy as np
import configs
from util import load_pickle, makedirs
import hparams
import setups
from tabulate import tabulate

argparser = argparse.ArgumentParser()
argparser.add_argument("--data_dir", 
                        default='/scratch/leco/',
                        help="Where the dataset are saved.")
SEED_LIST = [None, 1, 10, 100, 1000]


def get_all_result_lists(result_dict):
    lsts = []
    for train_mode_str in result_dict:
        for hparam_str in result_dict[train_mode_str]:
            for seed_str in result_dict[train_mode_str][hparam_str]:
                tp_0_epoch = result_dict[train_mode_str][hparam_str][seed_str][0]['best_train_loss_epoch']
                tp_0_train = result_dict[train_mode_str][hparam_str][seed_str][0]['train_acc']
                tp_0_test = result_dict[train_mode_str][hparam_str][seed_str][0]['test_acc']
                tp_1_epoch = result_dict[train_mode_str][hparam_str][seed_str][1]['best_train_loss_epoch']
                tp_1_train = result_dict[train_mode_str][hparam_str][seed_str][1]['train_acc']
                tp_1_test = result_dict[train_mode_str][hparam_str][seed_str][1]['test_acc']
                lst = [train_mode_str, hparam_str, seed_str, f"{tp_0_epoch}", f"{tp_0_train:.2%}", f"{tp_0_test:.2%}", f"{tp_1_epoch}", f"{tp_1_train:.2%}", f"{tp_1_test:.2%}"]
                lsts.append(lst)
    return lsts

def mean_and_std(lst, formatter="%"):
    if formatter == '%':
        return f"{np.array(lst).mean():.2%}+-{np.array(lst).std():.2%}"
    else:
        return f"{np.array(lst).mean():.2f}+-{np.array(lst).std():.2f}"

def get_avg_result_lists(result_dict):
    lsts = []
    for train_mode_str in result_dict:
        for hparam_str in result_dict[train_mode_str]:
            tp_0_epochs = []
            tp_0_trains = []
            tp_0_tests = []
            tp_1_epochs = []
            tp_1_trains = []
            tp_1_tests = []
            for seed_str in result_dict[train_mode_str][hparam_str]:
                tp_0_epochs.append(result_dict[train_mode_str][hparam_str][seed_str][0]['best_train_loss_epoch'])
                tp_0_trains.append(result_dict[train_mode_str][hparam_str][seed_str][0]['train_acc'])
                tp_0_tests.append(result_dict[train_mode_str][hparam_str][seed_str][0]['test_acc'])
                tp_1_epochs.append(result_dict[train_mode_str][hparam_str][seed_str][1]['best_train_loss_epoch'])
                tp_1_trains.append(result_dict[train_mode_str][hparam_str][seed_str][1]['train_acc'])
                tp_1_tests.append(result_dict[train_mode_str][hparam_str][seed_str][1]['test_acc'])
            lst = [train_mode_str, hparam_str, f"{len(tp_1_tests)}", mean_and_std(tp_0_epochs, formatter='f'), mean_and_std(tp_0_trains), mean_and_std(tp_0_tests), mean_and_std(tp_1_epochs, formatter='f'), mean_and_std(tp_1_trains), mean_and_std(tp_1_tests)]
            lsts.append(lst)
    return lsts
                        

def gather_exp(data_dir: str,
               seed_list=SEED_LIST):

    table_dir = os.path.join(data_dir, 'tables')
    makedirs(table_dir)
    setup_list = list(setups.SETUPS.keys())
    train_mode_list = list(configs.TRAIN_MODES.keys())
    hparam_str_list = list(hparams.HPARAMS.keys())

    result_dict = {}
    for setup_mode in setup_list:
        result_dict[setup_mode] = {}
        for seed in seed_list:
            seed_str = str(seed)
            setup_dir = os.path.join(data_dir, setup_mode, f"seed_{seed_str}")
            if not os.path.exists(setup_dir):
                import pdb; pdb.set_trace()
            for train_mode_str in train_mode_list:
                train_mode_dir = os.path.join(setup_dir, train_mode_str)
                for hparam_str in hparam_str_list:
                    exp_dir = os.path.join(train_mode_dir, hparam_str)
                    tp_results = []
                    is_ready = True
                    for tp_idx in range(2):
                        exp_dir_tp_idx = os.path.join(exp_dir, str(tp_idx))
                        exp_result_path = os.path.join(exp_dir_tp_idx, "result.ckpt")
                        if os.path.exists(exp_result_path):
                            exp_result = load_pickle(exp_result_path)
                            best_epoch = exp_result['best_result']['best_epoch']
                            train_acc = exp_result['acc_result']['train']
                            test_acc = exp_result['acc_result']['test']
                            tp_results.append({
                                'best_train_loss_epoch' : best_epoch,
                                'train_acc' : train_acc,
                                'test_acc' : test_acc
                            })
                        else:
                            is_ready = False
                            break
                    
                    if is_ready:
                        if not train_mode_str in result_dict[setup_mode]:
                            result_dict[setup_mode][train_mode_str] = {}
                        if not hparam_str in result_dict[setup_mode][train_mode_str]:
                            result_dict[setup_mode][train_mode_str][hparam_str] = {}
                        result_dict[setup_mode][train_mode_str][hparam_str][seed_str] = tp_results
                        
        all_headers = ['Train mode', 'Hyperparameter', 'Seed', 'TP0 Ckpt Epoch', 'TP0 Train Acc', 'TP0 Test Acc', 'TP1 Ckpt Epoch', 'TP1 Train Acc', 'TP1 Test Acc']                    
        all_results = get_all_result_lists(result_dict[setup_mode])
        all_table_path = os.path.join(table_dir, f"{setup_mode}.txt")
        with open(all_table_path, "w+") as file:
            file.write(tabulate(all_results, headers=all_headers, tablefmt='orgtbl'))
            print(f"Save at {all_table_path}")
        
        avg_headers = ['Train mode', 'Hyperparameter', 'Finished', 'TP0 Ckpt Epoch', 'TP0 Train Acc', 'TP0 Test Acc', 'TP1 Ckpt Epoch', 'TP1 Train Acc', 'TP1 Test Acc']                    
        avg_results = get_avg_result_lists(result_dict[setup_mode])
        avg_table_path = os.path.join(table_dir, f"{setup_mode}_avg.txt")
        with open(avg_table_path, "w+") as file:
            file.write(tabulate(avg_results, headers=avg_headers, tablefmt='orgtbl'))
            print(f"Save at {avg_table_path}")


if __name__ == '__main__':
    args = argparser.parse_args()
    gather_exp(args.data_dir)
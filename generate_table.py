import os
import argparse
import random
import numpy as np
import configs
from util import load_pickle, makedirs
import hparams
import setups
from tabulate import tabulate
from train import get_exp_str_from_train_mode, get_exp_str_from_hparam_strs
from train_partial_feedback import get_exp_str_from_partial_feedback
import configs

argparser = argparse.ArgumentParser()
argparser.add_argument("--data_dir", 
                        default='/scratch/leco/',
                        help="Where the dataset are saved.")
argparser.add_argument("--partial_feedback_mode", 
                        default=None, type=str,
                        help="If used partial feedback, input the mode")
SEED_LIST = [None, 1, 10, 100, 1000]


def get_all_result_lists(result_dict):
    lsts = []
    for train_mode_str in result_dict:
        for hparam_0_str in result_dict[train_mode_str]:
            for hparam_1_str in result_dict[train_mode_str][hparam_0_str]:
                for seed_str in result_dict[train_mode_str][hparam_0_str][hparam_1_str]:
                    tp_0_epoch = result_dict[train_mode_str][hparam_0_str][hparam_1_str][seed_str][0]['best_train_loss_epoch']
                    tp_0_train = result_dict[train_mode_str][hparam_0_str][hparam_1_str][seed_str][0]['train_acc']
                    tp_0_test = result_dict[train_mode_str][hparam_0_str][hparam_1_str][seed_str][0]['test_acc']
                    tp_1_epoch = result_dict[train_mode_str][hparam_0_str][hparam_1_str][seed_str][1]['best_train_loss_epoch']
                    tp_1_train = result_dict[train_mode_str][hparam_0_str][hparam_1_str][seed_str][1]['train_acc']
                    tp_1_test = result_dict[train_mode_str][hparam_0_str][hparam_1_str][seed_str][1]['test_acc']
                    lst = [train_mode_str, seed_str, hparam_0_str, f"{tp_0_epoch}", f"{tp_0_train:.2%}", f"{tp_0_test:.2%}", hparam_1_str, f"{tp_1_epoch}", f"{tp_1_train:.2%}", f"{tp_1_test:.2%}"]
                    lsts.append(lst)
    return lsts

def mean_and_std(lst, formatter="%"):
    if formatter == '%':
        return f"{np.array(lst).mean():.2%}+-{np.array(lst).std():.2%}"
    else:
        return f"{np.array(lst).mean():.2f}+-{np.array(lst).std():.2f}"

def to_str(result, formatter='%'):
    mean, std = result
    if formatter == '%':
        return f"{mean:.2%}+-{std:.2%}"
    else:
        return f"{mean:.2f}+-{std:.2f}"

def mean(lst):
    return float(np.array(lst).mean())

def std(lst):
    return float(np.array(lst).std())

def get_avg_result_lists(result_dict):
    lsts = []
    for train_mode_str in result_dict:
        for hparam_0_str in result_dict[train_mode_str]:
            for hparam_1_str in result_dict[train_mode_str][hparam_0_str]:
                tp_0_epochs = []
                tp_0_trains = []
                tp_0_tests = []
                tp_1_epochs = []
                tp_1_trains = []
                tp_1_tests = []
                for seed_str in result_dict[train_mode_str][hparam_0_str][hparam_1_str]:
                    tp_0_epochs.append(result_dict[train_mode_str][hparam_0_str][hparam_1_str][seed_str][0]['best_train_loss_epoch'])
                    tp_0_trains.append(result_dict[train_mode_str][hparam_0_str][hparam_1_str][seed_str][0]['train_acc'])
                    tp_0_tests.append(result_dict[train_mode_str][hparam_0_str][hparam_1_str][seed_str][0]['test_acc'])
                    tp_1_epochs.append(result_dict[train_mode_str][hparam_0_str][hparam_1_str][seed_str][1]['best_train_loss_epoch'])
                    tp_1_trains.append(result_dict[train_mode_str][hparam_0_str][hparam_1_str][seed_str][1]['train_acc'])
                    tp_1_tests.append(result_dict[train_mode_str][hparam_0_str][hparam_1_str][seed_str][1]['test_acc'])
                lst = [train_mode_str, f"{len(tp_1_tests)}", hparam_0_str, mean_and_std(tp_0_epochs, formatter='f'), mean_and_std(tp_0_trains), mean_and_std(tp_0_tests), hparam_1_str, mean_and_std(tp_1_epochs, formatter='f'), mean_and_std(tp_1_trains), mean_and_std(tp_1_tests)]
                lsts.append(lst)
    return lsts

def get_best_result_lists(result_dict, num_of_exp=5):
    lsts = []
    exp_dict = {}
    for train_mode_str in result_dict:
        exp_dict[train_mode_str] = {}
        for hparam_0_str in result_dict[train_mode_str]:
            exp_dict[train_mode_str][hparam_0_str] = {}
            for hparam_1_str in result_dict[train_mode_str][hparam_0_str]:
                exp_dict[train_mode_str][hparam_0_str][hparam_1_str] = {}
                tp_0_epochs = []
                tp_0_trains = []
                tp_0_tests = []
                tp_1_epochs = []
                tp_1_trains = []
                tp_1_tests = []
                for seed_str in result_dict[train_mode_str][hparam_0_str][hparam_1_str]:
                    tp_0_epochs.append(result_dict[train_mode_str][hparam_0_str][hparam_1_str][seed_str][0]['best_train_loss_epoch'])
                    tp_0_trains.append(result_dict[train_mode_str][hparam_0_str][hparam_1_str][seed_str][0]['train_acc'])
                    tp_0_tests.append(result_dict[train_mode_str][hparam_0_str][hparam_1_str][seed_str][0]['test_acc'])
                    tp_1_epochs.append(result_dict[train_mode_str][hparam_0_str][hparam_1_str][seed_str][1]['best_train_loss_epoch'])
                    tp_1_trains.append(result_dict[train_mode_str][hparam_0_str][hparam_1_str][seed_str][1]['train_acc'])
                    tp_1_tests.append(result_dict[train_mode_str][hparam_0_str][hparam_1_str][seed_str][1]['test_acc'])
                if not len(tp_0_epochs) == num_of_exp:
                    print(f"Train mode {train_mode_str}, HParam 0 {hparam_0_str}, HParam 1 {hparam_1_str}, not having enough experiments finished ({len(tp_0_epochs)} out of {num_of_exp})")
                    import pdb; pdb.set_trace()
                    continue
                
                tp_0_epoch_result = mean(tp_0_epochs), std(tp_0_epochs)
                tp_0_train_acc = mean(tp_0_trains), std(tp_0_trains)
                tp_0_test_acc = mean(tp_0_tests), std(tp_0_tests)
                tp_1_epoch_result = mean(tp_1_epochs), std(tp_1_epochs)
                tp_1_train_acc = mean(tp_1_trains), std(tp_1_trains)
                tp_1_test_acc = mean(tp_1_tests), std(tp_1_tests)
                if not 'result' in exp_dict[train_mode_str][hparam_0_str]:
                    exp_dict[train_mode_str][hparam_0_str]['result'] = {
                        'epochs' : tp_0_epoch_result,
                        'train_acc' : tp_0_train_acc,
                        'test_acc' : tp_0_test_acc,
                    }
                else:
                    if not exp_dict[train_mode_str][hparam_0_str]['result']['epochs'] == tp_0_epoch_result:
                        import pdb; pdb.set_trace()
                    if not exp_dict[train_mode_str][hparam_0_str]['result']['train_acc'] == tp_0_train_acc:
                        import pdb; pdb.set_trace()
                    if not exp_dict[train_mode_str][hparam_0_str]['result']['test_acc'] == tp_0_test_acc:
                        import pdb; pdb.set_trace()
                    
                exp_dict[train_mode_str][hparam_0_str][hparam_1_str] = {
                    'epochs' : tp_1_epoch_result,
                    'train_acc' : tp_1_train_acc,
                    'test_acc' : tp_1_test_acc
                }
    
    for train_mode_str in exp_dict:
        best_hparam_0_test_acc = None
        for hparam_0_str in exp_dict[train_mode_str]:
            hparam_0_test_acc = exp_dict[train_mode_str][hparam_0_str]['result']['test_acc'][0]
            if type(best_hparam_0_test_acc) == type(None) or hparam_0_test_acc > best_hparam_0_test_acc[0]:
                best_hparam_0_str = hparam_0_str
                best_hparam_0_test_acc = exp_dict[train_mode_str][hparam_0_str]['result']['test_acc']
                best_hparam_0_train_acc = exp_dict[train_mode_str][hparam_0_str]['result']['train_acc']
                best_hparam_0_epochs = exp_dict[train_mode_str][hparam_0_str]['result']['epochs']

        best_hparam_1_test_acc = None
        for hparam_1_str in exp_dict[train_mode_str][best_hparam_0_str]:
            if hparam_1_str == 'result':
                continue
            hparam_1_test_acc = exp_dict[train_mode_str][best_hparam_0_str][hparam_1_str]['test_acc'][0]
            if type(best_hparam_1_test_acc) == type(None) or hparam_1_test_acc > best_hparam_1_test_acc[0]:
                best_hparam_1_str = hparam_1_str
                best_hparam_1_test_acc = exp_dict[train_mode_str][best_hparam_0_str][hparam_1_str]['test_acc']
                best_hparam_1_train_acc = exp_dict[train_mode_str][best_hparam_0_str][hparam_1_str]['train_acc']
                best_hparam_1_epochs = exp_dict[train_mode_str][best_hparam_0_str][hparam_1_str]['epochs']
        
        lst = [train_mode_str, best_hparam_0_str, to_str(best_hparam_0_epochs, formatter='f'), to_str(best_hparam_0_train_acc), to_str(best_hparam_0_test_acc), best_hparam_1_str, to_str(best_hparam_1_epochs, formatter='f'), to_str(best_hparam_1_train_acc), to_str(best_hparam_1_test_acc)]
        lsts.append(lst)
    return lsts
                        

def gather_exp(data_dir: str,
               partial_feedback_mode=None,
               seed_list=SEED_LIST):

    table_dir = os.path.join(data_dir, 'tables')
    if partial_feedback_mode:
        table_dir = os.path.join(table_dir, partial_feedback_mode)
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
            for hparam_0_str in hparam_str_list:
                for hparam_1_str in hparam_str_list:
                    for train_mode_str in train_mode_list:
                        tp_results = []
                        is_ready = True
                        train_mode = configs.TRAIN_MODES[train_mode_str]
                        for tp_idx in range(2):
                            exp_result_path = os.path.join(setup_dir,
                                                            get_exp_str_from_train_mode(train_mode, tp_idx=tp_idx),
                                                            get_exp_str_from_partial_feedback(partial_feedback_mode, tp_idx=tp_idx), #CHANGE
                                                            get_exp_str_from_hparam_strs([hparam_0_str, hparam_1_str], tp_idx=tp_idx),
                                                            'result.ckpt')
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
                            if not hparam_0_str in result_dict[setup_mode][train_mode_str]:
                                result_dict[setup_mode][train_mode_str][hparam_0_str] = {}
                            if not hparam_1_str in result_dict[setup_mode][train_mode_str][hparam_0_str]:
                                result_dict[setup_mode][train_mode_str][hparam_0_str][hparam_1_str] = {}
                            result_dict[setup_mode][train_mode_str][hparam_0_str][hparam_1_str][seed_str] = tp_results
                        
        all_headers = ['Train mode', 'Seed', 'Hparam 0',  'TP0 Ckpt Epoch', 'TP0 Train Acc', 'TP0 Test Acc', 'Hparam 0', 'TP1 Ckpt Epoch', 'TP1 Train Acc', 'TP1 Test Acc']                    
        all_results = get_all_result_lists(result_dict[setup_mode])
        all_table_path = os.path.join(table_dir, f"{setup_mode}.txt")
        with open(all_table_path, "w+") as file:
            file.write(tabulate(all_results, headers=all_headers, tablefmt='orgtbl'))
            print(f"Save at {all_table_path}")
        
        avg_headers = ['Train mode', 'Finished', 'Hparam 0', 'TP0 Ckpt Epoch', 'TP0 Train Acc', 'TP0 Test Acc', 'Hparam 1', 'TP1 Ckpt Epoch', 'TP1 Train Acc', 'TP1 Test Acc']                    
        avg_results = get_avg_result_lists(result_dict[setup_mode])
        avg_table_path = os.path.join(table_dir, f"{setup_mode}_avg.txt")
        with open(avg_table_path, "w+") as file:
            file.write(tabulate(avg_results, headers=avg_headers, tablefmt='orgtbl'))
            print(f"Save at {avg_table_path}")
        
        best_headers = ['Train mode', 'Best Hparam 0', 'TP0 Ckpt Epoch', 'TP0 Train Acc', 'TP0 Test Acc', 'Best Hparam 1', 'TP1 Ckpt Epoch', 'TP1 Train Acc', 'TP1 Test Acc']                    
        best_results = get_best_result_lists(result_dict[setup_mode])
        best_table_path = os.path.join(table_dir, f"{setup_mode}_best.txt")
        with open(best_table_path, "w+") as file:
            file.write(tabulate(best_results, headers=best_headers, tablefmt='orgtbl'))
            print(f"Save at {best_table_path}")


if __name__ == '__main__':
    args = argparser.parse_args()
    gather_exp(args.data_dir,
               args.partial_feedback_mode)
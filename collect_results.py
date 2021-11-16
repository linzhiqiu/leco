import os
import argparse
import random
import numpy as np
import configs
from util import load_pickle, makedirs
import hparams
import setups
from tabulate import tabulate
from print_utils import get_exp_str_from_train_mode, get_exp_str_from_hparam_strs, get_exp_str_from_partial_feedback
import configs

argparser = argparse.ArgumentParser()
argparser.add_argument("--data_dir", 
                        default='/scratch/leco/',
                        help="Where the dataset are saved.")
argparser.add_argument("--partial_feedback_mode", 
                        default=None, type=str,
                        help="If used partial feedback, input the mode")
argparser.add_argument("--hparam_candidate",
                        type=str,
                        default='cifar',
                        choices=hparams.HPARAM_CANDIDATES.keys(),
                        help="The hyperparameter candidates (str) for next time period")
SEED_LIST = [None, 1, 10, 100, 1000]

def mean_std_from_dict(lst_of_dict, key):
    lst = np.array([d[key] for d in lst_of_dict])
    return lst.mean(), lst.std()

def get_result(tp_result):
    epoch = str(tp_result['best_epoch'])
    train_acc = f"{tp_result['train_acc']:.2%}"
    test_acc = f"{tp_result['test_acc']:.2%}"
    return epoch, train_acc, test_acc

def get_mean_std_from_dict(d, formatter="%"):
    mean = d['mean']
    std = d['std']
    if formatter == "%":
        return f"{mean:.2%}+-{std:.2%}"
    else:
        return f"{mean:.2f}+-{std:.2f}"

def get_mean_std_result(all_results):
    epoch = get_mean_std_from_dict(all_results['best_epoch'], formatter="f")
    train_acc = get_mean_std_from_dict(all_results['train_acc'])
    test_acc = get_mean_std_from_dict(all_results['test_acc'])
    return epoch, train_acc, test_acc

def save_all_results(result_dir, result_dict, setup_mode, all_tp_info):
    save_dir = os.path.join(result_dir, setup_mode, "all")
    makedirs(save_dir)
    for tp_info in all_tp_info:
        tp_idx = tp_info['tp_idx']
        tp_name = tp_info['tp_name']
        tp_str = str(tp_idx) + "_" + tp_name
        tp_file = os.path.join(save_dir, f'{tp_str}.txt')
    
        all_headers = ['Train mode', 'Run']
        for i in range(tp_idx, -1, -1):
            all_headers += [f'Hparam {i}', f'TP{i} Ckpt Epoch', f'TP{i} Train Acc', f'TP{i} Test Acc']
        all_rows = []

        for train_mode_str in result_dict[setup_mode]:
            if len(result_dict[setup_mode][train_mode_str]) <= tp_idx:
                continue
            else:
                assert result_dict[setup_mode][train_mode_str][tp_idx]['tp_idx'] == tp_idx
            all_results = result_dict[setup_mode][train_mode_str][tp_idx]['all_results']
            hparam_strs = result_dict[setup_mode][train_mode_str][tp_idx]['hparam_strs']
            prev_results = [result_dict[setup_mode][train_mode_str][i]['all_results'][hparam_strs[i]]['tp_results'] for i in range(tp_idx)]
            for hparam_str in all_results:
                tp_results = all_results[hparam_str]['tp_results']
                for seed_idx, tp_result in enumerate(tp_results):
                    row = [train_mode_str, str(seed_idx)]
                    for prev_i in range(tp_idx-1, -1, -1):
                        prev_result = prev_results[prev_i]
                        prev_result_i = prev_result[seed_idx]
                        prev_hparam = hparam_strs[prev_i]
                        epoch, train_acc, test_acc = get_result(prev_result_i)
                        row += [prev_hparam, epoch, train_acc, test_acc]
                    
                    epoch, train_acc, test_acc = get_result(tp_result)
                    row += [hparam_str, epoch, train_acc, test_acc]
                    all_rows.append(row)
        
        with open(tp_file, "w+") as file:
            file.write(tabulate(all_rows, headers=all_headers, tablefmt='orgtbl'))
            print(f"Save at {tp_file}")

def save_avg_results(result_dir, result_dict, setup_mode, all_tp_info):
    save_dir = os.path.join(result_dir, setup_mode, "avg")
    makedirs(save_dir)
    for tp_info in all_tp_info:
        tp_idx = tp_info['tp_idx']
        tp_name = tp_info['tp_name']
        tp_str = str(tp_idx) + "_" + tp_name
        tp_file = os.path.join(save_dir, f'{tp_str}.txt')
    
        all_headers = ['Train mode']
        for i in range(tp_idx, -1, -1):
            all_headers += [f'Hparam {i}', f'TP{i} Ckpt Epoch', f'TP{i} Train Acc', f'TP{i} Test Acc']
        all_rows = []

        for train_mode_str in result_dict[setup_mode]:
            if len(result_dict[setup_mode][train_mode_str]) <= tp_idx:
                continue
            else:
                assert result_dict[setup_mode][train_mode_str][tp_idx]['tp_idx'] == tp_idx
            all_results = result_dict[setup_mode][train_mode_str][tp_idx]['all_results']
            hparam_strs = result_dict[setup_mode][train_mode_str][tp_idx]['hparam_strs']
            prev_results = [result_dict[setup_mode][train_mode_str][i]['all_results'][hparam_strs[i]] for i in range(tp_idx)]
            for hparam_str in all_results:
                row = [train_mode_str]
                
                for prev_i in range(tp_idx-1, -1, -1):
                    prev_result = prev_results[prev_i]
                    prev_hparam = hparam_strs[prev_i]
                    epoch, train_acc, test_acc = get_mean_std_result(prev_result)
                    row += [prev_hparam, epoch, train_acc, test_acc]
                    
                epoch, train_acc, test_acc = get_mean_std_result(all_results[hparam_str])
                row += [hparam_str, epoch, train_acc, test_acc]
                all_rows.append(row)
        
        with open(tp_file, "w+") as file:
            file.write(tabulate(all_rows, headers=all_headers, tablefmt='orgtbl'))
            print(f"Save at {tp_file}")

def save_best_results(result_dir, result_dict, setup_mode, all_tp_info):
    save_dir = os.path.join(result_dir, setup_mode, "best")
    makedirs(save_dir)
    for tp_info in all_tp_info:
        tp_idx = tp_info['tp_idx']
        tp_name = tp_info['tp_name']
        tp_str = str(tp_idx) + "_" + tp_name
        tp_file = os.path.join(save_dir, f'{tp_str}.txt')
    
        all_headers = ['Train mode']
        for i in range(tp_idx, -1, -1):
            all_headers += [f'Best Hparam {i}', f'TP{i} Ckpt Epoch', f'TP{i} Train Acc', f'TP{i} Test Acc']
        all_rows = []

        for train_mode_str in result_dict[setup_mode]:
            if len(result_dict[setup_mode][train_mode_str]) <= tp_idx:
                continue
            else:
                assert result_dict[setup_mode][train_mode_str][tp_idx]['tp_idx'] == tp_idx
            hparam_strs = result_dict[setup_mode][train_mode_str][tp_idx]['hparam_strs']
            assert len(hparam_strs) == tp_idx + 1
            row = [train_mode_str]
            best_results = [result_dict[setup_mode][train_mode_str][i]['all_results'][hparam_strs[i]] for i in range(tp_idx+1)]
            for idx in range(tp_idx, -1, -1):
                best_result = best_results[idx]
                best_hparam = hparam_strs[idx]
                epoch, train_acc, test_acc = get_mean_std_result(best_result)
                row += [best_hparam, epoch, train_acc, test_acc]
            all_rows.append(row)
        
        with open(tp_file, "w+") as file:
            file.write(tabulate(all_rows, headers=all_headers, tablefmt='orgtbl'))
            print(f"Save at {tp_file}")

def prepare_scripts(data_dir, setup_mode, train_mode_str, hparam_strs, hparam_candidate, seed_list, partial_feedback_mode):
    scripts = []
    script_file = "train.py" if not partial_feedback_mode else "train_partial_feedback.py --partial_feedback_mode {partial_feedback_mode}"
    for seed in seed_list:
        seed_str = f"--seed {seed}" if seed else ""
        hparam_list_str = "--hparam_strs " + " ".join(hparam_strs) if len(hparam_strs) > 0 else ""
        script = f"python {script_file} --setup_mode {setup_mode} --train_mode {train_mode_str} {seed_str} {hparam_list_str} --hparam_candidate {hparam_candidate}"
        scripts.append(script)
    return scripts  

def write_scripts_to_file(scripts_to_run, result_dir, setup_mode):
    script_dir = os.path.join(result_dir, setup_mode)
    makedirs(script_dir)
    script_file = os.path.join(script_dir, "scripts.txt")
    with open(script_file, "w+") as f:
        for script in scripts_to_run:
            f.write(script + "\n")
    print(f"Saved {len(scripts_to_run)} scripts (to run) in {script_file}")

def gather_exp(data_dir: str,
               partial_feedback_mode=None,
               hparam_candidate='cifar',
               seed_list=SEED_LIST):

    result_dir = os.path.join(data_dir, 'results')
    if partial_feedback_mode:
        result_dir = os.path.join(result_dir, partial_feedback_mode)
    makedirs(result_dir)
    setup_list = list(setups.SETUPS.keys())
    train_mode_list = list(configs.TRAIN_MODES.keys())
    hparam_str_list = list(hparams.HPARAMS.keys())

    result_dict = {}
    for setup_mode in setup_list:
        result_dict[setup_mode] = {}
        setup_dirs = [os.path.join(data_dir, setup_mode, f"seed_{str(seed)}") for seed in seed_list]
        dataset_paths = [os.path.join(setup_dir, 'dataset.pt') for setup_dir in setup_dirs]
        is_ready = True
        for dataset_path in dataset_paths:
            if not os.path.exists(dataset_path):
                is_ready = False
        if not is_ready:
            print(f"{setup_mode} setup has not finished.")
            break
        
        dataset = load_pickle(dataset_path)
        _, _, all_tp_info, leaf_idx_to_all_class_idx = dataset
        
        scripts_to_run = []
        for train_mode_str in train_mode_list:
            train_mode = configs.TRAIN_MODES[train_mode_str]
            hparam_strs = [] # Best hparam per time period
            all_results = {}
            for tp_idx in range(len(all_tp_info)):
                all_candidates = hparams.HPARAM_CANDIDATES[hparam_candidate]
                best_hparam_str = None
                best_test_acc_mean = None
                all_candidates_are_ready = True
                for hparam_str in all_candidates:
                    tp_results = []
                    for setup_dir in setup_dirs:
                        exp_result_path = os.path.join(setup_dir,
                                                       get_exp_str_from_train_mode(train_mode, tp_idx=tp_idx),
                                                       get_exp_str_from_partial_feedback(partial_feedback_mode, tp_idx=tp_idx),
                                                       get_exp_str_from_hparam_strs(hparam_strs+[hparam_str], tp_idx=tp_idx),
                                                       'result.ckpt')
                        if os.path.exists(exp_result_path):
                            exp_result = load_pickle(exp_result_path)
                            best_epoch = exp_result['best_result']['best_epoch']
                            train_acc = exp_result['acc_result']['train']
                            test_acc = exp_result['acc_result']['test']
                            tp_results.append({
                                'best_epoch' : best_epoch,
                                'train_acc' : train_acc,
                                'test_acc' : test_acc
                            })
                        else:
                            all_candidates_are_ready = False
                            break
                    
                    if all_candidates_are_ready:
                        # summarize the results and put into all_results
                        best_epoch_mean, best_epoch_std = mean_std_from_dict(tp_results, 'best_epoch')
                        train_acc_mean, train_acc_std = mean_std_from_dict(tp_results, 'train_acc')
                        test_acc_mean, test_acc_std = mean_std_from_dict(tp_results, 'test_acc')
                        all_results[hparam_str] = {
                            'best_epoch' : {'mean' : best_epoch_mean,
                                            'std'  : best_epoch_std},
                            'train_acc' : {'mean' : train_acc_mean,
                                            'std'  : train_acc_std},
                            'test_acc' : {'mean' : test_acc_mean,
                                            'std'  : test_acc_std},
                            'tp_results' : tp_results,
                        }
                        if best_test_acc_mean is None or test_acc_mean > best_test_acc_mean:
                            best_test_acc_mean = test_acc_mean
                            best_hparam_str = hparam_str
                    else:
                        break
                if all_candidates_are_ready:
                    # write the results
                    if not train_mode_str in result_dict[setup_mode]:
                        result_dict[setup_mode][train_mode_str] = []
                    print(f"For setup {setup_mode}, train mode {train_mode_str}, tp {tp_idx}, the best hparam is {best_hparam_str} with test acc {best_test_acc_mean}")
                    hparam_strs += [best_hparam_str]
                    result_dict[setup_mode][train_mode_str].append({
                        'tp_idx' : tp_idx,
                        'best_hparam_str' : best_hparam_str,
                        'hparam_strs' : hparam_strs, # up to this tp_idx
                        'all_results' : all_results,
                    })
                        
                    
                else:
                    # prepare scripts for this tp_idx
                    current_scripts = prepare_scripts(data_dir, setup_mode, train_mode_str, hparam_strs, hparam_candidate, seed_list, partial_feedback_mode)
                    scripts_to_run += current_scripts
                    print(f"Setup {setup_mode}: {len(current_scripts)} scripts for train mode {train_mode_str}.")
                    break
        print(f"Setup {setup_mode}: Total {len(scripts_to_run)} scripts.")
        write_scripts_to_file(scripts_to_run, result_dir, setup_mode)
        
        save_all_results(result_dir, result_dict, setup_mode, all_tp_info)
        
        save_avg_results(result_dir, result_dict, setup_mode, all_tp_info)
    
        save_best_results(result_dir, result_dict, setup_mode, all_tp_info)


if __name__ == '__main__':
    args = argparser.parse_args()
    gather_exp(args.data_dir,
               args.partial_feedback_mode,
               args.hparam_candidate)
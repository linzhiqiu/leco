# python collect_for_more_tps.py --data_dir /scratch/leco/ --hparam_candidate inat --result_dir /data3/zhiqiul/leco_results/ --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --ema_decay 0.999 --sampling half
# python collect_for_more_tps.py --data_dir /scratch/leco/ --hparam_candidate inat --result_dir /data3/zhiqiul/leco_results/ --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --ema_decay 0.999 --sampling avg
import os
import argparse
import random
import numpy as np
import configs
from util import load_pickle, makedirs
import hparams
import setups
from tabulate import tabulate
import train_for_more_tps
from train_for_more_tps import get_setup_dir, get_train_dir, get_semi_supervised_dir
import configs
import copy
import leco_mode
from print_utils import get_exp_str_from_ema_decay

LATEX_FORMAT = True

argparser = argparse.ArgumentParser()
argparser.add_argument("--data_dir", 
                        default='/scratch/leco/',
                        help="Where the dataset are saved.")
argparser.add_argument("--result_dir",
                       default="/data3/zhiqiul/leco_results/",
                       help="Where the dataset split and results are saved")
argparser.add_argument("--model_save_dir", 
                        default='/data3/zhiqiul/self_supervised_models/resnet50/',
                        help="Where the self-supervised pre-trained models will be saved.")
argparser.add_argument('--ema_decay',
                       default=None,
                       type=float,
                       help='EMA decay rate. If none, then no ModelEMA is used.')
argparser.add_argument("--hparam_candidate",
                       type=str,
                       default='inat',
                       choices=hparams.HPARAM_CANDIDATES.keys(),
                       help="The hyperparameter candidates (str) for next time period")
argparser.add_argument('--sampling', default='half', choices=['half', 'avg'],
                       help='How to sample per-TP data for each batch ')

SETUP_LIST = ['semi_inat_strongaug_4_tps']
# For Inat all CL modes
SEED_LIST = [None, 1, 10, 100, 1000]
TRAIN_MODE_LIST = configs.ALL_TRAIN_MODES['inat_4_tps']
HIERARCHICAL_SEMI_SUPERVISION = train_for_more_tps.HIERARCHICAL_SEMI_SUPERVISION

LECO_MODES = ['label_new', 'relabel_old', 'upper_bound']
PL_THRESHOLDS = [None]
PARTIAL_FEEDBACK_MODE = [None]
SEMI_SUPERVISED_ALG = [None]

# LECO_MODES = ['label_new']
# PL_THRESHOLDS = [0.95]
# SEMI_SUPERVISED_ALG=["DistillHard", "DistillSoft", "Fixmatch", "PL", None]
# PARTIAL_FEEDBACK_MODE=['lpl', 'joint', None]

def latex_str(s):
    # $89.76\%\pm0.48\%$
    # s = "54.98%+-1.01%"
    mean, std = s.split("+-")
    mean = float(mean.strip("%"))
    std = float(std.strip("%"))
    final_str = f"${mean:.2f}\pm{std:.2f}$"
    return final_str


def mean_std_from_dict(lst_of_dict, key):
    lst = np.array([d[key] for d in lst_of_dict])
    if len(lst) == 1:
        return lst[0], 0.0
    else:
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


def save_all_results(print_result_dir, result_dict, setup_mode, all_tp_info):
    save_dir = os.path.join(print_result_dir, setup_mode, "all")
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
                    epoch, train_acc, test_acc = get_result(tp_result)
                    row += [hparam_str, epoch, train_acc, test_acc]

                    for prev_i in range(tp_idx-1, -1, -1):
                        prev_result = prev_results[prev_i]
                        prev_result_i = prev_result[seed_idx]
                        prev_hparam = hparam_strs[prev_i]
                        epoch, train_acc, test_acc = get_result(prev_result_i)
                        row += [prev_hparam, epoch, train_acc, test_acc]
                    
                    all_rows.append(row)
        
        with open(tp_file, "w+") as file:
            file.write(tabulate(all_rows, headers=all_headers, tablefmt='orgtbl'))
            # print(f"Save at {tp_file}")

def save_avg_results(print_result_dir, result_dict, setup_mode, all_tp_info):
    save_dir = os.path.join(print_result_dir, setup_mode, "avg")
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
                
                epoch, train_acc, test_acc = get_mean_std_result(all_results[hparam_str])
                row += [hparam_str, epoch, train_acc, test_acc]
                all_rows.append(row)

                for prev_i in range(tp_idx-1, -1, -1):
                    prev_result = prev_results[prev_i]
                    prev_hparam = hparam_strs[prev_i]
                    epoch, train_acc, test_acc = get_mean_std_result(prev_result)
                    row += [prev_hparam, epoch, train_acc, test_acc]
                    
        
        with open(tp_file, "w+") as file:
            file.write(tabulate(all_rows, headers=all_headers, tablefmt='orgtbl'))
            # print(f"Save at {tp_file}")

def save_best_results(print_result_dir, result_dict, setup_mode, all_tp_info):
    save_dir = os.path.join(print_result_dir, setup_mode, "best")
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
            if not len(hparam_strs) - 1 >= tp_idx:
                import pdb; pdb.set_trace()
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
            # print(f"Save at {tp_file}")

def prepare_scripts(data_dir, result_dir, model_save_dir, setup_mode, train_mode_str, hparam_candidate, seed_list, ema_decay):
    scripts = []
    script_file = "train_for_more_tps.py"
    if ema_decay:
        script_file += f" --ema_decay {ema_decay}"
    for seed in seed_list:
        seed_str = f"--seed {seed}" if seed else ""
        script = f"python {script_file} --setup_mode {setup_mode} --train_mode {train_mode_str} {seed_str} --hparam_candidate {hparam_candidate} --data_dir {data_dir} --model_save_dir {model_save_dir} --result_dir {result_dir}"
        scripts.append(script)
    return scripts

def prepare_scripts_for_time_1(data_dir,
                               result_dir,
                               model_save_dir,
                               setup_mode,
                               train_mode_str,
                               hparam_candidate,
                               seed_list,
                               ema_decay,
                               hparam_strs,
                               semi_supervised_alg=None,
                               partial_feedback_mode=None,
                               leco_mode='upper_bound',
                               hierarchical_semi_supervision=None,
                               pl_threshold=None):
    assert len(hparam_strs) == 1
    scripts = []
    train_file = "train_for_more_tps.py"
    script_file = f"{train_file} --leco_mode {leco_mode} "
    if ema_decay:
        script_file += f" --ema_decay {ema_decay}"
    script_file += f" --setup_mode {setup_mode} --train_mode {train_mode_str} --hparam_candidate {hparam_candidate} --data_dir {data_dir} --model_save_dir {model_save_dir} --result_dir {result_dir}"
    script_file += " --hparam_strs " + " ".join(hparam_strs)
    if semi_supervised_alg:
        script_file += f" --semi_supervised_alg {semi_supervised_alg}"
    
    if hierarchical_semi_supervision:
        script_file += f" --hierarchical_ssl {hierarchical_semi_supervision}"
    
    if pl_threshold:
        script_file += f" --pl_threshold {pl_threshold}"
        
    if partial_feedback_mode:
        script_file += f" --partial_feedback_mode {partial_feedback_mode}"
        
    for seed in seed_list:
        seed_str = f"--seed {seed}" if seed else ""
        script = f"python {script_file} {seed_str}"
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

def save_tp_res(print_result_dir_tp1_to_3, 
                configuration_dict_as_key,
                res):
    save_dir = os.path.join(print_result_dir_tp1_to_3, "best")
    makedirs(save_dir)
    write_path = os.path.join(save_dir, f'tp_{tp_idx}.txt')
    train_mode_str = configuration_dict_as_key['train_mode_str']
    semi_supervised_alg = configuration_dict_as_key['semi_supervised_alg']
    partial_feedback_mode = configuration_dict_as_key['partial_feedback_mode']
    leco_mode = configuration_dict_as_key['leco_mode']
    hierarchical_semi_supervision = configuration_dict_as_key['hierarchical_semi_supervision']
    pl_threshold = configuration_dict_as_key['pl_threshold']
    sampling = configuration_dict_as_key['sampling']
            
    assert len(res) >= 2
    all_headers = ['Train mode', 'SSL Alg', 'Head Mode', 'LECO Mode', 'Hier-SSL', 'PL-Thre', 'Sample']
    row = [train_mode_str, semi_supervised_alg, partial_feedback_mode, leco_mode, hierarchical_semi_supervision, pl_threshold, sampling]
    for tp_idx in range(len(res)-1, 0, -1):
        all_headers += [f'Best Hparam {tp_idx}', f'TP{tp_idx} Ckpt Epoch', f'TP{tp_idx} Train Acc', f'TP{tp_idx} Test Acc']

        curr_result, curr_best_hparam = res[tp_idx]
            
        curr_epoch = get_mean_std_from_dict(curr_result['best_epoch'], formatter="f")
        curr_train_acc = get_mean_std_from_dict(curr_result['train_acc'])
        curr_test_acc = get_mean_std_from_dict(curr_result['test_acc'])
        if LATEX_FORMAT:
            curr_train_acc = latex_str(curr_train_acc)
            curr_test_acc = latex_str(curr_test_acc)
        
        row += [curr_best_hparam, curr_epoch, curr_train_acc, curr_test_acc]
        if tp_idx == len(res) - 1:
            all_headers += [f'Mask rate', f'Impurity', f'Coarse Acc', f'Coarse Acc (Masked)', 'Mask Rate (filter)', 'Impurity (filter)']
            curr_mask_rate = get_mean_std_from_dict(curr_result['mask_rate'])
            curr_impurity = get_mean_std_from_dict(curr_result['impurity'])
            curr_coarse_accuracy = get_mean_std_from_dict(curr_result['coarse_accuracy'])
            curr_coarse_accuracy_masked = get_mean_std_from_dict(curr_result['coarse_accuracy_masked'])
            curr_mask_rate_filtered = get_mean_std_from_dict(curr_result['mask_rate_filtered'])
            curr_impurity_filtered = get_mean_std_from_dict(curr_result['impurity_filtered'])
            row += [latex_str(curr_mask_rate), latex_str(curr_impurity), latex_str(curr_coarse_accuracy), latex_str(curr_coarse_accuracy_masked), latex_str(curr_mask_rate_filtered), latex_str(curr_impurity_filtered)]
    
    with open(write_path, "w+") as file:
        file.write(tabulate([row], headers=all_headers, tablefmt='orgtbl'))
        # print(f"Save at {write_path}")

def gather_exp(data_dir: str,
               result_dir: str,
               model_save_dir : str,
               sampling='half',
               ema_decay=None,
               PARTIAL_FEEDBACK_MODE=PARTIAL_FEEDBACK_MODE,
               SEMI_SUPERVISED_ALG=SEMI_SUPERVISED_ALG,
               hparam_candidate='cifar',
               train_mode_list=TRAIN_MODE_LIST,
               seed_list=SEED_LIST,
               PL_THRESHOLDS=PL_THRESHOLDS,
               HIERARCHICAL_SEMI_SUPERVISION=HIERARCHICAL_SEMI_SUPERVISION,
               LECO_MODES=LECO_MODES):
    
    print_result_dir = os.path.join(result_dir, 'results_4_tps')
    if ema_decay:
        print_result_dir = os.path.join(print_result_dir, get_exp_str_from_ema_decay(ema_decay))
    makedirs(print_result_dir)
    
    setup_list = SETUP_LIST
    t0_all_hparam_candidates = hparams.HPARAM_CANDIDATES[hparam_candidate]
    all_hparam_candidates = {
        tp_idx : hparams.HPARAM_CANDIDATES[hparam_candidate]
        for tp_idx in range(1,4)
    }
    leco_modes_list = LECO_MODES
    
    result_dict = {}
    for setup_mode in setup_list:
        result_dict[setup_mode] = {}
        setup_dirs = [get_setup_dir(result_dir, setup_mode, seed) for seed in seed_list]
        dataset_paths = [os.path.join(setup_dir, 'dataset.pt') for setup_dir in setup_dirs]
        is_ready = True
        for dataset_path in dataset_paths:
            if not os.path.exists(dataset_path):
                is_ready = False
        if not is_ready:
            print(f"{setup_mode} setup has not been created.")
            print(f"Please run the below {len(seed_list)} scripts first:")
            for seed in seed_list:
                if seed:
                    seed_str = f"--seed {seed}"
                else:
                    seed_str = ""
                print(f'\tpython train_for_more_tps.py --data_dir {data_dir} --setup_mode {setup_mode} {seed_str}')
            continue
        
        dataset = load_pickle(dataset_path)
        _, _, all_tp_info, _ = dataset
        
        scripts_to_run = []
        for train_mode_str in train_mode_list:
            train_mode = configs.TRAIN_MODES[train_mode_str]
            hparam_strs = [] # Best hparam per time period
            
            ## Find best candidate for TP 0
            all_results = {}
            best_hparam_str = None
            best_test_acc_mean = None
            all_hparam_candidates_are_ready = True
            
            for hparam_str in t0_all_hparam_candidates:
                tp_results = []
                for setup_dir in setup_dirs:
                    train_dir = get_train_dir(setup_dir,
                                              train_mode,
                                              ema_decay,
                                              [hparam_str],
                                              tp_idx=0)
                    exp_result_path = os.path.join(train_dir,
                                                   'result.ckpt')
                    if os.path.exists(exp_result_path):
                        try:
                            exp_result = load_pickle(exp_result_path)
                        except:
                            print(exp_result_path + " truncated?")
                            import pdb; pdb.set_trace()
                            # exp_result = load_pickle(exp_result_path)
                            all_hparam_candidates_are_ready = False
                            break
                        best_epoch = exp_result['best_result']['best_epoch']
                        # best_stats = exp_result['best_result']['best_stat']
                        train_acc = exp_result['acc_result']['train']
                        test_acc = exp_result['acc_result']['test']
                        tp_results.append({
                            'best_epoch' : best_epoch,
                            'train_acc' : train_acc,
                            'test_acc' : test_acc
                        })
                        if hparam_candidate == 'inat':
                            print(f"{hparam_str} {setup_mode} {train_mode_str}: Train {train_acc} and test acc {test_acc}")
                    else:
                        all_hparam_candidates_are_ready = False
                        break
                
                if not all_hparam_candidates_are_ready:
                    break
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
            
            if all_hparam_candidates_are_ready:
                # write the results
                if not train_mode_str in result_dict[setup_mode]:
                    result_dict[setup_mode][train_mode_str] = []
                # print(f"For setup {setup_mode}, train mode {train_mode_str}, tp 0, the best hparam is {best_hparam_str} with test acc {best_test_acc_mean}")
                hparam_strs += [best_hparam_str]
                result_dict[setup_mode][train_mode_str].append({
                    'tp_idx' : 0,
                    'ema_decay' : ema_decay,
                    'best_hparam_str' : best_hparam_str,
                    'hparam_strs' : copy.deepcopy(hparam_strs), # up to this tp_idx
                    'all_results' : copy.deepcopy(all_results),
                })
                save_all_results(print_result_dir, result_dict, setup_mode, all_tp_info)
        
                save_avg_results(print_result_dir, result_dict, setup_mode, all_tp_info)
            
                save_best_results(print_result_dir, result_dict, setup_mode, all_tp_info)
            else:
                # prepare scripts for this tp_idx
                current_scripts = prepare_scripts(
                    data_dir,
                    result_dir,
                    model_save_dir,
                    setup_mode,
                    train_mode_str,
                    hparam_candidate,
                    seed_list,
                    ema_decay,
                )
                scripts_to_run += current_scripts
                print(f"Setup {setup_mode}: {len(current_scripts)} scripts for train mode {train_mode_str}.")
                print(f"Caveat: This list may be redundant since some train_mode has same setup for TP0")
                import pdb; pdb.set_trace()
                continue
            
            best_hparam_list = [best_hparam_str]
            print(f"Best hparam for time 0 is {best_hparam_list}")
            ### For TP 1-3
            for leco_mode in leco_modes_list:
                if leco_mode in ['relabel_old', 'upper_bound']:
                    partial_feedback_mode_list = [None]
                    semi_supervised_alg_list = [None]
                elif leco_mode in ['label_new']:
                    partial_feedback_mode_list = PARTIAL_FEEDBACK_MODE
                    semi_supervised_alg_list = SEMI_SUPERVISED_ALG
                # elif leco_mode in ['upper_bound_with_multi_task']:
                #     partial_feedback_mode_list = PARTIAL_FEEDBACK_MODE
                #     semi_supervised_alg_list = [None]
                else:
                    raise NotImplementedError()
                for partial_feedback_mode in partial_feedback_mode_list:
                    for semi_supervised_alg in semi_supervised_alg_list:
                        if semi_supervised_alg:
                            hierarchical_semi_supervision_list = HIERARCHICAL_SEMI_SUPERVISION
                            if semi_supervised_alg in ['Fixmatch', 'PL']:
                                pl_thresholds_list = PL_THRESHOLDS
                            elif semi_supervised_alg in ['DistillHard', 'DistillSoft']:
                                pl_thresholds_list = [None]
                            else:
                                raise NotImplementedError()
                        else:
                            hierarchical_semi_supervision_list = [None]
                            pl_thresholds_list = [None]
                        
                        for hierarchical_semi_supervision in hierarchical_semi_supervision_list:
                            for pl_threshold in pl_thresholds_list:
                                ssl_str = f"ssl_{semi_supervised_alg}"
                                if hierarchical_semi_supervision is not None:
                                    ssl_str += f"_{hierarchical_semi_supervision}"
                                if pl_threshold is not None:
                                    ssl_str += f"_thre_{pl_threshold}"
                                print_result_dir_tp1_to_3 = os.path.join(
                                    print_result_dir,
                                    setup_mode,
                                    train_mode_str,
                                    leco_mode,
                                    f"partial_{partial_feedback_mode}",
                                    ssl_str,
                                    f"sampling_{sampling}",
                                )
                                makedirs(print_result_dir_tp1_to_3)
                                curr_best_hparam_list = copy.deepcopy(best_hparam_list)
                                curr_results = []
                                while len(curr_best_hparam_list) < len(all_tp_info):
                                    tp_idx = len(curr_best_hparam_list)
                                    all_results = {}
                                    best_hparam_str = None
                                    best_test_acc_mean = None
                                    all_hparam_candidates_are_ready = True
                                    
                                    for hparam_str in all_hparam_candidates[tp_idx]:
                                        tp_results = [] # all seeds results
                                        for setup_dir in setup_dirs:
                                            train_semi_supervised_dir = get_semi_supervised_dir(
                                                setup_dir,
                                                ema_decay=ema_decay,
                                                train_mode=train_mode,
                                                leco_mode=leco_mode,
                                                hparam_list=curr_best_hparam_list + [hparam_str],
                                                semi_supervised_alg=semi_supervised_alg,
                                                pl_threshold=pl_threshold,
                                                partial_feedback_mode=partial_feedback_mode,
                                                hierarchical_ssl=hierarchical_semi_supervision,
                                                sampling=sampling,
                                                tp_idx=tp_idx
                                            )
                                            exp_result_path = os.path.join(train_semi_supervised_dir,
                                                                           'result.ckpt')
                                            if os.path.exists(exp_result_path):
                                                try:
                                                    exp_result = load_pickle(exp_result_path)
                                                except:
                                                    print(exp_result_path + " truncated?")
                                                    import pdb; pdb.set_trace()
                                                    all_hparam_candidates_are_ready = False
                                                    break
                                                best_epoch = exp_result['best_result']['best_epoch']
                                                best_stats = exp_result['best_result']['best_stat']
                                                train_acc = exp_result['acc_result']['train']
                                                test_acc = exp_result['acc_result']['test']
                                                tp_results.append({
                                                    'best_epoch' : best_epoch,
                                                    'mask_rate' : best_stats['mask_rate'],
                                                    'impurity' : best_stats['impurity'],
                                                    'coarse_accuracy' : best_stats['coarse_accuracy'],
                                                    'coarse_accuracy_masked' : best_stats['coarse_accuracy_masked'],
                                                    'mask_rate_filtered' : best_stats['mask_rate_filtered'],
                                                    'impurity_filtered' : best_stats['impurity_filtered'],
                                                    'train_acc' : train_acc,
                                                    'test_acc' : test_acc
                                                })
                                                if hparam_candidate == 'inat':
                                                    print(f"TP{tp_idx}: {hparam_str} {setup_mode} {train_mode_str} {leco_mode} {sampling}: Train {train_acc} and test acc {test_acc}")
                                            else:
                                                all_hparam_candidates_are_ready = False
                                                # if hparam_candidate == 'inat':
                                                #     # import pdb; pdb.set_trace()
                                                #     continue
                                                break
                                        
                                        # if hparam_candidate == 'inat':
                                        #     continue
                                        if not all_hparam_candidates_are_ready:
                                            break
                                        
                                        # summarize the results and put into all_results
                                        best_epoch_mean, best_epoch_std = mean_std_from_dict(tp_results, 'best_epoch')
                                        train_acc_mean, train_acc_std = mean_std_from_dict(tp_results, 'train_acc')
                                        test_acc_mean, test_acc_std = mean_std_from_dict(tp_results, 'test_acc')
                                        mask_rate_mean, mask_rate_std = mean_std_from_dict(tp_results, 'mask_rate')
                                        impurity_mean, impurity_std = mean_std_from_dict(tp_results, 'impurity')
                                        coarse_accuracy_mean, coarse_accuracy_std = mean_std_from_dict(tp_results, 'coarse_accuracy')
                                        coarse_accuracy_masked_mean, coarse_accuracy_masked_std = mean_std_from_dict(tp_results, 'coarse_accuracy_masked')
                                        mask_rate_filtered_mean, mask_rate_filtered_std = mean_std_from_dict(tp_results, 'mask_rate_filtered')
                                        impurity_filtered_mean, impurity_filtered_std = mean_std_from_dict(tp_results, 'impurity_filtered')
                                        all_results[hparam_str] = {
                                            'best_epoch' : {'mean' : best_epoch_mean,
                                                            'std'  : best_epoch_std},
                                            'train_acc' : {'mean' : train_acc_mean,
                                                            'std'  : train_acc_std},
                                            'test_acc' : {'mean' : test_acc_mean,
                                                            'std'  : test_acc_std},
                                            'mask_rate' : {'mean' : mask_rate_mean,
                                                            'std'  : mask_rate_std},
                                            'impurity' : {'mean' : impurity_mean,
                                                            'std'  : impurity_std},
                                            'coarse_accuracy' : {'mean' : coarse_accuracy_mean,
                                                                    'std'  : coarse_accuracy_std},
                                            'coarse_accuracy_masked' : {'mean' : coarse_accuracy_masked_mean,
                                                                        'std'  : coarse_accuracy_masked_std},
                                            'mask_rate_filtered' : {'mean' : mask_rate_filtered_mean,
                                                                    'std'  : mask_rate_filtered_std},
                                            'impurity_filtered' : {'mean' : impurity_filtered_mean,
                                                                    'std'  : impurity_filtered_std},
                                            'tp_results' : tp_results,
                                        }
                                        if best_test_acc_mean is None or test_acc_mean > best_test_acc_mean:
                                            best_test_acc_mean = test_acc_mean
                                            best_hparam_str = hparam_str
                                    
                                    
                                    configuration_dict_as_key = {
                                        'print_result_dir_tp1_to_3' : print_result_dir_tp1_to_3,
                                        'train_mode_str' : train_mode_str,
                                        'semi_supervised_alg' : semi_supervised_alg,
                                        'partial_feedback_mode' : partial_feedback_mode,
                                        'leco_mode' : leco_mode,
                                        'hierarchical_semi_supervision' : hierarchical_semi_supervision,
                                        'pl_threshold' : pl_threshold,
                                        'sampling' : sampling,
                                    }
                                    if all_hparam_candidates_are_ready:
                                        # all haparams and all seeds are ready
                                        # write the results
                                        assert best_hparam_list[0] == result_dict[setup_mode][train_mode_str][0]['best_hparam_str']
                                        t_0_result = result_dict[setup_mode][train_mode_str][0]['all_results'][best_hparam_list[0]]
                                        for idx in range(len(best_hparam_list)-1):
                                            assert best_hparam_list[idx] == result_dict[setup_mode][train_mode_str][idx][semi_supervised_alg][partial_feedback_mode][leco_mode][hierarchical_semi_supervision][pl_threshold][sampling]['best_hparam_str']
                                        curr_best_hparam_list.append(best_hparam_str)
                                        curr_results.append(all_results[best_hparam_str])
                                        res = [(t_0_result, best_hparam_list[0])]
                                        for hparam_idx, hparam in enumerate(curr_best_hparam_list[1:]):
                                            res.append([curr_results[hparam_idx], curr_best_hparam_list[hparam_idx+1]])
                                            save_tp_res(
                                                os.path.join(print_result_dir_tp1_to_3, f"tp_{tp_idx}"),
                                                configuration_dict_as_key,
                                                res
                                            )
                                    else:
                                        # prepare scripts for this tp_idx
                                        current_scripts = prepare_scripts_for_time_1(
                                            data_dir,
                                            result_dir,
                                            model_save_dir,
                                            setup_mode,
                                            train_mode_str,
                                            hparam_candidate,
                                            seed_list,
                                            ema_decay,
                                            best_hparam_list,
                                            semi_supervised_alg=semi_supervised_alg,
                                            partial_feedback_mode=partial_feedback_mode,
                                            leco_mode=leco_mode,
                                            hierarchical_semi_supervision=hierarchical_semi_supervision,
                                            pl_threshold=pl_threshold,
                                        )
                                        scripts_to_run += current_scripts
                                        # print(f"Setup {setup_mode}: {len(current_scripts)} scripts for train mode {train_mode_str} and config {configuration_dict_as_key}.")
                                        break
                
            
        print(f"Setup {setup_mode}: Total {len(scripts_to_run)} scripts.")
        write_scripts_to_file(scripts_to_run, print_result_dir, setup_mode)
        

if __name__ == '__main__':
    args = argparser.parse_args()

    gather_exp(args.data_dir,
               args.result_dir,
               args.model_save_dir,
               sampling='half',
               ema_decay=args.ema_decay,
               hparam_candidate=args.hparam_candidate)
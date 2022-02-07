# 0-13: python collect_results.py --hparam_candidate cifar --train_mode_candidate cifar --data_dir /ssd1/leco/ --model_save_dir /data3/zhiqiul/self_supervised_models/resnet18/

# 0-13: python collect_results.py --partial_feedback_mode partial_feedback --hparam_candidate cifar --train_mode_candidate cifar --data_dir /ssd1/leco/ --model_save_dir /data3/zhiqiul/self_supervised_models/resnet18/
# 0-13: python collect_results.py --partial_feedback_mode partial_feedback_weight_history_0 --hparam_candidate cifar --train_mode_candidate cifar --data_dir /ssd1/leco/ --model_save_dir /data3/zhiqiul/self_supervised_models/resnet18/
# 0-13: python collect_results.py --partial_feedback_mode partial_feedback_weight_history_2 --hparam_candidate cifar --train_mode_candidate cifar --data_dir /ssd1/leco/ --model_save_dir /data3/zhiqiul/self_supervised_models/resnet18/
# 0-13: python collect_results.py --partial_feedback_mode partial_feedback_weight_history_0.5 --hparam_candidate cifar --train_mode_candidate cifar --data_dir /ssd1/leco/ --model_save_dir /data3/zhiqiul/self_supervised_models/resnet18/
# 0-13: python collect_results.py --partial_feedback_mode log_in_partial_feedback --hparam_candidate cifar --train_mode_candidate cifar --data_dir /ssd1/leco/ --model_save_dir /data3/zhiqiul/self_supervised_models/resnet18/
# 0-13: python collect_results.py --partial_feedback_mode log_in_partial_feedback_weight_history_0 --hparam_candidate cifar --train_mode_candidate cifar --data_dir /ssd1/leco/ --model_save_dir /data3/zhiqiul/self_supervised_models/resnet18/
# 0-13: python collect_results.py --partial_feedback_mode log_in_partial_feedback_weight_history_2 --hparam_candidate cifar --train_mode_candidate cifar --data_dir /ssd1/leco/ --model_save_dir /data3/zhiqiul/self_supervised_models/resnet18/
# 0-13: python collect_results.py --partial_feedback_mode log_in_partial_feedback_weight_history_0.5 --hparam_candidate cifar --train_mode_candidate cifar --data_dir /ssd1/leco/ --model_save_dir /data3/zhiqiul/self_supervised_models/resnet18/



# 0-15: python collect_results.py --hparam_candidate inat --train_mode_candidate inat --data_dir /ssd1/leco/ --model_save_dir /data3/zhiqiul/self_supervised_models/inat2021_resnet50/
import os
import argparse
import shutil
import random
import numpy as np
import configs
from util import load_pickle, makedirs
import hparams
import setups
from tabulate import tabulate
from print_utils import get_exp_str_from_train_mode, get_exp_str_from_hparam_strs, get_exp_str_from_partial_feedback
import configs
import copy

argparser = argparse.ArgumentParser()
argparser.add_argument("--data_dir", 
                        default='/ssd1/leco/',
                        help="Where the dataset are saved.")
argparser.add_argument("--model_save_dir", 
                        default='/data3/zhiqiul/self_supervised_models/resnet18/',
                        help="Where the self-supervised pre-trained models will be saved.")
argparser.add_argument("--partial_feedback_mode", 
                        default=None, type=str,
                        help="If used partial feedback, input the mode")
argparser.add_argument("--hparam_candidate",
                        type=str,
                        default='cifar',
                        choices=hparams.HPARAM_CANDIDATES.keys(),
                        help="The hyperparameter candidates (str) for next time period")
argparser.add_argument("--train_mode_candidate",
                        type=str,
                        default='cifar',
                        choices=configs.ALL_TRAIN_MODES.keys(),
                        help="The train mode candidates for this setup")
SEED_LIST = [None, 1, 10, 100, 1000]

def gather_exp(data_dir: str,
               model_save_dir : str,
               partial_feedback_mode=None,
               hparam_candidate='cifar',
               train_mode_candidate='cifar',
               seed_list=SEED_LIST):

    setup_list = list(setups.SETUPS.keys())
    train_mode_list = configs.ALL_TRAIN_MODES[train_mode_candidate]
    hparam_str_list = list(hparams.HPARAMS.keys())

    result_dirs = []
    for setup_mode in setup_list:
        setup_dirs = [os.path.join(data_dir, setup_mode, f"seed_{str(seed)}") for seed in seed_list]
        dataset_paths = [os.path.join(setup_dir, 'dataset.pt') for setup_dir in setup_dirs]
        is_ready = True
        for dataset_path in dataset_paths:
            if not os.path.exists(dataset_path):
                is_ready = False
        if not is_ready:
            print(f"{setup_mode} setup has not started.")
            continue
        
        for train_mode_str in train_mode_list:
            train_mode = configs.TRAIN_MODES[train_mode_str]
            for setup_dir in setup_dirs:
                exp_result_path = os.path.join(setup_dir,
                                            get_exp_str_from_train_mode(train_mode, tp_idx=1),
                                            get_exp_str_from_partial_feedback(partial_feedback_mode, tp_idx=1))
                if os.path.exists(exp_result_path):
                    # import pdb; pdb.set_trace()
                    result_dirs.append(exp_result_path)
                    shutil.rmtree(exp_result_path)
    print(result_dirs)

if __name__ == '__main__':
    args = argparser.parse_args()
    gather_exp(args.data_dir,
               args.model_save_dir,
               partial_feedback_mode=args.partial_feedback_mode,
               hparam_candidate=args.hparam_candidate,
               train_mode_candidate=args.train_mode_candidate)
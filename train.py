import os
import argparse
import random
import numpy as np
import torch
import configs
from utils import load_pickle, save_obj_as_pickle

argparser = argparse.ArgumentParser()
argparser.add_argument("--data_dir", 
                        default='/scratch/leco/',
                        help="Where the dataset will be saved.")
argparser.add_argument("--model_save_dir", 
                        default='/data3/zhiqiul/self_supervised_models/resnet18/',
                        help="Where the self-supervised pre-trained models will be saved.")
argparser.add_argument("--setup_mode",
                        type=str,
                        default='cifar10_buffer_2000',
                        choices=setups.SETUPS.keys(),
                        help="The dataset setup mode")  
argparser.add_argument("--train_mode",
                        type=str,
                        default=None,
                        choices=configs.TRAIN_MODES.keys(),
                        help="The train mode")   


def start_experiment(data_dir, model_save_dir, setup_mode_str: str, train_mode_str: str, seed=None):
    seed_str = f"seed_{seed}"
    if seed == None:
        print("Not using a random seed")
    else:
        print(f"Using random seed {seed}")
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    dataset_path = os.path.join(data_dir, setup_mode_str, seed_str, 'dataset.pt')
    if os.path.exists(dataset_path):
        dataset = load_pickle(dataset_path)
    else:
        setup_mode = setups.SETUPS[setup_mode_str]
        dataset = setups.generate_dataset(data_dir, setup_mode)
        save_obj_as_pickle(dataset_path, dataset)
        print(f"Dataset saved at {dataset_path}")

    train_subsets, testset = dataset
    
    train_mode = configs.TRAIN_MODES[train_mode]
    
    # how about dataloader and optimizer and learning rate?
    

if __name__ == '__main__':
    args = argparser.parse_args()
    start_experiment(train_mode)
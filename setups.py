import torch
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms
import random
import os
import datasets

class Setup():
    def __init__(self, dataset_name, tp_buffers=[2000, 2000], replace=False):
        self.dataset_name = dataset_name
        self.tp_buffers = tp_buffers
        self.replace = replace # whether to sample with replacement per time period

SETUPS = {
    'cifar10_weakaug_train_2000_val_500' : Setup(
        'CIFAR10WeakAug',
        tp_buffers=[(2000, 500), (2000, 500)], # each element is a tuple of (train_set_size:int, val_set_size:int)
        replace=False
    ),
    'cifar10_strongaug_train_2000_val_500' : Setup(
        'CIFAR10StrongAug',
        tp_buffers=[(2000, 500), (2000, 500)], # each element is a tuple of (train_set_size:int, val_set_size:int)
        replace=False
    ),
    # 'semi_inat_buffer_1000_380' : Setup(
    #     'SemiInat2021',
    #     tp_buffers=[
    #         (1000, 380),
    #         (1000, 380),
    #         (1000, 380),
    #         (1000, 380),
    #         (1000, 380),
    #         (1000, 380),
    #         (1000, 380),
    #     ],
    #     replace=False
    # ),
    # 'semi_inat_buffer_7721_2000_replace' : Setup(
    #     'SemiInat2021',
    #     tp_buffers=[
    #         (7721, 2000),
    #         (7721, 2000),
    #         (7721, 2000),
    #         (7721, 2000),
    #         (7721, 2000),
    #         (7721, 2000),
    #         (7721, 2000),
    #     ],
    #     replace=True
    # ),
}

class HierarchyDataset(Dataset):
    def __init__(self, dataset, leaf_idx_to_all_class_idx):
        self.dataset = dataset
        self.leaf_idx_to_all_class_idx = leaf_idx_to_all_class_idx
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,index):
        sample, leaf_idx = self.dataset[index]
        all_class_idx = self.leaf_idx_to_all_class_idx[leaf_idx]
        return sample, all_class_idx

def get_superclass_to_subclass(leaf_idx_to_all_class_idx):
    # superclass_to_subclass[sub_class_time][super_class_time][super_class_idx]
    # is the set of indices in sub_class_time that correspond to the superclass
    num_of_levels = len(leaf_idx_to_all_class_idx[list(leaf_idx_to_all_class_idx.keys())[0]])
    superclass_to_subclass = {}
    for tp_idx in range(num_of_levels-1, -1, -1):
        superclass_to_subclass[tp_idx] = {}
        for super_class_time in range(tp_idx+1):
            superclass_to_subclass[tp_idx][super_class_time] = {}
            for leaf_idx in leaf_idx_to_all_class_idx:
                sub_class_idx = leaf_idx_to_all_class_idx[leaf_idx][tp_idx]
                super_class_idx = leaf_idx_to_all_class_idx[leaf_idx][super_class_time]
                if not super_class_idx in superclass_to_subclass[tp_idx][super_class_time]:
                    superclass_to_subclass[tp_idx][super_class_time][super_class_idx] = [sub_class_idx]
                elif not sub_class_idx in superclass_to_subclass[tp_idx][super_class_time][super_class_idx]:
                    superclass_to_subclass[tp_idx][super_class_time][super_class_idx].append(sub_class_idx)
    return superclass_to_subclass


def generate_dataset(data_dir, setup : Setup, annotation_file=''):
    print(f"==> Preparing {setup.dataset_name} data..")
    dataset = getattr(datasets, setup.dataset_name)(data_dir)
    trainset, testset = dataset.get_dataset()

    all_tp_info, leaf_idx_to_all_class_idx = dataset.get_class_hierarchy()
    trainset = HierarchyDataset(trainset, leaf_idx_to_all_class_idx)
    testset = HierarchyDataset(testset, leaf_idx_to_all_class_idx)
    print(f"Length of trainset is {len(trainset)}")
    
    train_val_subsets = []
    if setup.replace:
    # if setup.replace == True:
        print('Sample with replacement!')
        len_of_trainset = len(trainset)
        
        for _, (tp_buffer_train, tp_buffer_val) in enumerate(setup.tp_buffers):
            indices_trainset = list(range(len_of_trainset))
            random.shuffle(indices_trainset)
            indices_tp_train, indices_tp_val = indices_trainset[:tp_buffer_train], indices_trainset[tp_buffer_train:tp_buffer_train+tp_buffer_val]
            train_val_subsets.append((torch.utils.data.Subset(trainset, indices_tp_train), torch.utils.data.Subset(trainset, indices_tp_val)) )
    else:
        len_of_trainset = len(trainset)
        indices_trainset = list(range(len_of_trainset))
        random.shuffle(indices_trainset)
        
        for _, (tp_buffer_train, tp_buffer_val) in enumerate(setup.tp_buffers):
            indices_tp_train, indices_tp_val = indices_trainset[:tp_buffer_train], indices_trainset[tp_buffer_train:tp_buffer_train+tp_buffer_val]
            indices_trainset = indices_trainset[tp_buffer_train+tp_buffer_val:]
            train_val_subsets.append((torch.utils.data.Subset(trainset, indices_tp_train), torch.utils.data.Subset(trainset, indices_tp_val)) )
    return train_val_subsets, testset, all_tp_info, leaf_idx_to_all_class_idx


if __name__ == "__main__":
    import pdb; pdb.set_trace()
    generate_dataset('/scratch/leco/', SETUPS['cifar10_buffer_2000_500_same_image_same_model'])
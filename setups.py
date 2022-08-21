import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import random
from copy import deepcopy

import datasets


class Setup():
    def __init__(self, dataset_name, tp_buffers=[2000, 2000], replace=False):
        self.dataset_name = dataset_name
        self.tp_buffers = tp_buffers


SETUPS = {
    # 'cifar10_weakaug_train_2000_val_500': Setup(
    #     'CIFAR10WeakAug',
    #     tp_buffers=[(2000, 500), (2000, 500)], # each element is a tuple of (train_set_size:int, val_set_size:int)
    # ),
    'cifar10_strongaug_train_2000_val_500': Setup(
        'CIFAR10StrongAug',
        tp_buffers=[(2000, 500), (2000, 500)], # each element is a tuple of (train_set_size:int, val_set_size:int)
    ),
    'cifar100_strongaug_train_2000_val_500': Setup(
        'CIFAR100StrongAug',
        tp_buffers=[(2000, 500), (2000, 500)], # each element is a tuple of (train_set_size:int, val_set_size:int)
    ),
    'cifar100_strongaug_train_1000_val_250': Setup(
        'CIFAR100StrongAug',
        tp_buffers=[(1000, 250), (1000, 250)], # each element is a tuple of (train_set_size:int, val_set_size:int)
    ),
    'cifar100_strongaug_train_10000_val_2500': Setup(
        'CIFAR100StrongAug',
        tp_buffers=[(10000, 2500), (10000, 2500)], # each element is a tuple of (train_set_size:int, val_set_size:int)
    ),
    # 'semi_inat_weakaug': Setup(
    #     'SemiInat2021WeakAug',
    #     tp_buffers=None,
    # ),
    'semi_inat_strongaug': Setup(
        'SemiInat2021StrongAug',
        tp_buffers=None,
    ),
    'semi_inat_strongaug_4_tps': Setup(
        'SemiInat2021StrongAug4TPs',
        tp_buffers=None,
    ),
}

class _TransformWeakStrong(object):
    def __init__(self, weak, strong):
        self.weak = weak
        self.strong = strong

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return weak, strong

class HierarchyDataset(Dataset):
    def __init__(self,
                 dataset,
                 leaf_idx_to_all_class_idx,
                 weak_transform,
                 strong_transform,
                 test_transform):
        self.dataset = dataset
        self.leaf_idx_to_all_class_idx = leaf_idx_to_all_class_idx
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        self.weak_strong_transform = _TransformWeakStrong(
            weak_transform,
            strong_transform
        )
        self.test_transform = test_transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,index):
        sample, leaf_idx = self.dataset[index]
        all_class_idx = self.leaf_idx_to_all_class_idx[leaf_idx]
        return sample, all_class_idx
    
    def update_transform(self, transform):
        self.dataset.transform = transform
    
    def set_test_transform(self):
        self.dataset.transform = self.test_transform

class ConcatHierarchyDataset(Dataset):
    def __init__(self,
                 hierarchy_dataset_list):
        self.concat_set = torch.utils.data.ConcatDataset(hierarchy_dataset_list)
        self.hierarchy_dataset_list = hierarchy_dataset_list
        self.leaf_idx_to_all_class_idx = hierarchy_dataset_list[0].leaf_idx_to_all_class_idx
        self.weak_transform = hierarchy_dataset_list[0].weak_transform
        self.strong_transform = hierarchy_dataset_list[0].strong_transform
        self.weak_strong_transform = hierarchy_dataset_list[0].weak_strong_transform
        self.test_transform = hierarchy_dataset_list[0].test_transform
    
    def __len__(self):
        return len(self.concat_set)
    
    def __getitem__(self,index):
        return self.concat_set[index]

    def update_transform(self, transform):
        for hierarchy_dataset in self.hierarchy_dataset_list:
            hierarchy_dataset.update_transform(transform)
    
    def set_test_transform(self):
        for hierarchy_dataset in self.hierarchy_dataset_list:
            hierarchy_dataset.update_transform(self.test_transform)

class SubsetHierarchyDataset(Dataset):
    def __init__(self,
                 hierarchy_dataset,
                 indices):
        assert isinstance(hierarchy_dataset, HierarchyDataset)
        self.hierarchy_dataset = hierarchy_dataset
        self.subset = torch.utils.data.Subset(hierarchy_dataset, indices)
        # Below are the same as hierarchy_dataset
        self.leaf_idx_to_all_class_idx = hierarchy_dataset.leaf_idx_to_all_class_idx
        self.weak_transform = hierarchy_dataset.weak_transform
        self.strong_transform = hierarchy_dataset.strong_transform
        self.weak_strong_transform = hierarchy_dataset.weak_strong_transform
        self.test_transform = hierarchy_dataset.test_transform
    
    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self,index):
        return self.subset[index]
    
    def update_transform(self, transform):
        self.hierarchy_dataset.dataset.transform = transform
    
    def set_test_transform(self):
        self.hierarchy_dataset.dataset.transform = self.test_transform


def download_dataset(data_dir, setup: Setup):
    print(f"==> Preparing {setup.dataset_name} data..")
    dataset = getattr(datasets, setup.dataset_name)(data_dir)


def generate_dataset(data_dir, setup: Setup):
    print(f"==> Preparing {setup.dataset_name} data..")
    dataset = getattr(datasets, setup.dataset_name)(data_dir)
    all_tp_info, leaf_idx_to_all_class_idx = dataset.get_class_hierarchy()
    trainset, testset = dataset.get_dataset()
    weak_transform, strong_transform = dataset.get_weak_and_strong_transform()
    test_transform = dataset.get_transform_test()
    trainset = HierarchyDataset(trainset, leaf_idx_to_all_class_idx, weak_transform, strong_transform, test_transform)
    testset = HierarchyDataset(testset, leaf_idx_to_all_class_idx, weak_transform, strong_transform, test_transform)
    print(f"Length of trainset is {len(trainset)}")
    print(f"Length of testset is {len(testset)}")
    train_val_subsets = []
    train_val_indices = get_train_val_indices(dataset, trainset, tp_buffers=setup.tp_buffers)
    
    for _, (indices_tp_train, indices_tp_val) in enumerate(train_val_indices):
        train_subset = SubsetHierarchyDataset(deepcopy(trainset), indices_tp_train)
        val_subset = SubsetHierarchyDataset(deepcopy(trainset), indices_tp_val)
        val_subset.set_test_transform()
        train_val_subsets.append((train_subset, val_subset))
    
    return train_val_subsets, testset, all_tp_info, leaf_idx_to_all_class_idx


def get_train_val_indices(dataset, trainset, tp_buffers=None):
    """Must use 'random' to sample the dataset for seeding purpose
    """
    train_val_indices = []
    if tp_buffers is not None:
        len_of_trainset = len(trainset)
        indices_trainset = list(range(len_of_trainset))
        random.shuffle(indices_trainset)

        for _, (tp_buffer_train, tp_buffer_val) in enumerate(tp_buffers):
            indices_tp_train, indices_tp_val = indices_trainset[:tp_buffer_train], indices_trainset[tp_buffer_train:tp_buffer_train+tp_buffer_val]
            indices_trainset = indices_trainset[tp_buffer_train+tp_buffer_val:]
            train_val_indices.append((indices_tp_train, indices_tp_val))
    else:
        train_val_indices = dataset.get_train_val_indices(trainset)
    return train_val_indices

if __name__ == "__main__":
    # generate_dataset('/scratch/leco/', SETUPS['semi_inat_weakaug'])
    train_val_subsets, testset, all_tp_info, leaf_idx_to_all_class_idx = generate_dataset(
        '/scratch/leco/', SETUPS['semi_inat_strongaug_4_tps'])

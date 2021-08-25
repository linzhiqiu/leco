import torch
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms
import random
import os

def idx_of_superclass(subclass_idx, class_hierarchy, class_to_idx):
    """
    Args:
        subclass_idx (int) : Index of subclass
        class_hierarchy (dict) : class_hierarchy[int] is the list of subclass names
        class_to_idx (dict) : class_to_idx[subclass_name] is the index of subclass_name
    
    Returns:
        superclass_idx (int) : Index of superclass
    """
    for superclass_idx in class_hierarchy:
        for class_name in class_hierarchy[superclass_idx]:
            if subclass_idx == class_to_idx[class_name]:
                return superclass_idx

class Setup():
    def __init__(self, dataset_name, class_hierarchy, tp_buffers=[2000, 2000]):
        self.dataset_name = dataset_name
        self.class_hierarchy = class_hierarchy
        self.tp_buffers = tp_buffers

CIFAR10_HIERARCHY = {
    0 : ['airplane', 'ship', 'automobile', 'truck'], # vehicle
    1 : ['cat', 'frog', 'bird', 'deer', 'dog', 'horse'] # animal
}

SETUPS = {
    'cifar10_buffer_2000' : Setup(
        'CIFAR10',
        CIFAR10_HIERARCHY,
        tp_buffers=[2000, 2000]
    )
}

class TwoLevelDataset(Dataset):
    def __init__(self, dataset, idx_to_superclass_idx):
        self.dataset = dataset
        self.idx_to_superclass_idx = idx_to_superclass_idx
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,index):
        sample, subclass_idx = self.dataset[index]
        superclass_idx = self.idx_to_superclass_idx[subclass_idx]
        return sample, (superclass_idx, subclass_idx)

def generate_dataset(data_dir, setup : Setup):
    if setup.dataset_name == 'CIFAR10':
        print('==> Preparing CIFAR10 data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=transform_train
        )
        # trainloader = torch.utils.data.DataLoader(
        #     trainset, batch_size=128, shuffle=True, num_workers=2)
        testset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform_test)
        # testloader = torch.utils.data.DataLoader(
        #     testset, batch_size=100, shuffle=False, num_workers=2)

        class_to_idx = testset.class_to_idx
        idx_to_subclass = {class_to_idx[class_name] : class_name for class_name in class_to_idx}
        idx_to_superclass_idx = {idx : idx_of_superclass(idx, setup.class_hierarchy, class_to_idx)
                                 for idx in idx_to_subclass}
        num_of_superclasses = len(setup.class_hierarchy.keys())
        num_of_subclasses = len(idx_to_subclass.keys())
        trainset = TwoLevelDataset(trainset, idx_to_superclass_idx)
        testset = TwoLevelDataset(testset, idx_to_superclass_idx)
        
        len_of_trainset = len(trainset)
        indices_trainset = list(range(len_of_trainset))
        random.shuffle(indices_trainset)
        
        train_subsets = []
        for _, tp_buffer in enumerate(setup.tp_buffers):
            indices_tp, indices_trainset = indices_trainset[:tp_buffer], indices_trainset[tp_buffer:]
            train_subsets.append(torch.utils.data.Subset(trainset, indices_tp))
        return train_subsets, testset, (num_of_superclasses, num_of_subclasses)
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    pass
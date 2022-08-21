import json
import torch
from torch.utils.data import Dataset

import numpy as np
import torchvision
import torchvision.transforms as transforms
import random
import os
from randaugment import RandAugmentMC, RandAugmentInat
import zipfile
import gdown
from tqdm import tqdm
# annotation_file = ./semi_inat/annotation_v2.json

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
inat_mean = (0.485, 0.456, 0.406)
inat_std = (0.229, 0.224, 0.225)

def get_cifar_transform_train_strong_aug():
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        RandAugmentMC(n=2, m=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)]
    )
    return transform_train

def get_cifar_transform_train_weak_aug():
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    return transform_train

def get_cifar_transform_test():
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    return transform_test

def get_inat_transform_train_strong_aug():
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        RandAugmentInat(n=2, m=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=inat_mean, std=inat_std)]
    )
    return transform_train

def get_inat_transform_train_weak_aug():
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=inat_mean, std=inat_std)
    ])
    return transform_train

def get_inat_transform_test():
    transform_test = transforms.Compose([
        transforms.Resize(224), 
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=inat_mean, std=inat_std)
    ])
    return transform_test

class LecoDataset():
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_dataset(self):
        ### Must make sure that the trainset, testset returned by this func has a .transform field
        raise NotImplementedError()
    
    def get_weak_and_strong_transform(self):
        raise NotImplementedError() # For Fixmatch

    def get_class_hierarchy(self):
        leaf_idx_to_all_class_idx = None
        all_tp_info = None
        raise NotImplementedError() # return (all_tp_info, leaf_idx_to_all_class_idx)

def get_cifar_tp_info(cifar_classes,
                      cifar_hierachy,
                      tp_names):
    leaf_idx_to_all_class_idx = {} # leaf_idx_to_all_class_idx[leaf_idx][super_class_time] = super_class_idx at super_class_time of this leaf class 
    all_tp_info = []
    for tp_idx in range(0, 2):
        tp_info = {
            'tp_name' : tp_names[tp_idx],
            'tp_idx' : tp_idx,
            'all_classes' : [],
            'num_of_classes' : 0,
            'idx_to_leaf_name' : {},
            'leaf_name_to_idx' : {},
        }
        for class_name in cifar_classes[tp_idx]:
            class_id = cifar_classes[tp_idx][class_name]
            tp_info['idx_to_leaf_name'][class_id] = class_name
            tp_info['leaf_name_to_idx'][class_name] = class_id
        for class_id in sorted(list(tp_info['idx_to_leaf_name'].keys())):
            tp_info['all_classes'].append(tp_info['idx_to_leaf_name'][class_id]) 
        tp_info['num_of_classes'] = len(tp_info['all_classes'])
        
        all_tp_info.append(tp_info)
        if tp_idx == 1:
            for super_class_name in cifar_hierachy:
                super_class_idx = all_tp_info[0]['leaf_name_to_idx'][super_class_name]
                sub_class_names = cifar_hierachy[super_class_name]
                sub_class_indices = [all_tp_info[1]['leaf_name_to_idx'][sub_class_name]
                                     for sub_class_name in sub_class_names]
                for sub_class_idx in sub_class_indices:
                    leaf_idx_to_all_class_idx[sub_class_idx] = [super_class_idx, sub_class_idx]

    for leaf_idx in leaf_idx_to_all_class_idx:
        if not leaf_idx_to_all_class_idx[leaf_idx][-1] == leaf_idx:
            print("Wrong label format: Last index must be leaf index")
            import pdb; pdb.set_trace()
    for i, tp_info in enumerate(all_tp_info):
        assert tp_info['tp_idx'] == i
    return all_tp_info, leaf_idx_to_all_class_idx
    
class CIFAR10(LecoDataset):
    def __init__(self, data_dir):
        super().__init__(data_dir)
    
    def get_transform_train(self):
        raise NotImplementedError() #TODO
    
    def get_transform_test(self):
        return get_cifar_transform_test()

    def get_weak_and_strong_transform(self):
        return get_cifar_transform_train_weak_aug(), get_cifar_transform_train_strong_aug()

    def get_dataset(self):
        transform_train = self.get_transform_train()
        trainset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=transform_train
        )

        transform_test = self.get_transform_test()
        testset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=transform_test
        )
        return trainset, testset
    
    def get_class_hierarchy(self):
        cifar_classes = {
            0 : {'vehicle' : 0, 'animal' : 1},
            1 : {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
        }
        
        cifar_hierachy = {
            'vehicle' : ['airplane', 'ship', 'automobile', 'truck'], # vehicle
            'animal' : ['cat', 'frog', 'bird', 'deer', 'dog', 'horse'] # animal
        }

        tp_names = ['cifar10_tp_0', 'cifar10_tp_1']
        return get_cifar_tp_info(cifar_classes,
                                 cifar_hierachy,
                                 tp_names)
        
class CIFAR10WeakAug(CIFAR10):
    def __init__(self, data_dir):
        super().__init__(data_dir)
    
    def get_transform_train(self):
        return get_cifar_transform_train_weak_aug()

class CIFAR10StrongAug(CIFAR10):
    def __init__(self, data_dir):
        super().__init__(data_dir)
    
    def get_transform_train(self):
        return get_cifar_transform_train_strong_aug()

class CIFAR100(LecoDataset):
    def __init__(self, data_dir):
        super().__init__(data_dir)
    
    def get_transform_train(self):
        raise NotImplementedError() #TODO
    
    def get_transform_test(self):
        return get_cifar_transform_test()

    def get_weak_and_strong_transform(self):
        return get_cifar_transform_train_weak_aug(), get_cifar_transform_train_strong_aug()

    def get_dataset(self):
        transform_train = self.get_transform_train()
        trainset = torchvision.datasets.CIFAR100(
            root=self.data_dir,
            train=True,
            download=True,
            transform=transform_train
        )

        transform_test = self.get_transform_test()
        testset = torchvision.datasets.CIFAR100(
            root=self.data_dir,
            train=False,
            download=True,
            transform=transform_test
        )
        return trainset, testset
    
    def get_class_hierarchy(self):
        cifar_classes = {
            0 : {'aquatic_mammals' : 0,
                 'fish' : 1,
                 'flowers' : 2,
                 'food_containers' : 3,
                 'fruit_and_vegetables' : 4,
                 'household_electrical_devices' : 5,
                 'household_furniture' : 6,
                 'insects' : 7,
                 'large_carnivores' : 8,
                 'large_man_made_outdoor_things' : 9,
                 'large_natural_outdoor_scenes' : 10,
                 'large_omnivores_and_herbivores' : 11,
                 'medium_sized_mammals' : 12,
                 'non_insect_invertebrates' : 13,
                 'people' : 14,
                 'reptiles' : 15,
                 'small_mammals' : 16,
                 'trees' : 17,
                 'vehicles_1' : 18,
                 'vehicles_2' : 19},
            1 : {'apple': 0, 'aquarium_fish': 1, 'baby': 2, 'bear': 3, 'beaver': 4, 'bed': 5, 'bee': 6, 'beetle': 7, 
                 'bicycle': 8, 'bottle': 9, 'bowl': 10, 'boy': 11, 'bridge': 12, 
                 'bus': 13, 'butterfly': 14, 'camel': 15, 'can': 16, 'castle': 17, 'caterpillar': 18, 'cattle': 19, 'chair': 20, 'chimpanzee': 21,
                 'clock': 22, 'cloud': 23, 'cockroach': 24, 
                 'couch': 25, 'crab': 26, 'crocodile': 27, 'cup': 28, 'dinosaur': 29, 'dolphin': 30, 'elephant': 31, 'flatfish': 32, 
                 'forest': 33, 'fox': 34, 'girl': 35, 'hamster': 36, 'house': 37, 
                 'kangaroo': 38, 'keyboard': 39, 'lamp': 40, 'lawn_mower': 41, 'leopard': 42, 'lion': 43, 'lizard': 44, 
                 'lobster': 45, 'man': 46, 'maple_tree': 47, 'motorcycle': 48, 'mountain': 49, 
                 'mouse': 50, 'mushroom': 51, 'oak_tree': 52, 'orange': 53, 'orchid': 54, 'otter': 55, 'palm_tree': 56, 'pear': 57, 
                 'pickup_truck': 58, 'pine_tree': 59, 'plain': 60, 'plate': 61, 'poppy': 62, 
                 'porcupine': 63, 'possum': 64, 'rabbit': 65, 'raccoon': 66, 'ray': 67, 'road': 68, 'rocket': 69, 
                 'rose': 70, 'sea': 71, 'seal': 72, 'shark': 73, 'shrew': 74, 'skunk': 75, 'skyscraper': 76, 'snail': 77, 
                 'snake': 78, 'spider': 79, 'squirrel': 80, 'streetcar': 81, 'sunflower': 82, 'sweet_pepper': 83, 'table': 84, 
                 'tank': 85, 'telephone': 86, 'television': 87, 'tiger': 88, 'tractor': 89, 'train': 90, 
                 'trout': 91, 'tulip': 92, 'turtle': 93, 'wardrobe': 94, 'whale': 95, 'willow_tree': 96, 'wolf': 97, 'woman': 98, 'worm': 99}
        }
        
        cifar_hierachy = {
            'aquatic_mammals' : ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
            'fish' : ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
            'flowers' : ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
            'food_containers' : ['bottle', 'bowl', 'can', 'cup', 'plate'],
            'fruit_and_vegetables' : ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
            'household_electrical_devices' : ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
            'household_furniture' : ['bed', 'chair', 'couch', 'table', 'wardrobe'],
            'insects' : ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
            'large_carnivores' : ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
            'large_man_made_outdoor_things' : ['bridge', 'castle', 'house', 'road', 'skyscraper'],
            'large_natural_outdoor_scenes' : ['cloud', 'forest', 'mountain', 'plain', 'sea'],
            'large_omnivores_and_herbivores' : ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
            'medium_sized_mammals' : ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
            'non_insect_invertebrates' : ['crab', 'lobster', 'snail', 'spider', 'worm'],
            'people' : ['baby', 'boy', 'girl', 'man', 'woman'],
            'reptiles' : ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
            'small_mammals' :['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
            'trees' : ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
            'vehicles_1' : ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
            'vehicles_2' : ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'],
        }

        tp_names = ['cifar100_tp_0', 'cifar100_tp_1']
        return get_cifar_tp_info(cifar_classes,
                                 cifar_hierachy,
                                 tp_names)
        
class CIFAR100WeakAug(CIFAR100):
    def __init__(self, data_dir):
        super().__init__(data_dir)
    
    def get_transform_train(self):
        return get_cifar_transform_train_weak_aug()

class CIFAR100StrongAug(CIFAR100):
    def __init__(self, data_dir):
        super().__init__(data_dir)
    
    def get_transform_train(self):
        return get_cifar_transform_train_strong_aug()

        
# helper function to download and extract zip file from Google drive
def _extract_zip(from_path, to_path, compression):
    with zipfile.ZipFile(
        from_path, "r", compression=zipfile.ZIP_STORED
    ) as zip:
        zip.extractall(to_path)

def extract_archive(from_path, to_path=None, remove_finished=False):
    """Extract an archive.

    The archive type and a possible compression is automatically detected from the file name. If the file is compressed
    but not an archive the call is dispatched to :func:`decompress`.

    Args:
        from_path (str): Path to the file to be extracted.
        to_path (str): Path to the directory the file will be extracted to. If omitted, the directory of the file is
            used.
        remove_finished (bool): If ``True``, remove the file after the extraction.

    Returns:
        (str): Path to the directory the file was extracted to.
    """
    if to_path is None:
        to_path = os.path.dirname(from_path)

    _extract_zip(from_path, to_path, None)
    if remove_finished:
        os.remove(from_path)

    return to_path


class IndexFolder(torchvision.datasets.ImageFolder):
    def find_classes(self, directory):
        classes = sorted(int(entry.name) for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder with integer class name in {directory}.")

        class_to_idx = {str(cls_name): i for i, cls_name in enumerate(classes)}
        classes = [str(name) for name in classes]
        return classes, class_to_idx


def download_semi_inat(data_dir):
    inat_folder = os.path.join(data_dir, 'semi_inat')
    if not os.path.exists(inat_folder):
        zip_path = os.path.join(data_dir, 'semi_inat.zip')
        gdrive_url = "https://drive.google.com/u/0/uc?id=1kNWhy77tbet3HBrFrzy3Xw4S8uPVkDdb"
        gdown.download(gdrive_url, zip_path, quiet=False)
        gdown.cached_download(gdrive_url, zip_path,
                              md5="f97619d670278deaacce58ceaf2b5732")
        extract_archive(
            zip_path, to_path=data_dir, remove_finished=False
        )
    else:
        print(f"{inat_folder} already exists.")

class SemiInat2021(LecoDataset):
    def __init__(self, data_dir, levels, val_ratio):
        super().__init__(data_dir)
        download_semi_inat(self.data_dir)
        self.levels = levels # The hierarchy level for classification
        self.val_ratio = val_ratio # val ratio per class
    
    def get_transform_train(self):
        raise NotImplementedError() #TODO
    
    def get_transform_test(self):
        return get_inat_transform_test()

    def get_weak_and_strong_transform(self):
        return get_inat_transform_train_weak_aug(), get_inat_transform_train_strong_aug()
    
    def get_dataset(self):
        transform_train = self.get_transform_train()
        trainset = IndexFolder(
            os.path.join(self.data_dir, 'semi_inat', 'l_train_and_u_train_in'),
            transform=transform_train
        )
        
        transform_test = self.get_transform_test()
        testset = IndexFolder(
            os.path.join(self.data_dir, 'semi_inat', 'val'),
            transform=transform_test
        )
        return trainset, testset
    
    def get_train_val_indices(self, trainset):
        self.target_to_indices = {}
        for idx, target in tqdm(enumerate(trainset.dataset.targets)):
            if not target in self.target_to_indices:
                self.target_to_indices[target] = []
            self.target_to_indices[target].append(idx)
        
        all_t_0_train_indices = [] 
        all_t_0_val_indices = []
        all_t_1_train_indices = []
        all_t_1_val_indices = []
        for target in self.target_to_indices:
            random.shuffle(self.target_to_indices[target])
            
            len_of_indices = len(self.target_to_indices[target])
            len_of_t_0 = int(len_of_indices/2)
            len_of_t_1 = len_of_indices - len_of_t_0
             
            t_0_indices = self.target_to_indices[target][:len_of_t_0]
            t_1_indices = self.target_to_indices[target][len_of_t_0:]
            
            assert len(t_1_indices) == len_of_t_1
            assert len(t_0_indices) == len_of_t_0
            
            len_of_t_0_val = int(len_of_t_0 * self.val_ratio)
            len_of_t_0_train = len_of_t_0 - len_of_t_0_val
            t_0_val_indices = t_0_indices[:len_of_t_0_val]
            t_0_train_indices = t_0_indices[len_of_t_0_val:]
            assert len(t_0_train_indices) > 0
            assert len(t_0_val_indices) > 0
            assert len(t_0_train_indices) == len_of_t_0_train
            assert len(t_0_val_indices) == len_of_t_0_val
            
            len_of_t_1_val = int(len_of_t_1 * self.val_ratio)
            len_of_t_1_train = len_of_t_1 - len_of_t_1_val
            t_1_val_indices = t_1_indices[:len_of_t_1_val]
            t_1_train_indices = t_1_indices[len_of_t_1_val:]
            assert len(t_1_train_indices) > 0
            assert len(t_1_val_indices) > 0
            assert len(t_1_train_indices) == len_of_t_1_train
            assert len(t_1_val_indices) == len_of_t_1_val
            
            all_t_0_train_indices += t_0_train_indices
            all_t_0_val_indices += t_0_val_indices
            all_t_1_train_indices += t_1_train_indices
            all_t_1_val_indices += t_1_val_indices

        print(f"Time 0 Train Size: {len(all_t_0_train_indices)}")
        print(f"Time 0 Val Size: {len(all_t_0_val_indices)}")
        print(f"Time 1 Train Size: {len(all_t_1_train_indices)}")
        print(f"Time 1 Val Size: {len(all_t_1_val_indices)}")
        train_val_indices = [(all_t_0_train_indices, all_t_0_val_indices), (all_t_1_train_indices, all_t_1_val_indices)]
        return train_val_indices
    
    def _get_class_hierarchy(self):
        # An example of taxa
        # {'class_id': 10, 'species': 'Viola selkirkii', 'kingdom': 'Plantae', 'phylum': 'Tracheophyta', 'class': 'Magnoliopsida', 'order': 'Malpighiales', 'family': 'Violaceae', 'genus': 'Viola'}
        hierachy = {}
            
        annotation_file = os.path.join(self.data_dir, 'semi_inat', 'annotation_v2.json')
        assert os.path.exists(annotation_file), (f"File {annotation_file} does not exist.")
        annotation_category = json.load(open(annotation_file, 'r'))
        
        for taxa in annotation_category['categories']:
            if not taxa['kingdom'] in hierachy:
                hierachy[taxa['kingdom']] = {}
            if not taxa['phylum'] in hierachy[taxa['kingdom']]:
                hierachy[taxa['kingdom']][taxa['phylum']] = {}
            if not taxa['class'] in hierachy[taxa['kingdom']][taxa['phylum']]:
                hierachy[taxa['kingdom']][taxa['phylum']][taxa['class']] = {}
            if not taxa['order'] in hierachy[taxa['kingdom']][taxa['phylum']][taxa['class']]:
                hierachy[taxa['kingdom']][taxa['phylum']][taxa['class']][taxa['order']] = {}
            if not taxa['family'] in hierachy[taxa['kingdom']][taxa['phylum']][taxa['class']][taxa['order']]:
                hierachy[taxa['kingdom']][taxa['phylum']][taxa['class']][taxa['order']][taxa['family']] = {}
            if not taxa['genus'] in hierachy[taxa['kingdom']][taxa['phylum']][taxa['class']][taxa['order']][taxa['family']]:
                hierachy[taxa['kingdom']][taxa['phylum']][taxa['class']][taxa['order']][taxa['family']][taxa['genus']] = []
            hierachy[taxa['kingdom']][taxa['phylum']][taxa['class']][taxa['order']][taxa['family']][taxa['genus']].append(taxa)
        
        num_kingdom = 0
        num_phylum = 0
        num_class = 0
        num_order = 0
        num_family = 0
        num_genus = 0
        num_species = 0
        
        all_taxa = {
            'kingdom' : [],
            'phylum' : [],
            'class' : [],
            'order' : [],
            'family' : [],
            'genus' : [],
            'species' : []
        }
        for kingdom in hierachy:
            num_kingdom += 1
            all_taxa['kingdom'].append(kingdom)
            print(f"{kingdom} kingdom has {len(hierachy[kingdom])} phylum")
            for phylum in hierachy[kingdom]:
                num_phylum += 1
                all_taxa['phylum'].append(phylum)
                print(f"\t{phylum} phylum has {len(hierachy[kingdom][phylum])} classes")
                for the_class in hierachy[kingdom][phylum]:
                    num_class += 1
                    all_taxa['class'].append(the_class)
                    print(f"\t\t{the_class} class has {len(hierachy[kingdom][phylum][the_class])} orders")
                    for order in hierachy[kingdom][phylum][the_class]:
                        num_order += 1
                        all_taxa['order'].append(order)
                        print(f"\t\t\t{order} order has {len(hierachy[kingdom][phylum][the_class][order])} family")
                        for family in hierachy[kingdom][phylum][the_class][order]:
                            num_family += 1
                            all_taxa['family'].append(family)
                            print(f"\t\t\t\t{family} family has {len(hierachy[kingdom][phylum][the_class][order][family])} genus")
                            for genus in hierachy[kingdom][phylum][the_class][order][family]:
                                num_genus += 1
                                all_taxa['genus'].append(genus)
                                print(f"\t\t\t\t\t{genus} genus has {len(hierachy[kingdom][phylum][the_class][order][family][genus])} species")
                                for species in hierachy[kingdom][phylum][the_class][order][family][genus]:
                                    num_species += 1
                                    all_taxa['species'].append(species)
                                    pass
        print()
        print(f'Number of kingdom: {num_kingdom}')
        print(f'Number of phylum: {num_phylum}')
        print(f'Number of class: {num_class}')
        print(f'Number of order: {num_order}')
        print(f'Number of family: {num_family}')
        print(f'Number of genus: {num_genus}')
        print(f'Number of species: {num_species}')

        tp_names = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
        leaf_idx_to_all_class_idx = {} # leaf_idx_to_all_class_idx[leaf_idx][super_class_time] = super_class_idx at super_class_time of this leaf class 
        all_tp_info = []
        for tp_idx in range(len(tp_names)-1, -1, -1):
            tp_info = {
                'tp_name' : tp_names[tp_idx],
                'tp_idx' : tp_idx,
                'all_classes' : [],
                'num_of_classes' : 0,
                'idx_to_leaf_name' : {},
                'leaf_name_to_idx' : {},
            }
            if tp_idx == len(tp_names)-1:
                tp_name = tp_names[tp_idx]
                assert tp_name == 'species'
                for taxa in all_taxa['species']:
                    class_id = taxa['class_id']
                    species = taxa['species']
                    tp_info['idx_to_leaf_name'][class_id] = species
                    tp_info['leaf_name_to_idx'][species] = class_id
                    # genus = taxa['genus']
                    # family = taxa['family']
                    # order = taxa['order']
                    # the_class = taxa['class']
                    # phylum = taxa['phylum']
                    # kingdom = taxa['kingdom']
                    leaf_idx_to_all_class_idx[class_id] = [
                        all_taxa[level].index(taxa[level])
                        for level in tp_names[:-1]
                    ] + [class_id]
                
                for class_id in sorted(list(leaf_idx_to_all_class_idx.keys())):
                    tp_info['all_classes'].append(tp_info['idx_to_leaf_name'][class_id])
            else:
                tp_info['all_classes'] = all_taxa[tp_names[tp_idx]]
                for class_id, leaf_name in enumerate(tp_info['all_classes']):
                    tp_info['idx_to_leaf_name'][class_id] = leaf_name
                    tp_info['leaf_name_to_idx'][leaf_name] = class_id
            tp_info['num_of_classes'] = len(tp_info['all_classes'])

            all_tp_info = [tp_info] + all_tp_info

        # leaf_idx_to_all_class_idx = {} # leaf_idx_to_all_class_idx[leaf_idx][super_class_time] = super_class_idx at super_class_time of this leaf class 
        for leaf_idx in leaf_idx_to_all_class_idx:
            if not leaf_idx_to_all_class_idx[leaf_idx][-1] == leaf_idx:
                print("Wrong label format: Last index must be leaf index")
                import pdb; pdb.set_trace()
        for i, tp_info in enumerate(all_tp_info):
            assert tp_info['tp_idx'] == i
        return all_tp_info, leaf_idx_to_all_class_idx
    
    def get_class_hierarchy(self):
        all_tp_info, leaf_idx_to_all_class_idx = self._get_class_hierarchy()
        for level in self.levels:
            assert level >= 0 and level < len(all_tp_info)
        all_tp_info = [all_tp_info[i] for i in self.levels]
        leaf_idx_to_all_class_idx = {label_idx: 
                                        [leaf_idx_to_all_class_idx[label_idx][i]
                                         for i in self.levels]
                                     for label_idx in leaf_idx_to_all_class_idx}
        return all_tp_info, leaf_idx_to_all_class_idx

class SemiInat2021WeakAug(SemiInat2021):
    def __init__(self, data_dir, levels=[3,6], val_ratio=0.2):
        super().__init__(data_dir, levels, val_ratio)
    
    def get_transform_train(self):
        return get_inat_transform_train_weak_aug()

class SemiInat2021StrongAug(SemiInat2021):
    def __init__(self, data_dir, levels=[3,6], val_ratio=0.2):
        super().__init__(data_dir, levels, val_ratio)
    
    def get_transform_train(self):
        return get_inat_transform_train_strong_aug()

class SemiInat2021StrongAug4TPs(SemiInat2021):
    def __init__(self, data_dir, levels=[3,4,5,6], val_ratio=0.2):
        super().__init__(data_dir, levels=levels, val_ratio=val_ratio)
    
    def get_transform_train(self):
        return get_inat_transform_train_strong_aug()
    
    def get_train_val_indices(self, trainset):
        self.target_to_indices = {}
        for idx, target in tqdm(enumerate(trainset.dataset.targets)):
            if not target in self.target_to_indices:
                self.target_to_indices[target] = []
            self.target_to_indices[target].append(idx)
        
        num_levels = len(self.levels)
        all_tps_train_indices = [[] for _ in range(num_levels)]
        all_tps_val_indices = [[] for _ in range(num_levels)]
        for target in self.target_to_indices:
            random.shuffle(self.target_to_indices[target])
            
            len_of_indices = len(self.target_to_indices[target])
            all_tps_indices_target = np.array_split(
                self.target_to_indices[target],
                num_levels
            )
            
            for tp_idx, tp_indices in enumerate(all_tps_indices_target):
                len_of_tp_val = int(len(tp_indices) * self.val_ratio)
                if len_of_tp_val <= 0:
                    len_of_tp_val = 1
                    print(f"At TP{tp_idx}, val set for {target} is 1")
                elif len_of_tp_val <= 1:
                    print(f"At TP{tp_idx}, val set for {target} is {len_of_tp_val}")
                tp_val_indices = tp_indices[:len_of_tp_val]
                tp_train_indices = tp_indices[len_of_tp_val:]
                assert len(tp_val_indices) + len(tp_train_indices) == len(tp_indices)
                assert len(tp_val_indices) > 0 and len(tp_train_indices) > 0
            
                all_tps_train_indices[tp_idx] += tp_train_indices.tolist()
                all_tps_val_indices[tp_idx] += tp_val_indices.tolist()

        for tp_idx, tp_train_indices in enumerate(all_tps_train_indices):
            print(f"TP{tp_idx} Train Size: {len(tp_train_indices)}")
        
        for tp_idx, tp_val_indices in enumerate(all_tps_val_indices):
            print(f"TP{tp_idx} Val Size: {len(tp_val_indices)}")
        train_val_indices = [(all_tps_train_indices[tp_idx], all_tps_val_indices[tp_idx]) for tp_idx in range(num_levels)]
        return train_val_indices


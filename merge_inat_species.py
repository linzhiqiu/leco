import json
import torch
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms
import random
import os
from randaugment import RandAugmentMC, RandAugmentInat
import zipfile
import gdown
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


from pathlib import Path
from copy import deepcopy
class LeafClass():
    def __init__(self, species=[], taxa_info={}, samples=[]):
        self.species = species # list of int
        self.taxa_info = taxa_info # dict: key is species_id (int), value is taxa info
        self.samples = samples # list of paths
        
        
def convert_taxa_info_to_leaf_class(taxa, l_dir):
    species_dir = Path(l_dir) / str(taxa['class_id'])
    samples = [str(p) for p in species_dir.glob("*")]
    return LeafClass(species=[taxa['class_id']],
                     taxa_info={taxa['class_id'] : taxa},
                     samples=samples)

def merge_leaf_classes(leaf_1, leaf_2):
    leaf_1.taxa_info.update(leaf_2.taxa_info)
    return LeafClass(species=leaf_1.species + leaf_2.species,
                     taxa_info=leaf_1.taxa_info,
                     samples=leaf_1.samples + leaf_2.samples)

def num_of_leaf_of_tree(curr_tree):
    count = 0
    for k in curr_tree:
        if type(curr_tree[k]) == dict:
            count += num_of_leaf_of_tree(curr_tree[k])
        elif type(curr_tree[k]) == LeafClass:
            count += 1
    return count

def smallest_pair_of_leaf(curr_tree):
    """return the smallest pair of leafclass (under same parent)'s sum of samples
    Args:
        curr_tree: Must be collapsed()
    """
    min_pair_sample_num_of_subtrees = None
    curr_leaves = []
    for k in curr_tree:
        if type(curr_tree[k]) == dict:
            count_of_subtree = smallest_pair_of_leaf(curr_tree[k])
            assert count_of_subtree != None, "Subtree not collapsed?"
            if min_pair_sample_num_of_subtrees == None or min_pair_sample_num_of_subtrees > count_of_subtree:
                min_pair_sample_num_of_subtrees = count_of_subtree
        elif type(curr_tree[k]) == LeafClass:
            curr_leaves.append(curr_tree[k])
        else:
            raise ValueError("Node must be str or LeafClass")
    
    curr_leaves = sorted(curr_leaves, key=lambda x: len(x.samples))
    if len(curr_leaves) >= 2:
        min_pair_sample_num_of_curr_leaves = sum([len(x.samples) for x in curr_leaves[:2]])
        if min_pair_sample_num_of_subtrees == None:
            return min_pair_sample_num_of_curr_leaves
        else:
            return min(min_pair_sample_num_of_curr_leaves, min_pair_sample_num_of_subtrees)
    else:
        assert min_pair_sample_num_of_subtrees != None, "Subtree not collapsed"
        return min_pair_sample_num_of_subtrees

new_node_counter = 0
def merge(curr_tree, min_pair_sample_num):
    """Perform a merge for two leafclass with sample num == min_pair_sample_num
    Return (True) if this pair is found and merge is completed 
    Return (False) if otherwise and curr_tree is untouched
    """
    global new_node_counter
    assert type(curr_tree) == dict
    
    for k in curr_tree:
        if type(curr_tree[k]) == dict:
            success = merge(curr_tree[k], min_pair_sample_num)
            if success:
                return True
    
    curr_leaves = {}
    for k in curr_tree:
        if type(curr_tree[k]) == LeafClass:
            curr_leaves[k] = curr_tree[k]
    curr_leaves = sorted(curr_leaves, key=lambda k: len(curr_tree[k].samples))
    if len(curr_leaves) >= 2:
        min_pair_sample_num_of_curr_leaves = sum([len(curr_tree[k].samples) for k in curr_leaves[:2]])
        if min_pair_sample_num_of_curr_leaves == min_pair_sample_num:
            new_leaf = merge_leaf_classes(curr_tree[curr_leaves[0]], curr_tree[curr_leaves[1]])
            del curr_tree[curr_leaves[0]]
            del curr_tree[curr_leaves[1]]
            curr_tree[f"new_{new_node_counter}"] = new_leaf
            print(
                f"Merge {curr_leaves[0]} and {curr_leaves[1]} to {new_node_counter} with total {min_pair_sample_num} samples")
            new_node_counter += 1
            return True
    
    return False
    # success = merge(curr_tree[node], min_pair_sample_num)
    
def merge_smallest_pair_of_leaf(curr_tree):
    min_pair_sample_num = smallest_pair_of_leaf(curr_tree)
    success = merge(curr_tree, min_pair_sample_num)
    assert success, "Fail to merge"
    return curr_tree
    
def collapse(curr_tree):
    """Collapse all nodes with a single LeafClass
    """
    assert type(curr_tree) == dict
    if len(curr_tree) == 1:
        subtree = list(curr_tree.items())[0][1]
        if type(subtree) == dict:
            return collapse(subtree)
        elif type(subtree) == LeafClass:
            return subtree
        else:
            raise ValueError("Nodes can only be LeafClass or dict")
    
    for k in curr_tree:
        if type(curr_tree[k]) == dict:
            curr_tree[k] = collapse(curr_tree[k])
        elif type(curr_tree[k]) == LeafClass:
            pass
        else:
            raise ValueError("Nodes can only be LeafClass or dict")
    return curr_tree


def make_other_class_with_minor_nodes(tree, fewest):
    assert fewest > 0
    import import pdb; pdb.set_trace()

def merge_tree(hierachy, nums_of_leaf, fewest_num_samples=None):
    new_hierarchy = []
    curr_tree = collapse(hierachy)
    for num_of_leaf in reversed(nums_of_leaf):
        curr_num_of_leaf = num_of_leaf_of_tree(curr_tree)
        while curr_num_of_leaf > num_of_leaf:
            merge_smallest_pair_of_leaf(curr_tree)
            curr_tree = collapse(curr_tree)
            assert num_of_leaf_of_tree(curr_tree) == curr_num_of_leaf - 1
            curr_num_of_leaf -= 1
        
        if fewest_num_samples:
            curr_hierarchy = make_other_class_with_minor_nodes(deepcopy(curr_tree), fewest_num_samples)
        else:
            curr_hierarchy = deepcopy(curr_tree)
        new_hierarchy.append(curr_hierarchy)
    return list(reversed(new_hierarchy))


def get_all_tp_hierarchy(hierachy, nums_of_leaf, fewest_num_samples=None):
    num_of_species = num_of_leaf_of_tree(collapse(deepcopy(hierachy)))
    assert num_of_species == 810
    leaf_idx_to_all_class_idx = {
        species_idx : []
        for species_idx in range(num_of_species)
    } # leaf_idx_to_all_class_idx[leaf_idx][super_class_time] = super_class_idx at super_class_time of this leaf class 
    tp_hierarchy = merge_tree(hierachy, nums_of_leaf, fewest_num_samples)
    all_tp_info = []
    # return all_tp_info, leaf_idx_to_all_class_idx, tp_hierarchy
    # import pdb; pdb.set_trace()
    
    for tp_idx in range(len(nums_of_leaf)-1, -1, -1):
        leaves = flatten_tree(tp_hierarchy[tp_idx])
        tp_info = {
            'tp_name' : str(tp_idx),
            'tp_idx' : tp_idx,
            'all_classes' : list(range(len(leaves))),
            'num_of_classes': len(leaves),
            'idx_to_leaf_name': {idx: str(idx) for idx in range(len(leaves))},
            'leaf_name_to_idx': {str(idx): idx for idx in range(len(leaves))},
        }
        num_of_leaf = nums_of_leaf[tp_idx]
        tp_name = str(tp_idx)
        for species_idx in range(num_of_species):
            for idx, leaf in enumerate(leaves):
                if species_idx in leaf.species:
                    leaf_idx_to_all_class_idx[species_idx] = [idx] + leaf_idx_to_all_class_idx[species_idx]
        
    for i, tp_info in enumerate(all_tp_info):
        assert tp_info['tp_idx'] == i
    return all_tp_info, leaf_idx_to_all_class_idx, tp_hierarchy

def flatten_tree(curr_tree):
    """Return all LeafClass in a list
    """
    leaves = []
    expected_num = num_of_leaf_of_tree(curr_tree)
    for k in curr_tree:
        if type(curr_tree[k]) == dict:
            leaves += flatten_tree(curr_tree[k])
        elif type(curr_tree[k]) == LeafClass:
            leaves.append(curr_tree[k])
        else:
            raise ValueError("Subtree type must be dict or LeafClass")
    assert len(leaves) == expected_num
    return leaves
    
class SemiInat2021(LecoDataset):
    def __init__(self, data_dir, nums_of_leaf=list(range(3, 809)), fewest_num_samples=None):
        super().__init__(data_dir)
        download_semi_inat(self.data_dir)
        self.nums_of_leaf = nums_of_leaf # The number of leaf classes for classification each timestamp
        for i in range(len(self.nums_of_leaf) - 1):
            assert self.nums_of_leaf[i] < self.nums_of_leaf[i+1]
        self.fewest_num_samples = fewest_num_samples
    
    def get_transform_train(self):
        raise NotImplementedError() #TODO
    
    def get_transform_test(self):
        return get_inat_transform_test()

    def get_weak_and_strong_transform(self):
        return get_inat_transform_train_weak_aug(), get_inat_transform_train_strong_aug()
    
    def get_dataset(self):
        import pdb; pdb.set_trace() # TODO
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
    
    def _get_hierarchy(self):
        # An example of taxa
        # {'class_id': 10, 'species': 'Viola selkirkii', 'kingdom': 'Plantae', 'phylum': 'Tracheophyta', 'class': 'Magnoliopsida', 'order': 'Malpighiales', 'family': 'Violaceae', 'genus': 'Viola'}
        hierarchy = {}
        annotation_file = os.path.join(self.data_dir, 'semi_inat', 'annotation_v2.json')
        assert os.path.exists(annotation_file), (f"File {annotation_file} does not exist.")
        annotation_category = json.load(open(annotation_file, 'r'))
        
        l_dir = os.path.join(self.data_dir, 'semi_inat', 'l_train_and_u_train_in')
        
        for taxa in annotation_category['categories']:
            if not taxa['kingdom'] in hierarchy:
                hierarchy[taxa['kingdom']] = {}
            if not taxa['phylum'] in hierarchy[taxa['kingdom']]:
                hierarchy[taxa['kingdom']][taxa['phylum']] = {}
            if not taxa['class'] in hierarchy[taxa['kingdom']][taxa['phylum']]:
                hierarchy[taxa['kingdom']][taxa['phylum']][taxa['class']] = {}
            if not taxa['order'] in hierarchy[taxa['kingdom']][taxa['phylum']][taxa['class']]:
                hierarchy[taxa['kingdom']][taxa['phylum']][taxa['class']][taxa['order']] = {}
            if not taxa['family'] in hierarchy[taxa['kingdom']][taxa['phylum']][taxa['class']][taxa['order']]:
                hierarchy[taxa['kingdom']][taxa['phylum']][taxa['class']][taxa['order']][taxa['family']] = {}
            if not taxa['genus'] in hierarchy[taxa['kingdom']][taxa['phylum']][taxa['class']][taxa['order']][taxa['family']]:
                hierarchy[taxa['kingdom']][taxa['phylum']][taxa['class']][taxa['order']][taxa['family']][taxa['genus']] = {}
            hierarchy[taxa['kingdom']][taxa['phylum']][taxa['class']][taxa['order']][taxa['family']][taxa['genus']][taxa['species']] = convert_taxa_info_to_leaf_class(taxa, l_dir)
        return hierarchy
        
    def get_class_hierarchy(self):
        # An example of taxa
        # {'class_id': 10, 'species': 'Viola selkirkii', 'kingdom': 'Plantae', 'phylum': 'Tracheophyta', 'class': 'Magnoliopsida', 'order': 'Malpighiales', 'family': 'Violaceae', 'genus': 'Viola'}
        self.hierachy = self._get_hierarchy()
        all_tp_info, leaf_idx_to_all_class_idx, self.tp_hierarchy = get_all_tp_hierarchy(
            deepcopy(self.hierachy),
            self.nums_of_leaf,
            self.fewest_num_samples
        )
        return all_tp_info, leaf_idx_to_all_class_idx
    

class SemiInat2021WeakAug(SemiInat2021):
    def __init__(self, data_dir, nums_of_leaf=[100, 500]):
        super().__init__(data_dir, nums_of_leaf)
    
    def get_transform_train(self):
        return get_inat_transform_train_weak_aug()

class SemiInat2021StrongAug(SemiInat2021):
    def __init__(self, data_dir, nums_of_leaf=[100, 500]):
        super().__init__(data_dir, nums_of_leaf)
    
    def get_transform_train(self):
        return get_inat_transform_train_strong_aug()
    
def get_train_val_indices(dataset, trainset, tp_buffers=None):
    """Must use 'random' to sample the dataset for seeding purpose
    """
    train_val_indices = []
    if tp_buffers != None:
        len_of_trainset = len(trainset)
        indices_trainset = list(range(len_of_trainset))
        random.shuffle(indices_trainset)

        for _, (tp_buffer_train, tp_buffer_val) in enumerate(tp_buffers):
            indices_tp_train, indices_tp_val = indices_trainset[:tp_buffer_train], indices_trainset[tp_buffer_train:tp_buffer_train+tp_buffer_val]
            indices_trainset = indices_trainset[tp_buffer_train+tp_buffer_val:]
            train_val_indices.append((indices_tp_train, indices_tp_val))
    else:
        train_val_indices = dataset.get_train_val_indices()
    return train_val_indices

if __name__ == "__main__":
    import setups
    import sys
    data_dir = '/scratch/leco/'
    setup = setups.SETUPS['semi_inat_weakaug']
    print(f"==> Preparing {setup.dataset_name} data..")
    dataset = getattr(sys.modules[__name__], setup.dataset_name)(data_dir)
    all_tp_info, leaf_idx_to_all_class_idx = dataset.get_class_hierarchy()
    trainset, testset = dataset.get_dataset()
    weak_transform, strong_transform = dataset.get_weak_and_strong_transform()
    test_transform = dataset.get_transform_test()
    trainset = HierarchyDataset(trainset, leaf_idx_to_all_class_idx, weak_transform, strong_transform, test_transform)
    testset = HierarchyDataset(testset, leaf_idx_to_all_class_idx, weak_transform, strong_transform, test_transform)
    print(f"Length of trainset is {len(trainset)}")
    print(f"Length of testset is {len(testset)}")
    sample_train = samples_per_class(trainset, all_tp_info, leaf_idx_to_all_class_idx)
    # sample_test = samples_per_class(testset, all_tp_info, leaf_idx_to_all_class_idx)
    train_val_subsets = []
    train_val_indices = get_train_val_indices(dataset, trainset, tp_buffers=setup.tp_buffers)
    
    for _, (indices_tp_train, indices_tp_val) in enumerate(train_val_indices):
        train_subset = SubsetHierarchyDataset(deepcopy(trainset), indices_tp_train)
        val_subset = SubsetHierarchyDataset(deepcopy(trainset), indices_tp_val)
        sample_train = samples_per_class(train_subset, all_tp_info, leaf_idx_to_all_class_idx)
        sample_val = samples_per_class(val_subset, all_tp_info, leaf_idx_to_all_class_idx)
        val_subset.set_test_transform()
        train_val_subsets.append((train_subset, val_subset))
    
    import pdb; pdb.set_trace()
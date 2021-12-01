# To setup semi-inat: First download all the data in https://github.com/cvl-umass/semi-inat-2021
import json
import torch
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms
import random
import os
# annotation_file = ./semi_inat/annotation_v2.json

class LecoDataset():
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_dataset(self):
        raise NotImplementedError()

    def get_class_hierarchy(self):
        leaf_idx_to_all_class_idx = None
        all_tp_info = None
        raise NotImplementedError() # return (all_tp_info, leaf_idx_to_all_class_idx)

class CIFAR10(LecoDataset):
    def __init__(self, data_dir):
        super().__init__(data_dir)
    
    def get_dataset(self):
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
            root=self.data_dir,
            train=True,
            download=True,
            transform=transform_train
        )
        testset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=False, download=True, transform=transform_test)
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

        leaf_idx_to_all_class_idx = {} # leaf_idx_to_all_class_idx[leaf_idx][super_class_time] = super_class_idx at super_class_time of this leaf class 
        all_tp_info = []
        tp_names = ['cifar10_tp_0', 'cifar10_tp_1']
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

        # leaf_idx_to_all_class_idx = {} # leaf_idx_to_all_class_idx[leaf_idx][super_class_time] = super_class_idx at super_class_time of this leaf class 
        for leaf_idx in leaf_idx_to_all_class_idx:
            if not leaf_idx_to_all_class_idx[leaf_idx][-1] == leaf_idx:
                print("Wrong label format: Last index must be leaf index")
                import pdb; pdb.set_trace()
        for i, tp_info in enumerate(all_tp_info):
            assert tp_info['tp_idx'] == i
        return all_tp_info, leaf_idx_to_all_class_idx

class IndexFolder(torchvision.datasets.ImageFolder):
    def find_classes(self, directory):
        classes = sorted(int(entry.name) for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder with integer class name in {directory}.")

        class_to_idx = {str(cls_name): i for i, cls_name in enumerate(classes)}
        classes = [str(name) for name in classes]
        return classes, class_to_idx

class SemiInat2021(LecoDataset):
    def __init__(self, data_dir):
        super().__init__(data_dir)
    
    def get_dataset(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose(
            [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]
        )
        transform_test = transforms.Compose(
            [transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), normalize]
        )

        trainset = IndexFolder(
            os.path.join(self.data_dir, 'semi_inat', 'l_train'),
            transform=transform_train
        )
        testset = IndexFolder(
            os.path.join(self.data_dir, 'semi_inat', 'val'),
            transform=transform_test
        )
        return trainset, testset
    
    def get_class_hierarchy(self):
        # An example of taxa
        # {'class_id': 10, 'species': 'Viola selkirkii', 'kingdom': 'Plantae', 'phylum': 'Tracheophyta', 'class': 'Magnoliopsida', 'order': 'Malpighiales', 'family': 'Violaceae', 'genus': 'Viola'}
        hierachy = {}
        annotation_file = os.path.join(self.data_dir, 'semi_inat', 'annotation_v2.json')
        if not os.path.exists(annotation_file):
            raise Exception(f"File {annotation_file} does not exist.")
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
                
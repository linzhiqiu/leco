from util import *
import torch


def fix_dataset(dataset, dataset_path):
    train_val_subsets, testset, all_tp_info, leaf_idx_to_all_class_idx = dataset
    if leaf_idx_to_all_class_idx[267][-2] == 71:
        leaf_idx_to_all_class_idx[267][-2] = 639
        testset.leaf_idx_to_all_class_idx[267][-2] = 639
        for subset in train_val_subsets:
            subset[0].hierarchy_dataset.leaf_idx_to_all_class_idx[267][-2] = 639
            subset[1].hierarchy_dataset.leaf_idx_to_all_class_idx[267][-2] = 639

        all_tp_info[-2]['idx_to_leaf_name'][639] = 'Orchidaceae-Satyrium'
        all_tp_info[-2]['all_classes'][639] = 'Orchidaceae-Satyrium'
        all_tp_info[-2]['idx_to_leaf_name'][71] = 'Lycaenidae-Satyrium'
        all_tp_info[-2]['all_classes'][71] = 'Lycaenidae-Satyrium'

        del all_tp_info[-2]['leaf_name_to_idx']['Satyrium']
        all_tp_info[-2]['leaf_name_to_idx']['Lycaenidae-Satyrium'] = 71
        all_tp_info[-2]['leaf_name_to_idx']['Orchidaceae-Satyrium'] = 639
        dataset = train_val_subsets, testset, all_tp_info, leaf_idx_to_all_class_idx
        save_obj_as_pickle(dataset_path, dataset)
    else:
        pass


dataset_path_None = '/data3/zhiqiul/leco_results/semi_inat_strongaug_4_tps/seed_None/dataset.pt'
dataset_None = load_pickle(dataset_path_None)
fix_dataset(dataset_None, dataset_path_None)

dataset_path_1 = '/data3/zhiqiul/leco_results/semi_inat_strongaug_4_tps/seed_1/dataset.pt'
dataset_1 = load_pickle(dataset_path_1)
fix_dataset(dataset_1, dataset_path_1)

dataset_path_10 = '/data3/zhiqiul/leco_results/semi_inat_strongaug_4_tps/seed_10/dataset.pt'
dataset_10 = load_pickle(dataset_path_10)
fix_dataset(dataset_10, dataset_path_10)

dataset_path_100 = '/data3/zhiqiul/leco_results/semi_inat_strongaug_4_tps/seed_100/dataset.pt'
dataset_100 = load_pickle(dataset_path_100)
fix_dataset(dataset_100, dataset_path_100)

dataset_path_1000 = '/data3/zhiqiul/leco_results/semi_inat_strongaug_4_tps/seed_1000/dataset.pt'
dataset_1000 = load_pickle(dataset_path_1000)
fix_dataset(dataset_1000, dataset_path_1000)


from setups import generate_dataset
from pathlib import Path
from util import load_pickle, save_obj_as_pickle
from copy import deepcopy
import shutil

seed_list = [None, 1, 10, 100, 1000]
train_mode_list = ['none_0_scratch_linear', 'resnet18_byol_pl_0_finetune_pt_linear', 
                   'resnet18_byol_pl_0_freeze_pt_linear', 'resnet18_byol_pl_0_freeze_pt_mlp',
                   'resnet18_moco_v2_pl_0_finetune_pt_linear', 'resnet18_moco_v2_pl_0_freeze_pt_linear',
                   'resnet18_moco_v2_pl_0_freeze_pt_mlp', 'resnet18_simclr_pl_0_finetune_pt_linear',
                   'resnet18_simclr_pl_0_freeze_pt_linear', 'resnet18_simclr_pl_0_freeze_pt_mlp']

def main():

    data_dir = Path('/ssd1/leco')
    copy_setup = data_dir / 'cifar10_buffer_2000_500'
    setup = data_dir / 'cifar10_buffer_2000_500_same_image_same_model'
    setup.mkdir(exist_ok=True)
    for seed in seed_list:
        copy_setup_seed = copy_setup / f'seed_{seed}'
        setup_seed = setup / f'seed_{seed}'
        setup_seed.mkdir(exist_ok=True)
        copy_pt = load_pickle(copy_setup_seed / 'dataset.pt')
        copy_pt[0][1] = deepcopy(copy_pt[0][0])
        save_obj_as_pickle(setup_seed / 'dataset.pt', copy_pt)
        for train_mode in train_mode_list:
            copy_dir = copy_setup_seed / train_mode
            d = setup_seed / train_mode
            assert copy_dir.exists()
            if not d.exists():
                shutil.copytree(copy_dir, d)
            print(f"copy {copy_dir} to {d}")


if __name__ == '__main__':
    main()
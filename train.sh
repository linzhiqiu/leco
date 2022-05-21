## CIFAR10

## Scratch
# without ema
python train.py --data_dir /scratch/leco/ --setup_mode cifar10_weakaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear 
python train.py --data_dir /scratch/leco/ --setup_mode cifar10_weakaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear --seed 1
python train.py --data_dir /scratch/leco/ --setup_mode cifar10_weakaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear --seed 10
python train.py --data_dir /scratch/leco/ --setup_mode cifar10_weakaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear --seed 100
python train.py --data_dir /scratch/leco/ --setup_mode cifar10_weakaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear --seed 1000
python train.py --data_dir /scratch/leco/ --setup_mode cifar10_strongaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear 
python train.py --data_dir /scratch/leco/ --setup_mode cifar10_strongaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear --seed 1
python train.py --data_dir /scratch/leco/ --setup_mode cifar10_strongaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear --seed 10
python train.py --data_dir /scratch/leco/ --setup_mode cifar10_strongaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear --seed 100
python train.py --data_dir /scratch/leco/ --setup_mode cifar10_strongaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear --seed 1000

# with ema
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode cifar10_weakaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear 
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode cifar10_weakaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear --seed 1
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode cifar10_weakaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear --seed 10
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode cifar10_weakaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear --seed 100
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode cifar10_weakaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear --seed 1000
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode cifar10_strongaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear 
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode cifar10_strongaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear --seed 1
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode cifar10_strongaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear --seed 10
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode cifar10_strongaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear --seed 100
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode cifar10_strongaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear --seed 1000


## MoCo v2 stl 10
# without ema
python train.py --data_dir /scratch/leco/ --setup_mode cifar10_weakaug_train_2000_val_500 --train_mode wideres_28_2_moco_v2_stl10_0_finetune_pt_linear_1_finetune_pt_linear 
python train.py --data_dir /scratch/leco/ --setup_mode cifar10_weakaug_train_2000_val_500 --train_mode wideres_28_2_moco_v2_stl10_0_finetune_pt_linear_1_finetune_pt_linear --seed 1
python train.py --data_dir /scratch/leco/ --setup_mode cifar10_weakaug_train_2000_val_500 --train_mode wideres_28_2_moco_v2_stl10_0_finetune_pt_linear_1_finetune_pt_linear --seed 10
python train.py --data_dir /scratch/leco/ --setup_mode cifar10_weakaug_train_2000_val_500 --train_mode wideres_28_2_moco_v2_stl10_0_finetune_pt_linear_1_finetune_pt_linear --seed 100
python train.py --data_dir /scratch/leco/ --setup_mode cifar10_weakaug_train_2000_val_500 --train_mode wideres_28_2_moco_v2_stl10_0_finetune_pt_linear_1_finetune_pt_linear --seed 1000
python train.py --data_dir /scratch/leco/ --setup_mode cifar10_strongaug_train_2000_val_500 --train_mode wideres_28_2_moco_v2_stl10_0_finetune_pt_linear_1_finetune_pt_linear 
python train.py --data_dir /scratch/leco/ --setup_mode cifar10_strongaug_train_2000_val_500 --train_mode wideres_28_2_moco_v2_stl10_0_finetune_pt_linear_1_finetune_pt_linear --seed 1
python train.py --data_dir /scratch/leco/ --setup_mode cifar10_strongaug_train_2000_val_500 --train_mode wideres_28_2_moco_v2_stl10_0_finetune_pt_linear_1_finetune_pt_linear --seed 10
python train.py --data_dir /scratch/leco/ --setup_mode cifar10_strongaug_train_2000_val_500 --train_mode wideres_28_2_moco_v2_stl10_0_finetune_pt_linear_1_finetune_pt_linear --seed 100
python train.py --data_dir /scratch/leco/ --setup_mode cifar10_strongaug_train_2000_val_500 --train_mode wideres_28_2_moco_v2_stl10_0_finetune_pt_linear_1_finetune_pt_linear --seed 1000

# with ema
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode cifar10_weakaug_train_2000_val_500 --train_mode wideres_28_2_moco_v2_stl10_0_finetune_pt_linear_1_finetune_pt_linear 
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode cifar10_weakaug_train_2000_val_500 --train_mode wideres_28_2_moco_v2_stl10_0_finetune_pt_linear_1_finetune_pt_linear --seed 1
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode cifar10_weakaug_train_2000_val_500 --train_mode wideres_28_2_moco_v2_stl10_0_finetune_pt_linear_1_finetune_pt_linear --seed 10
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode cifar10_weakaug_train_2000_val_500 --train_mode wideres_28_2_moco_v2_stl10_0_finetune_pt_linear_1_finetune_pt_linear --seed 100
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode cifar10_weakaug_train_2000_val_500 --train_mode wideres_28_2_moco_v2_stl10_0_finetune_pt_linear_1_finetune_pt_linear --seed 1000
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode cifar10_strongaug_train_2000_val_500 --train_mode wideres_28_2_moco_v2_stl10_0_finetune_pt_linear_1_finetune_pt_linear 
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode cifar10_strongaug_train_2000_val_500 --train_mode wideres_28_2_moco_v2_stl10_0_finetune_pt_linear_1_finetune_pt_linear --seed 1
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode cifar10_strongaug_train_2000_val_500 --train_mode wideres_28_2_moco_v2_stl10_0_finetune_pt_linear_1_finetune_pt_linear --seed 10
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode cifar10_strongaug_train_2000_val_500 --train_mode wideres_28_2_moco_v2_stl10_0_finetune_pt_linear_1_finetune_pt_linear --seed 100
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode cifar10_strongaug_train_2000_val_500 --train_mode wideres_28_2_moco_v2_stl10_0_finetune_pt_linear_1_finetune_pt_linear --seed 1000



## CIFAR100

## Scratch
# without ema
python train.py --data_dir /scratch/leco/ --setup_mode cifar100_weakaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear 
python train.py --data_dir /scratch/leco/ --setup_mode cifar100_weakaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear --seed 1
python train.py --data_dir /scratch/leco/ --setup_mode cifar100_weakaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear --seed 10
python train.py --data_dir /scratch/leco/ --setup_mode cifar100_weakaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear --seed 100
python train.py --data_dir /scratch/leco/ --setup_mode cifar100_weakaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear --seed 1000
python train.py --data_dir /scratch/leco/ --setup_mode cifar100_strongaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear 
python train.py --data_dir /scratch/leco/ --setup_mode cifar100_strongaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear --seed 1
python train.py --data_dir /scratch/leco/ --setup_mode cifar100_strongaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear --seed 10
python train.py --data_dir /scratch/leco/ --setup_mode cifar100_strongaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear --seed 100
python train.py --data_dir /scratch/leco/ --setup_mode cifar100_strongaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear --seed 1000

# with ema
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode cifar100_weakaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear 
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode cifar100_weakaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear --seed 1
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode cifar100_weakaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear --seed 10
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode cifar100_weakaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear --seed 100
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode cifar100_weakaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear --seed 1000
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode cifar100_strongaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear 
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode cifar100_strongaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear --seed 1
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode cifar100_strongaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear --seed 10
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode cifar100_strongaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear --seed 100
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode cifar100_strongaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear --seed 1000


## MoCo v2 stl 10
# without ema
python train.py --data_dir /scratch/leco/ --setup_mode cifar100_weakaug_train_2000_val_500 --train_mode wideres_28_2_moco_v2_stl10_0_finetune_pt_linear_1_finetune_pt_linear 
python train.py --data_dir /scratch/leco/ --setup_mode cifar100_weakaug_train_2000_val_500 --train_mode wideres_28_2_moco_v2_stl10_0_finetune_pt_linear_1_finetune_pt_linear --seed 1
python train.py --data_dir /scratch/leco/ --setup_mode cifar100_weakaug_train_2000_val_500 --train_mode wideres_28_2_moco_v2_stl10_0_finetune_pt_linear_1_finetune_pt_linear --seed 10
python train.py --data_dir /scratch/leco/ --setup_mode cifar100_weakaug_train_2000_val_500 --train_mode wideres_28_2_moco_v2_stl10_0_finetune_pt_linear_1_finetune_pt_linear --seed 100
python train.py --data_dir /scratch/leco/ --setup_mode cifar100_weakaug_train_2000_val_500 --train_mode wideres_28_2_moco_v2_stl10_0_finetune_pt_linear_1_finetune_pt_linear --seed 1000
python train.py --data_dir /scratch/leco/ --setup_mode cifar100_strongaug_train_2000_val_500 --train_mode wideres_28_2_moco_v2_stl10_0_finetune_pt_linear_1_finetune_pt_linear 
python train.py --data_dir /scratch/leco/ --setup_mode cifar100_strongaug_train_2000_val_500 --train_mode wideres_28_2_moco_v2_stl10_0_finetune_pt_linear_1_finetune_pt_linear --seed 1
python train.py --data_dir /scratch/leco/ --setup_mode cifar100_strongaug_train_2000_val_500 --train_mode wideres_28_2_moco_v2_stl10_0_finetune_pt_linear_1_finetune_pt_linear --seed 10
python train.py --data_dir /scratch/leco/ --setup_mode cifar100_strongaug_train_2000_val_500 --train_mode wideres_28_2_moco_v2_stl10_0_finetune_pt_linear_1_finetune_pt_linear --seed 100
python train.py --data_dir /scratch/leco/ --setup_mode cifar100_strongaug_train_2000_val_500 --train_mode wideres_28_2_moco_v2_stl10_0_finetune_pt_linear_1_finetune_pt_linear --seed 1000

# with ema
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode cifar100_weakaug_train_2000_val_500 --train_mode wideres_28_2_moco_v2_stl10_0_finetune_pt_linear_1_finetune_pt_linear 
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode cifar100_weakaug_train_2000_val_500 --train_mode wideres_28_2_moco_v2_stl10_0_finetune_pt_linear_1_finetune_pt_linear --seed 1
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode cifar100_weakaug_train_2000_val_500 --train_mode wideres_28_2_moco_v2_stl10_0_finetune_pt_linear_1_finetune_pt_linear --seed 10
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode cifar100_weakaug_train_2000_val_500 --train_mode wideres_28_2_moco_v2_stl10_0_finetune_pt_linear_1_finetune_pt_linear --seed 100
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode cifar100_weakaug_train_2000_val_500 --train_mode wideres_28_2_moco_v2_stl10_0_finetune_pt_linear_1_finetune_pt_linear --seed 1000
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode cifar100_strongaug_train_2000_val_500 --train_mode wideres_28_2_moco_v2_stl10_0_finetune_pt_linear_1_finetune_pt_linear 
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode cifar100_strongaug_train_2000_val_500 --train_mode wideres_28_2_moco_v2_stl10_0_finetune_pt_linear_1_finetune_pt_linear --seed 1
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode cifar100_strongaug_train_2000_val_500 --train_mode wideres_28_2_moco_v2_stl10_0_finetune_pt_linear_1_finetune_pt_linear --seed 10
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode cifar100_strongaug_train_2000_val_500 --train_mode wideres_28_2_moco_v2_stl10_0_finetune_pt_linear_1_finetune_pt_linear --seed 100
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode cifar100_strongaug_train_2000_val_500 --train_mode wideres_28_2_moco_v2_stl10_0_finetune_pt_linear_1_finetune_pt_linear --seed 1000


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


## Inat
# First run pretrain.py to save the scratch model
python pretrain.py --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --arch resnet50 --pretrain scratch
## Scratch
# without ema
python train.py --data_dir /scratch/leco/ --setup_mode semi_inat_weakaug_train_4000_860 --hparam_candidate inat --train_mode resnet50_scratch_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ 
python train.py --data_dir /scratch/leco/ --setup_mode semi_inat_weakaug_train_4000_860 --hparam_candidate inat --train_mode resnet50_scratch_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --seed 1
python train.py --data_dir /scratch/leco/ --setup_mode semi_inat_weakaug_train_4000_860 --hparam_candidate inat --train_mode resnet50_scratch_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --seed 10
python train.py --data_dir /scratch/leco/ --setup_mode semi_inat_weakaug_train_4000_860 --hparam_candidate inat --train_mode resnet50_scratch_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --seed 100
python train.py --data_dir /scratch/leco/ --setup_mode semi_inat_weakaug_train_4000_860 --hparam_candidate inat --train_mode resnet50_scratch_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --seed 1000

# with ema
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode semi_inat_weakaug_train_4000_860 --hparam_candidate inat --train_mode resnet50_scratch_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ 
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode semi_inat_weakaug_train_4000_860 --hparam_candidate inat --train_mode resnet50_scratch_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --seed 1
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode semi_inat_weakaug_train_4000_860 --hparam_candidate inat --train_mode resnet50_scratch_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --seed 10
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode semi_inat_weakaug_train_4000_860 --hparam_candidate inat --train_mode resnet50_scratch_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --seed 100
python train.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode semi_inat_weakaug_train_4000_860 --hparam_candidate inat --train_mode resnet50_scratch_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --seed 1000

## Inat
# First run pretrain.py to save the scratch model
python pretrain.py --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --arch resnet50 --pretrain scratch
## Scratch Weak Aug
# without ema
python train_per_class.py --data_dir /scratch/leco/ --setup_mode semi_inat_weakaug --hparam_candidate inat --train_mode resnet50_scratch_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ 
python train_per_class.py --data_dir /scratch/leco/ --setup_mode semi_inat_weakaug --hparam_candidate inat --train_mode resnet50_scratch_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --seed 1
python train_per_class.py --data_dir /scratch/leco/ --setup_mode semi_inat_weakaug --hparam_candidate inat --train_mode resnet50_scratch_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --seed 10
python train_per_class.py --data_dir /scratch/leco/ --setup_mode semi_inat_weakaug --hparam_candidate inat --train_mode resnet50_scratch_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --seed 100
python train_per_class.py --data_dir /scratch/leco/ --setup_mode semi_inat_weakaug --hparam_candidate inat --train_mode resnet50_scratch_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --seed 1000

# with ema
python train_per_class.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode semi_inat_weakaug --hparam_candidate inat --train_mode resnet50_scratch_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ 
python train_per_class.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode semi_inat_weakaug --hparam_candidate inat --train_mode resnet50_scratch_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --seed 1
python train_per_class.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode semi_inat_weakaug --hparam_candidate inat --train_mode resnet50_scratch_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --seed 10
python train_per_class.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode semi_inat_weakaug --hparam_candidate inat --train_mode resnet50_scratch_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --seed 100
python train_per_class.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode semi_inat_weakaug --hparam_candidate inat --train_mode resnet50_scratch_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --seed 1000


## Scratch Strong aug
# without ema
python train_per_class.py --data_dir /scratch/leco/ --setup_mode semi_inat_strongaug --hparam_candidate inat --train_mode resnet50_scratch_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ 
python train_per_class.py --data_dir /scratch/leco/ --setup_mode semi_inat_strongaug --hparam_candidate inat --train_mode resnet50_scratch_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --seed 1
python train_per_class.py --data_dir /scratch/leco/ --setup_mode semi_inat_strongaug --hparam_candidate inat --train_mode resnet50_scratch_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --seed 10
python train_per_class.py --data_dir /scratch/leco/ --setup_mode semi_inat_strongaug --hparam_candidate inat --train_mode resnet50_scratch_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --seed 100
python train_per_class.py --data_dir /scratch/leco/ --setup_mode semi_inat_strongaug --hparam_candidate inat --train_mode resnet50_scratch_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --seed 1000

# with ema
python train_per_class.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode semi_inat_strongaug --hparam_candidate inat --train_mode resnet50_scratch_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ 
python train_per_class.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode semi_inat_strongaug --hparam_candidate inat --train_mode resnet50_scratch_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --seed 1
python train_per_class.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode semi_inat_strongaug --hparam_candidate inat --train_mode resnet50_scratch_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --seed 10
python train_per_class.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode semi_inat_strongaug --hparam_candidate inat --train_mode resnet50_scratch_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --seed 100
python train_per_class.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode semi_inat_strongaug --hparam_candidate inat --train_mode resnet50_scratch_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --seed 1000


## ImageNet Weak Aug
# without ema
python train_per_class.py --data_dir /scratch/leco/ --setup_mode semi_inat_weakaug --hparam_candidate inat --train_mode resnet50_imagenet_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ 
python train_per_class.py --data_dir /scratch/leco/ --setup_mode semi_inat_weakaug --hparam_candidate inat --train_mode resnet50_imagenet_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --seed 1
python train_per_class.py --data_dir /scratch/leco/ --setup_mode semi_inat_weakaug --hparam_candidate inat --train_mode resnet50_imagenet_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --seed 10
python train_per_class.py --data_dir /scratch/leco/ --setup_mode semi_inat_weakaug --hparam_candidate inat --train_mode resnet50_imagenet_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --seed 100
python train_per_class.py --data_dir /scratch/leco/ --setup_mode semi_inat_weakaug --hparam_candidate inat --train_mode resnet50_imagenet_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --seed 1000

# with ema
python train_per_class.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode semi_inat_weakaug --hparam_candidate inat --train_mode resnet50_imagenet_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ 
python train_per_class.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode semi_inat_weakaug --hparam_candidate inat --train_mode resnet50_imagenet_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --seed 1
python train_per_class.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode semi_inat_weakaug --hparam_candidate inat --train_mode resnet50_imagenet_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --seed 10
python train_per_class.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode semi_inat_weakaug --hparam_candidate inat --train_mode resnet50_imagenet_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --seed 100
python train_per_class.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode semi_inat_weakaug --hparam_candidate inat --train_mode resnet50_imagenet_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --seed 1000


## ImageNet Strong aug
# without ema
python train_per_class.py --data_dir /scratch/leco/ --setup_mode semi_inat_strongaug --hparam_candidate inat --train_mode resnet50_imagenet_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ 
python train_per_class.py --data_dir /scratch/leco/ --setup_mode semi_inat_strongaug --hparam_candidate inat --train_mode resnet50_imagenet_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --seed 1
python train_per_class.py --data_dir /scratch/leco/ --setup_mode semi_inat_strongaug --hparam_candidate inat --train_mode resnet50_imagenet_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --seed 10
python train_per_class.py --data_dir /scratch/leco/ --setup_mode semi_inat_strongaug --hparam_candidate inat --train_mode resnet50_imagenet_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --seed 100
python train_per_class.py --data_dir /scratch/leco/ --setup_mode semi_inat_strongaug --hparam_candidate inat --train_mode resnet50_imagenet_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --seed 1000

# with ema
python train_per_class.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode semi_inat_strongaug --hparam_candidate inat --train_mode resnet50_imagenet_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ 
python train_per_class.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode semi_inat_strongaug --hparam_candidate inat --train_mode resnet50_imagenet_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --seed 1
python train_per_class.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode semi_inat_strongaug --hparam_candidate inat --train_mode resnet50_imagenet_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --seed 10
python train_per_class.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode semi_inat_strongaug --hparam_candidate inat --train_mode resnet50_imagenet_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --seed 100
python train_per_class.py --ema_decay 0.999 --data_dir /scratch/leco/ --setup_mode semi_inat_strongaug --hparam_candidate inat --train_mode resnet50_imagenet_0_finetune_pt_linear_1_finetune_pt_linear --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --seed 1000

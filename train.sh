python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --train_mode none_0_scratch_linear_1_scratch_linear
python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --train_mode none_0_scratch_linear_1_scratch_linear
python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1 --train_mode none_0_scratch_linear_1_scratch_linear
python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1 --train_mode none_0_scratch_linear_1_scratch_linear
python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 10 --train_mode none_0_scratch_linear_1_scratch_linear
python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 10 --train_mode none_0_scratch_linear_1_scratch_linear
python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 100 --train_mode none_0_scratch_linear_1_scratch_linear
python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 100 --train_mode none_0_scratch_linear_1_scratch_linear
python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1000 --train_mode none_0_scratch_linear_1_scratch_linear
python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1000 --train_mode none_0_scratch_linear_1_scratch_linear


python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --train_mode none_0_scratch_linear_1_finetune_prev_linear
python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --train_mode none_0_scratch_linear_1_finetune_prev_linear
python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1 --train_mode none_0_scratch_linear_1_finetune_prev_linear
python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1 --train_mode none_0_scratch_linear_1_finetune_prev_linear
python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 10 --train_mode none_0_scratch_linear_1_finetune_prev_linear
python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 10 --train_mode none_0_scratch_linear_1_finetune_prev_linear
python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 100 --train_mode none_0_scratch_linear_1_finetune_prev_linear
python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 100 --train_mode none_0_scratch_linear_1_finetune_prev_linear
python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1000 --train_mode none_0_scratch_linear_1_finetune_prev_linear
python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1000 --train_mode none_0_scratch_linear_1_finetune_prev_linear


CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --train_mode none_0_scratch_linear_1_finetune_prev_mlp
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --train_mode none_0_scratch_linear_1_finetune_prev_mlp
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1 --train_mode none_0_scratch_linear_1_finetune_prev_mlp
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1 --train_mode none_0_scratch_linear_1_finetune_prev_mlp
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 10 --train_mode none_0_scratch_linear_1_finetune_prev_mlp
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 10 --train_mode none_0_scratch_linear_1_finetune_prev_mlp
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 100 --train_mode none_0_scratch_linear_1_finetune_prev_mlp
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 100 --train_mode none_0_scratch_linear_1_finetune_prev_mlp
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1000 --train_mode none_0_scratch_linear_1_finetune_prev_mlp
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1000 --train_mode none_0_scratch_linear_1_finetune_prev_mlp


CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --train_mode resnet18_simclr_0_freeze_pt_linear_1_freeze_pt_linear
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --train_mode resnet18_simclr_0_freeze_pt_linear_1_freeze_pt_linear
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1 --train_mode resnet18_simclr_0_freeze_pt_linear_1_freeze_pt_linear
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1 --train_mode resnet18_simclr_0_freeze_pt_linear_1_freeze_pt_linear
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 10 --train_mode resnet18_simclr_0_freeze_pt_linear_1_freeze_pt_linear
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 10 --train_mode resnet18_simclr_0_freeze_pt_linear_1_freeze_pt_linear
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 100 --train_mode resnet18_simclr_0_freeze_pt_linear_1_freeze_pt_linear
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 100 --train_mode resnet18_simclr_0_freeze_pt_linear_1_freeze_pt_linear
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1000 --train_mode resnet18_simclr_0_freeze_pt_linear_1_freeze_pt_linear
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1000 --train_mode resnet18_simclr_0_freeze_pt_linear_1_freeze_pt_linear

CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --train_mode resnet18_simclr_0_freeze_pt_mlp_1_freeze_pt_mlp
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --train_mode resnet18_simclr_0_freeze_pt_mlp_1_freeze_pt_mlp
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1 --train_mode resnet18_simclr_0_freeze_pt_mlp_1_freeze_pt_mlp
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1 --train_mode resnet18_simclr_0_freeze_pt_mlp_1_freeze_pt_mlp
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 10 --train_mode resnet18_simclr_0_freeze_pt_mlp_1_freeze_pt_mlp
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 10 --train_mode resnet18_simclr_0_freeze_pt_mlp_1_freeze_pt_mlp
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 100 --train_mode resnet18_simclr_0_freeze_pt_mlp_1_freeze_pt_mlp
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 100 --train_mode resnet18_simclr_0_freeze_pt_mlp_1_freeze_pt_mlp
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1000 --train_mode resnet18_simclr_0_freeze_pt_mlp_1_freeze_pt_mlp
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1000 --train_mode resnet18_simclr_0_freeze_pt_mlp_1_freeze_pt_mlp

CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --train_mode resnet18_simclr_0_freeze_pt_mlp_1_freeze_pt_mlp_replace_last
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --train_mode resnet18_simclr_0_freeze_pt_mlp_1_freeze_pt_mlp_replace_last
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1 --train_mode resnet18_simclr_0_freeze_pt_mlp_1_freeze_pt_mlp_replace_last
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1 --train_mode resnet18_simclr_0_freeze_pt_mlp_1_freeze_pt_mlp_replace_last
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 10 --train_mode resnet18_simclr_0_freeze_pt_mlp_1_freeze_pt_mlp_replace_last
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 10 --train_mode resnet18_simclr_0_freeze_pt_mlp_1_freeze_pt_mlp_replace_last
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 100 --train_mode resnet18_simclr_0_freeze_pt_mlp_1_freeze_pt_mlp_replace_last
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 100 --train_mode resnet18_simclr_0_freeze_pt_mlp_1_freeze_pt_mlp_replace_last
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1000 --train_mode resnet18_simclr_0_freeze_pt_mlp_1_freeze_pt_mlp_replace_last
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1000 --train_mode resnet18_simclr_0_freeze_pt_mlp_1_freeze_pt_mlp_replace_last

CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --train_mode resnet18_simclr_0_finetune_pt_linear_1_finetune_pt_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --train_mode resnet18_simclr_0_finetune_pt_linear_1_finetune_pt_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1 --train_mode resnet18_simclr_0_finetune_pt_linear_1_finetune_pt_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1 --train_mode resnet18_simclr_0_finetune_pt_linear_1_finetune_pt_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 10 --train_mode resnet18_simclr_0_finetune_pt_linear_1_finetune_pt_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 10 --train_mode resnet18_simclr_0_finetune_pt_linear_1_finetune_pt_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 100 --train_mode resnet18_simclr_0_finetune_pt_linear_1_finetune_pt_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 100 --train_mode resnet18_simclr_0_finetune_pt_linear_1_finetune_pt_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1000 --train_mode resnet18_simclr_0_finetune_pt_linear_1_finetune_pt_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1000 --train_mode resnet18_simclr_0_finetune_pt_linear_1_finetune_pt_linear

CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --train_mode resnet18_simclr_0_finetune_pt_linear_1_finetune_prev_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --train_mode resnet18_simclr_0_finetune_pt_linear_1_finetune_prev_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1 --train_mode resnet18_simclr_0_finetune_pt_linear_1_finetune_prev_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1 --train_mode resnet18_simclr_0_finetune_pt_linear_1_finetune_prev_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 10 --train_mode resnet18_simclr_0_finetune_pt_linear_1_finetune_prev_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 10 --train_mode resnet18_simclr_0_finetune_pt_linear_1_finetune_prev_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 100 --train_mode resnet18_simclr_0_finetune_pt_linear_1_finetune_prev_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 100 --train_mode resnet18_simclr_0_finetune_pt_linear_1_finetune_prev_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1000 --train_mode resnet18_simclr_0_finetune_pt_linear_1_finetune_prev_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1000 --train_mode resnet18_simclr_0_finetune_pt_linear_1_finetune_prev_linear

python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --train_mode resnet18_simclr_0_finetune_pt_linear_1_finetune_prev_mlp
python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --train_mode resnet18_simclr_0_finetune_pt_linear_1_finetune_prev_mlp
python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1 --train_mode resnet18_simclr_0_finetune_pt_linear_1_finetune_prev_mlp
python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1 --train_mode resnet18_simclr_0_finetune_pt_linear_1_finetune_prev_mlp
python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 10 --train_mode resnet18_simclr_0_finetune_pt_linear_1_finetune_prev_mlp
python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 10 --train_mode resnet18_simclr_0_finetune_pt_linear_1_finetune_prev_mlp
python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 100 --train_mode resnet18_simclr_0_finetune_pt_linear_1_finetune_prev_mlp
python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 100 --train_mode resnet18_simclr_0_finetune_pt_linear_1_finetune_prev_mlp
python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1000 --train_mode resnet18_simclr_0_finetune_pt_linear_1_finetune_prev_mlp
python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1000 --train_mode resnet18_simclr_0_finetune_pt_linear_1_finetune_prev_mlp


# resnet18_simclr_pl
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --train_mode resnet18_simclr_pl_0_freeze_pt_linear_1_freeze_pt_linear
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --train_mode resnet18_simclr_pl_0_freeze_pt_linear_1_freeze_pt_linear
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1 --train_mode resnet18_simclr_pl_0_freeze_pt_linear_1_freeze_pt_linear
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1 --train_mode resnet18_simclr_pl_0_freeze_pt_linear_1_freeze_pt_linear
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 10 --train_mode resnet18_simclr_pl_0_freeze_pt_linear_1_freeze_pt_linear
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 10 --train_mode resnet18_simclr_pl_0_freeze_pt_linear_1_freeze_pt_linear
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 100 --train_mode resnet18_simclr_pl_0_freeze_pt_linear_1_freeze_pt_linear
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 100 --train_mode resnet18_simclr_pl_0_freeze_pt_linear_1_freeze_pt_linear
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1000 --train_mode resnet18_simclr_pl_0_freeze_pt_linear_1_freeze_pt_linear
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1000 --train_mode resnet18_simclr_pl_0_freeze_pt_linear_1_freeze_pt_linear

CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --train_mode resnet18_simclr_pl_0_freeze_pt_mlp_1_freeze_pt_mlp
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --train_mode resnet18_simclr_pl_0_freeze_pt_mlp_1_freeze_pt_mlp
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1 --train_mode resnet18_simclr_pl_0_freeze_pt_mlp_1_freeze_pt_mlp
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1 --train_mode resnet18_simclr_pl_0_freeze_pt_mlp_1_freeze_pt_mlp
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 10 --train_mode resnet18_simclr_pl_0_freeze_pt_mlp_1_freeze_pt_mlp
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 10 --train_mode resnet18_simclr_pl_0_freeze_pt_mlp_1_freeze_pt_mlp
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 100 --train_mode resnet18_simclr_pl_0_freeze_pt_mlp_1_freeze_pt_mlp
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 100 --train_mode resnet18_simclr_pl_0_freeze_pt_mlp_1_freeze_pt_mlp
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1000 --train_mode resnet18_simclr_pl_0_freeze_pt_mlp_1_freeze_pt_mlp
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1000 --train_mode resnet18_simclr_pl_0_freeze_pt_mlp_1_freeze_pt_mlp

CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --train_mode resnet18_simclr_pl_0_freeze_pt_mlp_1_freeze_pt_mlp_replace_last
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --train_mode resnet18_simclr_pl_0_freeze_pt_mlp_1_freeze_pt_mlp_replace_last
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1 --train_mode resnet18_simclr_pl_0_freeze_pt_mlp_1_freeze_pt_mlp_replace_last
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1 --train_mode resnet18_simclr_pl_0_freeze_pt_mlp_1_freeze_pt_mlp_replace_last
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 10 --train_mode resnet18_simclr_pl_0_freeze_pt_mlp_1_freeze_pt_mlp_replace_last
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 10 --train_mode resnet18_simclr_pl_0_freeze_pt_mlp_1_freeze_pt_mlp_replace_last
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 100 --train_mode resnet18_simclr_pl_0_freeze_pt_mlp_1_freeze_pt_mlp_replace_last
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 100 --train_mode resnet18_simclr_pl_0_freeze_pt_mlp_1_freeze_pt_mlp_replace_last
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1000 --train_mode resnet18_simclr_pl_0_freeze_pt_mlp_1_freeze_pt_mlp_replace_last
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1000 --train_mode resnet18_simclr_pl_0_freeze_pt_mlp_1_freeze_pt_mlp_replace_last

CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --train_mode resnet18_simclr_pl_0_finetune_pt_linear_1_finetune_pt_linear
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --train_mode resnet18_simclr_pl_0_finetune_pt_linear_1_finetune_pt_linear
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1 --train_mode resnet18_simclr_pl_0_finetune_pt_linear_1_finetune_pt_linear
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1 --train_mode resnet18_simclr_pl_0_finetune_pt_linear_1_finetune_pt_linear
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 10 --train_mode resnet18_simclr_pl_0_finetune_pt_linear_1_finetune_pt_linear
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 10 --train_mode resnet18_simclr_pl_0_finetune_pt_linear_1_finetune_pt_linear
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 100 --train_mode resnet18_simclr_pl_0_finetune_pt_linear_1_finetune_pt_linear
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 100 --train_mode resnet18_simclr_pl_0_finetune_pt_linear_1_finetune_pt_linear
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1000 --train_mode resnet18_simclr_pl_0_finetune_pt_linear_1_finetune_pt_linear
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1000 --train_mode resnet18_simclr_pl_0_finetune_pt_linear_1_finetune_pt_linear

CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --train_mode resnet18_simclr_pl_0_finetune_pt_linear_1_finetune_prev_linear
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --train_mode resnet18_simclr_pl_0_finetune_pt_linear_1_finetune_prev_linear
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1 --train_mode resnet18_simclr_pl_0_finetune_pt_linear_1_finetune_prev_linear
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1 --train_mode resnet18_simclr_pl_0_finetune_pt_linear_1_finetune_prev_linear
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 10 --train_mode resnet18_simclr_pl_0_finetune_pt_linear_1_finetune_prev_linear
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 10 --train_mode resnet18_simclr_pl_0_finetune_pt_linear_1_finetune_prev_linear
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 100 --train_mode resnet18_simclr_pl_0_finetune_pt_linear_1_finetune_prev_linear
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 100 --train_mode resnet18_simclr_pl_0_finetune_pt_linear_1_finetune_prev_linear
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1000 --train_mode resnet18_simclr_pl_0_finetune_pt_linear_1_finetune_prev_linear
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1000 --train_mode resnet18_simclr_pl_0_finetune_pt_linear_1_finetune_prev_linear

CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --train_mode resnet18_simclr_pl_0_finetune_pt_linear_1_finetune_prev_mlp
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --train_mode resnet18_simclr_pl_0_finetune_pt_linear_1_finetune_prev_mlp
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1 --train_mode resnet18_simclr_pl_0_finetune_pt_linear_1_finetune_prev_mlp
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1 --train_mode resnet18_simclr_pl_0_finetune_pt_linear_1_finetune_prev_mlp
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 10 --train_mode resnet18_simclr_pl_0_finetune_pt_linear_1_finetune_prev_mlp
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 10 --train_mode resnet18_simclr_pl_0_finetune_pt_linear_1_finetune_prev_mlp
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 100 --train_mode resnet18_simclr_pl_0_finetune_pt_linear_1_finetune_prev_mlp
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 100 --train_mode resnet18_simclr_pl_0_finetune_pt_linear_1_finetune_prev_mlp
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1000 --train_mode resnet18_simclr_pl_0_finetune_pt_linear_1_finetune_prev_mlp
CUDA_VISIBLE_DEVICES=1 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1000 --train_mode resnet18_simclr_pl_0_finetune_pt_linear_1_finetune_prev_mlp

# resnet18_moco_v2_pl
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --train_mode resnet18_moco_v2_pl_0_freeze_pt_linear_1_freeze_pt_linear
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --train_mode resnet18_moco_v2_pl_0_freeze_pt_linear_1_freeze_pt_linear
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1 --train_mode resnet18_moco_v2_pl_0_freeze_pt_linear_1_freeze_pt_linear
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1 --train_mode resnet18_moco_v2_pl_0_freeze_pt_linear_1_freeze_pt_linear
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 10 --train_mode resnet18_moco_v2_pl_0_freeze_pt_linear_1_freeze_pt_linear
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 10 --train_mode resnet18_moco_v2_pl_0_freeze_pt_linear_1_freeze_pt_linear
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 100 --train_mode resnet18_moco_v2_pl_0_freeze_pt_linear_1_freeze_pt_linear
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 100 --train_mode resnet18_moco_v2_pl_0_freeze_pt_linear_1_freeze_pt_linear
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1000 --train_mode resnet18_moco_v2_pl_0_freeze_pt_linear_1_freeze_pt_linear
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1000 --train_mode resnet18_moco_v2_pl_0_freeze_pt_linear_1_freeze_pt_linear

CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --train_mode resnet18_moco_v2_pl_0_freeze_pt_mlp_1_freeze_pt_mlp
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --train_mode resnet18_moco_v2_pl_0_freeze_pt_mlp_1_freeze_pt_mlp
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1 --train_mode resnet18_moco_v2_pl_0_freeze_pt_mlp_1_freeze_pt_mlp
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1 --train_mode resnet18_moco_v2_pl_0_freeze_pt_mlp_1_freeze_pt_mlp
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 10 --train_mode resnet18_moco_v2_pl_0_freeze_pt_mlp_1_freeze_pt_mlp
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 10 --train_mode resnet18_moco_v2_pl_0_freeze_pt_mlp_1_freeze_pt_mlp
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 100 --train_mode resnet18_moco_v2_pl_0_freeze_pt_mlp_1_freeze_pt_mlp
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 100 --train_mode resnet18_moco_v2_pl_0_freeze_pt_mlp_1_freeze_pt_mlp
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1000 --train_mode resnet18_moco_v2_pl_0_freeze_pt_mlp_1_freeze_pt_mlp
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1000 --train_mode resnet18_moco_v2_pl_0_freeze_pt_mlp_1_freeze_pt_mlp

CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --train_mode resnet18_moco_v2_pl_0_freeze_pt_mlp_1_freeze_pt_mlp_replace_last
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --train_mode resnet18_moco_v2_pl_0_freeze_pt_mlp_1_freeze_pt_mlp_replace_last
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1 --train_mode resnet18_moco_v2_pl_0_freeze_pt_mlp_1_freeze_pt_mlp_replace_last
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1 --train_mode resnet18_moco_v2_pl_0_freeze_pt_mlp_1_freeze_pt_mlp_replace_last
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 10 --train_mode resnet18_moco_v2_pl_0_freeze_pt_mlp_1_freeze_pt_mlp_replace_last
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 10 --train_mode resnet18_moco_v2_pl_0_freeze_pt_mlp_1_freeze_pt_mlp_replace_last
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 100 --train_mode resnet18_moco_v2_pl_0_freeze_pt_mlp_1_freeze_pt_mlp_replace_last
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 100 --train_mode resnet18_moco_v2_pl_0_freeze_pt_mlp_1_freeze_pt_mlp_replace_last
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1000 --train_mode resnet18_moco_v2_pl_0_freeze_pt_mlp_1_freeze_pt_mlp_replace_last
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1000 --train_mode resnet18_moco_v2_pl_0_freeze_pt_mlp_1_freeze_pt_mlp_replace_last

CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --train_mode resnet18_moco_v2_pl_0_finetune_pt_linear_1_finetune_pt_linear
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --train_mode resnet18_moco_v2_pl_0_finetune_pt_linear_1_finetune_pt_linear
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1 --train_mode resnet18_moco_v2_pl_0_finetune_pt_linear_1_finetune_pt_linear
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1 --train_mode resnet18_moco_v2_pl_0_finetune_pt_linear_1_finetune_pt_linear
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 10 --train_mode resnet18_moco_v2_pl_0_finetune_pt_linear_1_finetune_pt_linear
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 10 --train_mode resnet18_moco_v2_pl_0_finetune_pt_linear_1_finetune_pt_linear
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 100 --train_mode resnet18_moco_v2_pl_0_finetune_pt_linear_1_finetune_pt_linear
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 100 --train_mode resnet18_moco_v2_pl_0_finetune_pt_linear_1_finetune_pt_linear
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1000 --train_mode resnet18_moco_v2_pl_0_finetune_pt_linear_1_finetune_pt_linear
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1000 --train_mode resnet18_moco_v2_pl_0_finetune_pt_linear_1_finetune_pt_linear

CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --train_mode resnet18_moco_v2_pl_0_finetune_pt_linear_1_finetune_prev_linear
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --train_mode resnet18_moco_v2_pl_0_finetune_pt_linear_1_finetune_prev_linear
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1 --train_mode resnet18_moco_v2_pl_0_finetune_pt_linear_1_finetune_prev_linear
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1 --train_mode resnet18_moco_v2_pl_0_finetune_pt_linear_1_finetune_prev_linear
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 10 --train_mode resnet18_moco_v2_pl_0_finetune_pt_linear_1_finetune_prev_linear
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 10 --train_mode resnet18_moco_v2_pl_0_finetune_pt_linear_1_finetune_prev_linear
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 100 --train_mode resnet18_moco_v2_pl_0_finetune_pt_linear_1_finetune_prev_linear
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 100 --train_mode resnet18_moco_v2_pl_0_finetune_pt_linear_1_finetune_prev_linear
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1000 --train_mode resnet18_moco_v2_pl_0_finetune_pt_linear_1_finetune_prev_linear
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1000 --train_mode resnet18_moco_v2_pl_0_finetune_pt_linear_1_finetune_prev_linear

CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --train_mode resnet18_moco_v2_pl_0_finetune_pt_linear_1_finetune_prev_mlp
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --train_mode resnet18_moco_v2_pl_0_finetune_pt_linear_1_finetune_prev_mlp
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1 --train_mode resnet18_moco_v2_pl_0_finetune_pt_linear_1_finetune_prev_mlp
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1 --train_mode resnet18_moco_v2_pl_0_finetune_pt_linear_1_finetune_prev_mlp
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 10 --train_mode resnet18_moco_v2_pl_0_finetune_pt_linear_1_finetune_prev_mlp
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 10 --train_mode resnet18_moco_v2_pl_0_finetune_pt_linear_1_finetune_prev_mlp
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 100 --train_mode resnet18_moco_v2_pl_0_finetune_pt_linear_1_finetune_prev_mlp
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 100 --train_mode resnet18_moco_v2_pl_0_finetune_pt_linear_1_finetune_prev_mlp
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1000 --train_mode resnet18_moco_v2_pl_0_finetune_pt_linear_1_finetune_prev_mlp
CUDA_VISIBLE_DEVICES=2 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1000 --train_mode resnet18_moco_v2_pl_0_finetune_pt_linear_1_finetune_prev_mlp


# resnet18_byol_pl
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --train_mode resnet18_byol_pl_0_freeze_pt_linear_1_freeze_pt_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --train_mode resnet18_byol_pl_0_freeze_pt_linear_1_freeze_pt_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1 --train_mode resnet18_byol_pl_0_freeze_pt_linear_1_freeze_pt_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1 --train_mode resnet18_byol_pl_0_freeze_pt_linear_1_freeze_pt_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 10 --train_mode resnet18_byol_pl_0_freeze_pt_linear_1_freeze_pt_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 10 --train_mode resnet18_byol_pl_0_freeze_pt_linear_1_freeze_pt_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 100 --train_mode resnet18_byol_pl_0_freeze_pt_linear_1_freeze_pt_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 100 --train_mode resnet18_byol_pl_0_freeze_pt_linear_1_freeze_pt_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1000 --train_mode resnet18_byol_pl_0_freeze_pt_linear_1_freeze_pt_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1000 --train_mode resnet18_byol_pl_0_freeze_pt_linear_1_freeze_pt_linear

CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --train_mode resnet18_byol_pl_0_freeze_pt_mlp_1_freeze_pt_mlp
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --train_mode resnet18_byol_pl_0_freeze_pt_mlp_1_freeze_pt_mlp
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1 --train_mode resnet18_byol_pl_0_freeze_pt_mlp_1_freeze_pt_mlp
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1 --train_mode resnet18_byol_pl_0_freeze_pt_mlp_1_freeze_pt_mlp
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 10 --train_mode resnet18_byol_pl_0_freeze_pt_mlp_1_freeze_pt_mlp
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 10 --train_mode resnet18_byol_pl_0_freeze_pt_mlp_1_freeze_pt_mlp
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 100 --train_mode resnet18_byol_pl_0_freeze_pt_mlp_1_freeze_pt_mlp
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 100 --train_mode resnet18_byol_pl_0_freeze_pt_mlp_1_freeze_pt_mlp
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1000 --train_mode resnet18_byol_pl_0_freeze_pt_mlp_1_freeze_pt_mlp
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1000 --train_mode resnet18_byol_pl_0_freeze_pt_mlp_1_freeze_pt_mlp

CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --train_mode resnet18_byol_pl_0_freeze_pt_mlp_1_freeze_pt_mlp_replace_last
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --train_mode resnet18_byol_pl_0_freeze_pt_mlp_1_freeze_pt_mlp_replace_last
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1 --train_mode resnet18_byol_pl_0_freeze_pt_mlp_1_freeze_pt_mlp_replace_last
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1 --train_mode resnet18_byol_pl_0_freeze_pt_mlp_1_freeze_pt_mlp_replace_last
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 10 --train_mode resnet18_byol_pl_0_freeze_pt_mlp_1_freeze_pt_mlp_replace_last
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 10 --train_mode resnet18_byol_pl_0_freeze_pt_mlp_1_freeze_pt_mlp_replace_last
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 100 --train_mode resnet18_byol_pl_0_freeze_pt_mlp_1_freeze_pt_mlp_replace_last
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 100 --train_mode resnet18_byol_pl_0_freeze_pt_mlp_1_freeze_pt_mlp_replace_last
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1000 --train_mode resnet18_byol_pl_0_freeze_pt_mlp_1_freeze_pt_mlp_replace_last
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1000 --train_mode resnet18_byol_pl_0_freeze_pt_mlp_1_freeze_pt_mlp_replace_last

CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --train_mode resnet18_byol_pl_0_finetune_pt_linear_1_finetune_pt_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --train_mode resnet18_byol_pl_0_finetune_pt_linear_1_finetune_pt_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1 --train_mode resnet18_byol_pl_0_finetune_pt_linear_1_finetune_pt_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1 --train_mode resnet18_byol_pl_0_finetune_pt_linear_1_finetune_pt_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 10 --train_mode resnet18_byol_pl_0_finetune_pt_linear_1_finetune_pt_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 10 --train_mode resnet18_byol_pl_0_finetune_pt_linear_1_finetune_pt_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 100 --train_mode resnet18_byol_pl_0_finetune_pt_linear_1_finetune_pt_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 100 --train_mode resnet18_byol_pl_0_finetune_pt_linear_1_finetune_pt_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1000 --train_mode resnet18_byol_pl_0_finetune_pt_linear_1_finetune_pt_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1000 --train_mode resnet18_byol_pl_0_finetune_pt_linear_1_finetune_pt_linear

CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --train_mode resnet18_byol_pl_0_finetune_pt_linear_1_finetune_prev_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --train_mode resnet18_byol_pl_0_finetune_pt_linear_1_finetune_prev_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1 --train_mode resnet18_byol_pl_0_finetune_pt_linear_1_finetune_prev_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1 --train_mode resnet18_byol_pl_0_finetune_pt_linear_1_finetune_prev_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 10 --train_mode resnet18_byol_pl_0_finetune_pt_linear_1_finetune_prev_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 10 --train_mode resnet18_byol_pl_0_finetune_pt_linear_1_finetune_prev_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 100 --train_mode resnet18_byol_pl_0_finetune_pt_linear_1_finetune_prev_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 100 --train_mode resnet18_byol_pl_0_finetune_pt_linear_1_finetune_prev_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1000 --train_mode resnet18_byol_pl_0_finetune_pt_linear_1_finetune_prev_linear
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1000 --train_mode resnet18_byol_pl_0_finetune_pt_linear_1_finetune_prev_linear

CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --train_mode resnet18_byol_pl_0_finetune_pt_linear_1_finetune_prev_mlp
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --train_mode resnet18_byol_pl_0_finetune_pt_linear_1_finetune_prev_mlp
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1 --train_mode resnet18_byol_pl_0_finetune_pt_linear_1_finetune_prev_mlp
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1 --train_mode resnet18_byol_pl_0_finetune_pt_linear_1_finetune_prev_mlp
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 10 --train_mode resnet18_byol_pl_0_finetune_pt_linear_1_finetune_prev_mlp
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 10 --train_mode resnet18_byol_pl_0_finetune_pt_linear_1_finetune_prev_mlp
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 100 --train_mode resnet18_byol_pl_0_finetune_pt_linear_1_finetune_prev_mlp
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 100 --train_mode resnet18_byol_pl_0_finetune_pt_linear_1_finetune_prev_mlp
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_default --seed 1000 --train_mode resnet18_byol_pl_0_finetune_pt_linear_1_finetune_prev_mlp
CUDA_VISIBLE_DEVICES=3 python train.py --setup_mode cifar10_buffer_2000 --hparam_str cifar10_decay --seed 1000 --train_mode resnet18_byol_pl_0_finetune_pt_linear_1_finetune_prev_mlp
CUDA_VISIBLE_DEVICES=3 
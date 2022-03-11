python train_semi.py --data_dir /scratch/leco/ --setup_mode cifar10_weakaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear 
python train_semi.py --data_dir /scratch/leco/ --setup_mode cifar10_strongaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear 
python train_semi.py --data_dir /scratch/leco/ --setup_mode cifar10_weakaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear --seed 1
python train_semi.py --data_dir /scratch/leco/ --setup_mode cifar10_strongaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear --seed 1
python train_semi.py --data_dir /scratch/leco/ --setup_mode cifar10_weakaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear --seed 10
python train_semi.py --data_dir /scratch/leco/ --setup_mode cifar10_strongaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear --seed 10
python train_semi.py --data_dir /scratch/leco/ --setup_mode cifar10_weakaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear --seed 100
python train_semi.py --data_dir /scratch/leco/ --setup_mode cifar10_strongaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear --seed 100
python train_semi.py --data_dir /scratch/leco/ --setup_mode cifar10_weakaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear --seed 1000
python train_semi.py --data_dir /scratch/leco/ --setup_mode cifar10_strongaug_train_2000_val_500 --train_mode wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear --seed 1000

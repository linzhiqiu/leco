# To train wideres_28_2 on STL10
python pretrain.py --model_save_dir /data3/zhiqiul/self_supervised_models/wideres_28_2/ --arch wideres_28_2 --pretrain scratch
python pretrain.py --model_save_dir /data3/zhiqiul/self_supervised_models/wideres_28_2/ --arch wideres_28_2 --pretrain byol_stl10
python pretrain.py --model_save_dir /data3/zhiqiul/self_supervised_models/wideres_28_2/ --arch wideres_28_2 --pretrain moco_v2_stl10
python pretrain.py --model_save_dir /data3/zhiqiul/self_supervised_models/wideres_28_2/ --arch wideres_28_2 --pretrain simclr_stl10 


# # To train resnet50 on SemiInat2021
python pretrain.py --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --arch resnet50 --pretrain scratch
python pretrain.py --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --arch resnet50 --pretrain imagenet
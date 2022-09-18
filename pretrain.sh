# To save wideres_28_2 from scratch for CIFAR
python pretrain.py --model_save_dir /data3/zhiqiul/self_supervised_models/wideres_28_2/ --arch wideres_28_2 --pretrain scratch

# To save resnet50 from scratch for SemiInat2021
python pretrain.py --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --arch resnet50 --pretrain scratch
# python pretrain.py --model_save_dir /data3/zhiqiul/self_supervised_models/resnet50/ --arch resnet50 --pretrain imagenet
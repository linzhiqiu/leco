# To train resnet18 on STL10
python pretrain.py --model_save_dir /data3/zhiqiul/self_supervised_models/resnet18/ --dataset_name stl10 --model resnet18_moco_v2_pl --max_epochs 320 --queue_size 65536
python pretrain.py --model_save_dir /data3/zhiqiul/self_supervised_models/resnet18/ --dataset_name stl10 --model resnet18_simclr_pl --max_epochs 320
python pretrain.py --model_save_dir /data3/zhiqiul/self_supervised_models/resnet18/ --dataset_name stl10 --model resnet18_byol_pl --max_epochs 320 --queue_size 65536

# To train resnet50 on SemiInat2021
python pretrain.py --model_save_dir /data3/zhiqiul/self_supervised_models/inat2021_resnet50/ --dataset_name inat2021 --transform_crop_size 224 --lr 0.03 --batch_size 64 --queue_size 2048 --model resnet50_moco_v2_pl --max_epochs 800
# These two cannot run on one gpu
python pretrain.py --model_save_dir /data3/zhiqiul/self_supervised_models/inat2021_resnet50/ --dataset_name inat2021 --transform_crop_size 224 --lr 0.03 --batch_size 64 --model resnet50_simclr_pl --max_epochs 800
python pretrain.py --model_save_dir /data3/zhiqiul/self_supervised_models/inat2021_resnet50/ --dataset_name inat2021 --transform_crop_size 224 --lr 0.03 --batch_size 64 --queue_size 2048 --model resnet50_byol_pl --max_epochs 800
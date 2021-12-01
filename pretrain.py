import os
import pytorch_lightning as pl
from self_supervised.moco import MoCoMethod 
from self_supervised.moco import MoCoMethodParams

import argparse
import configs

argparser = argparse.ArgumentParser()
argparser.add_argument("--data_dir", 
                        default='/scratch/leco/',
                        help="Where the dataset will be saved.")
argparser.add_argument("--model_save_dir", 
                        default='/data3/zhiqiul/self_supervised_models/resnet18/',
                        help="Where the self-supervised pre-trained models will be saved.")
argparser.add_argument("--dataset_name",
                        type=str,
                        default='stl10',
                        choices=['stl10', 'inat2021'],
                        help="The unlabeled data")   
argparser.add_argument("--model",
                        type=str,
                        default='resnet18_moco_v2_pl',
                        choices=configs.PRETRAINED_MODES,
                        help="The pre-trained model") 
argparser.add_argument("--transform_crop_size",
                        type=int,
                        default=96,
                        help="The transform crop size")
argparser.add_argument("--lr",
                        type=float,
                        default=0.5,
                        help="The learning rate")     
argparser.add_argument("--gpus",
                        type=int,
                        default=1,
                        help="The numbers of gpus to use") 
argparser.add_argument("--queue_size",
                        type=int,
                        default=65536,
                        help="The queue size for MoCo/BYOL") 
argparser.add_argument("--batch_size",
                        type=int,
                        default=256,
                        help="The batch size")   
argparser.add_argument("--max_epochs",
                        type=int,
                        default=320,
                        help="The numbers of epochs")   

args = argparser.parse_args()

os.environ["DATA_PATH"] = args.data_dir

if args.model == 'resnet18_moco_v2_pl':
    params = MoCoMethodParams(
        encoder_arch='resnet18', 
        embedding_dim=512,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        transform_crop_size=args.transform_crop_size,
        lr=args.lr,
        K=args.queue_size,
    )
    model = MoCoMethod(params)
elif args.model == 'resnet18_byol_pl':
    params = MoCoMethodParams(
        encoder_arch='resnet18',
        embedding_dim=512,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        transform_crop_size=args.transform_crop_size,
        lr=args.lr,
        K=args.queue_size,
        prediction_mlp_layers = 2,
        mlp_normalization = "bn",
        loss_type = "ip",
        use_negative_examples_from_queue = False,
        use_both_augmentations_as_queries = True,
        use_momentum_schedule = True,
        optimizer_name = "lars",
        exclude_matching_parameters_from_lars = [".bias", ".bn"],
        loss_constant_factor = 2
    )
    model = MoCoMethod(params)
elif args.model == 'resnet18_simclr_pl':
    hparams = MoCoMethodParams(
        encoder_arch='resnet18',
        embedding_dim=512,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        transform_crop_size=args.transform_crop_size,
        lr=args.lr,
        use_negative_examples_from_batch=True,
        use_negative_examples_from_queue=False,
        K=0,
        m=0.0,
        use_both_augmentations_as_queries=True,
    )
    model = MoCoMethod(hparams)
elif args.model == 'resnet50_moco_v2_pl':
    params = MoCoMethodParams(
        encoder_arch='resnet50', 
        embedding_dim=2048,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        transform_crop_size=args.transform_crop_size,
        lr=args.lr,
        K=args.queue_size,
    )
    model = MoCoMethod(params)
elif args.model == 'resnet50_byol_pl':
    params = MoCoMethodParams(
        encoder_arch='resnet50',
        embedding_dim=2048,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        transform_crop_size=args.transform_crop_size,
        lr=args.lr,
        K=args.queue_size,
        prediction_mlp_layers = 2,
        mlp_normalization = "bn",
        loss_type = "ip",
        use_negative_examples_from_queue = False,
        use_both_augmentations_as_queries = True,
        use_momentum_schedule = True,
        optimizer_name = "lars",
        exclude_matching_parameters_from_lars = [".bias", ".bn"],
        loss_constant_factor = 2
    )
    model = MoCoMethod(params)
elif args.model == 'resnet50_simclr_pl':
    hparams = MoCoMethodParams(
        encoder_arch='resnet50',
        embedding_dim=2048,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        transform_crop_size=args.transform_crop_size,
        lr=args.lr,
        use_negative_examples_from_batch=True,
        use_negative_examples_from_queue=False,
        K=0,
        m=0.0,
        use_both_augmentations_as_queries=True,
    )
    model = MoCoMethod(hparams)

trainer = pl.Trainer(gpus=args.gpus, max_epochs=args.max_epochs)    
trainer.fit(model) 
trainer.save_checkpoint(f"{args.model_save_dir}/{args.model}.ckpt")
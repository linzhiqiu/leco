import os
import pytorch_lightning as pl
from self_supervised.moco import MoCoMethod 
from self_supervised.moco import MoCoMethodParams

import torch
import argparse
import configs
import model_zoo

argparser = argparse.ArgumentParser()
argparser.add_argument("--data_dir", 
                        default='/scratch/leco/',
                        help="Where the dataset will be saved.")
argparser.add_argument("--model_save_dir", 
                        default='/data3/zhiqiul/self_supervised_models/wideres_28_2/',
                        help="Where the self-supervised pre-trained models will be saved.")
# argparser.add_argument("--dataset_name",
#                         type=str,
#                         default='stl10',
#                         choices=['stl10', 'inat2021'],
#                         help="The unlabeled data") 
argparser.add_argument("--arch",
                        type=str,
                        default='wideres_28_2',
                        choices=configs.ARCHS,
                        help="The backbone architecture")   
argparser.add_argument("--pretrain",
                        type=str,
                        default='moco_v2',
                        choices=configs.PRETRAINED_MODES,
                        help="The pre-trained mode") 
# argparser.add_argument("--transform_crop_size",
#                         type=int,
#                         default=96,
#                         help="The transform crop size")
# argparser.add_argument("--lr",
#                         type=float,
#                         default=0.5,
#                         help="The learning rate")     
argparser.add_argument("--gpus",
                        type=int,
                        default=1,
                        help="The numbers of gpus to use") 
# argparser.add_argument("--queue_size",
#                         type=int,
#                         default=65536,
#                         help="The queue size for MoCo/BYOL") 
# argparser.add_argument("--batch_size",
#                         type=int,
#                         default=256,
#                         help="The batch size")   
# argparser.add_argument("--max_epochs",
#                         type=int,
#                         default=320,
#                         help="The numbers of epochs")   

def pretrain(data_dir,
             model_save_dir,
             arch,
             pretrain):
    os.environ["DATA_PATH"] = data_dir

    embedding_dim = get_fc_size(arch)

    checkpoint_path = get_checkpoint_path(model_save_dir,
                                          arch,
                                          pretrain)
    if os.path.exists(checkpoint_path):
        print(f"{checkpoint_path} already exists. ")
        raise FileExistsError()

    if args.pretrain == 'scratch':
        model = getattr(model_zoo, arch)()
        torch.save(model, checkpoint_path)
        return
    
    # TODO: Fix the self_supervised library
    if args.pretrain == 'moco_v2_stl10':
        params = MoCoMethodParams(
            encoder_arch=arch, 
            embedding_dim=embedding_dim,
            dataset_name='stl10',
            batch_size=256,
            transform_crop_size=96,
            lr=0.5,
            K=65536,
        )
        model = MoCoMethod(params)
    elif args.pretrain == 'byol_stl10':
        params = MoCoMethodParams(
            encoder_arch=arch,
            embedding_dim=embedding_dim,
            dataset_name='stl10',
            batch_size=256,
            transform_crop_size=96,
            lr=0.5,
            K=65536,
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
    elif args.pretrain == 'simclr_stl10':
        hparams = MoCoMethodParams(
            encoder_arch=arch,
            embedding_dim=embedding_dim,
            dataset_name='stl10',
            batch_size=256,
            transform_crop_size=96,
            lr=0.5,
            use_negative_examples_from_batch=True,
            use_negative_examples_from_queue=False,
            K=0,
            m=0.0,
            use_both_augmentations_as_queries=True,
        )
        model = MoCoMethod(hparams)
    else:
        raise NotImplementedError()

    trainer = pl.Trainer(gpus=args.gpus, max_epochs=320)    
    trainer.fit(model) 
    trainer.save_checkpoint(checkpoint_path) # TODO: Look into this func, see how the checkpoint is saved

def get_checkpoint_path(model_save_dir,
                        arch,
                        pretrain):
    checkpoint_path = os.path.join(model_save_dir,
                                   f"{arch}_{pretrain}.ckpt")
    return checkpoint_path

def load_from_checkpoint(model_save_dir,
                         arch,
                         pretrain):
    checkpoint_path = get_checkpoint_path(model_save_dir,
                                          arch,
                                          pretrain)
    if pretrain == 'scratch':
        return torch.load(checkpoint_path)
    elif pretrain in ['moco_v2_stl10', 'byol_stl10', 'simclr_stl10']:
        model = MoCoMethod.load_from_checkpoint(checkpoint_path)
        model = model.__dict__['_modules']['model']
        return model
    else:
        raise NotImplementedError()

def get_fc_size(arch):
    if arch == 'wideres_28_2':
        return 128
    # elif arch == 'resnet18':
    #     embedding_dim = 512
    # elif arch == 'resnet50':
    #     embedding_dim = 2048
    else:
        raise NotImplementedError()

if __name__ == '__main__':
    args = argparser.parse_args()
    pretrain(args.data_dir,
             args.model_save_dir,
             args.arch,
             args.pretrain)
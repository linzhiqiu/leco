import os

import torch
import argparse
import configs
import model_zoo

argparser = argparse.ArgumentParser()
argparser.add_argument("--data_dir", 
                       default='/scratch/leco/',
                       help="Where the dataset is saved.")
argparser.add_argument("--model_save_dir", 
                       default='/data3/zhiqiul/self_supervised_models/wideres_28_2/',
                       help="Where the self-supervised pre-trained models will be saved.")
argparser.add_argument("--arch",
                       type=str,
                       default='wideres_28_2',
                       choices=configs.ARCHS,
                       help="The backbone architecture")   
argparser.add_argument("--pretrain",
                       type=str,
                       default='scratch',
                       choices=configs.PRETRAINED_MODES,
                       help="The pre-trained mode") 
argparser.add_argument("--gpus",
                       type=int,
                       default=1,
                       help="The numbers of gpus to use") 


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
    elif args.pretrain == 'imagenet':
        model = getattr(model_zoo, arch+"_imagenet")()
        torch.save(model, checkpoint_path)
        return
    else:
        raise NotImplementedError()


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
    elif pretrain == 'imagenet':
        return torch.load(checkpoint_path)
    else:
        raise NotImplementedError()


def get_fc_size(arch):
    if arch == 'wideres_28_2':
        return 128
    # elif arch == 'resnet18':
    #     embedding_dim = 512
    elif arch == 'resnet50':
        return 2048
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    args = argparser.parse_args()
    pretrain(args.data_dir,
             args.model_save_dir,
             args.arch,
             args.pretrain)
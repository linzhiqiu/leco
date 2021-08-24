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
argparser.add_argument("--model",
                        type=str,
                        default='resnet18_moco_v2_pl',
                        choices=configs.PRETRAINED_MODES,
                        help="The pre-trained model")   

args = argparser.parse_args()

os.environ["DATA_PATH"] = args.data_dir

if args.model == 'resnet18_moco_v2_pl':
    model = MoCoMethod()
elif args.model == 'resnet18_byol_pl':
    params = MoCoMethodParams(
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
        use_negative_examples_from_batch=True,
        use_negative_examples_from_queue=False,
        K=0,
        m=0.0,
        use_both_augmentations_as_queries=True,
    )
    model = MoCoMethod(hparams)

trainer = pl.Trainer(gpus=1, max_epochs=320)    
trainer.fit(model) 
trainer.save_checkpoint(f"{args.model_save_dir}/{args.model}.ckpt")
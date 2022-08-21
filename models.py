import torch
import torch.nn as nn
import torchvision
from torch.optim.lr_scheduler import LambdaLR
from self_supervised.moco import MoCoMethod
import copy
import math
from pretrain import get_fc_size, load_from_checkpoint

def update_model_time_0(model_save_dir,
                        train_mode,
                        num_class):
    extractor_mode = train_mode.tp_configs[0].extractor_mode
    classifier_mode = train_mode.tp_configs[0].classifier_mode
    
    fc_size = get_fc_size(train_mode.arch)
    if extractor_mode in ['finetune_pt', 'freeze_pt']:
        model = load_from_checkpoint(
            model_save_dir,
            train_mode.arch,
            train_mode.pretrain
        )
    elif extractor_mode in ['finetune_prev', 'freeze_prev']:
        print("resume from previous feature extractor checkpoint")
        # assert model != None and tp_idx > 0
        raise ValueError("Update model for time 0. Cannot do finetuning")
    else:
        raise NotImplementedError()

    if classifier_mode == 'linear':
        model.fc = torch.nn.Linear(fc_size, num_class)
    else:
        raise NotImplementedError()
    
    if extractor_mode in ['finetune_pt', 'finetune_prev']:
        for p in model.parameters():
            p.requires_grad = True
    elif extractor_mode in ['freeze_pt', 'freeze_prev']:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True
    else:
        raise NotImplementedError()

    return model

class MultiHead(torch.nn.Module):
    def __init__(self, list_of_fcs):
        super().__init__()
        self.list_of_fcs = list_of_fcs
        for idx, fc in enumerate(list_of_fcs):
            setattr(self, f"fc{idx}", fc)
    
    def forward(self, x):
        return [getattr(self, f"fc{idx}")(x) for idx in range(len(self.list_of_fcs))]
    

def update_model(model,
                 model_save_dir,
                 tp_idx,
                 train_mode,
                 num_class,
                 partial_feedback_mode):
    assert tp_idx > 0
    assert model is not None
    extractor_mode = train_mode.tp_configs[tp_idx].extractor_mode
    classifier_mode = train_mode.tp_configs[tp_idx].classifier_mode
    
    old_fc = copy.deepcopy(model.fc)
    
    fc_size = get_fc_size(train_mode.arch)
    if extractor_mode in ['finetune_pt', 'freeze_pt']:
        model = load_from_checkpoint(
            model_save_dir,
            train_mode.arch,
            train_mode.pretrain
        )
    elif extractor_mode in ['finetune_prev', 'freeze_prev']:
        print("resume from previous feature extractor checkpoint")
        pass
    else:
        raise NotImplementedError()

    if classifier_mode == 'linear':
        new_fc = torch.nn.Linear(fc_size, num_class)
    else:
        raise NotImplementedError()
    
    if partial_feedback_mode == 'joint':
        model.fc = MultiHead([old_fc, new_fc])
    elif partial_feedback_mode in [None, 'lpl']:
        model.fc = new_fc
    else:
        raise NotImplementedError()
    
    if extractor_mode in ['finetune_pt', 'finetune_prev']:
        for p in model.parameters():
            p.requires_grad = True
    elif extractor_mode in ['freeze_pt', 'freeze_prev']:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True
    else:
        raise NotImplementedError()

    return model


def update_model_multiple_tp(model,
                             model_save_dir,
                             tp_idx,
                             train_mode,
                             num_class,
                             partial_feedback_mode):
    assert tp_idx > 0
    assert model is not None
    extractor_mode = train_mode.tp_configs[tp_idx].extractor_mode
    classifier_mode = train_mode.tp_configs[tp_idx].classifier_mode
    
    old_fc = copy.deepcopy(model.fc)
    
    fc_size = get_fc_size(train_mode.arch)
    if extractor_mode in ['finetune_pt', 'freeze_pt']:
        model = load_from_checkpoint(
            model_save_dir,
            train_mode.arch,
            train_mode.pretrain
        )
    elif extractor_mode in ['finetune_prev', 'freeze_prev']:
        print("resume from previous feature extractor checkpoint")
        pass
    else:
        raise NotImplementedError()

    if classifier_mode == 'linear':
        new_fc = torch.nn.Linear(fc_size, num_class)
    else:
        raise NotImplementedError()
    
    if partial_feedback_mode == 'joint':
        if type(old_fc) == MultiHead:
            assert tp_idx > 1
            model.fc = MultiHead(old_fc.list_of_fcs + [new_fc])
        elif type(old_fc) == type(new_fc):
            assert tp_idx == 1
            model.fc = MultiHead([old_fc, new_fc])
        else:
            raise NotImplementedError()
    elif partial_feedback_mode in [None, 'lpl']:
        model.fc = new_fc
    else:
        raise NotImplementedError()
    
    if extractor_mode in ['finetune_pt', 'finetune_prev']:
        for p in model.parameters():
            p.requires_grad = True
    elif extractor_mode in ['freeze_pt', 'freeze_prev']:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True
    else:
        raise NotImplementedError()

    return model


def make_optimizer(network, optim, lr, weight_decay, momentum=0.9):
    if optim == 'sgd':
        optimizer = torch.optim.SGD(
            list(filter(lambda x: x.requires_grad, network.parameters())),
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum
        )
    else:
        raise NotImplementedError()
    return optimizer

def make_scheduler(optimizer,
                   decay,
                   warmup_steps,
                   total_steps,
                   num_cycles=7./16.,
                   last_epoch=-1):
    if decay == 'cosine':
        def _lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            no_progress = float(current_step - warmup_steps) / \
                float(max(1, total_steps - warmup_steps))
            return max(0., math.cos(math.pi * num_cycles * no_progress))

        return LambdaLR(optimizer, _lr_lambda, last_epoch)
    else:
        raise NotImplementedError()
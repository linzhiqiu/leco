"""Train LECO-CIFAR or LECO-iNat for two TPs
"""
from typing import List
import os
import argparse
import random
import numpy as np
import copy
import math

from tqdm import tqdm

import configs
from util import load_pickle, save_obj_as_pickle, makedirs, save_as_json
from models import make_optimizer, \
                   make_scheduler, \
                   update_model_time_0, \
                   update_model, \
                   MultiHead
import hparams
import setups
from setups import ConcatHierarchyDataset
import leco_mode
from print_utils import get_exp_str_from_semi_args, \
                        get_exp_str_from_partial_feedback_args, \
                        get_exp_str_from_leco_mode, \
                        get_exp_str_from_train_mode, \
                        get_exp_str_from_hparam_strs, \
                        get_exp_str_from_ema_decay


from semi_supervised import DistillHard, \
                            DistillSoft, \
                            PseudoLabel, \
                            NoSSL, \
                            Fixmatch
from ema import ModelEMA

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_BEST_MODEL_FOR_TP = [0]  # only save model up to TP0 to save memory

SEMI_SUPERVISED_ALG = [
    # Not adding SSL loss
    None,
    # Pseudo-labelling with hard thresholding
    'PL',
    # Fixmatch with hard thresholding
    'Fixmatch',
    # Self-training with distillation (hard-label cross entropy)
    'DistillHard',
    # Self-training with distillation (soft-label KL divergence)
    'DistillSoft',
]

PL_THRESHOLDS = [
    0.25,
    0.5,
    0.75,
    0.95
]

PARTIAL_FEEDBACK_MODE = [
    # Ignoring history coarse-labels
    None,
    # Using LPL loss on single classification head 
    'lpl',
    # Using Joint loss on two seperate heads 
    'joint'
]

HIERARCHICAL_SEMI_SUPERVISION = [
    # Ignoring the coarse label for history samples
    None,
    # Filter out history samples with wrong predicted coarse-label
    'filtering',
    # Only apply the pseudo-label loss to correct predicted coarse-label
    'conditioning'
]

argparser = argparse.ArgumentParser()
argparser.add_argument("--data_dir",
                       default='/scratch/leco/',
                       help="Where the dataset will be saved.")
argparser.add_argument("--result_dir",
                       default="/data3/zhiqiul/leco_results/",
                       help="Where the dataset split and results are saved")
argparser.add_argument("--model_save_dir",
                       default='/data3/zhiqiul/self_supervised_models/wideres_28_2/',
                       help="Where the pre-trained/random init models were saved.")
argparser.add_argument("--setup_mode",
                       type=str,
                       default='cifar10_weakaug_train_2000_val_500',
                       choices=setups.SETUPS.keys(),
                       help="The dataset setup mode")
argparser.add_argument("--leco_mode",
                       type=str,
                       default='label_new',
                       choices=leco_mode.LECO_MODES,
                       help="The continual labelling setup. See leco_mode.py.")
argparser.add_argument('--ema_decay',
                       default=None,
                       type=float,
                       help='EMA decay rate. If none, then no ModelEMA is used.')
argparser.add_argument("--semi_supervised_alg",
                       type=str,
                       default=None,
                       choices=SEMI_SUPERVISED_ALG,
                       help="The semi-supervised algorithm to use")
argparser.add_argument("--pl_threshold",
                       type=float,
                       default=None,
                       choices=PL_THRESHOLDS,
                       help="The threshold to use for (PL, Fixmatch)")
argparser.add_argument("--partial_feedback_mode",
                       type=str,
                       default=None,
                       choices=PARTIAL_FEEDBACK_MODE,
                       help="The partial feedback loss (None/lpl/joint) to use")
argparser.add_argument("--hierarchical_ssl",
                       type=str,
                       default=None,
                       choices=HIERARCHICAL_SEMI_SUPERVISION,
                       help="The hierarchical semi-supervised mode to use")
argparser.add_argument("--train_mode",
                       type=str,
                       default='wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear',
                       choices=configs.TRAIN_MODES.keys(),
                       help="The train mode")
argparser.add_argument('--hparam_strs', nargs='+', default=[],
                       help='The hparam to use for each time period. '
                            'If not specified, then use hparam_candidate. '
                            'It should be used to specify the best hparam for previous TPs.')
argparser.add_argument("--hparam_candidate",
                       type=str,
                       default='cifar',
                       choices=hparams.HPARAM_CANDIDATES.keys(),
                       help="The hyperparameter candidates (str) for next TP")
argparser.add_argument('--seed', default=None, type=int,
                       help='seed for initializing training. ')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_setup_dir(result_dir,
                  setup_mode_str,
                  seed):
    seed_str = f"seed_{seed}"
    setup_dir = os.path.join(result_dir, setup_mode_str, seed_str)
    return setup_dir


def get_train_dir(setup_dir,
                  train_mode,
                  ema_decay,
                  hparam_list,
                  tp_idx=0):
    assert tp_idx == 0
    dir_name = os.path.join(setup_dir,
                            get_exp_str_from_train_mode(train_mode, tp_idx=tp_idx),
                            get_exp_str_from_ema_decay(ema_decay),
                            get_exp_str_from_hparam_strs(hparam_list, tp_idx=tp_idx))
    return dir_name


def get_semi_supervised_dir(setup_dir,
                            ema_decay: float = None,
                            train_mode: configs.TrainMode = None,
                            leco_mode: str = None,
                            hparam_list: List[str] = [],
                            semi_supervised_alg: str = None,
                            pl_threshold: float = None,
                            partial_feedback_mode: str = None,
                            hierarchical_ssl: str = None,
                            tp_idx: int = 1):
    assert tp_idx == 1
    assert len(hparam_list) == tp_idx+1
    dir_name = os.path.join(setup_dir,
                            get_exp_str_from_train_mode(
                                train_mode,
                                tp_idx=tp_idx
                            ),
                            get_exp_str_from_leco_mode(leco_mode),
                            get_exp_str_from_partial_feedback_args(
                                partial_feedback_mode
                            ),
                            get_exp_str_from_semi_args(
                                semi_supervised_alg=semi_supervised_alg,
                                pl_threshold=pl_threshold,
                                hierarchical_ssl=hierarchical_ssl,
                            ),
                            get_exp_str_from_ema_decay(ema_decay),
                            get_exp_str_from_hparam_strs(
                                hparam_list,
                                tp_idx=tp_idx
                            ))
    return dir_name


def is_better(select_criterion, curr_value, best_value):
    # Return True if curr_value is better than best_value
    assert select_criterion in ['loss_per_epoch', 'acc_per_epoch']
    if select_criterion in ['loss_per_epoch']:
        return curr_value < best_value
    else:
        return curr_value > best_value


def update_per_class_corrects(corrects, preds, labels):
    correct = preds == labels.data
    for idx, label in enumerate(labels):
        corrects[int(label)] += correct[idx]


def update_per_class_count(count, labels):
    for label in labels:
        count[int(label)] += 1


def per_class_acc(corrects, count):
    avg_per_class_acc = 0.0
    num_of_class = len(list(count.keys()))
    for k in corrects:
        if count[k] > 0:
            avg_per_class_acc += float(corrects[k] / count[k])
        else:
            num_of_class -= 1
    
    return avg_per_class_acc / num_of_class


def train(loaders,
          model,
          optimizer,
          scheduler,
          epochs,
          eval_steps,
          tp_idx,
          num_of_class,
          select_criterion='acc_per_epoch',
          ema_decay=None):
    model = model.to(device)
    if ema_decay is not None:
        ema_model = ModelEMA(model, ema_decay, device)
    else:
        ema_model = None
    criterion = torch.nn.NLLLoss(reduction='mean')

    avg_results = {'train': {'loss_per_epoch': [], 'acc_per_epoch': [], 'per_class_correct_per_epoch': [], 'per_class_count_per_epoch': []},
                   'val':   {'loss_per_epoch': [], 'acc_per_epoch': [], 'per_class_correct_per_epoch': [], 'per_class_count_per_epoch': []},
                   'test':  {'loss_per_epoch': [], 'acc_per_epoch': [], 'per_class_correct_per_epoch': [], 'per_class_count_per_epoch': []}}
    phases = ['train', 'val', 'test']

    # Save best model based on select_criterion
    best_result = {'best_value': None,  # Best value of select_criterion
                   'best_epoch': None,
                   'best_model': None,
                   'best_criterion': select_criterion}
    
    best_model = None
    for epoch in range(0, epochs):
        print(f"Epoch {epoch}")
        ### Training phase ###
        model.train()

        running_loss = 0.0
        # Overall correct count per class
        running_corrects = {cls_idx: 0. for cls_idx in range(num_of_class)}
        # Count per class
        count = {cls_idx: 0 for cls_idx in range(num_of_class)}
        # Total count for all classes
        total = 0

        loader_iter = iter(loaders['train'])
        p_bar = tqdm(range(eval_steps))
        for batch in range(eval_steps):
            try:
                data = loader_iter.next()
            except Exception as e:
                loader_iter = iter(loaders['train'])
                data = loader_iter.next()

            inputs, labels = data

            inputs = inputs.to(device)
            total += inputs.shape[0]
            labels = labels[tp_idx].to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                log_probability = torch.nn.functional.log_softmax(outputs, dim=1)
                loss = criterion(log_probability, labels)

                loss.backward()
                optimizer.step()
                scheduler.step()
                if ema_decay:
                    ema_model.update(model)
            
            # statistics
            running_loss += loss.item() * inputs.size(0)
            update_per_class_count(count, labels)
            update_per_class_corrects(running_corrects, preds, labels)
            p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Loss: {loss:.4f}.".format(
                epoch=epoch + 1,
                epochs=epochs,
                batch=batch + 1,
                iter=eval_steps,
                lr=scheduler.get_last_lr()[0],
                loss=float(running_loss)/total))
            p_bar.update()

        p_bar.close()
        avg_loss = float(running_loss)/total
        avg_acc = per_class_acc(running_corrects, count)
        avg_results['train']['loss_per_epoch'].append(avg_loss)
        avg_results['train']['acc_per_epoch'].append(avg_acc)
        avg_results['train']['per_class_correct_per_epoch'].append(running_corrects)
        avg_results['train']['per_class_count_per_epoch'].append(count)
        print(f"Epoch {epoch}: Average train Loss {avg_loss:.4f}, Per-class Acc {avg_acc:.2%}")
        ### End of Training phase ###
        
        ### Val+Test phase ###
        if ema_model:
            eval_model = ema_model.ema  # keep a pointer to current model for train
        else:
            eval_model = model

        val_loss, val_acc, val_correct, val_count = test(loaders['val'], eval_model, tp_idx, num_of_class)
        avg_results['val']['loss_per_epoch'].append(val_loss)
        avg_results['val']['acc_per_epoch'].append(val_acc)
        avg_results['val']['per_class_correct_per_epoch'].append(val_correct)
        avg_results['val']['per_class_count_per_epoch'].append(val_count)
        curr_value = avg_results['val'][select_criterion][-1]
        if (best_result['best_value'] is None or
                is_better(select_criterion, curr_value, best_result['best_value'])):
            print(f"Best val {select_criterion} at epoch {epoch} being {curr_value}")
            best_result['best_epoch'] = epoch
            best_result['best_value'] = curr_value
            best_model = copy.deepcopy(eval_model.state_dict())
            if tp_idx in SAVE_BEST_MODEL_FOR_TP:
                best_result['best_model'] = best_model
        print(f"Epoch {epoch}: Average val Loss {val_loss:.4f}, Per-class Acc {val_acc:.2%}")
        
        test_loss, test_acc, test_correct, test_count = test(
            loaders['test'],
            eval_model,
            tp_idx,
            num_of_class
        )
        avg_results['test']['loss_per_epoch'].append(test_loss)
        avg_results['test']['acc_per_epoch'].append(test_acc)
        avg_results['test']['per_class_correct_per_epoch'].append(test_correct)
        avg_results['test']['per_class_count_per_epoch'].append(test_count)
        print(f"Epoch {epoch}: Average test Loss {test_loss:.4f}, Per-class Acc {test_acc:.2%}")
        ### End of Val+Test phase ###
        print()
    print(f"Test Per-class Accuracy (for best val {select_criterion} model): "
          f"{avg_results['test']['acc_per_epoch'][best_result['best_epoch']]:.2%}")
    print(f"Best Test Per-class Accuracy overall: {max(avg_results['test']['acc_per_epoch']):.2%}")
    model.load_state_dict(best_model)
    test_loss, test_acc, test_correct, test_count = test(
        loaders['test'],
        model,
        tp_idx,
        num_of_class
    )
    print(f"Verify the best test per-class accuracy for best val {select_criterion} is indeed {test_acc:.2%}")
    acc_result = {phase: avg_results[phase]['acc_per_epoch'][best_result['best_epoch']]
                  for phase in phases}
    return model, acc_result, best_result, avg_results


def get_l_loss_func():
    nll_criterion = torch.nn.NLLLoss(reduction='mean')
    
    def l_loss_func(outputs, labels):
        log_probability = torch.nn.functional.log_softmax(outputs, dim=1)
        return nll_criterion(log_probability, labels)
    return l_loss_func


def get_coarse_loss_func(edge_matrix, partial_feedback_mode):
    nll_criterion = torch.nn.NLLLoss(reduction='mean')
    if partial_feedback_mode is None:
        return None
    
    def coarse_loss_func(outputs, labels):
        if partial_feedback_mode == 'lpl':
            outputs = outputs - outputs.max(1)[0].unsqueeze(1)
            prob = torch.nn.Softmax(dim=1)(outputs)
            coarse_prob = torch.matmul(prob, edge_matrix)
            return nll_criterion(torch.log(coarse_prob), labels)
        elif partial_feedback_mode == 'joint':
            outputs_0, _ = outputs
            outputs_0 = outputs_0 - outputs_0.max(1)[0].unsqueeze(1)
            log_prob = torch.nn.LogSoftmax(dim=1)(outputs_0)
            # if nll_criterion(log_prob, labels) > 10:
            #     import pdb; pdb.set_trace()
            return nll_criterion(log_prob, labels)
        else:
            raise NotImplementedError()
    return coarse_loss_func


def get_ssl_loss_func(
        loaders,
        model,
        hparam_mode,
        tp_idx,
        num_of_classes,
        ema_decay,
        edge_matrix: torch.Tensor = None,
        semi_supervised_alg: str = None,
        pl_threshold: float = None,
        hierarchical_ssl: str = None,
        select_criterion: str = 'acc_per_epoch'):
    if semi_supervised_alg:
        if semi_supervised_alg in ['DistillHard', 'DistillSoft']:
            model_T = copy.deepcopy(model)
            if isinstance(model_T.fc, MultiHead):
                assert tp_idx == 1
                model_T.fc = getattr(model_T.fc, f"fc{tp_idx}")
            
            epochs = math.ceil(hparam_mode['total_steps'] / hparam_mode['eval_steps'])
            eval_steps = hparam_mode['eval_steps']
            print(f"Train for {epochs} epochs each with {eval_steps} steps")

            optimizer = make_optimizer(model_T,
                                       hparam_mode['optim'],
                                       hparam_mode['lr'],
                                       weight_decay=hparam_mode['weight_decay'],
                                       momentum=hparam_mode['momentum'])
            scheduler = make_scheduler(optimizer,
                                       hparam_mode['decay'],
                                       hparam_mode['warmup_steps'],
                                       hparam_mode['total_steps'])
            
            loaders['train'] = loaders['labeled']
            model_T, acc_result, best_result, avg_results = train(
                loaders,
                model_T,
                optimizer,
                scheduler,
                epochs,
                eval_steps,
                tp_idx,
                num_of_classes[tp_idx],
                ema_decay=ema_decay
            )
            print(f"Teacher achieves {acc_result['val']} val acc")
            
            if semi_supervised_alg == 'DistillHard':
                ssl_objective = DistillHard(
                    model_T,
                    hierarchical_ssl=hierarchical_ssl,
                    edge_matrix=edge_matrix
                )
            elif semi_supervised_alg == 'DistillSoft':
                ssl_objective = DistillSoft(
                    model_T,
                    hierarchical_ssl=hierarchical_ssl,
                    edge_matrix=edge_matrix
                )
            else:
                raise NotImplementedError()
        elif semi_supervised_alg == 'PL':
            ssl_objective = PseudoLabel(
                pl_threshold,
                hierarchical_ssl=hierarchical_ssl,
                edge_matrix=edge_matrix
            )
        elif semi_supervised_alg == 'Fixmatch':
            ssl_objective = Fixmatch(
                pl_threshold,
                hierarchical_ssl=hierarchical_ssl,
                edge_matrix=edge_matrix
            )
        else:
            raise NotImplementedError()
        
        return ssl_objective
    else:
        print("No SSL loss")
        return NoSSL(
            hierarchical_ssl=hierarchical_ssl,
            edge_matrix=edge_matrix
        )

    
def remove_multi_head(model, tp_idx):
    if isinstance(model.fc, MultiHead):
        return lambda inputs: model(inputs)[tp_idx]
    else:
        return model


def calc_stats(counter, pl_threshold):
    assert 'max_probs' in counter
    assert 'pred_labels' in counter
    assert 'gt_labels' in counter
    
    if pl_threshold is None:
        pl_threshold = 0.0
    count = counter['max_probs'][0].size(0)
    unmasked = torch.BoolTensor(counter['max_probs'][0] < pl_threshold)
    masked = torch.BoolTensor(counter['max_probs'][0] >= pl_threshold)
    corrects_coarse = torch.BoolTensor(counter['pred_labels'][0] == counter['gt_labels'][0])
    corrects = torch.BoolTensor(counter['pred_labels'][1] == counter['gt_labels'][1])
    
    masked_sum = float(masked.sum())
    masked_filtered_sum = float((corrects_coarse & masked).sum())
    
    curr_stats = {
        'mask_rate': float(masked.sum()) / count,
        'impurity': float((~corrects)[masked].sum()) / masked_sum if masked_sum > 0 else 1.,
        'coarse_accuracy': float(corrects_coarse.sum()) / count,
        'coarse_accuracy_masked': float(corrects_coarse[masked].sum()) / masked_sum if masked_sum > 0 else 1.,
        'mask_rate_filtered': float(masked[corrects_coarse].sum()) / count,
        'impurity_filtered': float((~corrects)[corrects_coarse & masked].sum()) / masked_filtered_sum if masked_filtered_sum > 0 else 1.,
    }
    return curr_stats


def train_semi_supervised(
    loaders,
    model,
    optimizer,
    scheduler,
    epochs,
    eval_steps,
    tp_idx,
    num_of_classes,
    l_loss_func,
    coarse_loss_func,
    ssl_loss_func,
    pl_threshold: float = 1.0,
    select_criterion: str = 'acc_per_epoch',
    ema_decay: float = None,
    # True if using Fixmatch, where the unlabeled_inputs consists of both weak and strong aug images
    use_both_weak_and_strong: bool = False,
    leco_mode: str = 'label_new'
):
    assert tp_idx == 1

    model = model.to(device)
    if ema_decay is not None:
        ema_model = ModelEMA(model, ema_decay, device)
    else:
        ema_model = None
        
    # Below is just a lambda func to wrap the output
    model_lpl = remove_multi_head(model, tp_idx)
    
    avg_results = {'train': {'loss_per_epoch': [], 'acc_per_epoch': [], 'per_class_correct_per_epoch': [], 'per_class_count_per_epoch': []},  # only measuring the labeled portion
                   'val':   {'loss_per_epoch': [], 'acc_per_epoch': [], 'per_class_correct_per_epoch': [], 'per_class_count_per_epoch': []},
                   'test':  {'loss_per_epoch': [], 'acc_per_epoch': [], 'per_class_correct_per_epoch': [], 'per_class_count_per_epoch': []}}
    stats = {  # stats per epoch
        'mask_rate': [],
        'impurity': [],
        'coarse_accuracy': [],
        'coarse_accuracy_masked': [],
        'mask_rate_filtered': [],
        'impurity_filtered': []
    }
    
    def update_stats(counter):
        curr_stats = calc_stats(counter, pl_threshold)
        for k in stats:
            stats[k].append(curr_stats[k])
            
    def best_stats(best_epoch):
        best_stat = {
            k: stats[k][best_epoch] for k in stats
        }
        return best_stat

    phases = ['train', 'val', 'test']

    # Save best model based on select_criterion
    best_result = {'best_value': None,  # Best value of select_criterion
                   'best_epoch': None,
                   'best_model': None,
                   'best_stat': None,
                   'best_criterion': select_criterion}
    
    best_model = None
    for epoch in range(0, epochs):
        print(f"Epoch {epoch}")
        ### Training Phase ###
        model.train()
        loader = loaders['labeled']
        unlabeled_loader_iter = iter(loaders['unlabeled'])
        if use_both_weak_and_strong:
            weak_and_strong_loader_iter = iter(loaders['unlabeled_weak_strong'])
        counter = {  # stats this epoch
            'max_probs': torch.Tensor([[]]),  # size 1xN array
            'pred_labels': torch.Tensor([[],[]]),  # size 2xN array (0 is coarse, 1 is fine)
            'gt_labels': torch.Tensor([[],[]]),  # size 2xN array (0 is coarse, 1 is fine)
        }
        
        loader_iter = iter(loader)

        running_loss = 0.0
        running_loss_ssl = 0.0
        running_loss_coarse = 0.0
        running_corrects = {cls_idx: 0. for cls_idx in range(num_of_classes[1])}  # Overall correct count per class
        count = {cls_idx: 0 for cls_idx in range(num_of_classes[1])}  # Count per class
        total = 0 # total count
        count_unlabeled_ssl = 0
        count_unlabeled_partial_feedback = 0
        
        # for batch, data in enumerate(loader):
        p_bar = tqdm(range(eval_steps))
        for batch in range(eval_steps):
            try:
                labeled_inputs, labeled_labels = loader_iter.next()
            except Exception as _:
                loader_iter = iter(loader)
                labeled_inputs, labeled_labels = loader_iter.next()
                
            labeled_inputs = labeled_inputs.to(device)
            labeled_labels = [labeled_labels[i].to(device) for i in range(len(labeled_labels))]
            total += labeled_inputs.size(0)
            
            try:
                unlabeled_inputs, unlabeled_labels = unlabeled_loader_iter.next()
            except:
                unlabeled_loader_iter = iter(loaders['unlabeled'])
                unlabeled_inputs, unlabeled_labels = unlabeled_loader_iter.next()
            
            count_unlabeled_ssl += unlabeled_inputs.size(0)
            unlabeled_inputs = unlabeled_inputs.to(device)
            
            if leco_mode == "label_new":
                curr_partial_count = unlabeled_inputs.size(0) + labeled_inputs.size(0)
            elif leco_mode == "upper_bound_with_multi_task":
                curr_partial_count = labeled_inputs.size(0)
            else:
                curr_partial_count = unlabeled_inputs.size(0)
            count_unlabeled_partial_feedback += curr_partial_count
            
            if use_both_weak_and_strong:
                try:
                    w_s_unlabeled_inputs, w_s_unlabeled_labels = weak_and_strong_loader_iter.next()
                except:
                    weak_and_strong_loader_iter = iter(loaders['unlabeled_weak_strong'])
                    w_s_unlabeled_inputs, w_s_unlabeled_labels = weak_and_strong_loader_iter.next()
                
                w_s_unlabeled_inputs[0] = w_s_unlabeled_inputs[0].to(device)
                w_s_unlabeled_inputs[1] = w_s_unlabeled_inputs[1].to(device)
                assert unlabeled_inputs.size(0) == w_s_unlabeled_inputs[0].size(0)
                assert unlabeled_inputs.size(0) == w_s_unlabeled_inputs[1].size(0)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                labeled_outputs = model(labeled_inputs)
                if type(labeled_outputs) == list:
                    labeled_outputs_for_ce = labeled_outputs[tp_idx]
                else:
                    labeled_outputs_for_ce = labeled_outputs
                _, labeled_preds = torch.max(labeled_outputs_for_ce, 1)

                labeled_loss = l_loss_func(labeled_outputs_for_ce, labeled_labels[tp_idx].to(device))
                loss = labeled_loss
                
                if coarse_loss_func:
                    unlabeled_outputs = model(unlabeled_inputs)
                    if leco_mode == 'label_new':
                        if type(unlabeled_outputs) == list:
                            unlabeled_outputs_partial_feedback = [torch.cat([u_out, l_out]) for u_out, l_out in zip(unlabeled_outputs, labeled_outputs)]
                        else:
                            unlabeled_outputs_partial_feedback = torch.cat([unlabeled_outputs, labeled_outputs])
                        unlabeled_labels_partial_feedback = [torch.cat([u_label.to(device), l_label]) for u_label, l_label in zip(unlabeled_labels, labeled_labels)]
                    elif leco_mode == 'upper_bound_with_multi_task':
                        unlabeled_outputs_partial_feedback = labeled_outputs
                        unlabeled_labels_partial_feedback = labeled_labels
                    else:
                        unlabeled_outputs_partial_feedback = unlabeled_outputs
                        unlabeled_labels_partial_feedback = unlabeled_labels
                    coarse_loss = coarse_loss_func(unlabeled_outputs_partial_feedback, unlabeled_labels_partial_feedback[tp_idx-1].cuda())
                else:
                    coarse_loss = 0.0
                    
                # import pdb; pdb.set_trace()
                ssl_stats, ssl_loss = ssl_loss_func(
                                            model_lpl,
                                            unlabeled_inputs if not use_both_weak_and_strong else w_s_unlabeled_inputs,
                                            unlabeled_labels if not use_both_weak_and_strong else w_s_unlabeled_labels
                                        )
                counter = {
                    k : torch.cat([counter[k], ssl_stats[k]], dim=1)
                    for k in ssl_stats
                }
                loss = loss + ssl_loss + coarse_loss
                running_loss_coarse += float(coarse_loss) * curr_partial_count
                running_loss_ssl += float(ssl_loss) * unlabeled_inputs.size(0)

                loss.backward()
                optimizer.step()
                scheduler.step()
                if ema_model:
                    ema_model.update(model)
            
            # statistics
            running_loss += labeled_loss.item() * labeled_inputs.size(0)
            update_per_class_count(count, labeled_labels[tp_idx])
            update_per_class_corrects(running_corrects, labeled_preds, labeled_labels[tp_idx])
            # running_corrects += torch.sum(labeled_preds == labeled_labels[tp_idx].data)
            if count_unlabeled_ssl > 0:
                auxilary_str = "Coarse: {coarse_loss_to_print:.4f}. SSL: {ssl_loss_to_print:.4f}.".format(
                    coarse_loss_to_print=float(running_loss_coarse)/count_unlabeled_partial_feedback,
                    ssl_loss_to_print=float(running_loss_ssl)/count_unlabeled_ssl
                )
            else:
                auxilary_str = ""
            p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Loss: {loss:.4f}. {auxilary_str}".format(
                epoch=epoch + 1,
                epochs=epochs,
                batch=batch + 1,
                iter=eval_steps,
                lr=scheduler.get_last_lr()[0],
                loss=float(running_loss)/total,
                auxilary_str=auxilary_str,
                # loss_ssl_pbar=loss_ssl_pbar
                ))
            p_bar.update()
        
        p_bar.close()
        avg_loss = float(running_loss)/total
        avg_acc = per_class_acc(running_corrects, count)
        avg_results['train']['loss_per_epoch'].append(avg_loss)
        avg_results['train']['acc_per_epoch'].append(avg_acc)
        avg_results['train']['per_class_correct_per_epoch'].append(running_corrects)
        avg_results['train']['per_class_count_per_epoch'].append(count)
        update_stats(counter)
        ### End of Training Phase ###
        
        ### Val+Test phase ###
        if ema_model:
            eval_model = ema_model.ema
        else:
            eval_model = model
            
        eval_model = copy.deepcopy(eval_model)
        if isinstance(eval_model.fc, MultiHead):
            eval_model.fc = getattr(eval_model.fc, f"fc{tp_idx}")
            
        val_loss, val_acc, val_correct, val_count = test(loaders['val'], eval_model, tp_idx, num_of_classes[tp_idx])
        avg_results['val']['loss_per_epoch'].append(val_loss)
        avg_results['val']['acc_per_epoch'].append(val_acc)
        avg_results['val']['per_class_correct_per_epoch'].append(val_correct)
        avg_results['val']['per_class_count_per_epoch'].append(val_count)
        curr_value = avg_results['val'][select_criterion][-1]
        if best_result['best_value'] == None or is_better(select_criterion, curr_value, best_result['best_value']):
            print(
                f"Best val {select_criterion} at epoch {epoch} being {curr_value}")
            best_result['best_epoch'] = epoch
            # best_result['best_acc'] = val_acc
            # best_result['best_loss'] = val_loss
            best_result['best_value'] = curr_value
            best_result['best_stat'] = best_stats(epoch)
            best_model = copy.deepcopy(eval_model.state_dict())
            if tp_idx in SAVE_BEST_MODEL_FOR_TP:
                best_result['best_model'] = best_model
        print(f"Epoch {epoch}: Average val Loss {val_loss:.4f}, Per-class Acc {val_acc:.2%}")
        
        test_loss, test_acc, test_correct, test_count = test(loaders['test'], eval_model, tp_idx, num_of_classes[tp_idx])
        avg_results['test']['loss_per_epoch'].append(test_loss)
        avg_results['test']['acc_per_epoch'].append(test_acc)
        avg_results['test']['per_class_correct_per_epoch'].append(test_correct)
        avg_results['test']['per_class_count_per_epoch'].append(test_count)
        print(f"Epoch {epoch}: Average test Loss {test_loss:.4f}, Acc {test_acc:.2%}")
        ### End of Val+Test phase ###
        print()
    print(
        f"Test Per-class Accuracy (for best val {select_criterion} model): {avg_results['test']['acc_per_epoch'][best_result['best_epoch']]:.2%}")
    print(
        f"Best Test Per-class Accuracy overall: {max(avg_results['test']['acc_per_epoch']):.2%}")
    
    model_test = copy.deepcopy(model)
    if isinstance(model_test.fc, MultiHead):
        model_test.fc = getattr(model_test.fc, f"fc{tp_idx}")
    model_test.load_state_dict(best_model)
    test_loss, test_acc, test_correct, test_count = test(loaders['test'], eval_model, tp_idx, num_of_classes[tp_idx])
    print(f"Verify the best test per-class accuracy for best val {select_criterion} is indeed {test_acc:.2%}")
    acc_result = {phase: avg_results[phase]['acc_per_epoch'][best_result['best_epoch']]
                  for phase in phases}
    
    return model, acc_result, best_result, avg_results, stats

def test(test_loader,
         model,
         tp_idx,
         num_of_class):
    model = model.to(device).eval()
    criterion = torch.nn.NLLLoss(reduction='mean')
    running_corrects = {cls_idx: 0. for cls_idx in range(num_of_class)}
    count = {cls_idx: 0 for cls_idx in range(num_of_class)}
    running_loss = 0.
    total = 0

    for batch, data in enumerate(test_loader):
        inputs, labels = data
        total += inputs.size(0)

        inputs = inputs.to(device)
        labels = labels[tp_idx].to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            log_probability = torch.nn.functional.log_softmax(outputs, dim=1)
            loss = criterion(log_probability, labels)

        # statistics
        update_per_class_count(count, labels)
        update_per_class_corrects(running_corrects, preds, labels)
        # running_corrects += torch.sum(preds == labels.data)
        running_loss += loss.item() * inputs.size(0)
        
    avg_acc = per_class_acc(running_corrects, count)
    avg_loss = float(running_loss)/total
    return avg_loss, avg_acc, running_corrects, count
        
def get_dataset(train_val_subsets,
                tp_idx,
                leco_mode=None):
    assert tp_idx == 1
    assert leco_mode != None
    if leco_mode == 'label_new':
        labeled_set = train_val_subsets[tp_idx][0]
        val_set = train_val_subsets[tp_idx][1]
        unlabeled_set = copy.deepcopy(train_val_subsets[0][0])
    elif leco_mode == 'relabel_old':
        labeled_set = train_val_subsets[0][0]
        val_set = train_val_subsets[0][1]
        unlabeled_set = copy.deepcopy(train_val_subsets[0][0])
    elif leco_mode in ['upper_bound', 'upper_bound_with_multi_task']:
        labeled_set = ConcatHierarchyDataset([train_val_subsets[i][0] for i in range(tp_idx+1)])
        val_set = ConcatHierarchyDataset([train_val_subsets[i][1] for i in range(tp_idx+1)])
        unlabeled_set = copy.deepcopy(train_val_subsets[0][0])
    else:
        raise NotImplementedError()
    return labeled_set, unlabeled_set, val_set

def get_loaders(labeled_set,
                unlabeled_set,
                val_set,
                testset,
                batch_size,
                workers):
    loaders = {}
    batch_size = int(batch_size / 2.0)
    unlabeled_batch_size = int(batch_size)
    loaders['unlabeled'] = torch.utils.data.DataLoader(
                                unlabeled_set,
                                batch_size=unlabeled_batch_size,
                                shuffle=True,
                                num_workers=workers,
                                drop_last=True
                            )
    print(f"Labeled batch size is {batch_size}; unlabeled is {unlabeled_batch_size}")
    
    # print("Need to check whether the new set has new transform + old set has old transform!")
    assert isinstance(unlabeled_set, setups.HierarchyDataset) or \
           isinstance(unlabeled_set, setups.SubsetHierarchyDataset) or \
           isinstance(unlabeled_set, setups.ConcatHierarchyDataset)
    unlabeled_set_weak_strong = copy.deepcopy(unlabeled_set)
    unlabeled_set_weak_strong.update_transform(unlabeled_set_weak_strong.weak_strong_transform)
    loaders['unlabeled_weak_strong'] = torch.utils.data.DataLoader(
                                            unlabeled_set_weak_strong,
                                            batch_size=unlabeled_batch_size,
                                            shuffle=True,
                                            num_workers=workers,
                                            drop_last=True
                                       )
    
    loaders['labeled'] = torch.utils.data.DataLoader(
                             labeled_set,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=workers,
                             drop_last=True
                         )
    loaders['val'] = torch.utils.data.DataLoader(
                         val_set,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=workers
                     )
    loaders['test'] = torch.utils.data.DataLoader(
                           testset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=workers
                       )
    return loaders

def start_training(model_save_dir,
                   tp_idx,
                   train_mode,
                   hparams_mode,
                   train_val_subsets,
                   num_of_classes,
                   testset,
                   ema_decay=None):
    """Train for time period 0
    """
    assert tp_idx == 0
    model = update_model_time_0(
        model_save_dir,
        train_mode,
        num_of_classes[0]
    )
    
    batch_size = hparams_mode['batch']
    workers = hparams_mode['workers']
    
    hparam_mode = hparams_mode['hparams']
    epochs = math.ceil(hparam_mode['total_steps'] / hparam_mode['eval_steps'])
    eval_steps = hparam_mode['eval_steps']

    optimizer = make_optimizer(model,
                               hparam_mode['optim'],
                               hparam_mode['lr'],
                               weight_decay=hparam_mode['weight_decay'],
                               momentum=hparam_mode['momentum'])
    scheduler = make_scheduler(optimizer,
                               hparam_mode['decay'],
                               hparam_mode['warmup_steps'],
                               hparam_mode['total_steps'])

    train_set = train_val_subsets[tp_idx][0]
    val_set = train_val_subsets[tp_idx][1]
    
    loaders = {}
    loaders['train'] = torch.utils.data.DataLoader(
                           train_set,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=workers
                       )
    loaders['val'] = torch.utils.data.DataLoader(
                         val_set,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=workers
                     )
    loaders['test'] = torch.utils.data.DataLoader(
                           testset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=workers
                       )
    
    model, acc_result, best_result, avg_results = train(
        loaders,
        model,
        optimizer,
        scheduler,
        epochs,
        eval_steps,
        tp_idx,
        num_of_classes[0],
        ema_decay=ema_decay
    )
    return model, acc_result, best_result, avg_results

def start_training_semi_supervised(model,
                                   model_save_dir,
                                   tp_idx,
                                   train_mode,
                                   leco_mode,
                                   hparams_mode,
                                   train_val_subsets,
                                   num_of_classes,
                                   testset,
                                   edge_matrix,
                                   ema_decay: float=None,
                                   semi_supervised_alg: str=None,
                                   pl_threshold: float=None,
                                   partial_feedback_mode: str=None,
                                   hierarchical_ssl: str=None):
    """Train for time period 0
    """
    assert tp_idx > 0
    
    model = update_model(
                model,
                model_save_dir,
                tp_idx,
                train_mode, 
                num_of_classes[tp_idx],
                partial_feedback_mode
            )
    
    batch_size = hparams_mode['batch']
    workers = hparams_mode['workers']
    
    hparam_mode = hparams_mode['hparams']
    epochs = math.ceil(hparam_mode['total_steps'] / hparam_mode['eval_steps'])
    eval_steps = hparam_mode['eval_steps']
    print(f"Train for {epochs} epochs each with {eval_steps} steps")

    optimizer = make_optimizer(model,
                               hparam_mode['optim'],
                               hparam_mode['lr'],
                               weight_decay=hparam_mode['weight_decay'],
                               momentum=hparam_mode['momentum'])
    scheduler = make_scheduler(optimizer,
                               hparam_mode['decay'],
                               hparam_mode['warmup_steps'],
                               hparam_mode['total_steps'])
    
    labeled_set, unlabeled_set, val_set = get_dataset(train_val_subsets, tp_idx, leco_mode)

    loaders = get_loaders(labeled_set,
                          unlabeled_set,
                          val_set,
                          testset,
                          batch_size,
                          workers)

    edge_matrix = edge_matrix.to(device)
    l_loss_func = get_l_loss_func()
    coarse_loss_func = get_coarse_loss_func(edge_matrix, partial_feedback_mode)
    ssl_loss_func = get_ssl_loss_func(
        loaders,
        model,
        hparam_mode,
        tp_idx,
        num_of_classes,
        ema_decay,
        edge_matrix=edge_matrix,
        semi_supervised_alg=semi_supervised_alg,
        pl_threshold=pl_threshold,
        hierarchical_ssl=hierarchical_ssl,
    )
    
    use_both_weak_and_strong = semi_supervised_alg == 'Fixmatch'
    if use_both_weak_and_strong:
        print("Running fixmatch..")
    model, acc_result, best_result, avg_results, stats = train_semi_supervised(
        loaders,
        model,
        optimizer,
        scheduler,
        epochs,
        eval_steps,
        tp_idx,
        num_of_classes,
        l_loss_func,
        coarse_loss_func,
        ssl_loss_func,
        ema_decay=ema_decay,
        pl_threshold=pl_threshold,
        use_both_weak_and_strong=use_both_weak_and_strong,
        leco_mode=leco_mode
    )
    
    return model, acc_result, best_result, avg_results, stats

def get_edge_matrix(leaf_idx_to_all_class_idx, superclass_time=0):
    num_leaf = len(leaf_idx_to_all_class_idx.keys())
    parents = set()
    for leaf_idx in leaf_idx_to_all_class_idx:
        parent = leaf_idx_to_all_class_idx[leaf_idx][superclass_time]
        parents.add(parent)
    
    num_parents = len(parents)
    edge_matrix = torch.zeros((num_leaf, num_parents))
    
    for leaf_idx in leaf_idx_to_all_class_idx:
        parent = leaf_idx_to_all_class_idx[leaf_idx][superclass_time]
        edge_matrix[leaf_idx][parent] = 1.
    return edge_matrix

def start_experiment(data_dir: str, # where the data are saved
                     result_dir: str, # where dataset split + model + final accuracy results will be saved
                     model_save_dir: str, # where the self-supervised pretrained models are saved
                     setup_mode_str: str, 
                     train_mode_str: str,
                     leco_mode: str,
                     hparam_strs, # A list of hparam str to load
                     hparam_candidate : str, # The list of hparam to try for next time period (tp_idx = len(hparam_strs))
                     semi_supervised_alg,
                     pl_threshold,
                     partial_feedback_mode,
                     hierarchical_ssl,
                     finetuning_mode,
                     ema_decay,
                     seed=None):
    if leco_mode in ['upper_bound', 'relabel_old']:
        assert semi_supervised_alg == None
        assert partial_feedback_mode == None
    
    if seed == None:
        print("Not using a random seed")
    else:
        print(f"Using random seed {seed}")
        set_seed(seed)
    
    setup_dir = get_setup_dir(result_dir,
                              setup_mode_str,
                              seed)
    makedirs(setup_dir)
    dataset_path = os.path.join(setup_dir, 'dataset.pt')
    setup_mode = setups.SETUPS[setup_mode_str]
    if os.path.exists(dataset_path):
        setups.download_dataset(data_dir, setup_mode)
        dataset = load_pickle(dataset_path)
    else:
        dataset = setups.generate_dataset(data_dir, setup_mode)
        save_obj_as_pickle(dataset_path, dataset)
        print(f"Dataset saved at {dataset_path}")

    train_val_subsets, testset, all_tp_info, leaf_idx_to_all_class_idx = dataset

    train_mode = configs.TRAIN_MODES[train_mode_str]
    print("Only support two time periods for now..")
    for tp_idx in [0,1]:
        if tp_idx == 0:
            if len(hparam_strs) == 0:
                for hparams_str in hparams.HPARAM_CANDIDATES[hparam_candidate]:
                    hparams_mode = hparams.HPARAMS[hparams_str]
                    interim_exp_dir_tp_idx = get_train_dir(setup_dir,
                                                           train_mode,
                                                           ema_decay,
                                                           [hparams_str],
                                                           tp_idx=tp_idx)
                    makedirs(interim_exp_dir_tp_idx)
                    exp_result_path = os.path.join(interim_exp_dir_tp_idx, "result.ckpt")

                    if os.path.exists(exp_result_path):
                        result = load_pickle(exp_result_path)
                        print(f"{tp_idx} time period already finished for {hparams_str}. Best epoch: {result['best_result']['best_epoch']}. Best Test Per-Class Acc: {result['acc_result']['test']:.2%}")
                    else:
                        print(f"Run {train_mode_str} for TP {tp_idx}")
                        # exp_result do not exist, therefore start training
                        new_model, acc_result, best_result, avg_results = start_training(
                            model_save_dir,
                            tp_idx, # must be tp_idx == 0
                            train_mode,
                            hparams_mode,
                            train_val_subsets,
                            [info['num_of_classes'] for info in all_tp_info],
                            testset,
                            ema_decay=ema_decay,
                        )

                        save_obj_as_pickle(exp_result_path, {
                            'model' : new_model, #TODO
                            'acc_result' : acc_result,
                            'best_result' : best_result,
                            'avg_results' : avg_results
                        })
                print(f"Finished for {tp_idx} time period. Please run collect.py")
                break
            else:
                interim_exp_dir_tp_idx = get_train_dir(setup_dir,
                                                       train_mode,
                                                       ema_decay,
                                                       hparam_strs,
                                                       tp_idx=tp_idx)
                exp_result_path = os.path.join(interim_exp_dir_tp_idx, "result.ckpt")
                # Load the model
                if not os.path.exists(exp_result_path):
                    print("Please specify the hparam for an experiment that is finished.")
                    import pdb; pdb.set_trace()
                    kill(0)
                else:
                    print(f"{tp_idx} time period already finished. Load from {exp_result_path}")
                    exp_result = load_pickle(exp_result_path)
                    model = exp_result['model']
        else:
            print(f"Working on time period {tp_idx}")
            for hparams_str in hparams.HPARAM_CANDIDATES[hparam_candidate]:
                hparams_mode = hparams.HPARAMS[hparams_str]
                interim_exp_dir_tp_idx = get_semi_supervised_dir(
                                             setup_dir,
                                             leco_mode=leco_mode,
                                             train_mode=train_mode,
                                             hparam_list=hparam_strs+[hparams_str],
                                             semi_supervised_alg=semi_supervised_alg,
                                             pl_threshold=pl_threshold,
                                             partial_feedback_mode=partial_feedback_mode,
                                             hierarchical_ssl=hierarchical_ssl,
                                             ema_decay=ema_decay,
                                             tp_idx=tp_idx
                                         )
                makedirs(interim_exp_dir_tp_idx)
                exp_result_path = os.path.join(interim_exp_dir_tp_idx, "result.ckpt")
                stats_path = os.path.join(interim_exp_dir_tp_idx, "stats.json")
                
                edge_matrix = get_edge_matrix(leaf_idx_to_all_class_idx, superclass_time=tp_idx-1)
                
                if os.path.exists(exp_result_path):
                    # print(f"{tp_idx} time period already finished for {hparams_str}")
                    res = load_pickle(exp_result_path)
                    print(
                        f"{tp_idx} time period already finished for {hparams_str}. Best epoch: {res['best_result']['best_epoch']}. Best Test Per-Class Acc: {res['acc_result']['test']:.2%}")
                else:
                    print(f"Run {train_mode_str} for TP {tp_idx}")
                    # exp_result do not exist, therefore start training
                    new_model, acc_result, best_result, avg_results, stats = start_training_semi_supervised(
                        copy.deepcopy(model),
                        model_save_dir,
                        tp_idx,
                        train_mode,
                        leco_mode,
                        hparams_mode,
                        train_val_subsets,
                        [info['num_of_classes'] for info in all_tp_info],
                        testset,
                        edge_matrix,
                        ema_decay=ema_decay,
                        semi_supervised_alg=semi_supervised_alg,
                        pl_threshold=pl_threshold,
                        partial_feedback_mode=partial_feedback_mode,
                        hierarchical_ssl=hierarchical_ssl,
                    )

                    save_obj_as_pickle(exp_result_path, {
                        # 'model' : new_model, #TODO
                        'acc_result' : acc_result,
                        'best_result' : best_result,
                        'avg_results' : avg_results
                    })
                    save_as_json(
                        stats_path,
                        stats
                    )
            print(f"Finished for {tp_idx} time period.")
            break

if __name__ == '__main__':
    args = argparser.parse_args()
    start_experiment(args.data_dir,
                     args.result_dir,
                     args.model_save_dir,
                     args.setup_mode,
                     args.train_mode,
                     args.leco_mode,
                     args.hparam_strs,
                     args.hparam_candidate,
                     args.semi_supervised_alg,
                     args.pl_threshold,
                     args.partial_feedback_mode,
                     args.hierarchical_ssl,
                     args.ema_decay,
                     seed=args.seed)

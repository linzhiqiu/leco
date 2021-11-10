import os
import argparse
import random
import numpy as np
import configs
from util import load_pickle, save_obj_as_pickle, makedirs
from models import update_model, make_optimizer, make_scheduler
import hparams
import setups
import copy


import torch
import torchvision
device = "cuda" if torch.cuda.is_available() else "cpu"

PARTIAL_FEEDBACK_MODES = [
    'partial_feedback', # Naively mix all history and current samples, using modified (log out) peiyun's loss function
    'partial_feedback_weight_history_0', # Naively mix all history and current samples, using modified (log out) peiyun's loss function, but weight history sample with weight 0 (sanity check)
    'partial_feedback_weight_history_2', # Naively mix all history and current samples, using modified (log out) peiyun's loss function, but weight history sample with weight 2 and current sample with 0
    'partial_feedback_weight_history_0.5', # Naively mix all history and current samples, using modified (log out) peiyun's loss function, but weight history sample with weight 0.5 and current sample with 1.5
    'log_in_partial_feedback', # Naively mix all history and current samples, using peiyun's loss function with the summation outside log func
    'log_in_partial_feedback_weight_history_0', # Naively mix all history and current samples, using peiyun's loss function, but weight history sample with weight 0 (sanity check)
    'log_in_partial_feedback_weight_history_2', # Naively mix all history and current samples, using peiyun's loss function, but weight history sample with weight 2 and current sample with 0
    'log_in_partial_feedback_weight_history_0.5', # Naively mix all history and current samples, using peiyun's loss function, but weight history sample with weight 0.5 and current sample with 1.5
    'no_history', # For test purpose, no history samples, using softmax loss (our modified version)
]

argparser = argparse.ArgumentParser()
argparser.add_argument("--data_dir", 
                        default='/scratch/leco/',
                        help="Where the dataset and final accuracy results will be saved.")
argparser.add_argument("--model_save_dir", 
                        default='/data3/zhiqiul/self_supervised_models/resnet18/',
                        help="Where the self-supervised pre-trained models were saved.")
argparser.add_argument("--setup_mode",
                        type=str,
                        default='cifar10_buffer_2000_500',
                        choices=setups.SETUPS.keys(),
                        help="The dataset setup mode")  
argparser.add_argument("--partial_feedback_mode",
                        type=str,
                        default='partial_feedback',
                        choices=PARTIAL_FEEDBACK_MODES,
                        help="The partial feedback mode")  
argparser.add_argument("--train_mode",
                        type=str,
                        default='resnet18_simclr_0_freeze_pt_linear_1_freeze_pt_linear',
                        choices=configs.TRAIN_MODES.keys(),
                        help="The train mode") 
argparser.add_argument("--hparam_0_str",
                        type=str,
                        default='cifar10_01',
                        choices=hparams.HPARAMS.keys(),
                        help="The hyperparameter mode for time period 0")  
argparser.add_argument("--hparam_1_str",
                        type=str,
                        default='cifar10_01',
                        choices=hparams.HPARAMS.keys(),
                        help="The hyperparameter mode for time period 1")   
argparser.add_argument('--seed', default=None, type=int,
                       help='seed for initializing training. ')

def get_exp_str_from_train_mode(train_mode: configs.TrainMode, tp_idx: int):
    if tp_idx == -1:
        if train_mode.pretrained_mode == None:
            return "none"
        else:
            return train_mode.pretrained_mode
    elif tp_idx >= 0:
        curr_config = train_mode.tp_configs[tp_idx]
        curr_str = "_".join([curr_config.extractor_mode, curr_config.classifier_mode])
        return f"_{tp_idx}_".join([get_exp_str_from_train_mode(train_mode, tp_idx-1), curr_str])

def get_exp_str_from_hparam_strs(hparam_strs, tp_idx: int):
    hparam_strs = hparam_strs[:tp_idx+1]
    return str(os.path.sep).join(hparam_strs)

def get_exp_str_from_partial_feedback(partial_feedback_mode : str, tp_idx : int):
    if tp_idx == 0 or partial_feedback_mode == None:
        return ""
    else:
        return partial_feedback_mode

def is_better(select_criterion, curr_value, best_value):
    # Return True if curr_value is better than best_value
    assert select_criterion in ['loss_per_epoch', 'acc_per_epoch']
    if select_criterion in ['loss_per_epoch']:
        return curr_value < best_value
    else:
        return curr_value > best_value

def train(loaders,
          model,
          optimizer,
          scheduler,
          epochs,
          tp_idx,
          loss_func, # loss_func takes model's output, time index, label as input, and return loss value
          hot_vector_func, # make a hot vector
          select_criterion='acc_per_epoch'
          ):
    model = model.to(device)

    avg_results = {'train': {'history_loss_per_epoch': [], 'history_acc_per_epoch': [],
                             'current_loss_per_epoch': [], 'current_acc_per_epoch': [],
                             'loss_per_epoch': [], 'acc_per_epoch': [],},
                   'val':   {'loss_per_epoch': [], 'acc_per_epoch': []},
                   'test':  {'loss_per_epoch': [], 'acc_per_epoch': []}}
    phases = ['train', 'val', 'test']

    # Save best model based on select_criterion
    best_result = {'best_loss': None, # overall loss at best epoch
                   'best_acc': 0, # overall acc at best epoch
                   'best_history_loss': None, # history loss at best epoch
                   'best_history_acc': 0, # history acc at best epoch
                   'best_current_loss': None, # current loss at best epoch
                   'best_current_acc': 0, # current acc at best epoch
                   'best_value' : None, # Best value of select_criterion
                   'best_epoch': None,
                   'best_model': None,
                   'best_criterion':select_criterion}

    for epoch in range(0, epochs):
        print(f"Epoch {epoch}")
        for phase in phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_loss_history = 0. # History sample loss (only on train set)
            running_loss_current = 0. # Current sample loss (only on train set)
            running_corrects = 0. # Overall correct count
            running_corrects_history = 0. # History sample correct count (only on train set)
            running_corrects_current = 0. # Current sample correct count (only on train set)
            
            count = 0
            count_history = 0 # (only on train set)
            count_current = 0 # (only on train set)

            pbar = loaders[phase]

            for batch, data in enumerate(pbar):
                if phase in ['train', 'val']:
                    inputs, time_indices, labels = data
                else:
                    inputs, labels = data
                    time_indices = torch.zeros((inputs.size(0))).long() + tp_idx
                count += time_indices.size(0)
                count_history += float((time_indices != tp_idx).sum())
                count_current += float((time_indices == tp_idx).sum())

                inputs = inputs.to(device)
                # labels = labels[tp_idx].to(device)

                if phase == 'train':
                    optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    # import pdb; pdb.set_trace()
                    hot_vector = hot_vector_func(time_indices, labels, device=outputs.device)
                    # The loss should be averaged across all samples
                    loss = loss_func(outputs, hot_vector)
                    
                    if phase == 'train':
                        loss.mean().backward()
                        optimizer.step()

                # statistics
                running_loss += loss.sum().item()
                # if running_loss != running_loss:
                #     import pdb; pdb.set_trace()
                # else:
                #     print(f"{batch} has loss {running_loss}")
                running_loss_current += loss[time_indices==tp_idx].sum().item()
                running_loss_history += loss[time_indices!=tp_idx].sum().item()
                corrects = torch.tensor([hot_vector[i, pred].item() for i, pred in enumerate(preds)])
                running_corrects += corrects.sum().item()
                running_corrects_current += corrects[time_indices==tp_idx].sum().item()
                running_corrects_history += corrects[time_indices!=tp_idx].sum().item()

            # Note that val set still has previous timestamp data  

            if phase == 'train':
                scheduler.step()
                avg_loss_history = float(running_loss_history)/count_history if count_history != 0 else 0.
                avg_loss_current = float(running_loss_current)/count_current
                avg_results[phase]['history_loss_per_epoch'].append(avg_loss_history)
                avg_results[phase]['current_loss_per_epoch'].append(avg_loss_current)
                
                avg_acc_history = float(running_corrects_history)/count_history if count_history != 0 else 0.
                avg_acc_current = float(running_corrects_current)/count_current
                avg_results[phase]['history_acc_per_epoch'].append(avg_acc_history)
                avg_results[phase]['current_acc_per_epoch'].append(avg_acc_current)
                phase_str = f"(Hist) Loss {avg_loss_history:.4f}, Acc {avg_acc_history:.2%}; (Curr) Loss {avg_loss_current:.4f}, Acc {avg_acc_current:.2%}"

                avg_loss = float(running_loss)/count
                avg_acc = float(running_corrects)/count
            else:
                phase_str = ""
                avg_loss = float(running_loss_current)/count_current
                avg_acc = float(running_corrects_current)/count_current
            
            avg_results[phase]['loss_per_epoch'].append(avg_loss)
            avg_results[phase]['acc_per_epoch'].append(avg_acc)

            if phase == 'val':
                curr_value = avg_results[phase][select_criterion][-1]
                if best_result['best_value'] == None or is_better(select_criterion, curr_value, best_result['best_value']):
                    print(
                        # f"Best val {select_criterion} at epoch {epoch} being {curr_value} with current sample accuracy {avg_acc_current:.2%} and history sample accuracy {avg_acc_history:.2%}")
                        f"Best val {select_criterion} at epoch {epoch} being {curr_value} | On train set this model achieves current sample accuracy {avg_acc_current:.2%} and history sample accuracy {avg_acc_history:.2%}")
                    best_result['best_epoch'] = epoch
                    best_result['best_acc'] = avg_acc
                    best_result['best_loss'] = avg_loss
                    # best_result['best_history_acc'] = avg_acc_history
                    # best_result['best_history_loss'] = avg_loss_history
                    # best_result['best_current_acc'] = avg_acc_current
                    # best_result['best_current_loss'] = avg_loss_current
                    best_result['best_value'] = curr_value
                    best_result['best_model'] = copy.deepcopy(model.state_dict())
            print(
                f"Epoch {epoch}: Average {phase} Loss {avg_loss:.4f}, Acc {avg_acc:.2%}; {phase_str}")
        print()
    print(
        f"Test Accuracy (for best val {select_criterion} model): {avg_results['test']['acc_per_epoch'][best_result['best_epoch']]:.2%}")
    print(
        f"Best Test Accuracy overall: {max(avg_results['test']['acc_per_epoch']):.2%}")
    model.load_state_dict(best_result['best_model'])
    test_acc = test(loaders['test'], model, tp_idx)
    print(f"Verify the best test accuracy for best val {select_criterion} is indeed {test_acc:.2%}")
    acc_result = {phase: avg_results[phase]['acc_per_epoch'][best_result['best_epoch']]
                  for phase in phases}
    return model, acc_result, best_result, avg_results

def test(test_loader,
         model,
         tp_idx):
    model = model.to(device).eval()
    running_corrects = 0.
    count = 0

    for batch, data in enumerate(test_loader):
        inputs, labels = data
        count += inputs.size(0)

        inputs = inputs.to(device)
        labels = labels[tp_idx].to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        # statistics
        running_corrects += torch.sum(preds == labels.data)

    avg_acc = float(running_corrects)/count
    print(f"Best Test Accuracy on test set: {avg_acc}")
    return avg_acc

class TimestampDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets # A dictionary of datasets (key is timestamp, value is dataset)
        self.dataset_times = sorted(datasets.keys())
        self.dataset_lengths = [len(datasets[time]) for time in self.dataset_times] # Length of each dataset
        self.length = sum(self.dataset_lengths) # Total length
        
        self.idx_to_dataset_idx = [] # map idx to item index in specific dataset
        self.idx_to_time = [] # map idx to the dataset index it belongs to
        for time_idx, dataset_length in enumerate(self.dataset_lengths):
            self.idx_to_dataset_idx += [i for i in range(dataset_length)]
            self.idx_to_time += [self.dataset_times[time_idx] for _ in range(dataset_length)]
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        time = self.idx_to_time[index]
        dataset_index = self.idx_to_dataset_idx[index]
        sample, label = self.datasets[time][dataset_index]
        return sample, time, label
        
def get_train_set(partial_feedback_mode,
                  train_val_subsets,
                  tp_idx):
    if partial_feedback_mode == 'no_history':
        train_datasets = {tp_idx : train_val_subsets[tp_idx][0]} # Only the current train set is used
        val_datasets = {tp_idx : train_val_subsets[tp_idx][1]} # Only the current val set is used
    elif partial_feedback_mode in ['partial_feedback', 
                                   'partial_feedback_weight_history_0',
                                   'partial_feedback_weight_history_2',
                                   'partial_feedback_weight_history_0.5',
                                   'log_in_partial_feedback',
                                   'log_in_partial_feedback_weight_history_0',
                                   'log_in_partial_feedback_weight_history_2',
                                   'log_in_partial_feedback_weight_history_0.5',
                                    ]:
        train_datasets = {idx : train_val_subsets[idx][0]
                                for idx in range(tp_idx+1)}
        val_datasets = {tp_idx : train_val_subsets[tp_idx][1]}  # Only the current val set is used
    return TimestampDataset(train_datasets), TimestampDataset(val_datasets)

def get_make_hot_vector_func(superclass_to_subclass_idx,
                             num_of_classes,
                             tp_idx):
    # Return a function that makes a one hot vector from timestamp and labels
    num_of_leaf_classes = num_of_classes[tp_idx]
    def make_hot_vector(time_indices, labels, device='cuda'):
        hot_vector = torch.zeros((time_indices.shape[0], num_of_leaf_classes)).to(device)
        for idx, time_idx in enumerate(time_indices):
            label_idx = int(labels[int(time_idx)][idx])
            if time_idx < tp_idx:
                label_indices = superclass_to_subclass_idx[label_idx]
                hot_vector[idx, label_indices] = 1
            elif time_idx == tp_idx:
                hot_vector[idx, label_idx] = 1
            else:
                raise ValueError('Invalid time index')
        return hot_vector
    return make_hot_vector

def get_loss_func(partial_feedback_mode):
    if partial_feedback_mode in ['no_history',
                                 'partial_feedback',
                                 'partial_feedback_weight_history_0',
                                 'partial_feedback_weight_history_2',
                                 'partial_feedback_weight_history_0.5',
                                 ]:
        # Regular softmax loss if no_history
        prefix_str = 'partial_feedback_weight_history_'
        if prefix_str in partial_feedback_mode:
            history_weight = float(partial_feedback_mode[len(prefix_str):])
            current_weight = 2. - history_weight
        else:
            history_weight = current_weight = 1.

        def loss_func(outputs, hot_vector):
            hot_vector = hot_vector.to(outputs.device)
            history_mask = hot_vector.sum(1) != 1.
            current_mask = hot_vector.sum(1) == 1.
            
            # prob = torch.nn.Softmax(dim=1)(outputs)
            log_prob = torch.nn.LogSoftmax(dim=1)(outputs)
            # loss = -(torch.log(prob) * hot_vector).sum(dim=1) # This is not numerically stable
            loss = -(log_prob * hot_vector).sum(dim=1)
            loss[history_mask] = loss[history_mask] * history_weight
            loss[current_mask] = loss[current_mask] * current_weight
            return loss
        return loss_func
    elif partial_feedback_mode in ['log_in_partial_feedback',
                                   'log_in_partial_feedback_weight_history_0',
                                   'log_in_partial_feedback_weight_history_2',
                                   'log_in_partial_feedback_weight_history_0.5',]:
        prefix_str = 'log_in_partial_feedback_weight_history_'
        if prefix_str in partial_feedback_mode:
            history_weight = float(partial_feedback_mode[len(prefix_str):])
            current_weight = 2. - history_weight
        else:
            history_weight = current_weight = 1.

        def loss_func(outputs, hot_vector):
            history_mask = hot_vector.sum(1) != 1.
            current_mask = hot_vector.sum(1) == 1.

            hot_vector = hot_vector.to(outputs.device)

            outputs = outputs - outputs.max(1)[0].unsqueeze(1)
            loss = - ( (outputs * hot_vector).sum(1) - torch.log((torch.exp(outputs)).sum(1)))

            loss[history_mask] = loss[history_mask] * history_weight
            loss[current_mask] = loss[current_mask] * current_weight
            return loss
        return loss_func                        
    else:
        raise NotImplementedError()

def start_training_partial_feedback(model,
                                    model_save_dir,
                                    tp_idx,
                                    train_mode,
                                    hparams_mode,
                                    train_val_subsets,
                                    num_of_classes,
                                    testset,
                                    partial_feedback_mode,
                                    superclass_to_subclass_idx):
    if tp_idx == 0:
        assert model == None
    else:
        assert model != None
    
    model = update_model(model, model_save_dir, tp_idx, train_mode, num_of_classes)
    
    batch_size = hparams_mode['batch']
    workers = hparams_mode['workers']
    
    hparam_mode = hparams_mode['hparams']
    epochs = hparam_mode['epochs']

    optimizer = make_optimizer(model,
                               hparam_mode['optim'],
                               hparam_mode['lr'],
                               weight_decay=hparam_mode['weight_decay'],
                               momentum=hparam_mode['momentum'])
    scheduler = make_scheduler(optimizer,
                               step_size=hparam_mode['decay_epochs'],
                               gamma=hparam_mode['decay_by'])

    hot_vector_func = get_make_hot_vector_func(superclass_to_subclass_idx,
                                               num_of_classes,
                                               tp_idx)
    loss_func = get_loss_func(partial_feedback_mode)
    train_set, val_set = get_train_set(partial_feedback_mode, train_val_subsets, tp_idx)
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
        tp_idx,
        loss_func,
        hot_vector_func
    )
    return model, acc_result, best_result, avg_results

def get_superclass_to_subclass_idx(idx_to_superclass_idx,
                                   num_of_superclasses):
    superclass_to_subclass_idx = {super_idx : [sub_idx 
                                               for sub_idx in idx_to_superclass_idx.keys()
                                               if idx_to_superclass_idx[sub_idx] == super_idx]
                                  for super_idx in range(num_of_superclasses)}
    return superclass_to_subclass_idx

def start_experiment(data_dir: str, # where the data are saved, and datasets + model + final accuracy results will be saved
                     model_save_dir: str, # where the self-supervised pretrained models are saved
                     setup_mode_str: str, 
                     train_mode_str: str,
                     hparam_strs, # List of strs
                     partial_feedback_mode : str, # str
                     seed=None):
    seed_str = f"seed_{seed}"
    if seed == None:
        print("Not using a random seed")
    else:
        print(f"Using random seed {seed}")
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    setup_dir = os.path.join(data_dir, setup_mode_str, seed_str)
    makedirs(setup_dir)
    dataset_path = os.path.join(setup_dir, 'dataset.pt')
    if os.path.exists(dataset_path):
        dataset = load_pickle(dataset_path)
    else:
        setup_mode = setups.SETUPS[setup_mode_str]
        dataset = setups.generate_dataset(data_dir, setup_mode)
        save_obj_as_pickle(dataset_path, dataset)
        print(f"Dataset saved at {dataset_path}")

    train_val_subsets, testset, num_of_classes = dataset

    superclass_to_subclass_idx = get_superclass_to_subclass_idx(testset.idx_to_superclass_idx,
                                                                num_of_classes[0])
    
    train_mode = configs.TRAIN_MODES[train_mode_str]

    train_mode_str_check = get_exp_str_from_train_mode(train_mode, tp_idx=1)
    assert train_mode_str_check == train_mode_str
    
    model = None
    assert len(hparam_strs) == len(train_mode.tp_configs)
    for tp_idx in range(len(train_mode.tp_configs)):
        interim_exp_dir_tp_idx = os.path.join(setup_dir,
                                              get_exp_str_from_train_mode(train_mode, tp_idx=tp_idx),
                                              get_exp_str_from_partial_feedback(partial_feedback_mode, tp_idx=tp_idx),
                                              get_exp_str_from_hparam_strs(hparam_strs, tp_idx=tp_idx))
        makedirs(interim_exp_dir_tp_idx)
        
        hparams_mode = hparams.HPARAMS[hparam_strs[tp_idx]]
        exp_result_path = os.path.join(interim_exp_dir_tp_idx, "result.ckpt")

        if os.path.exists(exp_result_path):
            print(f"{tp_idx} time period already finished. Load from {exp_result_path}")
            exp_result = load_pickle(exp_result_path)
            model = exp_result['model']
            acc_result = exp_result['acc_result']
            best_result = exp_result['best_result']
            avg_results = exp_result['avg_results']
        else:
            # both prev_result and exp_result do not exist, therefore start training
            model, acc_result, best_result, avg_results = start_training_partial_feedback(
                model,
                model_save_dir,
                tp_idx,
                train_mode,
                hparams_mode,
                train_val_subsets,
                num_of_classes,
                testset,
                partial_feedback_mode,
                superclass_to_subclass_idx
            )

            save_obj_as_pickle(exp_result_path, {
                'model' : model,
                'acc_result' : acc_result,
                'best_result' : best_result,
                'avg_results' : avg_results
            })


if __name__ == '__main__':
    args = argparser.parse_args()
    start_experiment(args.data_dir,
                     args.model_save_dir,
                     args.setup_mode,
                     args.train_mode,
                     [args.hparam_0_str, args.hparam_1_str],
                     args.partial_feedback_mode,
                     seed=args.seed)
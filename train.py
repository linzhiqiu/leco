import os
import argparse
import random
import numpy as np
import configs
from util import load_pickle, save_obj_as_pickle, makedirs
from models import update_model, make_optimizer, make_scheduler
import hparams
import setups
import print_utils
import copy


import torch
import torchvision
device = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_BEST_MODEL = False

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
argparser.add_argument("--train_mode",
                        type=str,
                        default='resnet18_simclr_0_freeze_pt_linear_1_freeze_pt_linear',
                        choices=configs.TRAIN_MODES.keys(),
                        help="The train mode") 
argparser.add_argument('--hparam_strs', nargs='+', default=[],
                       help='The hparam to use for each time period. If not specified, then use hparam_candidate. Should be used to specify the best hparam for all previous time periods.')
argparser.add_argument("--hparam_candidate",
                        type=str,
                        default='cifar',
                        choices=hparams.HPARAM_CANDIDATES.keys(),
                        help="The hyperparameter candidates (str) for next time period")
argparser.add_argument('--seed', default=None, type=int,
                       help='seed for initializing training. ')

def train(loaders,
          model,
          optimizer,
          scheduler,
          epochs,
          tp_idx):
    model = model.to(device)
    criterion = torch.nn.NLLLoss(reduction='mean')

    avg_results = {'train': {'loss_per_epoch': [], 'acc_per_epoch': []},
                   'val':   {'loss_per_epoch': [], 'acc_per_epoch': []},
                   'test':  {'loss_per_epoch': [], 'acc_per_epoch': []}}
    phases = ['train', 'val', 'test']

    # Save best validation accuracy model
    best_result = {'best_loss': None,
                   'best_acc': 0,
                   'best_epoch': None,
                   'best_model': None
                   }

    for epoch in range(0, epochs):
        print(f"Epoch {epoch}")
        # import pdb; pdb.set_trace()
        for phase in phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.
            count = 0

            pbar = loaders[phase]

            for batch, data in enumerate(pbar):
                inputs, labels = data
                count += inputs.size(0)

                inputs = inputs.to(device)
                labels = labels[tp_idx].to(device)

                if phase == 'train':
                    optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    log_probability = torch.nn.functional.log_softmax(outputs, dim=1)
                    loss = criterion(log_probability, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            avg_loss = float(running_loss)/count
            avg_acc = float(running_corrects)/count
            avg_results[phase]['loss_per_epoch'].append(avg_loss)
            avg_results[phase]['acc_per_epoch'].append(avg_acc)
            if phase == 'train':
                scheduler.step()
            if phase == 'val':
                if best_result['best_acc'] == None or avg_acc > best_result['best_acc']:
                    print(
                        f"Best validation accuracy at epoch {epoch} being {avg_acc} with loss {avg_loss}")
                    best_result['best_epoch'] = epoch
                    best_result['best_acc'] = avg_acc
                    best_result['best_loss'] = avg_loss
                    if SAVE_BEST_MODEL:
                        best_result['best_model'] = copy.deepcopy(model.state_dict())

            print(
                f"Epoch {epoch}: Average {phase} Loss {avg_loss}, Accuracy {avg_acc:.2%}")
        print()
    print(
        f"Test Accuracy (for best validation accuracy model): {avg_results['test']['acc_per_epoch'][best_result['best_epoch']]:.2%}")
    print(
        f"Best Test Accuracy overall: {max(avg_results['test']['acc_per_epoch']):.2%}")
    if SAVE_BEST_MODEL:
        model.load_state_dict(best_result['best_model'])
    test_acc = test(loaders['test'], model, tp_idx)
    print(f"Verify the best test accuracy for best validation accuracy is indeed {test_acc:.2%}")
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

def start_training(model,
                   model_save_dir,
                   tp_idx,
                   train_mode,
                   hparams_mode,
                   train_val_subsets,
                   num_of_classes,
                   testset):
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
    
    loaders = {}
    loaders['train'] = torch.utils.data.DataLoader(
                           train_val_subsets[tp_idx][0],
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=workers
                       )
    loaders['val'] = torch.utils.data.DataLoader(
                         train_val_subsets[tp_idx][1],
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
        tp_idx
    )
    return model, acc_result, best_result, avg_results
    

def start_experiment(data_dir: str, # where the data are saved, and datasets + model + final accuracy results will be saved
                     model_save_dir: str, # where the self-supervised pretrained models are saved
                     setup_mode_str: str, 
                     train_mode_str: str,
                     hparam_strs, # A list of hparam str to load
                     hparam_candidate : str, # The list of hparam to try for next time period (tp_idx = len(hparam_strs))
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

    train_val_subsets, testset, all_tp_info, leaf_idx_to_all_class_idx = dataset
    
    train_mode = configs.TRAIN_MODES[train_mode_str]

    # train_mode_str_check = print_utils.get_exp_str_from_train_mode(train_mode, tp_idx=1)
    # assert train_mode_str_check == train_mode_str
    
    model = None
    # assert len(hparam_strs) == len(train_mode.tp_configs)
    # for tp_idx in range(len(train_mode.tp_configs)):
    for tp_idx in range(len(train_val_subsets)):
        if tp_idx >= len(hparam_strs):
            for hparams_str in hparams.HPARAM_CANDIDATES[hparam_candidate]:
                hparams_mode = hparams.HPARAMS[hparams_str]
                interim_exp_dir_tp_idx = os.path.join(setup_dir,
                                                      print_utils.get_exp_str_from_train_mode(train_mode, tp_idx=tp_idx),
                                                      print_utils.get_exp_str_from_hparam_strs(hparam_strs+[hparams_str], tp_idx=tp_idx))
                makedirs(interim_exp_dir_tp_idx)
                exp_result_path = os.path.join(interim_exp_dir_tp_idx, "result.ckpt")
                if os.path.exists(exp_result_path):
                    print(f"{tp_idx} time period already finished for {hparams_str}")
                else:
                    new_model, acc_result, best_result, avg_results = start_training(
                        copy.deepcopy(model),
                        model_save_dir,
                        tp_idx,
                        train_mode,
                        hparams_mode,
                        train_val_subsets,
                        [info['num_of_classes'] for info in all_tp_info],
                        testset,
                    )

                    save_obj_as_pickle(exp_result_path, {
                        'model' : new_model,
                        'acc_result' : acc_result,
                        'best_result' : best_result,
                        'avg_results' : avg_results
                    })
            print(f"Finished for {tp_idx} time period.")
            break
        else:
            interim_exp_dir_tp_idx = os.path.join(setup_dir,
                                                  print_utils.get_exp_str_from_train_mode(train_mode, tp_idx=tp_idx),
                                                  print_utils.get_exp_str_from_hparam_strs(hparam_strs, tp_idx=tp_idx))
            makedirs(interim_exp_dir_tp_idx)
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

if __name__ == '__main__':
    args = argparser.parse_args()
    start_experiment(args.data_dir,
                     args.model_save_dir,
                     args.setup_mode,
                     args.train_mode,
                     args.hparam_strs,
                     args.hparam_candidate,
                     seed=args.seed)
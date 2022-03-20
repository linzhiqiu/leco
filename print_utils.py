### Several functions to help printing
import configs
import os

def get_exp_str_from_train_mode(train_mode: configs.TrainMode, tp_idx: int):
    """Given a TrainMode object, return a string with all train mode information up to [tp_idx]
    """
    if tp_idx == -1:
        return "_".join([train_mode.arch, train_mode.pretrain])
    elif tp_idx >= 0:
        curr_config = train_mode.tp_configs[tp_idx]
        curr_str = "_".join([curr_config.extractor_mode, curr_config.classifier_mode])
        return f"_{tp_idx}_".join([get_exp_str_from_train_mode(train_mode, tp_idx-1), curr_str])

def get_exp_str_from_hparam_strs(hparam_strs, tp_idx: int):
    hparam_strs = hparam_strs[:tp_idx+1]
    return str(os.path.sep).join(hparam_strs)

def get_exp_str_from_partial_feedback_args(partial_feedback_mode: str=None):
    if partial_feedback_mode:
        return partial_feedback_mode
    else:
        return ""

def get_exp_str_from_cl_mode(cl_mode):
    if cl_mode:
        return cl_mode
    else:
        raise NotImplementedError()

def get_exp_str_from_ema_decay(ema_decay):
    if ema_decay:
        return f"ema_decay_{ema_decay}"
    else:
        return ""

def get_exp_str_from_semi_args(ratio_unlabeled_to_labeled: float=1.,
                               semi_supervised_alg: str=None,
                               pl_threshold: float=None,
                               hierarchical_ssl: str=None,
                               finetuning_mode: str=None):
    if semi_supervised_alg:
        name = f"{semi_supervised_alg}_R_{ratio_unlabeled_to_labeled}"
        if semi_supervised_alg in ['PL', "Fixmatch"]:
            name += f"_T_{pl_threshold}"
        elif semi_supervised_alg in ['DistillHard', 'DistillSoft']:
            pass
        else:
            raise NotImplementedError()

        if hierarchical_ssl:
            name += f"_H_{hierarchical_ssl}"
        
        if finetuning_mode:
            name += f"_F_{finetuning_mode}"
        return name
    else:
        return ""
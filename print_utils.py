### Several functions to help printing
import configs
import os

def get_exp_str_from_train_mode(train_mode: configs.TrainMode, tp_idx: int):
    """Given a TrainMode object, return a string with all train mode information up to [tp_idx]
    """
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

def get_exp_str_from_multi_head(multi_head_mode : str, tp_idx : int):
    if tp_idx == 0 or multi_head_mode == None:
        return ""
    else:
        return multi_head_mode
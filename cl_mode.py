CL_MODES = [
    'label_new', # Use current TP images for labeled set, use all cumulative images for Joint/LPL, and use old images for SSL
    'relabel_old', # Keep using images from TP0 (do not support Joint/LPL/SSL)
    'upper_bound', # Assume all TP images have finest labels (do not support Joint/LPL/SSL)
    'upper_bound_with_multi_task', # (Only support two TPs) Use T0+T1 sampled images for labeled set + Joint or LPL (no ssl)
]
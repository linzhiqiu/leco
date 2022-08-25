LECO_MODES = [
    # LabelNew: Label New Images
    #     Use current TP images for CE loss
    #     Use all cumulative images for Joint/LPL loss
    #     Use previous TP images for SSL loss
    'label_new',
    
    # RelabelOld: Relabel Old Images
    #     Keep using images from TP0 (do not support Joint/LPL/SSL)
    'relabel_old',
    
    # UpperBound: All Images have Finest Labels
    #     Assume all TP images have finest labels (do not support Joint/LPL/SSL)
    'upper_bound',
    
    # UpperBoundWithMultiTask: All Images have Finest Labels + Coarse Labels
    #     (Only support two TPs) Use T0+T1 images for CE loss + Joint or LPL loss
    'upper_bound_with_multi_task',
]

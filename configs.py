
PRETRAINED_MODES = [
    None,
    'resnet18_simclr', # From (https://github.com/sthalles/SimCLR), download link is: https://drive.google.com/file/d/14_nH2FkyKbt61cieQDiSbBVNP8-gtwgF/view
    # Below are trained from github repo (with default parameters): https://github.com/untitled-ai/self_supervised
    'resnet18_simclr_pl',
    'resnet18_moco_v2_pl',
    'resnet18_byol_pl'
]

EXTRACTOR_MODES = [
    'scratch', # Initialize a new backbone
    'finetune_pt', # Finetune from pre-trained model
    'freeze_pt', # Freeze pre-trained model
    'finetune_prev', # Finetune model from last timestamp
    'freeze_prev' # Freeze model from last timestamp
]

CLASSIFER_MODES = [
    'linear', # Train a new linear layer
    'mlp', # Train a new MLP classifier
    'mlp_replace_last', # Replace the last layer of MLP classifier from previous timestamp
]

class Phase():
    def __init__(self, extractor_mode, classifier_mode):
        assert extractor_mode in EXTRACTOR_MODES
        assert classifier_mode in CLASSIFER_MODES
        self.extractor_mode = extractor_mode
        self.classifier_mode = classifier_mode

class TrainMode():
    def __init__(self, pretrained_mode : str, tp_0 : Phase, tp_1 : Phase):
        assert pretrained_mode in PRETRAINED_MODES
        self.pretrained_mode = pretrained_mode
        self.tp_0 = tp_0
        self.tp_1 = tp_1

TRAIN_MODES = {
    "none_0_scratch_1_scratch" : TrainMode(
        None,
        Phase('scratch', 'linear'),
        Phase('scratch', 'linear'),
    ),
    "none_0_scratch_1_finetune_prev_linear" : TrainMode(
        None,
        Phase('scratch', 'linear'),
        Phase('finetune_prev', 'linear'),
    ),
    "none_0_scratch_1_finetune_prev_mlp" : TrainMode(
        None,
        Phase('scratch', 'linear'),
        Phase('finetune_prev', 'mlp'),
    ),
    "resnet18_simclr_0_freeze_pt_linear_1_freeze_pt_linear" : TrainMode(
        'resnet18_simclr',
        Phase('freeze_pt', 'linear'),
        Phase('freeze_pt', 'linear'),
    ),
    "resnet18_simclr_0_freeze_pt_mlp_1_freeze_pt_mlp" : TrainMode(
        'resnet18_simclr',
        Phase('freeze_pt', 'mlp'),
        Phase('freeze_pt', 'mlp'),
    ),
    "resnet18_simclr_0_freeze_pt_mlp_1_freeze_pt_mlp_replace_last" : TrainMode(
        'resnet18_simclr',
        Phase('freeze_pt', 'mlp'),
        Phase('freeze_pt', 'mlp_replace_last'),
    ),
    "resnet18_simclr_0_finetune_pt_linear_1_finetune_pt_linear" : TrainMode(
        'resnet18_simclr',
        Phase('finetune_pt', 'linear'),
        Phase('finetune_pt', 'linear'),
    ),
    "resnet18_simclr_0_finetune_pt_linear_1_finetune_prev_linear" : TrainMode(
        'resnet18_simclr',
        Phase('finetune_pt', 'linear'),
        Phase('finetune_prev', 'linear'),
    ),
    "resnet18_simclr_0_finetune_pt_linear_1_finetune_prev_mlp" : TrainMode(
        'resnet18_simclr',
        Phase('finetune_pt', 'linear'),
        Phase('finetune_prev', 'mlp'),
    ),
}
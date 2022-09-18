from typing import List

ARCHS = [
    'wideres_28_2',
    'resnet50'
]

PRETRAINED_MODES = [
    'scratch',  # TrainScratch
    'imagenet',  # Start from imagenet-pretrained weight, if exists
]

EXTRACTOR_MODES = [
    'finetune_pt',  # Finetune from pre-trained (pt) weight, including scratch weight
    'freeze_pt',  # Freeze pre-trained (pt) weight, including scratch weight
    'finetune_prev',  # Finetune model from last timestamp
    'freeze_prev',  # Freeze model from last timestamp
]

CLASSIFER_MODES = [
    'linear',  # Train a new linear layer
]


class Phase():
    def __init__(self, extractor_mode, classifier_mode):
        assert extractor_mode in EXTRACTOR_MODES
        assert classifier_mode in CLASSIFER_MODES
        self.extractor_mode = extractor_mode
        self.classifier_mode = classifier_mode


class TrainMode():
    def __init__(self,
                 arch: str,
                 pretrain: str,
                 tp_configs: List[Phase]):
        assert pretrain in PRETRAINED_MODES
        assert arch in ARCHS
        self.arch = arch
        self.pretrain = pretrain
        self.tp_configs = tp_configs


CIFAR_MODES = {
    "wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_pt_linear": TrainMode(
        "wideres_28_2",
        'scratch',
        [Phase('finetune_pt', 'linear'), Phase('finetune_pt', 'linear'), ]
    ),
    "wideres_28_2_scratch_0_finetune_pt_linear_1_finetune_prev_linear": TrainMode(
        "wideres_28_2",
        'scratch',
        [Phase('finetune_pt', 'linear'), Phase('finetune_prev', 'linear'), ]
    ),
    "wideres_28_2_scratch_0_finetune_pt_linear_1_freeze_pt_linear": TrainMode(
        'wideres_28_2',
        'scratch',
        [Phase('finetune_pt', 'linear'), Phase('freeze_pt', 'linear'), ]
    ),
    "wideres_28_2_scratch_0_finetune_pt_linear_1_freeze_prev_linear": TrainMode(
        'wideres_28_2',
        'scratch',
        [Phase('finetune_pt', 'linear'), Phase('freeze_prev', 'linear'), ]
    ),
}

INAT_MODES = {
    # Resnet50 for inat
    "resnet50_scratch_0_finetune_pt_linear_1_finetune_pt_linear": TrainMode(
        "resnet50",
        'scratch',
        [Phase('finetune_pt', 'linear'), Phase('finetune_pt', 'linear'), ]
    ),
    "resnet50_scratch_0_finetune_pt_linear_1_finetune_prev_linear": TrainMode(
        "resnet50",
        'scratch',
        [Phase('finetune_pt', 'linear'), Phase('finetune_prev', 'linear'), ]
    ),
    "resnet50_scratch_0_finetune_pt_linear_1_freeze_pt_linear": TrainMode(
        'resnet50',
        'scratch',
        [Phase('finetune_pt', 'linear'), Phase('freeze_pt', 'linear'), ]
    ),
    "resnet50_scratch_0_finetune_pt_linear_1_freeze_prev_linear": TrainMode(
        'resnet50',
        'scratch',
        [Phase('finetune_pt', 'linear'), Phase('freeze_prev', 'linear'), ]
    ),
    # Resnet50 pre-trained imagenet for inat
    "resnet50_imagenet_0_finetune_pt_linear_1_finetune_pt_linear": TrainMode(
        "resnet50",
        'imagenet',
        [Phase('finetune_pt', 'linear'), Phase('finetune_pt', 'linear'), ]
    ),
    "resnet50_imagenet_0_finetune_pt_linear_1_finetune_prev_linear": TrainMode(
        "resnet50",
        'imagenet',
        [Phase('finetune_pt', 'linear'), Phase('finetune_prev', 'linear'), ]
    ),
    "resnet50_imagenet_0_finetune_pt_linear_1_freeze_pt_linear": TrainMode(
        'resnet50',
        'imagenet',
        [Phase('finetune_pt', 'linear'), Phase('freeze_pt', 'linear'), ]
    ),
    "resnet50_imagenet_0_finetune_pt_linear_1_freeze_prev_linear": TrainMode(
        'resnet50',
        'imagenet',
        [Phase('finetune_pt', 'linear'), Phase('freeze_prev', 'linear'), ]
    ),
}

INAT_MODES_4_TPS = {
    # Resnet50 for inat
    "resnet50_scratch_train_scratch": TrainMode(
        "resnet50",
        'scratch',
        [Phase('finetune_pt', 'linear'), Phase('finetune_pt', 'linear'),
         Phase('finetune_pt', 'linear'), Phase('finetune_pt', 'linear'), ]
    ),
    "resnet50_scratch_finetune_prev": TrainMode(
        "resnet50",
        'scratch',
        [Phase('finetune_pt', 'linear'), Phase('finetune_prev', 'linear'),
         Phase('finetune_prev', 'linear'), Phase('finetune_prev', 'linear'), ]
    ),
}

TRAIN_MODES = {
    **CIFAR_MODES,
    **INAT_MODES,
    **INAT_MODES_4_TPS,
}

ALL_TRAIN_MODES = {
    'cifar': CIFAR_MODES,
    'inat': INAT_MODES,
    'inat_4_tps': INAT_MODES_4_TPS,
}

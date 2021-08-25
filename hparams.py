
CIFAR10_DECAY_HPARAMS = {
    'batch' : 128,
    'workers' : 4,
    'hparams' : [{
        'optim' : 'sgd',
        'weight_decay' : 5e-4,
        'momentum' : 0.9,
        'lr' : 0.1,
        'epochs' : 200,
        'decay_epochs': 60,
        'decay_by' : 0.1,
    },
    {
        'optim' : 'sgd',
        'weight_decay' : 5e-4,
        'momentum' : 0.9,
        'lr' : 0.01,
        'epochs' : 200,
        'decay_epochs': 60,
        'decay_by' : 0.1,
    }],
}

CIFAR10_DEFAULT_HPARAMS = {
    'batch' : 128,
    'workers' : 4,
    'hparams' : [{
        'optim' : 'sgd',
        'weight_decay' : 5e-4,
        'momentum' : 0.9,
        'lr' : 0.1,
        'epochs' : 200,
        'decay_epochs': 60,
        'decay_by' : 0.1,
    },
    {
        'optim' : 'sgd',
        'weight_decay' : 5e-4,
        'momentum' : 0.9,
        'lr' : 0.1,
        'epochs' : 200,
        'decay_epochs': 60,
        'decay_by' : 0.1,
    }],
}

HPARAMS = {
    'cifar10_default' : CIFAR10_DEFAULT_HPARAMS,
    'cifar10_decay' : CIFAR10_DECAY_HPARAMS,
}
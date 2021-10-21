CIFAR10_HPARAMS_0001 = {
    'batch' : 128,
    'workers' : 4,
    'hparams' : {
        'optim' : 'sgd',
        'weight_decay' : 5e-4,
        'momentum' : 0.9,
        'lr' : 0.001,
        'epochs' : 200,
        'decay_epochs': 60,
        'decay_by' : 0.1,
    },
}

CIFAR10_HPARAMS_001 = {
    'batch' : 128,
    'workers' : 4,
    'hparams' : {
        'optim' : 'sgd',
        'weight_decay' : 5e-4,
        'momentum' : 0.9,
        'lr' : 0.01,
        'epochs' : 200,
        'decay_epochs': 60,
        'decay_by' : 0.1,
    },
}

CIFAR10_HPARAMS_01 = {
    'batch' : 128,
    'workers' : 4,
    'hparams' : {
        'optim' : 'sgd',
        'weight_decay' : 5e-4,
        'momentum' : 0.9,
        'lr' : 0.1,
        'epochs' : 200,
        'decay_epochs': 60,
        'decay_by' : 0.1,
    },
}

HPARAMS = {
    'cifar10_01' : CIFAR10_HPARAMS_01,
    'cifar10_001' : CIFAR10_HPARAMS_001,
    'cifar10_0001' : CIFAR10_HPARAMS_0001,
}
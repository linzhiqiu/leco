HPARAMS = {
    'cifar_00001_batch_128' : {
        'batch' : 128,
        'workers' : 4,
        'hparams' : {
            'optim' : 'sgd',
            'weight_decay' : 5e-4,
            'momentum' : 0.9,
            'lr' : 0.0001,
            'epochs' : 200,
            'decay_epochs': 60,
            'decay_by' : 0.1,
        },
    },
    'cifar_0001_batch_128' : {
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
    },
    'cifar_001_batch_128' : {
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
    },
    'cifar_01_batch_128' : {
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
    },
    'cifar_1_batch_128' : {
        'batch' : 128,
        'workers' : 4,
        'hparams' : {
            'optim' : 'sgd',
            'weight_decay' : 5e-4,
            'momentum' : 0.9,
            'lr' : 1.,
            'epochs' : 200,
            'decay_epochs': 60,
            'decay_by' : 0.1,
        },
    },
    'cifar_00001_batch_64' : {
        'batch' : 64,
        'workers' : 4,
        'hparams' : {
            'optim' : 'sgd',
            'weight_decay' : 5e-4,
            'momentum' : 0.9,
            'lr' : 0.0001,
            'epochs' : 200,
            'decay_epochs': 60,
            'decay_by' : 0.1,
        },
    },
    'cifar_0001_batch_64' : {
        'batch' : 64,
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
    },
    'cifar_001_batch_64' : {
        'batch' : 64,
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
    },
    'cifar_01_batch_64' : {
        'batch' : 64,
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
    },
    'cifar_1_batch_64' : {
        'batch' : 64,
        'workers' : 4,
        'hparams' : {
            'optim' : 'sgd',
            'weight_decay' : 5e-4,
            'momentum' : 0.9,
            'lr' : 1.,
            'epochs' : 200,
            'decay_epochs': 60,
            'decay_by' : 0.1,
        },
    },
}

HPARAM_CANDIDATES = {
    'cifar' : ['cifar_1_batch_128', 'cifar_01_batch_128', 'cifar_001_batch_128', 'cifar_0001_batch_128', 'cifar_00001_batch_128',
               'cifar_1_batch_64', 'cifar_01_batch_64', 'cifar_001_batch_64', 'cifar_0001_batch_64', 'cifar_00001_batch_64']
}

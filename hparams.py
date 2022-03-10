HPARAMS = {
    'cifar10_lr_06_batch_128' : {
        'batch' : 128,
        'workers' : 4,
        'hparams' : {
            'total_steps' : 160000,
            'eval_steps' : 1000,
            'warmup_steps' : 0,
            'optim' : 'sgd',
            'weight_decay' : 5e-4,
            'momentum' : 0.9,
            'lr' : 0.3,
            'decay' : 'cosine',
        },
    },
    'cifar10_lr_006_batch_128' : {
        'batch' : 128,
        'workers' : 4,
        'hparams' : {
            'total_steps' : 160000,
            'eval_steps' : 1000,
            'warmup_steps' : 0,
            'optim' : 'sgd',
            'weight_decay' : 5e-4,
            'momentum' : 0.9,
            'lr' : 0.06,
            'decay' : 'cosine',
        },
    },
    'cifar10_lr_0006_batch_128' : {
        'batch' : 128,
        'workers' : 4,
        'hparams' : {
            'total_steps' : 160000,
            'eval_steps' : 1000,
            'warmup_steps' : 0,
            'optim' : 'sgd',
            'weight_decay' : 5e-4,
            'momentum' : 0.9,
            'lr' : 0.006,
            'decay' : 'cosine',
        },
    },
    'cifar10_lr_00006_batch_128' : {
        'batch' : 128,
        'workers' : 4,
        'hparams' : {
            'total_steps' : 160000,
            'eval_steps' : 1000,
            'warmup_steps' : 0,
            'optim' : 'sgd',
            'weight_decay' : 5e-4,
            'momentum' : 0.9,
            'lr' : 0.0006,
            'decay' : 'cosine',
        },
    },
    'cifar10_lr_000006_batch_128' : {
        'batch' : 128,
        'workers' : 4,
        'hparams' : {
            'total_steps' : 160000,
            'eval_steps' : 1000,
            'warmup_steps' : 0,
            'optim' : 'sgd',
            'weight_decay' : 5e-4,
            'momentum' : 0.9,
            'lr' : 0.00006,
            'decay' : 'cosine',
        },
    },
}

HPARAM_CANDIDATES = {
    'cifar' : ['cifar10_lr_000006_batch_128', 'cifar10_lr_00006_batch_128', 'cifar10_lr_0006_batch_128', 'cifar10_lr_006_batch_128', 'cifar10_lr_06_batch_128']
}

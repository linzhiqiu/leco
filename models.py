import torch
import torch.nn as nn
import torchvision
from self_supervised.moco import MoCoMethod

class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            print("Architecture not supported")
            exit(0)
        else:
            return model

    def forward(self, x):
        return self.backbone(x)


class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.output_size = output_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        return output

def load_model(model_save_dir: str,
               pretrained_mode: str):
    """Load a pre-trained resnet model. 
    model.fc = nn.Identity()
    """
    if pretrained_mode == None:
        model = torchvision.models.resnet18(pretrained=False, num_classes=100)
        model.fc = torch.nn.Identity()
        return model
    else:
        if pretrained_mode == 'resnet18_simclr':
            from models import ResNetSimCLR
            model = ResNetSimCLR('resnet18', out_dim=128)
            model.load_state_dict(torch.load(f"{model_save_dir}/{pretrained_mode}.ckpt")['state_dict'])
            model = model.backbone
            model.fc = torch.nn.Identity()
        else:
            # pytorch lightning modules
            model = MoCoMethod.load_from_checkpoint(f"{model_save_dir}/{pretrained_mode}.ckpt")
            model = model.__dict__['_modules']['model']
        return model

def update_model(model, model_save_dir, tp_idx, train_mode, num_of_classes):
    extractor_mode = train_mode.tp_configs[tp_idx].extractor_mode
    classifier_mode = train_mode.tp_configs[tp_idx].classifier_mode
    num_class = num_of_classes[tp_idx]
    
    if tp_idx > 0 and classifier_mode == 'mlp_replace_last':
        prev_fc = copy.deepcopy(model.fc)
    
    if extractor_mode in ['scratch', 'finetune_pt', 'freeze_pt']:
        model = load_model(model_save_dir, train_mode.pretrained_mode)
    elif extractor_mode in ['finetune_prev', 'freeze_prev']:
        print("resume from previous feature extractor checkpoint")
        assert model != None and tp_idx > 0
    
    if classifier_mode == 'linear':
        model.fc = torch.nn.Linear(512, num_class)
    elif classifier_mode == 'mlp':
        model.fc = MLP(512, 1024, num_class)
    elif classifier_mode == 'mlp_replace_last':
        prev_fc.fc2 = torch.nn.Linear(prev_fc.hidden_size, num_class)
        model.fc = prev_fc
    
    if extractor_mode in ['scratch', 'finetune_pt', 'finetune_prev']:
        for p in model.parameters():
            p.requires_grad = True
    elif extractor_mode in ['freeze_pt', 'freeze_prev']:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True
    else:
        raise NotImplementedError()

    return model

def make_optimizer(network, optim, lr, weight_decay=1e-5, momentum=0.9):
    if optim == 'sgd':
        optimizer = torch.optim.SGD(
            list(filter(lambda x: x.requires_grad, network.parameters())),
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum
        )
    else:
        raise NotImplementedError()
    return optimizer

def make_scheduler(optimizer, step_size=50, gamma=0.1):
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma
    )
    return scheduler
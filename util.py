import pickle
import os
import torch

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print(path + " already exists.")

def save_obj_as_pickle(pickle_location, obj):
    pickle.dump(obj, open(pickle_location, 'wb+'), protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Save object as a pickle at {pickle_location}")

def load_pickle(pickle_location, default_obj=None):
    if os.path.exists(pickle_location):
        return pickle.load(open(pickle_location, 'rb'))
    else:
        return default_obj

def save_as_json(json_location, obj):
    with open(json_location, "w+") as f:
        json.dump(obj, f)

def load_json(json_location, default_obj=None):
    if os.path.exists(json_location):
        try:
            with open(json_location, 'r') as f:
                # import pdb; pdb.set_trace()
                obj = json.load(f)
            return obj
        except:
            print(f"Error loading {json_location}")
            return default_obj
    else:
        return default_obj

def make_optimizer(network, lr, weight_decay=1e-5, momentum=0.9):
    optimizer = torch.optim.SGD(
        list(filter(lambda x: x.requires_grad, network.parameters())),
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum
    )
    return optimizer

def make_scheduler(optimizer, step_size=50, gamma=0.1):
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma
    )
    return scheduler
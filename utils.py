import os
import yaml
import torch
import shutil
from datetime import datetime
from collections import OrderedDict

def save_to_yaml(args, filename='config.yaml'):
    file_path = os.path.join(args.save_dir, filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        yaml.dump(vars(args), file, default_flow_style=False)
        
def save_checkpoint(args, state, is_best, filename):
    model_path = os.path.join(args.save_dir, filename)
    torch.save(state, model_path)
    if is_best:
        shutil.copyfile(model_path, os.path.join(args.save_dir, "best_model.pth"))

def resume_model(resume_path, model, optimizer=None, strict=False):
    checkpoint = torch.load(resume_path, map_location='cuda')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        # The pre-trained models that we provide in the README do not have 'state_dict' in the keys as
        # the checkpoint is directly the state dict
        state_dict = checkpoint
    # if the model contains the prefix "module" which is appendend by
    # DataParallel, remove it to avoid errors when loading dict
    if list(state_dict.keys())[0].startswith('module'):
        state_dict = OrderedDict({k.replace('module.', ''): v for (k, v) in state_dict.items()})
    model.load_state_dict(state_dict)
    return model

def resume_train(args, model, optimizer=None, strict=False):
    """Load model, optimizer, and other training parameters"""
    checkpoint = torch.load(args.resume)
    start_epoch_num = checkpoint["epoch_num"]
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    best_r5 = checkpoint["best_r5"]
    not_improved_num = checkpoint["not_improved_num"]
    if args.resume.endswith("last_model.pth"):  # Copy best model to current save_dir
        shutil.copy(args.resume.replace("last_model.pth", "best_model.pth"), args.save_dir)
    return model, optimizer, best_r5, start_epoch_num, not_improved_num

def save_files(path):
    import subprocess
    models_dir = os.path.join(path, "save_codes")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    file_paths = ['Datasets','scripts','commons.py',
                  'datasets_ws.py', 'datasets_dual.py',
                  'eval.py', 'loss.py',' inference.py',' parser.py',
                  'train.py',' util.py', 'network.py']

    for file_path in file_paths:
        if os.path.exists(file_path):
            if not os.path.exists(os.path.join(models_dir, file_path)):
                subprocess.run(['cp', '-r', file_path, models_dir])
                
cached_timestamp = None
def get_timestamp():
    global cached_timestamp
    if cached_timestamp is None:
        cached_timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    return cached_timestamp

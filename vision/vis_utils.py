import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import pandas as pd
import sys
import datetime
import json
import yaml
import subprocess
import copy
import numpy as np


def is_jsonable(x):
    try:
        json.dumps(x, ensure_ascii=False, indent=4)
        return True
    except (TypeError, OverflowError):
        pass
    return False

def make_jsonable(x):
    if is_jsonable(x): return x
    if type(x)==dict:
        for k in list(x.keys()):
            newk = make_jsonable(k)
            x[newk] = make_jsonable(x[k])
            if newk!=k or type(newk)!=type(k):
                print("K:", k, x[k])
                del x[k]
    elif type(x)==str:
        return x
    elif hasattr(x, "__len__"):
        x = [make_jsonable(xx) for xx in x]
    elif hasattr(x,"__name__"):
        x = x.__name__
    else:
        try:
            x = str(x)
        except:
            print("Removing", x, "from json")
            x = ""
    return x

def save_json(data, file_name):
    """
    saves a dict to a json file

    data: dict
    file_name: str
        the path that you would like to save to
    """
    jdata = make_jsonable(copy.deepcopy(data))
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(jdata, f, ensure_ascii=False, indent=4)

def save_yaml(data, file_name):
    data = make_jsonable(copy.deepcopy(data))
    with open(file_name, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

def get_git_revision_hash():
    """
    Finds the current git hash
    """
    return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).decode('ascii').strip()


def read_yaml(file_name):
    try:
        with open(file_name) as readfile:
            config = yaml.safe_load(readfile)
    except:
        with open(file_name, "r") as f:
            lines = []
            for line in f.readlines():
                if "tuple" not in line and line.strip()!="":
                    lines.append(line)
        with open(file_name, "w") as f:
            f.write("".join(lines))
        with open(file_name) as readfile:
            config = yaml.safe_load(readfile)
    return config

def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def get_layer_name_from_model_name(model_name):
    if "resnet" in model_name:
        layer_name = "backbone.encoder.stages.1.layers.0"
    elif "vit" in model_name:
        layer_name = "backbone.encoder.layer.1"
    else:
        raise ValueError(f"Model {model_name} not supported")
    return layer_name

def get_valid_layer_names(model_names, models):
    """
    Get the valid layer names for each model.

    Args:
        model_names: list of model names
        models: list of models
    Returns:
        layer_names: list of list of layer names
    """
    layer_names = []
    for model_name, model in zip(model_names, models):
        layer_names.append([])
        if "resnet" in model_name:
            for i in range(len(model.backbone.encoder.stages)):
                for j in range(len(model.backbone.encoder.stages[i].layers)):
                    layer_names[-1].append(f"backbone.encoder.stages.{i}.layers.{j}")
        elif "vit" in model_name:
            for i in range(len(model.backbone.encoder.layer)):
                layer_names[-1].append(f"backbone.encoder.layer.{i}")
        else:
            raise ValueError(f"Model {model_name} not supported")
    return layer_names

def parse_type(val):
    """
    Determines the appropriate data type for the argued string value.

    Args:
        val: str
    Returns:
        val: any
    """
    val = str(val)
    if val.lower() in {"none", "null", "na"}:
        val = None
    elif "," in val:
        val = [parse_type(v) for v in val.split(",") if v!=""]
    elif val.lower() in {"true", "t"}:
        val = True
    elif val.lower() in {"false", "f"}:
        val = False
    elif val.isnumeric():
        val = int(val)
    elif val.replace(".", "").isnumeric():
        val = float(val)
    return val

def read_command_line_args(args=None):
    if args is None: args = sys.argv[1:]
    model_folders = []
    command_args = []
    command_kwargs = dict()

    for arg in args:
        if "checkpt" in arg and ".pt" in arg:
            model_folders.append(arg)
        elif ".yaml" in arg or ".json" in arg:
            command_kwargs = {**command_kwargs,
                              **io.load_json_or_yaml(arg)}
        elif "=" in arg:
            key,val = arg.split("=")
            command_kwargs[key] = parse_type(val)
        else:
            command_args.append(arg)
    return model_folders, command_args, command_kwargs

def mtx_cor(
        X, Y,
        batch_size=500,
        to_numpy=False,
        zscore=False,
        scale=False,
        device=None,
        verbose=True,
):
    """
    Creates a correlation matrix for X and Y using the GPU

    X: torch tensor or ndarray (N, C) or (N, C, H, W)
    Y: torch tensor or ndarray (N, K) or (N, K, H1, W1)
    batch_size: int
        batches the calculation if this is not None
    to_numpy: bool
        if true, returns matrix as ndarray
    zscore: bool
        if true, both X and Y are normalized over the T dimension
    scale: bool
        if true, the correlation matrix is scaled by the number of samples
    device: int
        optionally argue a device to use for the matrix multiplications
    verbose: bool
        if true, will print a progress bar
    Returns:
        cor_mtx: (C,K) or (C*H*W, K*H1*W1)
            the correlation matrix
    """
    if len(X.shape) < 2:
        X = X[:,None]
    if len(Y.shape) < 2:
        Y = Y[:,None]
    if len(X.shape) > 2:
        X = X.reshape(len(X), -1)
    if len(Y.shape) > 2:
        Y = Y.reshape(len(Y), -1)
    if type(X) == type(np.array([])):
        to_numpy = True
        X = torch.FloatTensor(X)
        Y = torch.FloatTensor(Y)
    if device is None:
        device = X.get_device()
        if device<0: device = "cpu"
    if zscore:
        xmean = X.mean(0)
        xstd = torch.sqrt(((X-xmean)**2).mean(0))
        ymean = Y.mean(0)
        ystd = torch.sqrt(((Y-ymean)**2).mean(0))
        xstd[xstd<=0] = 1
        X = (X-xmean)/(xstd+1e-5)
        ystd[ystd<=0] = 1
        Y = (Y-ymean)/(ystd+1e-5)

    with torch.no_grad():
        if batch_size is None:
            X = X.to(device)
            Y = Y.to(device)
            cor_mtx = torch.einsum("ti,tj->ij", X, Y).detach().cpu()
        else:
            cor_mtx = torch.zeros(X.shape[1], Y.shape[1])
            if verbose:
                pbar = tqdm(range(0,len(X),batch_size), desc="Computing correlation matrix")
            else:
                pbar = range(0,len(X),batch_size)
            for i in pbar: # loop over x neurons
                x = X[i:i+batch_size].to(device)
                y = Y[i:i+batch_size].to(device)
                mtx = torch.einsum("ti,tj->ij", x, y).detach().cpu()
                cor_mtx += mtx
    if scale:
        cor_mtx = cor_mtx/len(Y)
    if to_numpy:
        return cor_mtx.numpy()
    return cor_mtx

def mtx_pinv(X, batch_size=500, to_numpy=False, to_cpu=False, device=None, verbose=True):
    """
    Computes the pseudoinverse of a matrix using the GPU
    X: torch tensor (N, D)
    batch_size: int
        batches the calculation if this is not None
    to_numpy: bool
        if true, returns matrix as ndarray
    to_cpu: bool
        if true, returns matrix on the cpu
    device: int
        optionally argue a device to use for the matrix multiplications
    verbose: bool
        if true, will print a progress bar

    Returns:
        pinv: (D,N)
            the pseudoinverse of the matrix
    """
    if batch_size is not None:
        C = mtx_cor(
            X, X,
            zscore=False,
            scale=False,
            to_numpy=False,
            device=device,
            batch_size=batch_size,
            verbose=verbose,
        )
        mms = []
        pinv = torch.linalg.pinv(C)
        device = device_fxn(pinv.get_device())
        for i in range(0,len(X),batch_size):
            mtx = pinv @ X[i:i+batch_size].T.to(device)
            mms.append(mtx)
        pinv = torch.cat(mms, dim=1)
    else:
        pinv = torch.linalg.pinv(X)
    if to_numpy:
        return pinv.cpu().numpy()
    if to_cpu:
        return pinv.cpu()
    return pinv

def device_fxn(device):
    if device<0: return "cpu"
    return device

def get_newest_model_save_path(model_save_path):
    """
    Gets the newest model save path from the model save path.
    """
    while os.path.exists(model_save_path):
        model_save_path = model_save_path.replace(".pt", "1.pt")
    return model_save_path

def replace_module(root, predicate, factory):
    for name, module in list(root.named_children()):
        if predicate(module):
            setattr(root, name, factory(module))
        else:
            replace_module(module, predicate, factory)

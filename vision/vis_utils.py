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

def save_json(data, file_name):
    """
    saves a dict to a json file

    data: dict
    file_name: str
        the path that you would like to save to
    """
    failure = True
    n_loops = 0
    while failure and n_loops<10*len(data):
        failure = False
        jdata = make_jsonable(copy.deepcopy(data))
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(jdata, f, ensure_ascii=False, indent=4)

def save_yaml(data, file_name):
    with open(file_name, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

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


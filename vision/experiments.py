
"""
Experiment 1: Compare the performance of the MAS alignment on the CIFAR-10 dataset
    for different models and layers
"""

import os
import sys
import numpy as np
from compare_vision_models import default_config, compare_models
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

exp_config = {
    "overwrite": False,
    "pretrained": True,
    "finetune_full_model": True,
    "make_figs": True,
    "model_save_dir": "/data2/grantsrb/vision_mas/models",
    "data_root": "/data2/grantsrb/pytorch_datasets",
}

def experiment_1():
    config = {
        "model_names": [
            "microsoft/resnet-18",
            "microsoft/resnet-18",
        ],
        "dataset_name": "cifar10",
    }
    config = {**default_config, **exp_config, **config}
    layer_names = [
            [
                "backbone.encoder.stages.0.layers.0",
                "backbone.encoder.stages.1.layers.0",
                "backbone.encoder.stages.2.layers.0",
                "backbone.encoder.stages.3.layers.0",
            ],
            [
                "backbone.encoder.stages.0.layers.0",
                "backbone.encoder.stages.1.layers.0",
                "backbone.encoder.stages.2.layers.0",
                "backbone.encoder.stages.3.layers.0",
            ],
        ]
    for l1 in range(len(layer_names[0])):
        for l2 in range(l1,len(layer_names[1])):
            config["layer_names"] = [layer_names[0][l1], layer_names[1][l2]]
            compare_models(config)

def experiment_2():
    config = {
        "model_names": [
            "google/vit-base-patch16-224",
            "google/vit-base-patch16-224",
        ],
        "dataset_name": "cifar10",
    }
    config = {**default_config, **exp_config, **config}
    layer_names = [
            [
                "backbone.encoder.layer.0",
                "backbone.encoder.layer.4",
                "backbone.encoder.layer.8",
                "backbone.encoder.layer.11",
            ],
            [
                "backbone.encoder.layer.0",
                "backbone.encoder.layer.4",
                "backbone.encoder.layer.8",
                "backbone.encoder.layer.11",
            ],
        ]
    for l1 in range(len(layer_names[0])):
        for l2 in range(l1,len(layer_names[1])):
            config["layer_names"] = [layer_names[0][l1], layer_names[1][l2]]
            compare_models(config)

def experiment_3():
    config = {
        "model_names": [
            "microsoft/resnet-18",
            "google/vit-base-patch16-224",
        ],
        "dataset_name": "cifar10",
    }
    config = {**default_config, **exp_config, **config}
    layer_names = [
        [
            "backbone.encoder.stages.0.layers.0",
            "backbone.encoder.stages.1.layers.0",
            "backbone.encoder.stages.2.layers.0",
            "backbone.encoder.stages.3.layers.0",
        ],
        [
            "backbone.encoder.layer.0",
            "backbone.encoder.layer.4",
            "backbone.encoder.layer.8",
            "backbone.encoder.layer.11",
        ],
    ]
    for l1 in range(len(layer_names[0])):
        for l2 in range(l1,len(layer_names[1])):
            config["layer_names"] = [layer_names[0][l1], layer_names[1][l2]]
            compare_models(config)

if __name__ == "__main__":
    experiment_1()
    experiment_2()
    experiment_3()

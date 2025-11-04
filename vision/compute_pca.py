import os
import time
import gc

from vis_training import (
    get_model_and_processor, get_dataloaders,
    get_actvs_data, get_dataloader,
    get_datasets,
)
import torch
import torch.nn as nn
import torch.optim as optim
from alignment import MASAlignment, ModelStitch, load_alignment, LowRankTransformation
from hooks import hook_vision_model
import numpy as np
import pandas as pd
from vis_utils import (
    get_valid_layer_names, read_command_line_args,
    get_layer_name_from_model_name, get_timestamp,
    save_yaml, get_git_revision_hash, get_newest_model_save_path,
)
from vis_similarity import perform_pca

def train_linear_classifier(X_train, y_train, X_val, y_val, n_components, device, max_epochs=100):
    """
    Train a linear classifier on PCA-reduced features.
    
    Args:
        X_train: tensor (N_train, n_components) - PCA-projected training features
        y_train: tensor (N_train,) - training labels
        X_val: tensor (N_val, n_components) - PCA-projected validation features
        y_val: tensor (N_val,) - validation labels
        n_components: int - number of PCA components
        device: device to use
        max_epochs: int - maximum training epochs
        
    Returns:
        accuracy: float - validation accuracy
    """
    num_classes = len(torch.unique(y_train))
    classifier = nn.Linear(n_components, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
    
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    
    # Train the classifier
    classifier.train()
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        logits = classifier(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
    
    # Evaluate on validation set
    classifier.eval()
    with torch.no_grad():
        logits = classifier(X_val)
        preds = logits.argmax(dim=-1)
        accuracy = (preds == y_val).float().mean().item()
    
    return accuracy

def compute_pca_analysis(config):
    config["git_hash"] = get_git_revision_hash()
    config["datetime"] = get_timestamp()
    for k in sorted(config.keys()):
        print(f"{k} ({type(config[k]).__name__}): {config[k]}")
    print()

    exp_name = config.get("exp_name", "")
    if exp_name and exp_name != "":
        exp_name = f"{exp_name}_"
    model_names = config["model_names"]
    n_models = len(model_names)
    batch_size = config["batch_size"]
    val_batch_size = config["val_batch_size"]
    num_workers = config["num_workers"]
    num_epochs = config["num_epochs"]
    lr = config["train_lr"]
    overwrite = config["overwrite"]
    new_models = config.get("new_models", False)
    model_save_dir = config["model_save_dir"]

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir, exist_ok=True)

    models = []
    processors = []
    
    for model_name in model_names:
        model, proc = get_model_and_processor(
            model_name,
            pretrained=config["pretrained"],
            image_resize=config.get("image_resize", None),
        )
        models.append(model)
        processors.append(proc)

        if config["finetune_full_model"]:
            for p in model.parameters():
                p.requires_grad = True
    
    ####################################################
    #    Load the datasets
    ####################################################
    print("Loading datasets...")
    
    finetune_dataset_name = config["dataset_name"]
    if config.get("debug", False):
        config["n_train_samples"] = 100
        config["n_valid_samples"] = 100
    train_ds, test_ds = get_datasets(
        dataset_name=finetune_dataset_name,
        data_root=config["data_root"],
        n_train_samples=config["n_train_samples"],
        n_valid_samples=config["n_valid_samples"],
    )
    
    train_loaders = []
    test_loaders = []
    for i, (model, processor) in enumerate(zip(models, processors)):
        train_loader, test_loader = get_dataloaders(
            train_dataset=train_ds,
            test_dataset=test_ds,
            processor=processor,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            num_workers=num_workers,
            augment=True, # only use augmentation for the training set
        )
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
    
    
    ####################################################
    #    Load the models and finetune
    ####################################################
    seed = config.get("seed", None)
    if seed is not None: seed_str = f"_seed{seed}"
    else: seed_str = ""
    for i, (model, processor) in enumerate(zip(models, processors)):
        train_loader = train_loaders[i]
        test_loader = test_loaders[i]
        model_name = model_names[i].split("/")[-1]
        full_finetune = config["finetune_full_model"]
        model_save_path = f"{exp_name}{model_name}_{finetune_dataset_name}_finetune{full_finetune}{seed_str}_epochs{num_epochs}_sd_{i}.pt"
        model_save_path = os.path.join(model_save_dir, model_save_path)
        if not config["pretrained"]:
            model_save_path = model_save_path.replace(".pt", "_unpretrained.pt")
        if config.get("image_resize", None) is not None and config.get("image_resize", None) > 0:
            size = config["image_resize"]
            model_save_path = model_save_path.replace(".pt", f"_resize{size}.pt")
        if new_models and os.path.exists(model_save_path):
            model_save_path = get_newest_model_save_path(model_save_path)
        print(f"Model save path: {model_save_path}")
        config[f"model_save_path_{i}"] = model_save_path
        if os.path.exists(model_save_path) and not overwrite and not new_models:
            print(f"Loading model from {model_save_path}")
            model.load_state_dict(torch.load(model_save_path))
        else:
            print(f"Finetuning model {i} to {model_save_path}")
            try:
                from vis_training import train_model
                model, metrics = train_model(
                    model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    hyperparameters={
                        "train_lr": lr,
                        "num_epochs": num_epochs,
                        "early_stopping": True,
                        "label_smoothing": config.get("og_train_label_smoothing", 0.1),
                        "weight_decay": config.get("og_train_weight_decay", 0),
                    },
                )
                f = model_save_path.replace("pt", "metrics.csv")
                metrics = pd.DataFrame({k: [v] for k, v in metrics.items()})
                metrics.to_csv(f, index=False, header=True)
                save_yaml(config, f.replace("metrics.csv", "config.yaml"))
                print(f"Saved metrics to {f}")
            except KeyboardInterrupt:
                print("Interrupted training, continuing...")
                pass

            if not config.get("debug", False):
                torch.save(model.state_dict(), model_save_path)
            print(f"Saved model to {model_save_path}")
    
        # From here on, we will not update the model parameters
        for p in model.parameters():
            p.requires_grad = False
        
    file_save_dir = "_".join(model_save_path.split("_sd_")).split(".")[0]
    if not os.path.exists(file_save_dir):
        os.makedirs(file_save_dir, exist_ok=True)
    config["file_save_dir"] = file_save_dir
    
    ####################################################
    #    Collect model intermediates and outputs
    ####################################################
    print("Collecting model intermediates and outputs...")
    
    layer_names = config["layer_names"]
    
    n_models = len(models)
    assert n_models == len(layer_names)
    
    device = 0 if torch.cuda.is_available() else "cpu"
    actvs_train_sets = []
    actvs_valid_sets = []
    dataset_name = config["dataset_name"]
    use_train_for_classifier = config.get("use_train_for_classifier", True)
    z = zip(
        config["model_names"],
        config["layer_names"],
        models,
        processors,
    )
    for mi, (model_name, layer_name, model, processor) in enumerate(z):
        print(f"Processing {model_name}, layer {layer_name}")
        lname = layer_name.replace("backbone.", "").replace(".", "-")
        mname = model_name.split("/")[-1]
        dname = dataset_name.split("/")[-1]
        debug = config.get("debug", False)*"_debug"
        actvs_valid_name = f"{file_save_dir}/{exp_name}{mname}_{dname}_{lname}{seed_str}_m{mi}_actvs_valid{debug}.pt"
        actvs_train_name = f"{file_save_dir}/{exp_name}{mname}_{dname}_{lname}{seed_str}_m{mi}_actvs_train{debug}.pt"
        
        actvs_valid = None
        actvs_train = None
        
        if os.path.exists(actvs_valid_name) and not overwrite and not config["fresh_actvs"]:
            print(f"Loading validation actvs sets from disk...")
            actvs_valid = torch.load(actvs_valid_name)
        else:
            print(f"Collecting validation actvs for model {model_name}, layer {layer_name}, index {mi}")
            model.eval()
            model.to(device)
            print(f"Collecting validation set...")
            with torch.no_grad():
                actvs_valid = get_actvs_data(
                    model=model,
                    data_loader=test_loaders[mi],
                    layer_name=layer_name,
                    verbose=True,
                )
            model.cpu()
            if config.get("save_actvs", False) or (os.path.exists(actvs_valid_name) and overwrite):
                torch.save(actvs_valid, actvs_valid_name)
        
        if use_train_for_classifier:
            if os.path.exists(actvs_train_name) and not overwrite and not config["fresh_actvs"]:
                print(f"Loading training actvs sets from disk...")
                actvs_train = torch.load(actvs_train_name)
            else:
                print(f"Collecting training actvs for model {model_name}, layer {layer_name}, index {mi}")
                model.eval()
                model.to(device)
                train_loader = get_dataloader(
                    dataset=train_ds,
                    processor=processor,
                    batch_size=config.get("actvs_batch_size", 1000),
                    num_workers=num_workers,
                    shuffle=False,
                    augment=False,
                )
                print(f"Collecting training set...")
                with torch.no_grad():
                    actvs_train = get_actvs_data(
                        model=model,
                        data_loader=train_loader,
                        layer_name=layer_name,
                        verbose=True,
                    )
                model.cpu()
                if config.get("save_actvs", False) or (os.path.exists(actvs_train_name) and overwrite):
                    torch.save(actvs_train, actvs_train_name)
        
        actvs_valid_sets.append(actvs_valid)
        if use_train_for_classifier:
            actvs_train_sets.append(actvs_train)
        else:
            actvs_train_sets.append(None)

    ####################################################
    #    Compute PCA and analyze
    ####################################################
    print("Computing PCA and analyzing...")
    
    all_results = []
    
    for mi, (model_name, layer_name) in enumerate(zip(
        config["model_names"],
        config["layer_names"],
    )):
        print(f"\nComputing PCA for model {mi} ({model_name})")
        
        if actvs_train_sets[mi] is not None:
            actvs = actvs_train_sets[mi]["actvs"] # (B, D, H, W) or (B, S, D)
            labels = actvs_train_sets[mi]["logits"].argmax(dim=-1)  # (B,)
            if len(actvs.shape)==4:
                actvs = actvs.permute(0, 2, 3, 1)
                actvs = actvs.reshape(-1, actvs.shape[-1])
            else:
                actvs = actvs.reshape(-1, actvs.shape[-1])
            labels = torch.repeat_interleave(labels, actvs.shape[0]//labels.shape[0], dim=0)
        else:
            actvs = actvs_valid_sets[mi]["actvs"] # (B, D, H, W) or (B, S, D)
            labels = actvs_valid_sets[mi]["logits"].argmax(dim=-1)  # (B,)
            if len(actvs.shape)==4:
                actvs = actvs.permute(0, 2, 3, 1)
                actvs = actvs.reshape(-1, actvs.shape[-1])
            else:
                actvs = actvs.reshape(-1, actvs.shape[-1])
            labels = torch.repeat_interleave(labels, actvs.shape[0]//labels.shape[0], dim=0)
        
        # Compute PCA on validation set
        print(f"Computing PCA on {actvs.shape[0]} samples with {actvs.shape[1]} features...")
        pca_result = perform_pca(
            X=actvs,
            n_components=None,  # Use all components
            scale=True,
            center=True,
            transform_data=False,
            use_eigen=True,
            batch_size=config.get("pca_batch_size", 1000),
            verbose=True,
        )
        
        components = pca_result["components"]  # (n_components, D)
        explained_variance = pca_result["explained_variance"]  # (n_components,)
        proportion_expl_var = pca_result["proportion_expl_var"]  # (n_components,)
        means = pca_result["means"]  # (1, D)
        stds = pca_result["stds"]  # (1, D)
        
        n_components = len(explained_variance)
        print(f"Computed PCA with {n_components} components")
        
        # Normalize and center the data using PCA preprocessing
        actvs_centered = (actvs - means.squeeze(0)) / (stds.squeeze(0) + 1e-6)
        
        # Get training and validation data for classifier
        if use_train_for_classifier and actvs_train_sets[mi] is not None:
            # Use training data for classifier training, validation for evaluation
            actvs_train_full = actvs_train_sets[mi]["actvs"]
            labels_train_full = actvs_train_sets[mi]["logits"].argmax(dim=-1)
            if len(actvs_train_full.shape)==4:
                actvs_train_full = actvs_train_full.permute(0, 2, 3, 1)
                actvs_train_full = actvs_train_full.reshape(-1, actvs_train_full.shape[-1])
            else:
                actvs_train_full = actvs_train_full.reshape(-1, actvs_train_full.shape[-1])
            labels_train_full = torch.repeat_interleave(
                labels_train_full,
                actvs_train_full.shape[0]//labels_train_full.shape[0],
                dim=0
            )
            # Normalize training data using same means/stds from validation PCA
            actvs_train_centered = (actvs_train_full - means.squeeze(0)) / (stds.squeeze(0) + 1e-6)
            
            X_train = actvs_train_centered
            y_train = labels_train_full
            X_val = actvs_centered
            y_val = labels
        else:
            # Split validation set into train/val for classifier
            n_train = int(len(actvs) * 0.8)
            perm = torch.randperm(len(actvs))
            train_idx = perm[:n_train]
            val_idx = perm[n_train:]
            
            X_train = actvs_centered[train_idx]
            y_train = labels[train_idx]
            X_val = actvs_centered[val_idx]
            y_val = labels[val_idx]
        
        # Compute cumulative proportion
        cumulative_proportion = torch.cumsum(proportion_expl_var, dim=0)
        print(f"Cumulative proportion: {cumulative_proportion}")
        ncomps = torch.argmax(cumulative_proportion >= 1.0)
        print(f"Num Components to 100%: {ncomps} ({ncomps/n_components*100:.2f}%)")
        
        # For each number of components, compute accuracy
        print("Computing accuracy for different numbers of components...")
        max_components_to_test = config.get("max_pca_components", n_components)
        if max_components_to_test is None:
            max_components_to_test = n_components
        max_components_to_test = min(max_components_to_test, n_components)
        
        # Test at regular intervals to avoid testing every single component
        step_size = max(1, max_components_to_test // config.get("n_pca_steps", 50))
        component_counts = list(range(step_size, max_components_to_test + 1, step_size))
        if max_components_to_test not in component_counts:
            component_counts.append(max_components_to_test)
        component_counts = sorted(set(component_counts))
        
        for n_comp in component_counts:
            print(f"  Testing with {n_comp} components...")
            
            # Project to n_comp dimensions
            X_train_proj = X_train @ components[:n_comp].T  # (N_train, n_comp)
            X_val_proj = X_val @ components[:n_comp].T  # (N_val, n_comp)
            
            # Train linear classifier
            accuracy = train_linear_classifier(
                X_train_proj, y_train, X_val_proj, y_val,
                n_comp, device, max_epochs=config.get("classifier_epochs", 100)
            )
            
            # Store results
            all_results.append({
                "model_idx": mi,
                "singular_index": n_comp - 1,  # 0-indexed
                "explained_variance": explained_variance[n_comp - 1].item(),
                "proportion_explained": proportion_expl_var[n_comp - 1].item(),
                "cumulative_proportion_explained": cumulative_proportion[n_comp - 1].item(),
                "model_accuracy": accuracy,
            })
        
        # Also add results for all components
        print(f"  Testing with all {n_components} components...")
        X_train_proj = X_train @ components.T
        X_val_proj = X_val @ components.T
        accuracy = train_linear_classifier(
            X_train_proj, y_train, X_val_proj, y_val,
            n_components, device, max_epochs=config.get("classifier_epochs", 100)
        )
        all_results.append({
            "model_idx": mi,
            "singular_index": n_components - 1,
            "explained_variance": explained_variance[-1].item(),
            "proportion_explained": proportion_expl_var[-1].item(),
            "cumulative_proportion_explained": cumulative_proportion[-1].item(),
            "model_accuracy": accuracy,
        })
    
    ####################################################
    #    Save results
    ####################################################
    df = pd.DataFrame(all_results)
    
    timestamp = get_timestamp()
    hash_str = str(hash("".join([f"{k}={v}" for k,v in config.items()])))[-4:]
    m1 = model_names[0].replace("/", "_")
    m1 = m1+layer_names[0].replace("backbone", "").replace(".", "-")
    csv_name = f"pca_analysis_{m1}_{dataset_name}_{timestamp}_h{hash_str}.csv"
    csv_path = os.path.join(file_save_dir, csv_name)
    
    if not config.get("debug", False):
        df.to_csv(csv_path, index=False, header=True)
        save_yaml(config, csv_path.replace(".csv", ".yaml"))
        print(f"Saved results to {csv_path}")
    
    print("\nResults summary:")
    print(df)
    
    return df

default_config = {
    "overwrite": False,
    "new_models": False,
    "fresh_actvs": False,
    "pretrained": True,
    "image_resize": 64,
    "finetune_full_model": True,
    "make_figs": False,
    "debug": False,
    
    "dataset_name": "cifar10",
    "model_names": [
        "microsoft/resnet-18",
        "microsoft/resnet-18",
    ],
    "layer_names": [
        None,
        None,
    ],
    
    # Finetuning parameters
    "batch_size": 128,
    "val_batch_size": 1000,
    "num_workers": 1,
    "num_epochs": 25,
    "n_train_samples": 50000,
    "n_valid_samples": 10000,
    "train_lr": 0.005,
    "model_save_dir": "/data2/grantsrb/vision_mas/models",
    "data_root": "/data2/grantsrb/pytorch_datasets",
    "actvs_batch_size": 2056,
    "og_train_weight_decay": 0,
    "og_train_label_smoothing": 0.1,
    
    # PCA analysis parameters
    "pca_batch_size": 1000,
    "max_pca_components": None,  # None means use all components
    "n_pca_steps": 50,  # Number of component counts to test
    "classifier_epochs": 100,  # Number of epochs to train linear classifier
    "use_train_for_classifier": True,  # If True, use training data for classifier, validation for eval
    "save_actvs": False,
}

def prepare_config(config):
    # Similar to compare_vision_models.py
    if config.get("single_model", False):
        config["model_mode"] = "single_model"
    return config

if __name__ == "__main__":
    _,_,kwargs = read_command_line_args()
    for k in kwargs:
        if k not in default_config:
            print(f"WARNING: {k} is not in the default configuration parameters")
    config = {**default_config, **kwargs}
    config = prepare_config(config)
    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    layer_names = config.get("layer_names", None)
    if type(layer_names)==str:
        layer_names = [layer_names for _ in config["model_names"]]

    if layer_names is None:
        layer_names = [None for _ in config["model_names"]]
    
    # Get default layer names if not provided
    final_layer_names = []
    for i, (model_name, layer_name) in enumerate(zip(config["model_names"], layer_names)):
        if layer_name is None:
            from vis_utils import get_layer_name_from_model_name
            layer_name = get_layer_name_from_model_name(model_name)
        final_layer_names.append(layer_name)
    config["layer_names"] = final_layer_names
    
    compute_pca_analysis(config)


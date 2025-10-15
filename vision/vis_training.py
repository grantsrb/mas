import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch_modules import BackboneWithLinearHead, ViTWithLinearHead, IdentityModule
from transformers import AutoImageProcessor, AutoModel  # or AutoModelForImageClassification
from torchvision import datasets
from torch.utils.data import DataLoader
import pandas as pd
from hooks import hook_model_layer
from vis_similarity import get_cka

def device_fxn(device):
    if type(device)==int and device<0:
        return "cpu"
    return device

def get_model_and_processor(model_name, pretrained=True):
    """
    Get a model and processor for a given model name.

    Common model names:
    - "microsoft/resnet-50"
    - "microsoft/resnet-18"
    - "google/vit-base-patch16-224"
    - "google/vit-small-patch16-224"
    - "google/vit-large-patch16-224"
    - "google/vit-huge-patch16-224"
    """
    model = AutoModel.from_pretrained(model_name)
    if not pretrained:
        for p in model.parameters():
            fan_in = p.data.shape[0]
            p.data = torch.randn_like(p.data) * (2/fan_in)**0.5
    processor = AutoImageProcessor.from_pretrained(
        model_name, use_fast=True,
    )
    if "vit" in model_name.lower():
        pooler_name = None
        for name, _ in model.named_modules():
            if name.split(".")[-1]=="pooler":
                pooler_name = name
        if pooler_name is not None:
            parent_name = ".".join(pooler_name.split(".")[:-1])
            for name, modu in model.named_modules():
                if name==parent_name:
                    parent = modu
            setattr(parent, pooler_name.split(".")[-1], IdentityModule())
        model = ViTWithLinearHead(
            model,
            hidden_dim=model.config.hidden_size,
            num_classes=10,
            freeze_backbone=True,
        )
    else:
        hidden_dim = model.config.hidden_sizes[-1]
        model = BackboneWithLinearHead(
            model,
            hidden_dim=hidden_dim,
            num_classes=10,
            freeze_backbone=True,
        )
    return model, processor

def get_datasets(dataset_name, data_root, n_train_samples=None, n_valid_samples=None):
    print(f"Loading {dataset_name}...")
    if dataset_name == "cifar10":
        train_ds = datasets.CIFAR10(root=data_root, train=True, download=True)
        valid_ds = datasets.CIFAR10(root=data_root, train=False, download=True)
    elif dataset_name == "imagenet":
        train_ds = datasets.ImageNet(root=data_root, split="train")
        valid_ds = datasets.ImageNet(root=data_root, split="val")
    elif dataset_name == "mnist":
        train_ds = datasets.MNIST(root=data_root, train=True, download=True)
        valid_ds = datasets.MNIST(root=data_root, train=False, download=True)
    elif dataset_name == "fashion-mnist":
        train_ds = datasets.FashionMNIST(root=data_root, train=True, download=True)
        valid_ds = datasets.FashionMNIST(root=data_root, train=False, download=True)
    elif dataset_name == "svhn":
        train_ds = datasets.SVHN(root=data_root, split="train", download=True)
        valid_ds = datasets.SVHN(root=data_root, split="test", download=True)
    elif dataset_name == "stl10":
        train_ds = datasets.STL10(root=data_root, split="train", download=True)
        valid_ds = datasets.STL10(root=data_root, split="test", download=True)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    if n_train_samples is not None:
        perm = np.random.permutation(len(train_ds))[:n_train_samples]  
        train_ds.data = train_ds.data[perm]
        if hasattr(train_ds, "labels"):
            train_ds.labels = train_ds.labels[perm]
        elif hasattr(train_ds, "targets"):
            train_ds.targets = np.asarray(train_ds.targets)[perm]
    if n_valid_samples is not None:
        perm = np.random.permutation(len(valid_ds))[:n_valid_samples]
        valid_ds.data = valid_ds.data[perm]
        if hasattr(valid_ds, "labels"):
            valid_ds.labels = valid_ds.labels[perm]
        elif hasattr(valid_ds, "targets"):
            valid_ds.targets = np.asarray(valid_ds.targets)[perm]
    return train_ds, valid_ds

def get_dataloader(
        processor: AutoImageProcessor,
        batch_size: int,
        num_workers: int,
        dataset: datasets.CIFAR10 = None,
        data_root: str = "./data",
        shuffle: bool = True,
):
   if dataset is None:
       # CIFAR-10 returns PIL images; we'll use the processor inside a
       # collate_fn to handle resize + normalize.
       dataset = datasets.CIFAR10(root=data_root, train=True, download=True)

   def collate_fn(batch):
       images, labels = zip(*batch)  # images are PIL.Image
       enc = processor(images=list(images), return_tensors="pt")
       pixel_values = enc["pixel_values"]  # (B, 3, 224, 224)
       labels_t = torch.tensor(labels, dtype=torch.long)
       return pixel_values, labels_t

   data_loader = DataLoader(
       dataset,
       batch_size=batch_size,
       shuffle=shuffle,
       num_workers=num_workers,
       pin_memory=True,
       collate_fn=collate_fn
   )
   return data_loader

def get_dataloaders(
        processor: AutoImageProcessor,
        batch_size: int,
        num_workers: int,
        val_batch_size: int = None,
        train_dataset: datasets.CIFAR10 = None,
        test_dataset: datasets.CIFAR10 = None,
        data_root: str = "./data",
):
    train_loader = get_dataloader(
        dataset=train_dataset,
        processor=processor,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        data_root=data_root,
    )
    if val_batch_size is None:
        val_batch_size = batch_size
    test_loader = get_dataloader(
        dataset=test_dataset,
        processor=processor,
        batch_size=val_batch_size,
        num_workers=num_workers,
        shuffle=False,
        data_root=data_root,
    )
    return train_loader, test_loader

def train_model(
        model,
        train_loader,
        test_loader,
        verbose=True,
        hyperparameters: dict = {
            "train_lr": 0.001,
            "num_epochs": 10,
            "early_stopping": False,
        },
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=hyperparameters.get("train_lr", 0.001)
    )
    model.train()
    best_test_acc = 0
    best_epoch = 0
    if hyperparameters.get("early_stopping", False):
        patience = hyperparameters.get("patience", 10)
        patience_counter = 0
    for epoch in range(hyperparameters.get("num_epochs", 10)):
        avg_train_loss = 0
        avg_train_acc = 0
        for batch_idx, (pixel_values, labels) in enumerate(train_loader):
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(pixel_values)
            loss = nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            avg_train_loss += loss.item()
            acc = (outputs.argmax(dim=1) == labels).float().mean()
            avg_train_acc += acc.item()

            if batch_idx % 100 == 0 and verbose:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}, Acc: {acc}")
        avg_train_loss = avg_train_loss / (batch_idx + 1)
        avg_train_acc = avg_train_acc / (batch_idx + 1)
        avg_test_loss = 0
        avg_test_acc = 0
        if verbose:
            print("Validating...")
        itr = tqdm(enumerate(test_loader)) if verbose else enumerate(test_loader)
        for batch_idx, (pixel_values, labels) in itr:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = model(pixel_values)
                loss = nn.functional.cross_entropy(outputs, labels)
                acc = (outputs.argmax(dim=1) == labels).float().mean()
                avg_test_acc += acc.item()
                avg_test_loss += loss.item()
        avg_test_acc = avg_test_acc / (batch_idx + 1)
        avg_test_loss = avg_test_loss / (batch_idx + 1)
        if avg_test_acc > best_test_acc-0.001:
            best_test_acc = avg_test_acc
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        if verbose:
            print(f"Epoch {epoch}, VLoss: {avg_test_loss}, VAcc: {avg_test_acc}")
        if hyperparameters["early_stopping"]:
            if patience_counter >= patience\
                    or avg_test_acc >= 0.9999 and avg_train_acc >= 0.9999:
                print(f"Early stopping at epoch {epoch}")
                break
    metrics = {
        "train_loss": avg_train_loss,
        "train_acc": avg_train_acc,
        "test_loss": avg_test_loss,
        "test_acc": avg_test_acc ,
        "best_test_acc": best_test_acc,
        "best_epoch": best_epoch,
        "final_epoch": epoch,
    }
    return model.cpu(), metrics

def get_actvs_data(
        model,
        data_loader,
        layer_name,
        verbose=True,
):
    device = next(model.parameters()).device
    handle, comms_dict = hook_model_layer(model, layer_name)

    actvs_data = dict()
    actvs_data["inputs"] = []
    actvs_data["labels"] = []
    actvs_data["actvs"] = []
    actvs_data["logits"] = []
    actvs_data["preds"] = []
    itr = tqdm(data_loader) if verbose else data_loader
    for pixel_values, labels in itr:
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)
        outputs = model(pixel_values)
        actvs_data["inputs"].append(pixel_values.cpu())
        actvs_data["labels"].append(labels.cpu())
        actvs_data["actvs"].append(comms_dict[layer_name][-1].cpu())
        actvs_data["logits"].append(outputs.cpu())
        actvs_data["preds"].append(outputs.argmax(dim=-1).cpu())
    handle.remove()
    del comms_dict
    for k in actvs_data:
        actvs_data[k] = torch.cat(actvs_data[k], dim=0)
    return actvs_data

def get_input_outputs(
        model,
        data_loader,
        verbose=True,
):
    device = next(model.parameters()).device
    model.eval()
    model.to(device)
    with torch.no_grad():
        pixel_values_list = []
        logits_list = []
        labels_list = []
        itr = tqdm(data_loader) if verbose else data_loader
        for pixel_values, labels in itr:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            outputs = model(pixel_values)
            logits = extract_logits(outputs)
            pixel_values_list.append(pixel_values.cpu())
            logits_list.append(logits.cpu())
            labels_list.append(labels.cpu())
    pixel_values_list = torch.cat(pixel_values_list, dim=0)
    logits_list = torch.cat(logits_list, dim=0)
    labels_list = torch.cat(labels_list, dim=0)
    return{
        "inputs": pixel_values_list,
        "logits": logits_list,
        "labels": labels_list,
    }

def get_cl_vectors(activations, probs, method="mean", src_probs=None):
    """
    Args:
        activations: (N,D,H,W)
        probs: (N,C)
        method: str
            "mean" - take the mean of the activations for each class
            "sample" - sample a random activation for each class
            "most_similar" - take the activations with the highest similarity
                to the source class probabilities
        src_probs: (B,C)
            the source class probabilities
    Returns:
        cl_vectors: (B,D,H,W) or (C,D,H,W)
            the counterfactual latent vectors
    """
    if method in {"mean", "sample"}:
        labels = probs.argmax(dim=-1)
        n_classes = torch.max(labels)+1
        cl_vectors = torch.empty(n_classes, *activations.shape[1:])
        if method == "mean":
            for i in range(n_classes):
                cl_vectors[i] = activations[labels==i].mean(dim=0)
        elif method == "sample":
            for i in range(n_classes):
                bools = labels==i
                vectors = activations[bools]
                sample_idx = torch.randint(0, len(vectors), (1,))
                cl_vectors[i] = vectors[sample_idx]
    elif method == "most_similar":
        ranks = torch.matmul(src_probs, probs.T).argmax(dim=-1)
        cl_vectors = activations[ranks]
    return cl_vectors

def cl_loss_fxn(intrv_vectors, cl_vectors):
    return F.mse_loss(intrv_vectors, cl_vectors)\
        - F.cosine_similarity(intrv_vectors, cl_vectors, dim=-1).mean()

def extract_logits(outputs):
    logits = outputs
    if type(logits)==dict:
        try:
            logits = logits["last_hidden_state"]
        except:
            logits = logits["hidden_states"][-1]
    elif type(logits)==tuple:
        logits = logits[0]
    elif hasattr(logits, "last_hidden_state"):
        logits = logits.last_hidden_state
    elif hasattr(logits, "hidden_states"):
        logits = logits.hidden_states[-1]
    return logits

def train_mas_alignment_one_epoch(
        models,
        alignment,
        actvs_sets,
        batch_size,
        optimizer,
        varb_idx=None,
        verbose=True,
        one_hot_loss=False,
        train_directions=None,
        batches_per_optim_step=1,
        cl_directions=None,
        cl_eps=1,
        cl_method="sample",
        label_smoothing=0.0,
        use_ground_truth_labels=False,
        debug=False,
):
    device = next(alignment.parameters()).device
    models = [model.to(device) for model in models]
    models = [model.eval() for model in models]
    alignment.train()

    df_dict = dict()
    df_dict["actn_loss"] = []
    df_dict["cl_loss"] = []
    df_dict["acc"] = []
    df_dict["trg_idx"] = []
    df_dict["src_idx"] = []
    df_dict["varb_idx"] = []
    df_dict["batch_idx"] = []

    if use_ground_truth_labels:
        label_key = "labels"
    else:
        label_key = "preds"
    alignment.comms_dict["varb_idx"] = varb_idx
    src_perm = torch.randperm(len(actvs_sets[0]["inputs"])).long()
    trg_perm = torch.randperm(len(actvs_sets[0]["inputs"])).long()
    optimizer.zero_grad()
    for batch_idx in range(0,len(actvs_sets[0]["inputs"]),batch_size):
        start_time = time.time()
        src_batch = src_perm[batch_idx:batch_idx+batch_size]
        trg_batch = trg_perm[batch_idx:batch_idx+batch_size]
        accs = dict()
        cl_losses = dict()
        actn_losses = dict()
        for src_idx in range(len(models)):
            src_data = actvs_sets[src_idx]
            src_actvs = src_data["actvs"][src_batch]
            src_preds = src_data["preds"][src_batch]
            src_logits = src_data["logits"][src_batch]
            src_labels = src_data[label_key][src_batch]
            alignment.comms_dict["src_activations"] = src_actvs
            alignment.comms_dict["src_idx"] = src_idx
            for trg_idx in range(len(models)):
                trg_data = actvs_sets[trg_idx]
                trg_inputs = trg_data["inputs"][trg_batch]
                cl_vectors = trg_data.get("cl_vectors", None) # (n_classes,C,H,W)
                alignment.comms_dict["trg_idx"] = trg_idx

                if train_directions is not None and (src_idx,trg_idx) not in train_directions:
                    with torch.no_grad():
                        outputs = models[trg_idx](trg_inputs.to(device))
                else:
                    outputs = models[trg_idx](trg_inputs.to(device))
                logits = extract_logits(outputs)
                
                if one_hot_loss:
                    if label_smoothing > 0:
                        actn_loss = nn.functional.cross_entropy(
                            logits, src_labels.to(device), label_smoothing=label_smoothing
                        ).mean()
                    else:
                        actn_loss = nn.functional.cross_entropy(
                            logits, src_labels.to(device)
                        ).mean()
                else:
                    actn_loss = nn.functional.cross_entropy(
                        logits, src_logits.to(device).softmax(dim=-1)
                    ).mean()

                cl_loss = torch.zeros(1).to(device)
                if cl_directions is not None\
                        and (src_idx,trg_idx) in cl_directions:
                    if cl_vectors is None:
                        cl_vectors = get_cl_vectors(
                            activations=trg_data["actvs"],
                            probs=trg_data["logits"].softmax(dim=-1),
                            method=cl_method,
                            src_probs=src_logits.softmax(dim=-1),
                        )[src_labels]
                    else:
                        cl_vectors = cl_vectors[src_labels] # (B,C,H,W)
                    intrv_vectors = alignment.comms_dict["intrv_vectors"].to(device)
                    cl_loss = cl_loss_fxn(intrv_vectors, cl_vectors.to(device))

                loss = (actn_loss+cl_eps*cl_loss)/len(models)**2
                if loss.requires_grad:
                    loss.backward()
                
                acc = (logits.argmax(dim=-1) == src_preds.to(device)).float().mean()
                accs[(src_idx,trg_idx)] = acc.item()
                cl_losses[(src_idx,trg_idx)] = cl_loss.item()
                actn_losses[(src_idx,trg_idx)] = actn_loss.item()

        if batch_idx//batch_size % batches_per_optim_step == 0:
            optimizer.step()
            optimizer.zero_grad()
        end_time = time.time()
        if verbose:
            n_batches = len(actvs_sets[0]['inputs'])
            print(f"Batch {batch_idx}/{n_batches}",
                "IIA:", min(accs.values()),
                "Loss:", max(actn_losses.values()),
                "Time:", end_time - start_time,
                end=" "*50+"\r"
            )

        for (src_idx,trg_idx) in sorted(list(accs.keys())):
            df_dict["actn_loss"].append(actn_losses[(src_idx,trg_idx)])
            df_dict["cl_loss"].append(cl_losses[(src_idx,trg_idx)])
            df_dict["acc"].append(accs[(src_idx,trg_idx)])
            df_dict["trg_idx"].append(trg_idx)
            df_dict["src_idx"].append(src_idx)
            df_dict["varb_idx"].append(varb_idx)
            df_dict["batch_idx"].append(batch_idx)
        if debug and batch_idx>batch_size:
            return pd.DataFrame(df_dict)
    if batch_idx//batch_size % batches_per_optim_step != 0:
        optimizer.step()
        optimizer.zero_grad()
    return pd.DataFrame(df_dict)

def evaluate_mas_alignment(
        models,
        alignment,
        actvs_sets,
        batch_size,
        varb_idx=None,
        one_hot_loss=False,
        use_ground_truth_labels=False,
        cl_directions=None,
        cl_eps=1,
        cl_method="sample",
        verbose=True,
        debug=False,
):
    device = next(alignment.parameters()).device
    models = [model.to(device) for model in models]
    models = [model.eval() for model in models]
    alignment.eval()

    if use_ground_truth_labels:
        label_key = "labels"
    else:
        label_key = "preds"

    df_dict = dict()
    df_dict["actn_loss"] = []
    df_dict["cl_loss"] = []
    df_dict["acc"] = []
    df_dict["trg_idx"] = []
    df_dict["src_idx"] = []
    df_dict["varb_idx"] = []
    df_dict["batch_idx"] = []

    src_perm = torch.arange(len(actvs_sets[0]["inputs"])).long()
    trg_perm = torch.arange(len(actvs_sets[0]["inputs"])).long()
    for batch_idx in range(0,len(actvs_sets[0]["inputs"]),batch_size):
        src_batch = src_perm[batch_idx:batch_idx+batch_size]
        trg_batch = trg_perm[batch_idx:batch_idx+batch_size]
        accs = dict()
        cl_losses = dict()
        actn_losses = dict()
        for src_idx in range(len(models)):
            src_data = actvs_sets[src_idx]
            src_actvs = src_data["actvs"][src_batch]
            src_preds = src_data["preds"][src_batch]
            src_logits = src_data["logits"][src_batch]
            src_labels = src_data[label_key][src_batch]
            alignment.comms_dict["src_activations"] = src_actvs
            alignment.comms_dict["src_idx"] = src_idx
            for trg_idx in range(len(models)):
                trg_data = actvs_sets[trg_idx]
                trg_inputs = trg_data["inputs"][trg_batch]
                cl_vectors = trg_data.get("cl_vectors", None) # (n_classes,C,H,W)
                alignment.comms_dict["trg_idx"] = trg_idx

                with torch.no_grad():
                    outputs = models[trg_idx](trg_inputs.to(device))
                logits = extract_logits(outputs)

                if one_hot_loss:
                    actn_loss = nn.functional.cross_entropy(
                        logits, src_labels.to(device)
                    ).mean()
                else:
                    actn_loss = nn.functional.cross_entropy(
                        logits, src_logits.to(device).softmax(dim=-1)
                    ).mean()

                cl_loss = torch.zeros(1).to(device)
                if cl_directions is not None\
                        and (src_idx,trg_idx) in cl_directions:
                    if cl_vectors is None:
                        cl_vectors = get_cl_vectors(
                            activations=trg_data["actvs"],
                            probs=trg_data["logits"].softmax(dim=-1),
                            method=cl_method,
                            src_probs=src_logits.softmax(dim=-1),
                        )
                        if cl_vectors.shape[0]!=len(src_labels):
                            cl_vectors = cl_vectors[src_labels]
                    else:
                        cl_vectors = cl_vectors[src_labels] # (B,C,H,W)
                    intrv_vectors = alignment.comms_dict["intrv_vectors"].to(device)
                    cl_loss = cl_loss_fxn(intrv_vectors, cl_vectors.to(device))

                loss = (actn_loss+cl_eps*cl_loss)/len(models)**2
                
                acc = (logits.argmax(dim=-1) == src_preds.to(device)).float().mean()
                accs[(src_idx,trg_idx)] = acc.item()
                cl_losses[(src_idx,trg_idx)] = cl_loss.item()
                actn_losses[(src_idx,trg_idx)] = actn_loss.item()

        if verbose:
            n_batches = len(actvs_sets[0]['inputs'])
            print(f"Batch {batch_idx}/{n_batches}",
                "IIA:", min(accs.values()),
                "Loss:", max(actn_losses.values()),
                end=" "*50+"\r"
            )

        for (src_idx,trg_idx) in sorted(list(accs.keys())):
            df_dict["actn_loss"].append(actn_losses[(src_idx,trg_idx)])
            df_dict["cl_loss"].append(cl_losses[(src_idx,trg_idx)])
            df_dict["acc"].append(accs[(src_idx,trg_idx)])
            df_dict["trg_idx"].append(trg_idx)
            df_dict["src_idx"].append(src_idx)
            df_dict["varb_idx"].append(varb_idx)
            df_dict["batch_idx"].append(batch_idx)
        if debug and batch_idx>batch_size:
            return pd.DataFrame(df_dict)
    return pd.DataFrame(df_dict)

def minimize_cka_one_epoch(
        models,
        data_sets,
        batch_size,
        comms_dicts,
        optimizer=None,
        one_hot_loss=False,
        comms_key="actvs",
        cka_eps=0.5,
        verbose=True,
):
    """
    Args:
        models: list of models
        data_sets: list of data sets
        optimizer: optimizer or None
        batch_size: batch size
        comms_dicts: list of comms dicts
        one_hot_loss: bool
        comms_key: str
        cka_eps: float in [0,1]
            the weighting of the cka loss in the total weighted loss sum
    Returns:
        accs: list of tensors of shape (B,)
            the accuracies for each model
        actn_losses: list of tensors of shape (B,)
            the losses for each model
        ckas: tensor of shape (B,)
            the cka values for each model
    """
    models = [model.train() for model in models]
    device = device_fxn(next(models[0].parameters()).device)
    if optimizer is not None:
        optimizer.zero_grad()

    accs = [[],[]]
    actn_losses = [[],[]]
    ckas = []
    perm = torch.randperm(len(data_sets[0]["inputs"])).long()
    for batch_idx in range(0,len(data_sets[0]["inputs"]),batch_size):
        batch_idx = perm[batch_idx:batch_idx+batch_size]
        intermediates = []
        loss = 0

        # Forward pass for each model to collect intermediates and to maintain
        # model behavior
        for mi, (model, data_set) in enumerate(zip(models, data_sets)):
            inputs = data_set["inputs"][batch_idx]
            labels = data_set["labels"][batch_idx]
            og_logits = data_set["logits"][batch_idx]
            outputs = model(inputs.to(device))
            logits = extract_logits(outputs)
            intermediates.append(comms_dicts[mi][comms_key][-1].cpu())
            comms_dicts[mi][comms_key] = []
            if one_hot_loss:
                actn_loss = nn.functional.cross_entropy(
                    logits, labels.to(device))
            else:
                actn_loss = nn.functional.cross_entropy(
                    logits, og_logits.softmax(dim=-1).to(device)
                )
            loss += actn_loss
            acc = (logits.argmax(dim=-1) == og_logits.argmax(dim=-1).to(device)).float().mean()
            accs[mi].append(acc)
            actn_losses[mi].append(actn_loss)

        # Calculate CKA and add to loss
        intermediates = [
            intermediate.permute(0,2,3,1).reshape(-1,intermediate.shape[1])
            for intermediate in intermediates
        ]
        cka = get_cka(
            intermediates[0], intermediates[1],
            to_cpu=True,
            verbose=False,
        )
        ckas.append(cka)
        loss = (1-cka_eps)*loss + cka_eps*cka

        if loss.requires_grad:
            loss.backward()
            if optimizer is not None:
                optimizer.step()
                optimizer.zero_grad()
        if verbose:
            print(f"Batch {batch_idx}/{len(data_sets[0]['inputs'])}",
                "CKA:", cka,
                "Loss:", loss,
                end=" "*50+"\r"
            )

    return [torch.stack(acc) for acc in accs],\
        [torch.stack(actn_loss) for actn_loss in actn_losses],\
        torch.stack(ckas)


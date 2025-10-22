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
from vis_utils import mtx_cor
from vis_similarity import get_cka
from alignment import MASAlignment

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
        for name,modu in model.named_modules():
            if "Linear" in str(type(modu)) or "Conv" in str(type(modu)) or "Embedding" in str(type(modu)):
                for pname,p in modu.named_parameters():
                    if pname in {"bias"}:
                        p.data = torch.zeros_like(p.data)
                    else:
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
    else:
        raise ValueError(f"Invalid CL method: {method}")
    return cl_vectors

def cl_loss_fxn(intrv_vectors, cl_vectors, loss_type="both"):
    mse = 0
    cos = 0
    if loss_type in {"mse", "both"}:
        mse = F.mse_loss(intrv_vectors, cl_vectors)
    if loss_type in {"cos", "both"}:
        cos = 1-F.cosine_similarity(intrv_vectors, cl_vectors, dim=-1).mean()
    return mse + cos

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
        cl_method="sample", # {"mean", "sample", "most_similar", "same_as_target"}
        cl_loss_type="both", # {"mse", "cos", "both"}
        label_smoothing=0.0,
        use_ground_truth_labels=False,
        use_trg_labels=False,
        debug=False,
):
    """
    Args:
        models: list of models
        alignment: MASAlignment object
        actvs_sets: list of actvs sets
        batch_size: batch size
        optimizer: optimizer
        varb_idx: variable index
        verbose: bool
        one_hot_loss: bool
        use_ground_truth_labels: bool
            use the ground truth labels for the training objective (as
            opposed to the model predictions)
        use_trg_labels: bool
            use the target model labels for the training objective (as
            opposed to the source model predictions). Be careful combining
            this with low dimensional subspace sizes. It is possible to
            learn a trivial/null alignment where the target model simply
            passes its representations through the alignment module.
        debug: bool
        train_directions: list of tuples
        batches_per_optim_step: int
        cl_directions: list of tuples
        cl_eps: float
        cl_method: str
        cl_loss_type: str
        label_smoothing: float

    Returns:
        df: pandas DataFrame
            the dataframe containing the loss and accuracy for each model
            and each variable index
    """
    device = next(alignment.parameters()).device
    models = [model.to(device) for model in models]
    models = [model.eval() for model in models]
    alignment.train()

    df_dict = dict()
    df_dict["actn_loss"] = []
    df_dict["cl_loss"] = []
    df_dict["acc"] = []
    df_dict["penalty"] = []
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
        penalties = dict()
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
                if use_trg_labels:
                    src_logits = trg_data["logits"][src_batch]
                    src_preds =  trg_data["preds"][src_batch]
                    src_labels = trg_data[label_key][src_batch]
                trg_inputs = trg_data["inputs"][trg_batch]
                cl_vectors = trg_data.get("cl_vectors", None) # (N,C,H,W)
                if cl_vectors is not None:
                    cl_vectors = cl_vectors[src_batch] # (B,C,H,W)
                alignment.comms_dict["trg_idx"] = trg_idx

                prev_grad_state = torch.is_grad_enabled()
                do_train_loss = train_directions is None or (src_idx,trg_idx) in train_directions
                do_cl_loss = cl_directions is not None and (src_idx,trg_idx) in cl_directions
                if not do_train_loss:
                    with torch.no_grad():
                        alignment.comms_dict["req_grad"] = do_cl_loss
                        outputs = models[trg_idx](trg_inputs.to(device))
                        alignment.comms_dict["req_grad"] = None
                    torch.set_grad_enabled(prev_grad_state)
                else:
                    alignment.comms_dict["req_grad"] = None
                    outputs = models[trg_idx](trg_inputs.to(device))
                logits = extract_logits(outputs)

                if not do_train_loss:
                    torch.set_grad_enabled(False)
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
                torch.set_grad_enabled(prev_grad_state)

                cl_loss = torch.zeros(1).to(device)
                if do_cl_loss:
                    if cl_method == "same_as_target":
                        cl_vectors = trg_data["actvs"][src_batch] # (B,C,H,W)
                    elif cl_vectors is None:
                        cl_vectors = get_cl_vectors(
                            activations=trg_data["actvs"],
                            probs=trg_data["logits"].softmax(dim=-1),
                            method=cl_method,
                            src_probs=src_logits.softmax(dim=-1),
                        )
                        cl_vectors = cl_vectors[src_labels] # (B,C,H,W)
                    intrv_vectors = alignment.comms_dict["intrv_vectors"].to(device)
                    cl_loss = cl_loss_fxn(
                        intrv_vectors, cl_vectors.to(device), loss_type=cl_loss_type
                    )

                loss = (actn_loss+cl_eps*cl_loss)/len(models)**2
                if loss.requires_grad:
                    loss.backward()
                
                acc = (logits.argmax(dim=-1) == src_labels.to(device)).float().mean().item()
                accs[(src_idx,trg_idx)] = acc
                src_acc = (src_logits.argmax(dim=-1) == src_labels).float().mean()
                penalties[(src_idx,trg_idx)] = (src_acc-acc).item()
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
                "Penalty:", min(penalties.values()),
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
            df_dict["penalty"].append(penalties[(src_idx,trg_idx)])
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
        use_trg_labels=False,
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
    df_dict["penalty"] = []
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
        penalties = dict()
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
                if use_trg_labels:
                    src_logits = trg_data["logits"][src_batch]
                    src_preds = trg_data["preds"][src_batch]
                    src_labels = trg_data[label_key][src_batch]
                cl_vectors = trg_data.get("cl_vectors", None) # (n_classes,C,H,W)
                if cl_vectors is not None:
                    cl_vectors = cl_vectors[src_batch] # (B,C,H,W)
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
                    if cl_method == "same_as_target":
                        cl_vectors = trg_data["actvs"][src_batch] # (B,C,H,W)
                    elif cl_vectors is None:
                        cl_vectors = get_cl_vectors(
                            activations=trg_data["actvs"],
                            probs=trg_data["logits"].softmax(dim=-1),
                            method=cl_method,
                            src_probs=src_logits.softmax(dim=-1),
                        )
                        cl_vectors = cl_vectors[src_labels] # (B,C,H,W)
                    intrv_vectors = alignment.comms_dict["intrv_vectors"].to(device)
                    cl_loss = cl_loss_fxn(intrv_vectors, cl_vectors.to(device))

                loss = (actn_loss+cl_eps*cl_loss)/len(models)**2
                
                acc = (logits.argmax(dim=-1) == src_labels.to(device)).float().mean().item()
                accs[(src_idx,trg_idx)] = acc
                src_acc = (src_logits.argmax(dim=-1) == src_labels).float().mean()
                penalties[(src_idx,trg_idx)] = (src_acc-acc).item()
                cl_losses[(src_idx,trg_idx)] = cl_loss.item()
                actn_losses[(src_idx,trg_idx)] = actn_loss.item()

        if verbose:
            n_batches = len(actvs_sets[0]['inputs'])
            print(f"Batch {batch_idx}/{n_batches}",
                "IIA:", min(accs.values()),
                "Penalty:", min(penalties.values()),
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
            df_dict["penalty"].append(penalties[(src_idx,trg_idx)])
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

def orthogonal_procrustes(
    X, Y, center=True, scale=False, allow_reflection=True, batch_size=1000, verbose=True
):
    """
    Solve the (weighted) orthogonal Procrustes problem:
        minimize_R,t,s  || s * (X - mu_X) R - (Y - mu_Y) ||_F^2
    subject to R^T R = I, and (optionally) det(R)=+1 if allow_reflection=False.
    
    Args:
        X : torch tensor (N, D)
            Source points (rows = samples, columns = dimensions).
        Y : torch tensor (N, D)
            Target points, paired to X by row.
        center : bool, default True
            If True, estimate and remove means before solving and return translation t.
            If False, solve for R (and optional s) with no translation term.
        scale : bool, default False
            If True, also estimate a global scalar s >= 0 that minimizes the error.
        allow_reflection : bool, default False
            If False, constrain det(R)=1 (proper rotation). If True, R may reflect.
        batch_size: int, default 1000
            Batch size for the matrix correlation calculation.

    Returns:
        R: torch tensor (D, D)
            Orthogonal matrix (rotation/reflection).
        t: torch tensor (D,)
            Translation vector such that Y ≈ s * X R + t. If center=False, t is zeros.
        s: float
            Scale (1.0 if scale=False).
        info: dict
            Diagnostics including 'residual', 'detR', 'fro_error', and 'trace_sigma'.
    """
    X = X.to(torch.float)
    Y = Y.to(torch.float)
    assert X.shape == Y.shape and X.ndim == 2, "X and Y must be (N,D) with same shape."
    N, D = X.shape

    # Means (weighted or not)
    if center:
        mu_X = X.mean(dim=0)
        mu_Y = Y.mean(dim=0)
    else:
        mu_X = torch.zeros(D)
        mu_Y = torch.zeros(D)

    # Centered copies used for solving R (and s)
    X = X - mu_X
    Y = Y - mu_Y
    X0 = X
    Y0 = Y

    # Cross-covariance (D x D): we solve min_R ||X0 R - Y0||, so C = X0^T Y0
    if batch_size is not None:
        C = mtx_cor(
            X0, Y0,
            zscore=False,
            scale=False,
            to_numpy=False,
            batch_size=batch_size,
            verbose=verbose,
        )
    else:
        C = X0.T @ Y0

    # SVD of cross-covariance
    U, S, Vt = torch.linalg.svd(C, full_matrices=False)
    # Base solution
    R = U @ Vt

    # Enforce det(R)=1 if requested (no reflection)
    detR = torch.linalg.det(R)
    if not allow_reflection and detR < 0:
        # Flip last column of U to change sign of det
        U[:, -1] *= -1
        R = U @ Vt
        detR = torch.linalg.det(R)  # should now be +1

    # Optional optimal global scale
    if scale:
        # For min || s X0 R - Y0 ||, s* = trace(S) / ||X0||_F^2 (with weights handled above)
        num = S.sum()
        den = (X0 * X0).sum()  # ||X0||_F^2 with weights
        s = (num / den) if den > 0 else 1.0
    else:
        s = 1.0

    # Diagnostics
    Y_hat = s * (X @ R)
    fro_error = torch.linalg.norm(Y_hat - Y, ord='fro')
    info = {
        "R": R,
        "singular_values": S,
        "left_matrix": U,
        "right_matrix": Vt,
        "s": float(s),
        "residual": Y_hat - Y,
        "fro_error": fro_error,
        "detR": detR,
        "trace_sigma": float(S.sum()),
        "mu_X": mu_X,
        "mu_Y": mu_Y,
    }
    return info

def solve_alignment_procrustes(
        X, Y,
        center=True,
        scale=False,
        allow_reflection=True,
        n_samples=None,
        batch_size=1000,
        verbose=True,
):
    """
    Solve the alignment analytically using orthogonal procrustes.

    Args:
        alignment: MASAlignment
        X: torch tensor (N,D) or (B,C,H,W) or (B,S,D)
            Source points (rows = samples, columns = dimensions).
        Y: torch tensor (N,D) or (B,C,H,W) or (B,S,D)
            Target points, paired to X by row.
        center: bool, default True
        n_samples: int, default None
            Number of samples to use for the alignment. If None, will use all samples.
        verbose: bool, default True
            Whether to print verbose output.
    Returns:
        alignment: MASAlignment
            The aligned alignment object.
    """
    if len(X.shape)==4:
        X = X.permute(0, 2, 3, 1)
    if len(Y.shape)==4:
        Y = Y.permute(0, 2, 3, 1)
    D = X.shape[-1]
    X = X.reshape(-1,D)
    Y = Y.reshape(-1,D)
    if n_samples is not None:
        perm = torch.randperm(len(X))[:n_samples].long()
        X = X[perm]
        Y = Y[perm]

    alignment = MASAlignment(
        model_dims=[D, D],
        mtx_type="linear",
        dtype=X.dtype,
    )

    soln = orthogonal_procrustes(
        X, Y,
        center=center,
        scale=scale,
        allow_reflection=allow_reflection,
        batch_size=batch_size,
        verbose=verbose,
    )
    if verbose:
        print(f"Solved alignment using orthogonal procrustes")
    alignment.rot_mtxs[0].weight.data = soln["left_matrix"]*soln["s"]
    alignment.rot_mtxs[0].set_normalization_params(mu=soln["mu_X"])
    alignment.rot_mtxs[1].weight.data = soln["right_matrix"].T
    alignment.rot_mtxs[1].set_normalization_params(mu=soln["mu_Y"])
    return alignment

# -------------------------
# Minimal usage examples
# -------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N, D = 200, 5

    # Ground-truth transform
    A = rng.standard_normal((D, D))
    Q, _ = np.linalg.qr(A)  # random orthogonal
    s_true = 1.7
    t_true = rng.standard_normal(D)

    # Paired data
    X = rng.standard_normal((N, D))
    Y = s_true * (X @ Q) + t_true

    # Recover (rotation only)
    R1, t1, s1, info1 = orthogonal_procrustes(X, Y, center=True, scale=False)
    # Recover (similarity: scale + rotation + translation)
    R2, t2, s2, info2 = orthogonal_procrustes(X, Y, center=True, scale=True)

    print("Rotation-only  det(R):", np.linalg.det(R1))
    print("Rotation-only  |t|   :", np.linalg.norm(t1))
    print("Similarity     det(R):", np.linalg.det(R2))
    print("Recovered scale s2   :", s2, " (true:", s_true, ")")
    print("Rotation error (||R - Q||_F):", np.linalg.norm(R2 - Q, "fro"))
    print("Translation error (||t - t_true||):", np.linalg.norm(t2 - t_true))
    print("Frobenius fit error:", info2["fro_error"])

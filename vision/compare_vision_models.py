# ## Vision Model Alignment Search
# In this notebook, we will walk through how to apply MAS to simplistic vision models trained on cifar10.

# ## Overview
# 
# ### 1. Define models
# 
# In this tutorial, we will assume that the models are already trained. We can use any pytorch based model architecture. 
# 
# ### 2. Define input data (and train classification head)
# 
# We need a dataset for each model. In this tutorial, we will use the same dataset for both models. In principle, however, these datasets can be different. We will also be using cifar10 as our dataset. In this notebook, we will also train the classification head on the loaded data. In principle, this step is unnecessary for pretrained models. 
# 
# ### 3. Collect model intermediates and outputs
# 
# Once we have trained models, we will need to collect model responses for each data point in our dataset. This will give us tuples of inputs, intermediate representations, and outputs for each model that can be used to train the MAS transformation matrices.
# 
# ### 4. Train the MAS alignment (i.e. rotation matrices)
# 
# Using our data tuples, we can systematically select source and target models, patch from the source into the target intermediate representations using an interchange intervention, and then use the output distribution of the source model as the training objective for the transformation matrices. We train until convergence.
# 
# ### 5. Evaluate the alignment
# 
# Finally, we will evaluate the alignment on the cifar 10 test set.

# # 1. Define the Models


import os
import time
import gc

from vis_training import (
    get_model_and_processor, train_model, get_dataloaders,
    get_actvs_data, get_dataloader,
    train_mas_alignment_one_epoch, evaluate_mas_alignment,
    get_datasets, solve_alignment, evaluate_behavioral_relevance,
)
import torch
from torchvision import datasets
from alignment import MASAlignment, ModelStitch, load_alignment, LowRankTransformation
from hooks import hook_vision_model
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vis_utils import (
    get_valid_layer_names, read_command_line_args,
    get_layer_name_from_model_name, get_timestamp,
    save_yaml, get_git_revision_hash, get_newest_model_save_path,
)

def compare_models(config):
    config["git_hash"] = get_git_revision_hash()
    config["datetime"] = get_timestamp()
    for k in sorted(config.keys()):
        print(f"{k} ({type(config[k]).__name__}): {config[k]}")
    print()

    id_keys = [
        "exp_name", "model_names", "layer_names", "dataset_name",
        "model_stitch",  "mtx_type", "ground_truth_labels",
        "do_low_rank_transformation", "low_rank_transformation_type",
    ]
    dummy = {k: config.get(k, "") for k in id_keys if k in config}
    dummy = {k.replace("_","").split(".")[-1].split("/")[-1]: v if type(v) != list else v[0] for k, v in dummy.items()}
    id_str = "-".join([f"{k}={v}" for k, v in dummy.items()])

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
        # Models are defined as the backbone (frozen parameters, no head)
        # with an untrained head (unfrozen parameters) that will be trained
        # on the cifar10 dataset before training the MAS alignment
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
    z = zip(
        config["model_names"],
        config["layer_names"],
        models,
        processors,
    )
    for mi, (model_name, layer_name, model, processor) in enumerate(z):
        if config.get("model_mode","")=="single_model" and mi>0:
            continue
        print(f"Processing {model_name}, layer {layer_name}")
        lname = layer_name.replace("backbone.", "").replace(".", "-")
        mname = model_name.split("/")[-1]
        dname = dataset_name.split("/")[-1]
        debug = config.get("debug", False)*"_debug"
        actvs_name = f"{file_save_dir}/{exp_name}{mname}_{dname}_{lname}{seed_str}_m{mi}_actvs_train{debug}.pt"
        if os.path.exists(actvs_name) and not overwrite and not config["fresh_actvs"]:
            print(f"Loading actvs sets from disk...")
            actvs_train_sets.append(torch.load(actvs_name))
            actvs_valid_sets.append(torch.load(actvs_name.replace("train", "valid")))
        else:
            print(f"Collecting actvs for model {model_name}, layer {layer_name}, index {mi}")
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
            # the actvs data is a dict of "inputs" (B,C,H,W), "labels" (B,),
            # "actvs" (B,D,H,W) or (B,D), "logits" (B,C), "preds" (B,).
            # Note that because this is a non-sequential setting, we could make
            # the whole process more efficient by ignoring all processing layers
            # leading up to the layer of interest, but that requires us to know
            # the architecture of the model.
            with torch.no_grad():
                print(f"Collecting training set...")
                actvs_train = get_actvs_data(
                    model=model,
                    data_loader=train_loader,
                    layer_name=layer_name,
                    verbose=True,
                )
                actvs_train_sets.append(actvs_train)
    
                print(f"Collecting validation set...")
                actvs_valid = get_actvs_data(
                    model=model,
                    data_loader=test_loaders[mi],
                    layer_name=layer_name,
                    verbose=True,
                )
                actvs_valid_sets.append(actvs_valid)
            model.cpu()
    
            if config.get("save_actvs", False) or (os.path.exists(actvs_name) and overwrite):
                torch.save(actvs_train, actvs_name)
                torch.save(actvs_valid, actvs_name.replace("train", "valid"))

    for i,actvs_train_set in enumerate(actvs_train_sets):
        print("Set", i)
        for k in actvs_train_set.keys():
            print(k, actvs_train_set[k].shape)

    ####################################################
    #    Instantiate the MAS alignment object
    ####################################################
    
    og_models = [model for model in models]
    og_processors = [processor for processor in processors]
    og_layer_names = [layer_name for layer_name in layer_names]
    og_actvs_train_sets = [actvs_train_set for actvs_train_set in actvs_train_sets]
    og_actvs_valid_sets = [actvs_valid_set for actvs_valid_set in actvs_valid_sets]
    
    subspace_size = config["subspace_size"]
    mtx_type = config["mtx_type"]
    normalize = config["normalize"]
    batch_norm = config["batch_norm"]
    identity_rot = config["identity_rot"]
    debug = config["debug"]
    
    if config.get("model_mode","")=="single_model":
        models = [og_models[0] for _ in og_models]
        processors = [og_processors[0] for _ in og_models]
        layer_names = [og_layer_names[0] for _ in og_models]
        actvs_train_sets = [og_actvs_train_sets[0] for _ in og_models]
        actvs_valid_sets = [og_actvs_valid_sets[0] for _ in og_models]
        del og_models[1:]
        del og_processors[1:]
        del og_layer_names[1:]
        del og_actvs_train_sets[1:]
        del og_actvs_valid_sets[1:]
    else:
        models = [model for model in og_models]
        processors = [processor for processor in og_processors]
        layer_names = [layer_name for layer_name in og_layer_names]
        actvs_train_sets = [actvs_train_set for actvs_train_set in og_actvs_train_sets]
        actvs_valid_sets = [actvs_valid_set for actvs_valid_set in og_actvs_valid_sets]
    
    # the alignment object contains the transformation matrices for each model
    # and the patching masks for each model. It also contains the logic for
    # performing an alignment intervention (i.e. transferring causal activity
    # from a source model to a target model).
    og_dims = []
    for i in range(len(actvs_train_sets)):
        if len(actvs_train_sets[i]["actvs"].shape)==3:
            og_dims.append(actvs_train_sets[i]["actvs"].shape[-1])
        else:
            og_dims.append(actvs_train_sets[i]["actvs"].shape[1])
    if config.get("do_low_rank_transformation", False):
        if config.get("low_rank_transformation_type", "noise") == "dummy":
            config["low_rank_transformation_added_dimensions"] = og_dims[0]
        new_dims = config.get("low_rank_transformation_added_dimensions", 10)
        model_dims = [og_dims[i]+new_dims for i in range(len(og_dims))]
    else:
        model_dims = og_dims
    print("Using Model Dims:", model_dims)
    alignment_class = MASAlignment
    if config["model_stitch"] and config.get("direct_mapping", False):
        alignment_class = ModelStitch
    if type(subspace_size)==float:
        subspace_size = int(subspace_size*min(model_dims))
    alignment = alignment_class(
        model_dims=model_dims,
        mtx_type=mtx_type,
        subspace_sizes=subspace_size,
        normalize=normalize,
        batch_norm=batch_norm,
        identity_rot=identity_rot,
        dtype=config.get("mas_dtype", next(models[0].parameters()).dtype),
        same_matrix=config["same_matrix"],
    )
    low_rank_transformation = None
    if config.get("alignment_load_file",""):
        load_alignment(alignment, config["alignment_load_file"])
        print(f"Loaded alignment from {config['alignment_load_file']}")
    elif config.get("analytic_alignment", False):
        with torch.no_grad():
            alignment,low_rank_transformation = solve_alignment(
                X=actvs_train_sets[0]["actvs"],
                Y=actvs_train_sets[1]["actvs"],
                method=config.get("mtx_type","orthogonal"),
                center=True,
                scale=True,
                allow_reflection=True,
                batch_size=config.get("analytic_batch_size", 10000),
                n_samples=config.get("analytic_n_samples", 10000),
                do_low_rank_transformation=config.get("do_low_rank_transformation", False),
                low_rank_transformation_type=config.get("low_rank_transformation_type", "noise"),
                low_rank_transformation_added_dimensions=config.get("low_rank_transformation_added_dimensions", 10),
                verbose=config.get("verbose", True),
            )
        config["mas_epochs"] = 1
        config["train_directions"] = []
        config["cl_directions"] = []
        print(f"Solved alignment using analytic", config["mtx_type"], "method")
        print(f"Training directions: {config['train_directions']}")
        print(f"CL directions: {config['cl_directions']}")
    print(f"Alignment Object:")
    print(alignment)

    # We need to hook the models in order to perform the patching intervention
    # at the desired layer.
    hooks = []
    for model,layer in zip(models, layer_names):
        hooks.append(
            hook_vision_model(model, layer, alignment)
        )

    if config.get("do_low_rank_transformation", False):
        assert model_dims[0] == model_dims[1], "The low-rank transformation can only be used for two models of the same dimension"
        if low_rank_transformation is None:
            low_rank_transformation = LowRankTransformation(
                original_dimensions=og_dims[0],
                added_dimensions=config.get("low_rank_transformation_added_dimensions", 10),
                transformation_type=config.get("low_rank_transformation_type","noise"),
            )
        low_rank_transformation.to(device)
        alignment.comms_dict["low_rank_transformation"] = low_rank_transformation

    ####################################################
    #    Train the MAS alignment
    ####################################################
    
    num_epochs = config["mas_epochs"]
    batch_size = config["mas_batch_size"]
    lr = config["mas_lr"]
    one_hot_loss = config["one_hot_loss"]
    label_smoothing = config["label_smoothing"]
    train_directions = config["train_directions"]
    ground_truth_labels = config["ground_truth_labels"] # task matching
    use_trg_labels = config["use_trg_labels"]
    val_batch_size = config["mas_val_batch_size"]
    cl_directions = config["cl_directions"]
    cl_eps = config["cl_eps"]
    cl_method = config["cl_method"]
    cl_loss_type = config["cl_loss_type"]
    batches_per_optim_step = config["mas_batches_per_optim_step"]

    timestamp = get_timestamp()
    m1 = model_names[0].replace("/", "_")
    m1 = m1+layer_names[0].replace("backbone", "").replace(".", "-")
    m2 = model_names[1].replace("/", "_")
    m2 = m2+layer_names[1].replace("backbone", "").replace(".", "-")
    if config["model_stitch"]: label = "stitch"
    else: label = "mas"
    csv_name = f"{m1}_{m2}_{dataset_name}_{label}_{timestamp}.csv"
    config_name = csv_name.replace(".csv", ".yaml")
    mas_save_name = config_name.replace(".yaml", ".pt")
    mas_save_name = os.path.join(file_save_dir, mas_save_name)
    config["alignment_save_path"] = mas_save_name
    
    device = 0 if torch.cuda.is_available() else "cpu"
    alignment.to(device)
    alignment.train()
    optimizer = optim.RMSprop(alignment.parameters(), lr=lr)
    train_dfs = []
    valid_dfs = []
    rel_dfs = []
    best_train_acc = 0
    best_valid_acc = 0
    for epoch in range(num_epochs):
        try:
            print(f"Epoch {epoch} - Training", id_str)
            start_time = time.time()
            train_df = train_mas_alignment_one_epoch(
                models=models,
                alignment=alignment,
                actvs_sets=actvs_train_sets,
                batch_size=batch_size,
                optimizer=optimizer,
                batches_per_optim_step=batches_per_optim_step,
                one_hot_loss=one_hot_loss,
                label_smoothing=label_smoothing,
                train_directions=train_directions,
                use_ground_truth_labels=ground_truth_labels,
                use_trg_labels=use_trg_labels,
                cl_directions=cl_directions,
                cl_eps=cl_eps,
                cl_method=cl_method,
                cl_loss_type=cl_loss_type,
                debug=debug,
            )
            gc.collect()
            end_time = time.time()
            print(f"Epoch Duration: {end_time - start_time}s")
            print(f"Epoch {epoch} - Validating")
            valid_df = evaluate_mas_alignment(
                models=models,
                alignment=alignment,
                actvs_sets=actvs_valid_sets,
                batch_size=val_batch_size,
                one_hot_loss=one_hot_loss,
                use_ground_truth_labels=ground_truth_labels,
                cl_directions=cl_directions,
                cl_eps=cl_eps,
                cl_method=cl_method,
                cl_loss_type=cl_loss_type,
                verbose=True,
                debug=debug,
            )
            gc.collect()

            cols = ["actn_loss","penalty","cl_loss","acc","behav_acc","label_acc","src_acc"]
            groups = ["src_idx","trg_idx"]
    
            train = train_df.groupby(groups)[cols].mean().reset_index()
            train["epoch"] = epoch
            train_dfs.append(train)
    
            valid = valid_df.groupby(groups)[cols].mean().reset_index()
            valid["epoch"] = epoch
            valid_dfs.append(valid)

            print()
            print(train.sort_values(by=groups,ascending=True))
            valid.columns = ["valid_"+col if col in cols else col for col in valid.columns]
            print(valid.sort_values(by=groups,ascending=True))
            print(
                "Train IIA:", round(np.min(train["acc"]), 5),
                "|| Penalty:", round(np.min(train["penalty"]), 5),
                "|| Loss:", round(np.max(train["actn_loss"]), 5)
            )
            print(
                "Valid IIA:", round(np.min(valid["valid_acc"]), 5),
                "|| Penalty:", round(np.min(valid["valid_penalty"]), 5),
                "|| Loss:", round(np.max(valid["valid_actn_loss"]), 5)
            )

            if config.get("track_relevance", False) and (epoch % 10 == 0 or debug):
                print("Evaluating behavioral relevance...")
                rel_df = evaluate_behavioral_relevance(
                    models=models,
                    alignment=alignment,
                    actvs_sets=actvs_valid_sets,
                    batch_size=batch_size,
                    one_hot_loss=one_hot_loss,
                    use_ground_truth_labels=ground_truth_labels,
                    use_trg_labels=use_trg_labels,
                    ablate_low_rank_transformation=config.get("ablate_low_rank_transformation", False),
                    verbose=True,
                    debug=debug,
                )
                rel_groups = [c for c in groups if c in rel_df.columns]
                rel_cols = [c for c in cols if c in rel_df.columns] +\
                           ["grad_mse", "grad_cosine", "grad_correlation"]
                rel = rel_df.groupby(rel_groups)[rel_cols].mean().reset_index()
                rel["epoch"] = epoch
                rel_dfs.append(rel)
                print(rel.sort_values(by=rel_groups,ascending=True))
                print()
            
            if config["debug"]:
                continue
            sname = mas_save_name.replace(".pt", f"_best.pt")
            if config["model_stitch"] and valid["valid_acc"].max() > best_valid_acc:
                best_valid_acc = valid["valid_acc"].max()
                best_train_acc = train["acc"].max()
                torch.save(alignment.state_dict(), sname)
                print(f"Saved alignment to {mas_save_name}")
                print(f"New best validation accuracy: {best_valid_acc}")
                print(f"New best train accuracy: {best_train_acc}")
            elif not config["model_stitch"] and valid["valid_acc"].min() > best_valid_acc:
                best_valid_acc = valid["valid_acc"].min()
                best_train_acc = train["acc"].min()
                torch.save(alignment.state_dict(), sname)
                print(f"New best validation accuracy: {best_valid_acc}")
                print(f"New best train accuracy: {best_train_acc}")

        except KeyboardInterrupt:
            print("Interrupted training, exiting...")
            break
    
    
    train_df = pd.concat(train_dfs)
    valid_df = pd.concat(valid_dfs)
    
    
    train_df.columns = ["train_"+col if col in cols else col for col in train_df.columns]
    valid_df.columns = ["valid_"+col if col in cols else col for col in valid_df.columns]
    cols = [
        "train_label_acc", "valid_label_acc", "penalty",
        "train_acc","valid_acc",
        "train_actn_loss", "valid_actn_loss",
    ]
    main_df = pd.merge(train_df, valid_df, on=groups+["epoch"])

    ####################################################
    #    Evaluate behavioral relevance
    ####################################################
    ablate_df = None
    grad_df = None
    cols = ["actn_loss","penalty","cl_loss","acc","behav_acc","label_acc","src_acc"]
    groups = ["src_idx","trg_idx"]
    rel_groups = [c for c in groups if c in rel_df.columns]
    rel_cols = [c for c in cols if c in rel_df.columns] +\
            ["grad_mse", "grad_cosine", "grad_correlation"]
    grad_df = evaluate_behavioral_relevance(
        models=models,
        alignment=alignment,
        actvs_sets=actvs_valid_sets,
        batch_size=batch_size,
        one_hot_loss=one_hot_loss,
        use_ground_truth_labels=ground_truth_labels,
        ablate_low_rank_transformation=False,
        verbose=True,
        debug=debug,
    )
    print("Gradient based relevance")
    grad = grad_df.groupby(rel_groups)[rel_cols].mean().reset_index()
    print(grad.sort_values(by=rel_groups,ascending=True))
    dolow = config.get("do_low_rank_transformation", False)
    if dolow:
        ablate_df = evaluate_behavioral_relevance(
            models=models,
            alignment=alignment,
            actvs_sets=actvs_valid_sets,
            batch_size=batch_size,
            one_hot_loss=one_hot_loss,
            use_ground_truth_labels=ground_truth_labels,
            ablate_low_rank_transformation=True,
            verbose=True,
            debug=debug,
        )
        print("Ablated low-rank transformation")
        ablat = ablate_df.groupby(rel_groups)[rel_cols].mean().reset_index()
        print(ablat.sort_values(by=rel_groups,ascending=True))
    if dolow and config.get("ablate_low_rank_transformation", False):
        rel_df = pd.concat([grad_df, ablate_df])
    else:
        rel_df = grad_df
    rel = rel_df.groupby(rel_groups)[rel_cols].mean().reset_index()
    rel["epoch"] = epoch
    rel_dfs.append(rel)
    rel_df = pd.concat(rel_dfs)
    
    ####################################################
    #    Save results
    ####################################################
    if not config.get("debug", False):
        torch.save(alignment.state_dict(), mas_save_name)
        print(f"Saved alignment to {mas_save_name}")

        if config.get("track_relevance", False): # track the behavioral relevance of the alignment
            rel_csv_name = csv_name.replace(".csv", "_rel.csv")
            rel_save_name = os.path.join(file_save_dir, rel_csv_name)
            rel_df.to_csv(rel_save_name, index=False, header=True)
            print(f"Saved behavioral relevance results to {rel_save_name}")

            if ablate_df is not None:
                ablate_csv_name = csv_name.replace(".csv", "_ablate.csv")
                ablate_save_name = os.path.join(file_save_dir, ablate_csv_name)
                ablate_df.to_csv(ablate_save_name, index=False, header=True)
                print(f"Saved behavioral relevance results to {ablate_save_name}")
            if grad_df is not None:
                grad_csv_name = csv_name.replace(".csv", "_grad.csv")
                grad_save_name = os.path.join(file_save_dir, grad_csv_name)
                grad_df.to_csv(grad_save_name, index=False, header=True)
                print(f"Saved behavioral relevance results to {grad_save_name}")
    
        main_df.to_csv(f"csvs/{csv_name}", index=False, header=True)
        save_yaml(config, f"csvs/{config_name}")
        print(f"Saved results to {csv_name}")

    
    if not config["make_figs"]:
        print("Ending", id_str)
        print("--------------------------------")
        return main_df

    ####################################################
    #    Evaluate results
    ####################################################
    
    try:
        main_df["direction"] = main_df["src_idx"].astype(str) + "->" + main_df["trg_idx"].astype(str)
        all_df = main_df[["direction","epoch"]+cols]
        cols = ["train_acc","valid_acc", "train_actn_loss", "valid_actn_loss"]
        all_dfs = [all_df]
        for agg in ["min","max","mean"]:
            df = all_df.groupby(["epoch"])[cols].agg(agg).reset_index()
            df["epoch"] = df.index
            df["direction"] = agg
            all_dfs.append(df)
        all_df = pd.concat(all_dfs, sort=True)
    
        fig = plt.figure(figsize=(7,7))
        ax = plt.gca()
    
        plot_df = all_df[all_df["direction"].isin({"min", "1->0", "0->1"})]
        sns.lineplot(
            data=plot_df,
            x="epoch",
            y="train_acc",
            hue="direction",
            ax=ax,
            legend=False,
            palette=sns.color_palette("bright"),
        )
    
        sns.lineplot(
            data=plot_df,
            x="epoch",
            y="valid_acc",
            hue="direction",
            ax=ax,
            legend=True,
            palette=sns.color_palette("dark"),
        )
    
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy of MAS Alignment")
    
        plt.savefig("figs/mas_acc.png", dpi=600)
    
    
        fig = plt.figure(figsize=(7,7))
        ax = plt.gca()
    
        plot_df = all_df[all_df["direction"].isin({"max", "min", "1->0", "0->1"})]
        sns.lineplot(
            data=plot_df,
            x="epoch",
            y="train_actn_loss",
            hue="direction",
            legend=False,
            ax=ax,
            palette=sns.color_palette("bright"),
        )
    
        sns.lineplot(
            data=plot_df,
            x="epoch",
            y="valid_actn_loss",
            hue="direction",
            ax=ax,
            legend=True,
            palette=sns.color_palette("dark"),
        )
    
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss of MAS Alignment")
    
        if not config.get("debug", False) and config.get("make_figs", True):
            plt.savefig("figs/mas_loss.png", dpi=600)
    except Exception as e:
        print(f"Error making figures: {e}")
        print("Valid columns:", main_df.columns)
        pass
    return main_df

default_config = {
    "overwrite": False,
    "new_models": False, # if True, will create new model saves even if others exist.
    "fresh_actvs": False, # if True, will overwrite the actvs sets even if they exist on disk
    "pretrained": True, # if True, will use the pretrained model weights from huggingface
    "image_resize": 64, # if an int is provided, will resize the images to the specified size
    "finetune_full_model": True, # if True, will only finetune the classification head of the model
    "make_figs": False,
    "layer_sweep": False,
    "model_stitch": False, # If true, will change the training settings to
        # perform model stitching in stead of MAS, where model stitching
        # is performed in one direction, always using the first model as
        # the source and the second model as the target.
    "same_matrix": False, # If true, will use the same matrix for all models.
        # Only applies to model stitching, not MAS. Can only use two models.
        # Will use the inverse of the matrix for the second model when intervening.
    "latent_model_stitch": False, # If true and model_stitch is true,
        # will change the training settings to
        # perform latent model stitching in stead of MAS, where latent model
        # stitching is performed in both directions. model_stitch must be true
        # for this to take effect.
    "direct_mapping": False, # If true will use a ModelStitch object
        # instead of MASAlignment. Only applies if model_stitch is true.
    "alignment_load_file": "", # if provided, will load the alignment from the
        # specified file instead of training it from scratch.
    
    "low_rank_transformation_type": "noise", # the type of low-rank transformation to use.
        # choices:
        #   "noise": add noise to the representations
        #   "zeros": set the low-rank dimensions to zero
        #   "dummy": use the original representations are duplicated along
        #       the null dimensions.
    "do_low_rank_transformation": False, # if True, will use a transformation
        # that pads the representations with zeros or noise and then rotates
        # them into a new basis before the alignment intervention.
    "low_rank_transformation_added_dimensions": 10, # the number of dimensions to
        # add to the representations.
    "track_relevance": True, # if True, will track the behavioral relevance of the alignment
    "ablate_low_rank_transformation": True, # if True, will set the added low-rank
        # transformation dims to zeros instead of its previous type during
        # evaluation of behavioral relevance.

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
    "num_epochs": 25, # number of epochs to finetune the model for
    "train_lr": 0.005,
    "model_save_dir": "/data2/grantsrb/vision_mas/models",
    "data_root": "/data2/grantsrb/pytorch_datasets",
    "actvs_batch_size": 2056,
    "og_train_weight_decay": 0, # will use weight decay in the original training loss.
    "og_train_label_smoothing": 0.1, # will use label smoothing in the original training loss.
        # Only used if finetune_full_model is True
    
    # MAS alignment parameters
    "subspace_size": None,
    "mtx_type": "orthogonal",
    "normalize": False,
    "batch_norm": True, # applies invertible batch norm to the activations
        # before the transformation matrix
    "post_batch_norm": False, # applies invertible batch norm to the activations
        # after the transformation matrix. Mainly useful for model stitching.
    "identity_rot": False, # uses the identity rotation matrix (only use for debugging)
    "debug": False, # if True, will use the first model for all models
    "model_mode": "", # "single_model" or "multi_model"
        # "single_model" will use the first model for all models
        # "multi_model" will use the models specified in the model_names parameter
        # None defaults to "multi_model"

    # MAS training parameters
    "train_directions": None, # will only train the MAS alignment for the
        # specified directions (argue a list of tuples of (src_idx, trg_idx)).
        # None defaults to all directions.
    "n_train_samples": 50000, # will only train the MAS alignment on the specified
        # number of samples for each direction. None defaults to all samples.
    "n_valid_samples": 10000, # will only validate the MAS alignment on the specified
        # number of samples for each direction. None defaults to all samples.
    "mas_epochs": 50,
    "mas_batch_size": 256,
    "mas_lr": 0.0001,
    "mas_val_batch_size": 1000,
    "mas_batches_per_optim_step": 1, # will run the optimizer for the specified
        # number of batches per optimizer step.
    "mas_dtype": None, # the dtype to use for the MAS alignment
    "ground_truth_labels": False, # will use data labels instead of model
        # predictions as the training objective for the MAS alignment
    "use_trg_labels": False, # will use the target model logits/labels
        # for the training objective for the MAS alignment (instead of the
        # source model predictions). Be careful combining this with
        # low dimensional subspace sizes. It is possible to learn a trivial/null
        # alignment where the target model simply passes its representations
        # through the alignment module.
    "one_hot_loss": False, # will use one-hot encoded labels as the training
        # objective for the MAS alignment
    "label_smoothing": 0.0, # will use label smoothing in the MAS loss.
        # Only used if one_hot_loss is True

    # CL training parameters
    "bnn_mas": False, # if True, will train the MAS alignment on the behavioral
        # and the latent loss in a similar fashion to a biological neural network
        # compared to an artificial neural network.
    "latent_mas": False, # if True, will train the MAS alignment on the CL loss only.
    "cl_directions": None, # will only collect CL vectors for the specified
        # directions (argue a list of tuples of (src_idx, trg_idx)).
        # None defaults to no directions. Set cl_eps to 0 if you wish to
        # track the cl_loss without training it.
    "cl_eps": 1, # raw multiplicative factor for the cl_loss does not
        # affect the normal loss other than that it will be added to the
        # normal loss
    "cl_method": "same_as_target", # determines how the CL vectors are generated.
        # choices:
        #   "sample": sample a random activation for each class
        #   "mean": take the mean of the activations for each class
        #   "most_similar": take the activations with the highest similarity
        #     to the source class probabilities
        #   "same_as_target": use the source model's activations created
        #     under the same inputs as the source model
    "cl_loss_type": "both", # {"mse", "cos", "both"}
    "cl_causal_dims_only": False, # if True, will only use the causal dimensions
        # for the CL loss.
}

def prepare_config(config):
    if config["model_stitch"]:
        config["batch_norm"] = True
        config["post_batch_norm"] = True
        config["direct_mapping"] = True
        config["same_matrix"] = True
        config["mask_type"] = "ZeroMask"# Ensures that the extraneous dimensions
            # are set to zero during the patching
        if config["latent_model_stitch"]:
            config["train_directions"] = []
            config["cl_directions"] = [(0,1),(1,0)]
            config["cl_method"] = "same_as_target"
        else:
            config["train_directions"] = [(0,1)]
    elif config.get("unimas", False):
        config["train_directions"] = [(0,1)]
        config["cl_directions"] = []
    elif config.get("bnn_mas", False):
        config["train_directions"] = [(0,1),(1,1)]
        config["cl_directions"] = [(1,0),(1,1)]
    elif config.get("latent_mas", False):
        config["train_directions"] = []
        config["cl_directions"] = [(0,1),(1,0)]
    if config["train_directions"] is not None and config["train_directions"] == "":
        config["train_directions"] = []
    elif config["train_directions"] is None or config["train_directions"] == "all":
        config["train_directions"] = [(0,0),(0,1),(1,0),(1,1)]
    if not config["cl_directions"]:
        config["cl_directions"] = []

    if config["ground_truth_labels"]:
        config["one_hot_loss"] = True

    if config.get("single_model", False):
        config["model_mode"] = "single_model"

    if config.get("analytic_alignment", False):
        config["mas_epochs"] = 1
        config["train_directions"] = []
        config["cl_directions"] = []
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
    lname1 = layer_names[0]
    lname2 = layer_names[1]
    if lname1 is None:
        lname1 = get_layer_name_from_model_name(config["model_names"][0])
    if lname2 is None:
        lname2 = get_layer_name_from_model_name(config["model_names"][1])
    config["layer_names"] = [lname1, lname2]
    compare_models(config)

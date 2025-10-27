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
    get_datasets, solve_alignment,
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
    save_yaml, get_git_revision_hash,
)

def compare_models(config):
    config["git_hash"] = get_git_revision_hash()
    config["datetime"] = get_timestamp()
    for k in sorted(config.keys()):
        print(f"{k} ({type(config[k]).__name__}): {config[k]}")
    print()

    model_names = config["model_names"]
    n_models = len(model_names)
    batch_size = config["batch_size"]
    val_batch_size = config["val_batch_size"]
    num_workers = config["num_workers"]
    num_epochs = config["num_epochs"]
    lr = config["train_lr"]
    model_save_dir = config["model_save_dir"]
    data_root = config["data_root"]
    overwrite = config["overwrite"]

    models = []
    processors = []
    
    for model_name in model_names:
        # Models are defined as the backbone (frozen parameters, no head)
        # with an untrained head (unfrozen parameters) that will be trained
        # on the cifar10 dataset before training the MAS alignment
        model, proc = get_model_and_processor(
            model_name, pretrained=config["pretrained"]
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
        )
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
    
    
    ####################################################
    #    Load the models and finetune
    ####################################################
    print("Loading models...")
    for i, (model, processor) in enumerate(zip(models, processors)):
        train_loader = train_loaders[i]
        test_loader = test_loaders[i]
        model_name = model_names[i].split("/")[-1]
        full_finetune = config["finetune_full_model"]
        model_save_path = f"{model_save_dir}/{model_name}_{finetune_dataset_name}_finetune{full_finetune}_sd_{i}.pt"
        if not config["pretrained"]:
            model_save_path = model_save_path.replace(".pt", "_unpretrained.pt")
        if os.path.exists(model_save_path) and not config["overwrite"]:
            print(f"Loading model from {model_save_path}")
            model.load_state_dict(torch.load(model_save_path))
        else:
            print("Overwriting model")
            config["overwrite"] = True
            print(f"Finetuning model {i}")
            try:
                model, metrics = train_model(
                    model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    hyperparameters={
                        "train_lr": lr,
                        "num_epochs": num_epochs,
                        "early_stopping": True,
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
            
        # From here on, we will not update the model parameters
        for p in model.parameters():
            p.requires_grad = False
        
        torch.save(model.state_dict(), model_save_path)
    
    
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
        actvs_name = f"{model_save_dir}/{mname}_{dname}_{lname}_m{mi}_actvs_train{debug}.pt"
        if os.path.exists(actvs_name) and not config["overwrite"] and not config["fresh_actvs"]:
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
    
            if config.get("save_actvs", False) or (os.path.exists(actvs_name) and config["overwrite"]):
                torch.save(actvs_train, actvs_name)
                torch.save(actvs_valid, actvs_name.replace("train", "valid"))
    
    if config.get("debug", False):
        os.makedirs("figs", exist_ok=True)
        plt.imshow(actvs_train_sets[0]["inputs"][0].cpu().numpy().transpose(1,2,0))
        plt.savefig("figs/input_0.png", dpi=600)
        try:
            plt.imshow(actvs_train_sets[1]["inputs"][0].cpu().numpy().transpose(1,2,0))
            plt.savefig("figs/input_1.png", dpi=600)
        except:
            pass
        plt.imshow(actvs_valid_sets[0]["inputs"][0].cpu().numpy().transpose(1,2,0))
        plt.savefig("figs/input_0_valid.png", dpi=600)
        try:
            plt.imshow(actvs_valid_sets[1]["inputs"][0].cpu().numpy().transpose(1,2,0))
            plt.savefig("figs/input_1_valid.png", dpi=600)
        except:
            pass
    

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
        new_dims = config.get("low_rank_transformation_added_dimensions", 10)
        model_dims = [og_dims[i]+new_dims for i in range(len(og_dims))]
    else:
        model_dims = og_dims
    print("Using Model Dims:", model_dims)
    alignment_class = MASAlignment
    if config["model_stitch"] and config.get("direct_mapping", False):
        alignment_class = ModelStitch
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
    
    device = 0 if torch.cuda.is_available() else "cpu"
    alignment.to(device)
    alignment.train()
    optimizer = optim.RMSprop(alignment.parameters(), lr=lr)
    train_dfs = []
    valid_dfs = []
    for epoch in range(num_epochs):
        try:
            print(f"Epoch {epoch} - Training")
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

            cols = ["actn_loss","penalty","cl_loss","acc"]
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
        except KeyboardInterrupt:
            print("Interrupted training, exiting...")
            break
    
    
    train_df = pd.concat(train_dfs)
    valid_df = pd.concat(valid_dfs)
    
    
    train_df.columns = ["train_"+col if col in cols else col for col in train_df.columns]
    valid_df.columns = ["valid_"+col if col in cols else col for col in valid_df.columns]
    cols = ["train_acc","valid_acc", "train_actn_loss", "valid_actn_loss"]
    main_df = pd.merge(train_df, valid_df, on=groups+["epoch"])
    
    timestamp = get_timestamp()
    m1 = model_names[0].replace("/", "_")
    m1 = m1+layer_names[0].replace("backbone", "").replace(".", "-")
    m2 = model_names[1].replace("/", "_")
    m2 = m2+layer_names[1].replace("backbone", "").replace(".", "-")
    csv_name = f"{m1}_{m2}_{dataset_name}_mas_{timestamp}.csv"
    config_name = csv_name.replace(".csv", ".yaml")
    if not config.get("debug", False):
        main_df.to_csv(f"csvs/{csv_name}", index=False, header=True)
        save_yaml(config, f"csvs/{config_name}")
        print(f"Saved results to {csv_name}")
    
    
    if not config["make_figs"]:
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
    "fresh_actvs": False, # if True, will overwrite the actvs sets even if they exist on disk
    "pretrained": True, # if True, will use the pretrained model weights from huggingface
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
    "do_low_rank_transformation": False, # if True, will use a transformation
        # that pads the representations with zeros or noise and then rotates
        # them into a new basis before the alignment intervention.
    "low_rank_transformation_added_dimensions": 10, # the number of dimensions to
        # add to the representations.

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
    "num_epochs": 10, # number of epochs to finetune the model for
    "train_lr": 0.001,
    "model_save_dir": "/data2/grantsrb/vision_mas/models",
    "data_root": "/data2/grantsrb/pytorch_datasets",
    "actvs_batch_size": 2056,
    
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

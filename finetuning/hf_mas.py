import sys
sys.path.insert(1, "../")
import os
import torch
import numpy as np
import time
from datetime import datetime
from torch.utils.data import DataLoader
from datasets import Dataset

from datas import (
    get_dataset, tokenize_dataset,
    collate_fn, make_tokenized_info, add_token_ids_to_info,
    add_prompt, pad_data_dict, add_pad_masks, convert_to_tensors,
)
from utils import (
    collect_activations, get_command_line_args,
)
import seq_models as smods
from dl_utils.save_io import (
    get_save_name, load_checkpoint, get_folder_from_path, save_json,
    get_config,
)
from dl_utils.utils import (
    get_git_revision_hash, get_timestamp, analytical_linear_regression, 
)
from dl_utils.schedulers import PlateauTracker
from dl_utils.tokenizer import Tokenizer
from fca import perform_eigen_pca
from intrv_modules import InterventionModule
import filters
import causal_models
import constants as consts
from intrv_datas import make_intrv_data_from_src_data
from hooks import get_stepwise_hook, get_indywise_hook, get_hook_module
from intrv_training import (
    get_model_and_tokenizer, forward_pass, get_embedding_name,
    get_cl_latents,
)

from transformers import ( AutoTokenizer, AutoModelForCausalLM, )

import pandas as pd # import after transformers to avoid versioning bug

def config_prep(config):
    # Catch plural errors:
    for k in [
        "swap_keys", "mtx_types", "model_names",
        "dataset_names", "padding_sides", "layers", 
    ]:
        singular = k[:-1]
        assert singular not in config, f"Must use {k} instead of {singular}"

    if type(config["model_names"])==str:
        config["model_names"] = config["model_names"].split(",")

    n_models = len(config["model_names"])
    if type(config["mtx_types"])==str:
        config["mtx_types"] = [config["mtx_types"] for _ in range(n_models)]
    config["mtx_kwargs"] = [ {**config} for _ in range(n_models) ]
    config["mask_kwargs"] = {**config}
    config["filters"] = [
        getattr(filters, fname) for fname in config["filter_names"]
    ]

    kwargs = { "hold_outs": [], }

    if config["swap_keys"] is None:
        config["swap_keys"] = [["full"], ["full"]]
    elif type(config["swap_keys"])==list and type(config["swap_keys"][0])==str:
            config["swap_keys"] = [config["swap_keys"] for _ in range(n_models)]
    elif type(config["swap_keys"] )==str:
        skey = config["swap_keys"]
        config["swap_keys"] = [skey for _ in range(n_models)]
    for si,sks in enumerate(config["swap_keys"]):
        if sks[0] is None:
            config["swap_keys"][si] = ["full"]
        elif type(sks)==str:
            config["swap_keys"][si] = [sks]
        if config.get("incl_empty_varbs", False):
            config["swap_keys"][si].append("")
    if len(config["swap_keys"])<n_models:
        config["swap_keys"] = config["swap_keys"]*n_models
    print("Swap Keys:", config["swap_keys"])

    config["n_subspaces"] = len(config["swap_keys"][0])-int(config.get(
        "incl_empty_varbs", False))
    
    if type(config["train_directions"])==list:
        if len(config["train_directions"])>0:
            config["train_directions"] = [
                tuple(td) for td in config["train_directions"] if td is not None]
    elif config["train_directions"] in {None, "all", "none"}:
        config["train_directions"] = []
        for s in range(n_models):
            for t in range(n_models):
                config["train_directions"].append((s,t))

    if type(config["cl_directions"])==list:
        config["cl_directions"] = [tuple(td) for td in config["cl_directions"]]
    elif config["cl_directions"] in {None, "none"}:
        config["cl_directions"] = []
    elif config["cl_directions"] =="all":
        config["cl_directions"] = []
        for s in range(n_models):
            for t in range(n_models):
                config["cl_directions"].append((s,t))

    config["layers"] = [
            "inpt_identity" if l=="embeddings" else l for l in config["layers"]]
    if len(config["layers"])<n_models:
        config["layers"] = config["layers"]*n_models

    if "learning_rate" in config:
        print("use lr instead of learning_rate keyword")
        assert False

    if config.get("debugging", False):
        config["n_train_samples"] = 100
        config["n_valid_samples"] = 100
        config["print_every"] = 25
    return config

def fill_in_prompts_and_replacements(config):
    config["padding_sides"] = []
    for model_name in config["model_names"]:
        print("Model Name:", model_name)
        # Get padding side
        padding_side = consts.PADDING_SIDES.get(model_name, "right")
        config["padding_sides"].append(padding_side)
    return config

def main():
    arg_config, command_keys = get_command_line_args(sys.argv)
    ##########################
    #    Default configuration
    ##########################
    defaults = {
        "save_root": "/data2/grantsrb/mas_finetuning/",
        "exp_name": "myexp",
        "conserve_memory": True,

        "source_files": [
            "/data2/grantsrb/mas_finetuning/srcactvs_gpt2_Anthropic/hh-rlhf_toxic_d2025-07-24_t22-19-23.pt",
            "/data2/grantsrb/mas_finetuning/srcactvs_gpt2_Anthropic/hh-rlhf_toxic_d2025-07-24_t22-19-23.pt",
        ],

        "layers": [ # layers at which to attach the hooks
            "embeddings",
            "embeddings"
        ],  
        "swap_keys": [ ["full"], ["full"] ], # argue a list of
            # keys for each model.
        "incl_empty_varbs": False, # if true, includes an explicit
            # training of the extraneous information, encouraging
            # it to be a functionally null subspace.
        "mtx_types": ["RotationMatrix", "RotationMatrix"],
        "nonlin_align_fn": "identity", # Inverse of a function applied
            # before the rotation matrix during interventions. options:
            # "identity", "tanh", "sigmoid"
        ## Not Implmented Yet
        "zscore_alignment": False, # If true, will zscore the reps
            # before alignment. zscoring is applied after the
            # nonlinearity if using a nonlin_align_fn. Uses the source
            # activations to calculate the mu and std
        "pca_init": False, # If true, will use PCA to initialize the
            # rotation matrix. If false, will use a random orthogonal
            # matrix. If using a nonlin_align_fn, will apply the
            # nonlinearity to the activations before PCA.
        "identity_init": False,
        "identity_rot": False,
        "n_layers": 3, # if using rev resnets, number of res layers
        "mask_type":   "FixedMask", # BoundlessMask
        "n_units": None,
        "learnable_addition": False,
        "const_targ_inpt_id": False, # If true, will use the resp_id for all target input ids
        "fsr": False, # (Functionally sufficient representations) only applies
            # if using fca. Discards the excess components in the target
            # representation during an intervention. Ends up being equivalent
            # to using a vector of 0s for input embeddings at the intervention
            # positions

        "num_training_steps": 50000,
        "print_every": 100,
        "batch_size": 32,
        "grad_accumulation_steps": 8,
        "lr": 1e-3,
        "max_length": 128,                 # max token length for our (toy) examples
        "eval_batch_size": 16,             # batch size for correctness evaluation
        "patience": 10, # only evaluated on print_every epochs
        "plateau": 0.001,
        "measure": "loss", #plateau measure (acc or loss)
        "upper_acc_thresh": 0.995,

        "stepwise": False,
        "train_directions": None, # None and "all" do the same thing. Can
            # specify training direction tuples: [(0,0), (1,0), (0,1), (1,1)] where
            # the first index in the tuple specifies the src idx, and the second
            # specifies the target.
        "cl_directions": "all", # "all" will default to all directions.
            # specify training direction tuples: [(0,0), (1,0), (0,1), (1,1)] where
            # the first index in the tuple specifies the src idx, and the second
            # specifies the target for the counterfactual latent loss. None
            # defaults to no directions. Set cl_eps to 0 if you wish to
            # track the cl_loss without training it.
        "cl_loss_type": "both", # choices: mse, cos, both
        "cl_eps": 0, # raw multiplicative factor for the cl_loss does not
            # affect the normal loss
        "track_intrv_distr": False, # if true, will track the difference
            # between the intervention distribution and the original distribution.
            # Uses kl divergence, mse, and cosine similarity

        "save_keys": [
            "mtx_types", "layers", "n_units","stepwise", "swap_keys"
        ],
        "debugging": False,
    }
    config = {**defaults}
    config["git_hash"] = get_git_revision_hash()
    config["datetime"] = get_timestamp()
    for k in arg_config: config[k] = arg_config[k]
    for k in command_keys:
        config["save_keys"].append(k)
    config["save_keys"] = sorted(list(set(config["save_keys"])))
    print("Config:")
    for k in sorted(list(config.keys())):
        print(k, config[k])

    config = config_prep(config) # general error catching
    padding_sides = ["left" for _ in config["source_files"]]

    save_folder = get_folder_from_path(config["source_files"][0])
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    save_name = get_save_name(
        save_folder=save_folder,
        kwargs=arg_config,
        config=config)
    print("Saving to:", save_folder)
    print("Save Name:", save_name)

    jpath = os.path.join(save_folder, save_name + ".json")
    if not config.get("debugging", False):
        save_json(config, jpath)

    ##########################
    #    Load models and tokenizers
    ##########################
    n_models = len(config["source_files"])
    n_varbs = len(config["swap_keys"])
    config["swap_keys"] = config.get("swap_keys", [["full"],["full"]])
    if config["incl_empty_varbs"]:
        config["swap_keys"] = [sk + [""] for sk in config["swap_keys"]]
    poss_devices = ["cpu","cpu"]
    if torch.cuda.is_available():
        if torch.cuda.device_count()>1:
            poss_devices = [0,1]
        else:
            poss_devices = [0,0]
    models = []
    tokenizers = []
    model_configs = []
    m_sizes = []
    devices = []
    print("Loading datasets and models...")
    prompts = [] if config.get("prompts", None) is None else config["prompts"]
    prompt_lens = []
    src_datasets = { "train": [], "valid": [], } # lists will correspond
        # to the index of each model
    for mi,source_file in enumerate(config["source_files"]):
        source_data = torch.load(source_file)
        model_config = source_data["config"]
        model = AutoModelForCausalLM.from_pretrained(model_config["model_name"])
        tokenizer = AutoTokenizer.from_pretrained( model_config["tokenizer_name"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token


        model.eval()
        model_configs.append(model_config)

        # Freeze model parameters so that only our rotation matrix is trained.
        for param in model.parameters():
            param.requires_grad = False
        print("Model", mi, "-", model_config["model_name"])
        print(model)
        models.append(model)
        tokenizers.append(tokenizer)

        # Get appropriate model layers
        if config["layers"][mi] in {"embeddings", "inpt_identity"}:
            config["layers"][mi] = get_embedding_name(
                model, config["layers"][mi])
            print("Decided Layer Name:", config["layers"][mi])

        # Record appropriate devices for layers
        if hasattr(model, "hf_device_map") or hasattr(model, "hf_encoder"):
            try:
                dmap = model.hf_encoder.hf_device_map
            except:
                dmap = model.hf_device_map
            if config["layers"][mi] in dmap:
                devices.append(dmap[config["layers"][mi]])
            else:
                try:
                    devices.append(dmap[""])
                except:
                    print(dmap.keys())
                    devices.append(0)
        else:
            devices.append(poss_devices[mi])
            model.to(devices[-1])

        # Just collect a single step to determine the dimensionality of
        # the hooked layer
        with torch.no_grad():
            actvs = collect_activations(
                model,
                input_ids=torch.LongTensor([[1,2,3,4]]),
                layers=[config["layers"][mi]],
                batch_size=500,
                to_cpu=True,)
        m_sizes.append(actvs[config["layers"][mi]].shape[-1])

        ####################################################
        #    Read in RAW DATA
        ####################################################
        layer = config["layers"][mi]
        src_text = source_data["text"]
        z = zip(source_data["input_text"], source_data["generated_text"])
        raw_text = [in_txt + gen_txt for in_txt,gen_txt in z]
        logits = source_data["logits"]
        src_actvs = source_data["layer_states"][layer]
        prompt_len = source_data["propt_len"]
        prompt_lens.append(prompt_len)
        if config.get("prompts", None) is None:
            prompts.append(source_data["prompt"])

        trn_size = int(0.8*len(raw_text))
        val_size = len(raw_text)-trn_size

        # Split data into train/valid
        perm = torch.randperm(len(raw_text)).long()
        lperm = perm.tolist()

        trn_data = {
            "src_text": [src_text[p] for p in lperm[:trn_size]],
            "text": [raw_text[p] for p in lperm[:trn_size]],
            "logits": logits[perm[:trn_size]],
            "actvs": src_actvs[perm[:trn_size]],
            "prompt_len": prompt_len,
        }
        src_datasets["train"].append( trn_data )

        val_data = {
            "src_text": [src_text[p] for p in lperm[trn_size:]],
            "text": [raw_text[p] for p in lperm[trn_size:]],
            "logits": logits[perm[trn_size:]],
            "actvs": src_actvs[perm[trn_size:]],
            "prompt_len": prompt_len,
        }
        src_datasets["valid"].append( val_data )

    ####################################################
    #  MAKE INTERVENTION DATA
    ####################################################
    subspace_maps = { # maps between indices and string keys for each model
        "subspace2varbs": [],
        "varb2subspaces": [],
    }
    config["subspace_maps"] = subspace_maps
    all_src_activations = {k: dict() for k in src_datasets}
    intrv_datasets = {k: dict() for k in src_datasets}
    for si in range(n_models):
        subspace_maps["subspace2varbs"].append(
            {i: key for i,key in enumerate(config["swap_keys"][si])}
        )
        subspace_maps["varb2subspaces"].append(
            {key: i for i,key in enumerate(config["swap_keys"][si])}
        )
        for ti in range(n_models):
            for vi,varb in enumerate(config["swap_keys"][si]):
                dir_var_tup = (si,ti,vi)
                prompt = prompts[ti]
                for k in src_datasets:
                    intrv_data = make_intrv_data_from_src_data(
                        text=src_datasets[k][si]["text"],
                        shuffle=varb=="full",
                        trg_prompt=prompt,
                        src_logits=src_datasets[k][si]["logits"],
                        src_actvs=src_datasets[k][si]["actvs"],
                        src_prompt_len=src_datasets[k][si]["prompt_len"],
                        tokenizer=tokenizers[ti],
                        stepwise=config.get("stepwise", True),
                        ret_cl_data=(si,ti) in config["cl_directions"],
                    )
                    intrv_data = pad_data_dict(
                        intrv_data,
                        src_pad_id=tokenizers[si].pad_token_id,
                        trg_pad_id=tokenizers[ti].pad_token_id,
                        src_pad_side=padding_sides[si],
                        trg_pad_side=padding_sides[ti],
                    )
                    intrv_data = add_pad_masks(
                        intrv_data,
                        src_info={
                            "pad_token_id": getattr(tokenizers[si], "pad_token_id"),
                            "bos_token_id": getattr(tokenizers[si], "bos_token_id"),
                            "eos_token_id": getattr(tokenizers[si], "eos_token_id"),
                        },
                        trg_info={
                            "pad_token_id": getattr(tokenizers[ti], "pad_token_id"),
                            "bos_token_id": getattr(tokenizers[ti], "bos_token_id"),
                            "eos_token_id": getattr(tokenizers[ti], "eos_token_id"),
                        },
                    )
                    print("Intrv Src:", intrv_data["src_input_ids"][0])
                    print("Intrv Decode Src:", tokenizers[si].decode(intrv_data["src_input_ids"][0]))
                    print("Intrv Trg:", intrv_data["trg_input_ids"][0])
                    print("Intrv Decode Trg:", tokenizers[ti].decode(intrv_data["trg_input_ids"][0]))
                    print()
                    print()
                    intrv_data = convert_to_tensors(intrv_data)
                    all_src_activations[k][dir_var_tup] = intrv_data["src_actvs"]
                    intrv_datasets[k][dir_var_tup] = Dataset.from_dict(intrv_data)

    # Create a DataLoader that iterates over indices of the filtered dataset.
    indices = list(range(trn_size))
    train_loader = DataLoader(
        indices,
        batch_size=config["batch_size"],
        shuffle=True
    )

    indices = list(range(val_size))
    valid_loader = DataLoader(
        indices,
        batch_size=config["eval_batch_size"],
        shuffle=True
    )

    ####################################################
    #    Collect BASELINE LOSS AND ACC
    ####################################################

    criterion = torch.nn.CrossEntropyLoss()
    baseline_metrics = {k: {
        "loss": dict(),
        "acc": dict(),
    } for k in intrv_datasets}
    print("Baselines:")
    with torch.no_grad():
        for k in intrv_datasets:
            for sidx in range(n_models):
                for tidx in range(n_models):
                    for vidx in range(n_varbs):
                        dir_var_tup = (sidx,tidx,vidx)
                        input_ids = intrv_datasets[k][dir_var_tup]["src_input_ids"]
                        tmask = intrv_datasets[k][dir_var_tup]["src_task_masks"]
                        logits = src_datasets[k][sidx]["logits"]
                        preds = logits.argmax(-1)

                        loss = torch.zeros_like(input_ids[:,1:])
                        temp = criterion(logits[:,:-1], input_ids[:,1:], reduction="none")
                        loss[tmask] = temp[tmask]
                        loss = loss.sum(-1)/tmask.sum(-1)

                        acc = torch.zeros_like(input_ids[:,1:])
                        temp = (preds[:,:-1]==input_ids[:,1:]).float()
                        acc[tmask] = temp[tmask]
                        acc = acc.sum(-1)/tmask.sum(-1)

                        baseline_metrics[k]["loss"][dir_var_tup] = loss
                        baseline_metrics[k]["acc"][dir_var_tup] = acc
                        print(f"Tup: {sidx},{tidx},{vidx}; Loss: {loss.mean():.4f}; Acc: {acc.mean():.4f}")

    ####################################################
    #    Collect CL VECTORS
    ####################################################

    with torch.no_grad():
        cl_vectors = {k:dict() for k in intrv_datasets}
        print("Collecting CL Activations")
        for k in intrv_datasets:
            for dirvar_tup in intrv_datasets[k].keys():
                src_idx,trg_idx,varb_idx = dirvar_tup
                cl_vectors[k][dirvar_tup] = None
                if (src_idx,trg_idx) not in config["cl_directions"]:
                    continue

                startt = time.time()
                trg_model = models[trg_idx].eval()
                device = devices[trg_idx]
                print("Model", trg_idx, config["model_names"][trg_idx])
                batch = collate_fn(
                    torch.arange(trn_size).long(),
                    intrv_datasets[k][dirvar_tup],
                    incl_src=True,
                    device="cpu")

                ## Collect cl latents by generating them from cl sequences
                ## paired with cl indices to pick out the correct latents.
                cl_vectors[k][dirvar_tup] = get_cl_latents(
                    model=trg_model,
                    device=device,
                    swap_mask=batch["trg_swap_masks"]>=0,
                    input_ids=batch["cl_input_ids"],
                    idxs=batch["cl_idxs"],
                    layer=config["layers"][trg_idx],
                )
                print("Exec Time:", time.time()-startt)
                print()

    ##########################
    #    Define the intervention object, optimizer, and plateau tracker
    ##########################
    if config.get("mtx_kwargs", None) is None:
        mtx_kwarg_keys = {
            "rank", "identity_init", "bias", "mu",
            "sigma", "identity_rot", "orthogonal_map",
            "nonlin_align_fn", "n_layers",
        }
        mtx_kwargs = dict()
        for key in mtx_kwarg_keys:
            if key in config:
                mtx_kwargs[key] = config[key]
        config["mtx_kwargs"] = [mtx_kwargs for _ in models]
    config["sizes"] = m_sizes
    intrv_module = InterventionModule( **config, )
    intrv_module.eval()
    optimizer = torch.optim.Adam(
        intrv_module.parameters(),
        lr=config["lr"])
    plateau_tracker = PlateauTracker(**config)

    ##########################
    #    Z-Score the Source Activations and/or start rot matrices from PCA
    #    (if specified in the config)
    ##########################
    if config["zscore_alignment"] or config.get("pca_init",False):
        for midx in range(len(models)):
            for dirvar_tup in all_src_activations["train"].keys():
                sidx,tidx,vidx = dirvar_tup
                if sidx!=midx:
                    continue

                actvs = all_src_activations["train"][dirvar_tup]
                if config["nonlin_align_fn"]!="identity":
                    actvs = intrv_module.rot_mtxs[midx].nonlin_fwd(actvs)
                actvs = actvs.reshape(-1, actvs.shape[-1])
                break

            mu = 0
            std = 1
            if config["zscore_alignment"]:
                mu =  actvs.mean(0)
                std = actvs.std(0)
                intrv_module.set_normalization_params(
                    midx=midx, mu=mu, sigma=std, )
            if config.get("pca_init", False):
                actvs = (actvs - mu) / std
                ret_dict = perform_eigen_pca(
                    X=actvs,
                    n_components=actvs.shape[-1],
                    center=False, scale=False,
                )
                # Dims (features, components) after transpose
                pca_matrix = torch.tensor(ret_dict["components"].T, device=devices[midx])
                intrv_module.solve_and_set_rotation_matrix(
                    midx=midx, target_mtx=pca_matrix, verbose=True,
                )
            

    ##########################
    #    Define and attach forward hooks to a specified layer in each model.
    #    The hook for model 1 applies the rotation (matrix multiplication);
    #    the hook for model 2 applies the transpose.
    ##########################
    comms_dict = {
        "intrv_module": intrv_module,
        "src_activations": None,
        "src_idx": 0,
        "trg_idx": 1,
        "varb_idx": None,
    }
    if config.get("stepwise", True):
        hook_fns = [get_stepwise_hook(comms_dict) for _ in models]
    else:
        hook_fns = [get_indywise_hook(comms_dict) for _ in models]
    hook_modules = [
        get_hook_module(model, config["layers"][mi])
            for mi,model in enumerate(models)
    ]
    hook_handles = [hmod.register_forward_hook(hfn) for hmod,hfn in zip(hook_modules,hook_fns)]
    
    ##########################
    #    Training loop: adjust the rotation matrix so that the models (with hooked activations)
    #    autoregressively predict the (filtered) training text.
    #    (Since the underlying models are frozen, only the rotation is updated.)
    ##########################
    global_step = 0
    print("Starting training of the rotation matrix ...")
    optimizer.zero_grad()
    df_dict = {
        "global_step": [],
        "train_loss": [],
        "train_tok_acc": [],
        "train_trial_acc": [],
        "train_ploss": [],
        "train_ptok_acc": [],
        "valid_loss": [],
        "valid_tok_acc": [],
        "valid_trial_acc": [],
        "valid_tok_acc": [],
        "valid_ptok_acc": [],
        "cl_loss": [],
        "cl_divergence": [],
        "src_idx": [],
        "trg_idx": [],
        "varb_idx": [],
        "varb_name": [],
    }
    end_training = False
    try:
        while global_step < config["num_training_steps"] and not end_training:
            for batch_indices in train_loader:
                # Forward passes. The hook functions will transform activations at the chosen layer.
                losses = dict()
                trial_accs = dict()
                tok_accs = dict()
                plosses = dict()   # proportional losses
                ptok_accs = dict() # proportional accs
                tot_loss = 0
                tot_tok = 0
                tot_trial = 0
                tot_ploss = 0
                tot_ptok = 0

                val_losses = dict()
                val_trial_accs = dict()
                val_tok_accs = dict()
                val_plosses = dict()   # proportional losses
                val_ptok_accs = dict() # proportional accs
                val_cl_loss = dict()
                val_cl_div = dict()
                val_cl_sdx = dict() # Amount that the max value of the
                    # cl_vectors exceeds the max value and the natural 
                    # latents max value measured in standard deviations 
                    # from the natural mean
                
                # Reset the tracking dictionaries for this step.
                tracking_dicts = [
                    losses, trial_accs, tok_accs,
                    val_losses, val_trial_accs, val_tok_accs,
                    val_cl_loss, val_cl_div, val_cl_sdx,
                    plosses, ptok_accs, val_plosses, val_ptok_accs,
                ]
                for tidx in range(len(models)):
                    for sidx in range(len(models)):
                        for vidx in range(len(config["swap_keys"][sidx])):
                            dirvar_tup = (sidx, tidx, vidx)
                            for d in tracking_dicts:
                                d[dirvar_tup] = 0

                startt = time.time()
                accum = config.get("grad_accumulation_steps", 1)
                for dirvar_tup in intrv_datasets["train"]:
                    models = [model.train() for model in models]
                    runtime = time.time()
                    (sidx,tidx,vidx) = dirvar_tup
                    track_train = (sidx,tidx) in config["train_directions"]
                    track_cl = (sidx,tidx) in config["cl_directions"]
                    track_grad = track_train or track_cl
                    loss, cl_loss, tok_acc, trial_acc, ploss, ptok = forward_pass(
                        sidx=sidx,
                        tidx=tidx,
                        vidx=vidx,
                        model=models[tidx],
                        comms_dict=comms_dict,
                        batch_indices=batch_indices,
                        dataset=intrv_datasets["train"][dirvar_tup],
                        src_activations=all_src_activations["train"][dirvar_tup],
                        cl_vectors=cl_vectors["train"][dirvar_tup],
                        device=devices[tidx],
                        config=config,
                        tforce=True,
                        track_grad=track_grad,
                        baseline_accs=baseline_metrics["train"]["acc"][dirvar_tup],
                        baseline_losses=baseline_metrics["train"]["loss"][dirvar_tup]
                    )
                    cl_loss = cl_loss/accum/(len(models)**2)/n_varbs
                    loss = loss/accum/(len(models)**2)/n_varbs
                    combo_loss = torch.zeros_like(loss)
                    if track_train: combo_loss = loss
                    if track_cl:
                        eps = config.get("cl_eps",1)
                        combo_loss = combo_loss + eps*cl_loss

                    if config["conserve_memory"] and track_grad:
                        n_tups = len(list(intrv_datasets["train"].keys()))
                        (combo_loss/float(n_tups)).backward()

                    losses[dirvar_tup] = loss.item()
                    tot_loss += combo_loss.to(devices[0])

                    tot_trial += trial_acc.item()/(len(models)**2)
                    tot_tok += tok_acc.item()/(len(models)**2)
                    tot_ploss += ploss.item()/(len(models)**2)
                    tot_ptok += ptok.item()/(len(models)**2)
                    trial_accs[dirvar_tup] = trial_acc.item()
                    tok_accs[dirvar_tup] = tok_acc.item()
                    plosses[dirvar_tup] = tot_ploss.item()
                    ptok_accs[dirvar_tup] = tot_ptok.item()
                    print("Loss:", round(loss.item(), 5),
                        "- Time:", round(time.time()-runtime,5),
                        "- Step:", round(global_step),
                        end="                  \r"
                    )

                    # Print a sample generation every print_every steps.
                    if global_step % config["print_every"] == 0:
                        models = [model.eval() for model in models]
                        ####################################################
                        #### VALIDATION
                        ####################################################
                        print("\n\nSource Model", sidx, "- Target Model", tidx, "- Varbl:", vidx)
                        print("Validating...")
                        val_loss = 0
                        val_tok = 0
                        val_trial = 0
                        val_ploss = 0
                        val_ptok = 0
                        for val_indices in valid_loader:
                            vloss, vcl_loss, vtok, vtrial, vcl_div, vcl_sdx, vploss, vptok = forward_pass(
                                sidx=sidx,
                                tidx=tidx,
                                vidx=vidx,
                                model=models[tidx],
                                comms_dict=comms_dict,
                                batch_indices=val_indices,
                                dataset=intrv_datasets["valid"][dirvar_tup],
                                src_activations=all_src_activations["valid"][dirvar_tup],
                                cl_vectors=cl_vectors["valid"][dirvar_tup],
                                device=devices[tidx],
                                tokenizer=tokenizers[tidx],
                                config=config,
                                verbose=True,
                                tforce=False,
                                track_grad=False,
                                cl_divergence=True,
                                baseline_accs=baseline_metrics["valid"]["acc"][dirvar_tup],
                                baseline_losses=baseline_metrics["valid"]["loss"][dirvar_tup],
                            )
                            val_loss  += vloss.item() /len(valid_loader)
                            val_tok   += vtok.item()  /len(valid_loader)
                            val_trial += vtrial.item()/len(valid_loader)
                            val_ploss  += vploss.item() /len(valid_loader)
                            val_ptok   += vptok.item()  /len(valid_loader)
                        val_losses[dirvar_tup] = val_loss
                        val_tok_accs[dirvar_tup] = val_tok
                        val_trial_accs[dirvar_tup] = val_trial
                        val_plosses[dirvar_tup] = val_ploss
                        val_ptok_accs[dirvar_tup] = val_ptok
                        val_cl_loss[dirvar_tup] = vcl_loss.item() if vcl_loss is not None else 0
                        val_cl_div[dirvar_tup] =  vcl_div if vcl_div is not None else 0
                        val_cl_sdx[dirvar_tup] = vcl_sdx if vcl_sdx is not None else 0

                if not config["conserve_memory"]:
                    tot_loss.backward()
                if global_step % accum==0:
                    optimizer.step()
                    optimizer.zero_grad()

                end_training = False
                if global_step % config["print_every"] == 0:
                    print("Layers:", config["layers"])
                    print("\tSwap Keys:", config["swap_keys"])
                    print("Mtx  Type:", config["mtx_types"][0])
                    print("\tAlignFn:", config.get("nonlin_align_fn","identity"))
                    print("Mask Type:", type(intrv_module.swap_mask).__name__,
                            "- FSR:", config["fsr"],
                            "- Const Inpt:", config["const_targ_inpt_id"],
                            "- Units:", intrv_module.swap_mask.n_units)
                    print("Trn Dirs:",
                        " ".join(sorted(
                            [str(d) for d in config["train_directions"]])))
                    print("CL Dirs:",
                        " ".join(sorted(
                            [str(d) for d in config["cl_directions"]])))
                    print("\tCL Eps:", config.get("cl_eps", 0))
                    print("Step:", global_step, "| Train pLoss:", tot_ploss.item())
                    print()
                    for vidx in range(n_varbs):
                        print("Varbl", vidx, config["swap_keys"][sidx][vidx])

                        print("Valid CL Loss:")
                        s = "\tM1->M1: " + str(round(val_cl_loss[(0,0,vidx)], 5))
                        if len(models)>1:
                            s += " | M1->M2: " + str(round(val_cl_loss[(0,1,vidx)],5))
                            s += "\n\tM2->M1: " + str(round(val_cl_loss[(1,0,vidx)], 5))
                            s += " | M2->M2: " + str(round(val_cl_loss[(1,1,vidx)],5))
                        print(s)

                        max_diff = max([v for v in val_cl_sdx.values()])
                        print(f"Valid CL Divergence: (Excess in SDs: {max_diff})")
                        s = "\tM1->M1: " + str(round(val_cl_div[(0,0,vidx)], 5))
                        if len(models)>1:
                            s += " | M1->M2: " + str(round(val_cl_div[(0,1,vidx)],5))
                            s += "\n\tM2->M1: " + str(round(val_cl_div[(1,0,vidx)], 5))
                            s += " | M2->M2: " + str(round(val_cl_div[(1,1,vidx)],5))
                        print(s)

                        print("Train PTok Acc:",  tot_ptok)
                        s = "\tM1->M1: " + str(round(ptok_accs[(0,0,vidx)], 5))
                        if len(models)>1:
                            s += " | M1->M2: " + str(round(ptok_accs[(0,1,vidx)],5))
                            s += "\n\tM2->M1: " + str(round(ptok_accs[(1,0,vidx)], 5))
                            s += " | M2->M2: " + str(round(ptok_accs[(1,1,vidx)],5))
                        print(s)

                        # print("Train Trial Acc:",tot_trial)
                        # s = "\tM1->M1: " + str(round(trial_accs[(0,0,vidx)], 5))
                        # if len(models)>1:
                        #     s += " | M1->M2: " + str(round(trial_accs[(0,1,vidx)],5))
                        #     s += "\n\tM2->M1: " + str(round(trial_accs[(1,0,vidx)], 5))
                        #     s += " | M2->M2: " + str(round(trial_accs[(1,1,vidx)],5))
                        # print(s)
                        print()

                        print("Valid Tok Acc:")
                        s = "\tM1->M1: " + str(round(val_tok_accs[(0,0,vidx)], 5))
                        if len(models)>1:
                            s += " | M1->M2: " + str(round(val_tok_accs[(0,1,vidx)],5))
                            s += "\n\tM2->M1: " + str(round(val_tok_accs[(1,0,vidx)], 5))
                            s += " | M2->M2: " + str(round(val_tok_accs[(1,1,vidx)],5))
                        print(s)

                        print("Valid Trial Acc:")
                        s = "\tM1->M1: " + str(round(val_trial_accs[(0,0,vidx)], 5))
                        if len(models)>1:
                            s += " | M1->M2: " + str(round(val_trial_accs[(0,1,vidx)],5))
                            s += "\n\tM2->M1: " + str(round(val_trial_accs[(1,0,vidx)], 5))
                            s += " | M2->M2: " + str(round(val_trial_accs[(1,1,vidx)],5))
                        print(s)
                        print()

                        print()

                    print()
                    print("Layers:", config["layers"])
                    print("\tSwap Keys:", config["swap_keys"])
                    print("Mtx  Type:", config["mtx_types"][0])
                    print("\tAlignFn:", config.get("nonlin_align_fn","identity"))
                    print("Mask Type:", type(intrv_module.swap_mask).__name__,
                            "- FSR:", config["fsr"],
                            "- Const Inpt:", config["const_targ_inpt_id"],
                            "- Units:", intrv_module.swap_mask.n_units,
                            "- ZScore:", config["zscore_alignment"],
                            "- PCA Init:", config["pca_init"],
                    )
                    print("Trn Dirs:",
                        " ".join(sorted(
                            [str(d) for d in config["train_directions"]])))
                    print("CL Dirs:",
                        " ".join(sorted(
                            [str(d) for d in config["cl_directions"]])))
                    print("\tCL Eps:", config.get("cl_eps", 0))
                    print("Step:", global_step, "| Train pLoss:", tot_ploss.item())
                    print()
                    print("Experiment:", os.path.join(save_folder, save_name))
                    print("M1:", config["model_names"][0])
                    if len(config["model_names"])>1:
                        print("M2:", config["model_names"][1])
                    print("Exec Time:", time.time()-startt)
                    print()

                    for (s,t,v) in intrv_datasets["train"]:
                        tup = (s,t,v)
                        df_dict["global_step"].append(global_step)
                        df_dict["train_loss"].append(float(losses[tup]))
                        df_dict["train_ploss"].append(float(plosses[tup]))
                        df_dict["train_tok_acc"].append(float(tok_accs[tup]))
                        df_dict["train_ptok_acc"].append(float(ptok_accs[tup]))
                        df_dict["train_trial_acc"].append(float(trial_accs[tup]))
                        df_dict["valid_loss"].append(float(val_losses[tup]))
                        df_dict["valid_ploss"].append(float(val_plosses[tup]))
                        df_dict["valid_tok_acc"].append(float(val_tok_accs[tup]))
                        df_dict["valid_ptok_acc"].append(float(val_ptok_accs[tup]))
                        df_dict["valid_trial_acc"].append(float(val_trial_accs[tup]))
                        df_dict["cl_loss"].append(float(val_cl_loss[tup]))
                        df_dict["cl_divergence"].append(float(val_cl_div[tup]))
                        df_dict["src_idx"].append(s)
                        df_dict["trg_idx"].append(t)
                        df_dict["varb_idx"].append(v)
                        df_dict["varb_name"].append(config["swap_keys"][t][v])
                    val_loss = np.mean(
                        [float(l) for l in val_losses.values()])
                    vals = [float(l) for l in val_trial_accs.values()]
                    val_acc = np.mean(vals)
                    end_training = plateau_tracker.update(
                        val_loss=val_loss, 
                        val_acc=val_acc)

                    trns = [float(l) for l in trial_accs.values()]
                    trn_min = np.min(trns)
                    val_min = np.min(vals)
                    m = config.get("upper_acc_thresh", 0.99)
                    end_training = end_training or (val_min>=m and trn_min>=m)
                    if (val_min<0.1 and global_step>=1500):
                        print("Stopping due to poor performance!")
                        end_training = True

                
                ### Save loss and state dict
                svsteps = config.get("save_every_steps", 100)
                if config.get("debugging", False) and global_step%svsteps==0:
                    print("Skipping saving due to debugging flag")
                elif end_training or global_step%svsteps==0:
                    #print("Saving To", os.path.join(save_folder, save_name))
                    csv = os.path.join(save_folder, save_name + ".csv")
                    df = pd.DataFrame(df_dict)
                    df.to_csv(csv, header=True, index=False)

                    pt = os.path.join(save_folder, save_name + ".pt")
                    sd = {
                        "config": config,
                        "state_dict": intrv_module.state_dict(),
                    }
                    torch.save(sd, pt)

                ### Stop training
                global_step += 1
                if global_step >= config["num_training_steps"]:
                    break
                if end_training:
                    print("Early stopping due to performance plateau!!")
                    break
    except KeyboardInterrupt:
        print("Interrupted training!!")

    ##########################
    # 9. Clean up: remove hooks.
    ##########################
    for handle in hook_handles:
        handle.remove()
    print("Training complete.")

if __name__ == "__main__":
    main()

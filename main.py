import sys
import os
import torch
import numpy as np
import time
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset, load_from_disk
import torch.nn.functional as F

from datas import (
    get_dataset, tokenize_dataset, ensure_equal_length,
    collate_fn, default_replacement_dict,
    make_tokenized_info, add_token_ids_to_info
)
from utils import (
    collect_activations, device_fxn, get_command_line_args,
    default_to_list, tensor2str
)
import seq_models as smods
from dl_utils.save_io import (
    get_save_name, load_checkpoint, get_folder_from_path, save_json, load_yaml,
)
from dl_utils.utils import get_git_revision_hash, get_mask_past_arglast, arglast
from dl_utils.schedulers import PlateauTracker
from dl_utils.tokenizer import Tokenizer
from intrv_modules import InterventionModule
import filters
import causal_models
from intrv_datas import make_intrv_data_from_seqs
from train import make_tokenizer_from_info

import pandas as pd # import after transformers to avoid versioning bug

def config_prep(config):
    n_models = len(config["model_names"])
    config["mtx_kwargs"] = [ {**config} for _ in range(n_models) ]
    config["mask_kwargs"] = {**config}
    config["filters"] = [
        getattr(filters, fname) for fname in config["filter_names"]
    ]
    
    # can assume different cmodels will default to appropriate parameters. This
    # reduces risk of error. Just make a new causal model for new interventions
    kwargs = { "hold_outs": [], }
    config["cmodels"] = [
        getattr(causal_models, cname)(**kwargs) for cname in config["cmodel_names"]
    ]

    if config["swap_keys"] is None:
        config["swap_keys"] = [["full"], ["full"]]
    elif type(config["swap_keys"] )==str:
        skey = config["swap_keys"]
        config["swap_keys"] = [skey for _ in range(n_models)]
    for si,sks in enumerate(config["swap_keys"]):
        if sks[0] is None:
            config["swap_keys"][si] = ["full"]
        elif type(sks)==str:
            config["swap_keys"][si] = [sks]
    
    if config["train_directions"] in {None, "all"}:
        config["train_directions"] = []
        for s in range(n_models):
            for t in range(n_models):
                config["train_directions"].append((s,t))

    if not config.get("stepwise", True):
        config["layers"] = ["inpt_identity" if l=="embeddings" else l for l in config["layers"]]

    if "learning_rate" in config:
        print("use lr instead of learning_rate keyword")
        assert False

    if config.get("debugging", False):
        config["n_train_samples"] = 100
        config["n_valid_samples"] = 100
    return config

def fill_in_prompts_and_replacements(config, yaml_path="./constants.yaml"):
    consts = load_yaml(yaml_path)
    config["prompts"] = []
    config["replacements"] = []
    config["padding_sides"] = []
    for model_name in config["model_names"]:
        print("Model Name:", model_name)
        # Get padding side
        padding_side = consts["padding_sides"].get(model_name, "right")
        config["padding_sides"].append(padding_side)

        # Get prompts
        prompt = consts["prompts"].get(model_name, "")
        if not prompt:
            for k in consts["prompts"]:
                if k in model_name:
                    prompt = consts["prompts"][k]
        config["prompts"].append(prompt)
        print("Prompt:", prompt)

        # Get string replacement dict
        replacements = consts["replacements"].get(
                model_name,
                None
            )
        if not replacements:
            replacements = {**default_replacement_dict}
            for k in consts["replacements"]:
                if k in model_name:
                    replacements = {**replacements, **consts["replacements"][k]}
        config["replacements"].append(replacements)
        print("Replacements:")
        for k,v in replacements.items():
            print(f"\t{k}: {v}")
        print()
    return config

def get_stepwise_hook(comms_dict):
    def hook_fn(module, input, output):
        if "loop_count" not in comms_dict:
            comms_dict["loop_count"] = 0
        # output is assumed to be of shape (batch, seq_length, hidden_size)
        src_idx = comms_dict.get("src_idx",0)
        trg_idx = comms_dict.get("trg_idx",1)
        varb_idx = comms_dict.get("varb_idx",None)

        if hasattr(output,"hidden_states"):
            trg_actvs = output["hidden_states"]
        else:
            trg_actvs = output

        # Prep source vectors
        src_actvs = comms_dict["src_activations"]

        # Handle case where we have a specific swap mask
        trg_swap_mask = None
        if comms_dict.get("trg_swap_masks", None) is not None:
            trg_swap_mask = comms_dict["trg_swap_masks"]
            src_swap_mask = comms_dict["src_swap_masks"]

        comms_dict["loop_count"] += 1

        if trg_swap_mask is not None:
            placeholder = torch.empty_like(trg_actvs)
            placeholder[~trg_swap_mask] = trg_actvs[~trg_swap_mask]
            src_actvs = src_actvs[src_swap_mask]
            trg_actvs = trg_actvs[trg_swap_mask]

        intrv_module = comms_dict["intrv_module"]
        outs = intrv_module(
            target=trg_actvs,
            source=src_actvs,
            target_idx=trg_idx,
            source_idx=src_idx,
            varb_idx=varb_idx,
        )

        if trg_swap_mask is not None:
            placeholder[trg_swap_mask] = outs
            outs = placeholder

        if hasattr(output,"hidden_states"):
            output["hidden_states"] = outs
            return output
        else:
            return outs

    return hook_fn

def get_indywise_hook(comms_dict):
    def hook(module, inp, out):
        """
        out: tensor (B,M,D)
            the mamba recurrent states where M is the number of SSM
            states
        """
        h = out
        if type(out)==dict:
            h = h["hidden_states"]
        device = device_fxn(h.get_device())
        og_h_shape = h.shape
        intrv_modu = comms_dict["intrv_module"]
        src_idx = comms_dict.get("src_idx",0)
        trg_idx = comms_dict.get("trg_idx",1)
        varb_idx = comms_dict.get("varb_idx",None)

        #if comms_dict["pad_mask"] is None:
        #    pad_mask = torch.ones(len(h)).bool().to(device)
        #else:
        #    # assumes mask denotes pad ids as true
        #    pad_mask = ~(comms_dict["pad_mask"].bool()).to(device)
        #    pad_mask = pad_mask[:,comms_dict["loop_count"]]

        #if pad_mask.long().sum()==0:
        #    return out

        # General to multi-dimensional states or single vector states
        source_actvs = comms_dict["src_activations"]
        B,S = source_actvs.shape[:2]
        source_actvs = source_actvs.reshape(B,S,-1)
        source_actvs = source_actvs.to(device)

        #print("B,S:", B,S)
        #print("og_out shape:", og_h_shape)
        #print("pre pad:", comms_dict[source_actvs_key].shape)
        #print("pad:", pad_mask.shape)
        #print("ptype:", type(pad_mask), pad_mask.dtype)
        #print("h:", h.shape, type(h), h.dtype)

        # Get positional indices of the interchange for each sample in
        # the batch.
        source_seq_idxs = comms_dict["src_swap_idxs"].long()
        trg_seq_idxs = comms_dict["trg_swap_idxs"].long()
        batch_bools = trg_seq_idxs==comms_dict["loop_count"]
        h = h.reshape(B,-1) # assume no seq dim
        intr_out = h.clone()

        comms_dict["loop_count"] += 1
        if batch_bools.float().sum()==0:
            h = h.reshape(og_h_shape)
            if type(out)==dict:
                out["hidden_states"] = h
                h = out
            return h

        # Get appropriate inputs for interchange
        idxs = torch.arange(len(batch_bools)).long().to(device)
        idxs = idxs[batch_bools]
        #trg_idxs = trg_seq_idxs[batch_bools]
        source_idxs = source_seq_idxs[batch_bools]

        trg_inpts = h[idxs]
        source_inpts = source_actvs[idxs, source_idxs]

        #print("source_idxs:", source_idxs.shape)
        #print("h:", h.shape)
        #print("sactvs:", source_actvs.shape)
        #print("trg_inpts:", trg_inpts.shape)
        #print("source_inpts:", source_inpts.shape)

        # Perform causal interchange
        outs = intrv_modu(
            target=trg_inpts,
            source=source_inpts.to(device),
            target_idx=trg_idx,
            source_idx=src_idx,
            varb_idx=varb_idx,)

        ## If auxiliary targets are argued, then use them as a constraint
        ## on the intervened vectors.
        #if comms_dict.get(aux_targs_key, None) is not None:
        #    comms_dict[aux_loss_key] = aux_loss_fxn(
        #        outs,
        #        comms_dict[aux_targs_key][idxs].to(device),
        #        targs=torch.ones(len(outs)).to(device),
        #    )

        # Place causally intervened outputs into appropriate locations
        # in original output tensor. We do it this way to avoid auto-grad
        # errors for in-place operations
        intr_out[idxs] = 0
        intr_out[idxs] += outs

        intr_out = intr_out.reshape(og_h_shape)
        if type(out)==dict:
            out["hidden_states"] = intr_out
            intr_out = out
        return intr_out
    return hook

# Helper: get the module corresponding to the chosen layer.
def get_hook_module(model, hook_layer):
    if type(hook_layer)==str: # optionally specify module name string
        for name,modu in model.named_modules():
            if name==hook_layer:
                return modu
    # For LLaMA-style models the transformer layers might be stored in model.model.layers or model.transformer.h.
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[hook_layer]
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[hook_layer]
    else:
        raise ValueError("Cannot locate hook layer in the model.")

def get_model_and_tokenizer(model_name, padding_side="left"):
    print(f"Loading model and tokenizer for {model_name}...")
    try:
        checkpt = load_checkpoint(model_name)
        mconfig = checkpt["config"]
        temp = smods.make_model(mconfig)
        temp.load_state_dict(checkpt["state_dict"])
        model = temp.model
        if "word2id" in mconfig:
            print("word2id:", mconfig["word2id"])
            tokenizer = Tokenizer(
                word2id=mconfig["word2id"],
                **mconfig["info"],
                padding_side=padding_side)
        elif "info" in mconfig:
            tokenizer = make_tokenizer_from_info(mconfig["info"])
        else:
            tokenizer = Tokenizer(
                words=set(),
                unk_token=None,
                word2id=None,
                padding_side=padding_side)
    except:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                padding_side=padding_side)
        except:
            model_name = "/".join(model_name.split("/")[-2:])
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                padding_side=padding_side)
        if not tokenizer.pad_token:
            tokenizer.pad_token = "<PAD>"
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("okay")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto")
    model.eval()
    return model, tokenizer, mconfig

def forward_pass(
        sidx,
        tidx,
        model,
        batch_indices,
        dataset,
        comms_dict,
        src_activations,
        device,
        tokenizer=None,
        pad_mask=None,
        config=dict(),
        verbose=False,
        vidx=None,
    ):
    shuffle_targ_ids = config.get("shuffle_targ_ids", False)
    const_targ_inpt_id = config.get("const_targ_inpt_id", False)
    ## Get batch
    batch = collate_fn( batch_indices, dataset, device=device)

    ## Set Comms Dict Values
    comms_dict["src_idx"] = sidx
    comms_dict["trg_idx"] = tidx
    comms_dict["varb_idx"] = vidx
    comms_dict["loop_count"] = 0
    comms_dict["intrv_module"].to(device)
    comms_dict["intrv_module"].reset()
    comms_dict["src_activations"] =\
        src_activations[batch_indices].to(device)
    input_ids = batch["input_ids"].clone()

    mask = None
    if "trg_swap_masks" in batch:
        comms_dict["trg_swap_masks"] = batch["trg_swap_masks"]
        comms_dict["src_swap_masks"] = batch["src_swap_masks"]
        mask = batch["trg_swap_masks"]
    if mask is not None and config.get("stepwise", True):
        if const_targ_inpt_id:
            resp_id = config.get("resp_id", 6)
            input_ids[mask] = int(resp_id)
        elif shuffle_targ_ids:
            # Shuffles the input ids
            msums = mask.long().sum(-1)
            perms = [torch.randperm(s).long() for s in msums]
            perm = [perms[i+1]+len(perms[i]) for i in range(len(perms)-1)]
            perm = torch.cat([perms[0]] + perm)
            input_ids[mask] = input_ids[mask][perm.to(device)]
    if "trg_swap_idxs" in batch:
        ssm = batch["src_swap_idxs"].to(device)
        comms_dict["src_swap_idxs"] = ssm
        tsm = batch["trg_swap_idxs"].to(device)
        comms_dict["trg_swap_idxs"] = tsm

    ## Run model
    if config.get("stepwise", True):
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["inpt_attn_mask"],
        )
    else:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["inpt_attn_mask"],
            task_mask=batch["input_tmask"],
            tforce=False,
        )

    # Calc Loss
    if "logits" in outputs:
        logits = outputs["logits"]
    else:
        logits = outputs.logits

    V = logits.shape[-1]
    flat = logits.reshape(-1,V)
    labels = batch["labels"].reshape(-1)
    lmask = batch["outp_attn_mask"]
    if "trg_swap_masks" in batch and config.get("stepwise", True):
        smask = torch.roll(~batch["trg_swap_masks"], -1, dims=-1)
        smask[...,-1] = True
        lmask = lmask&(smask)

    ### TODO
    #pids = torch.argmax(logits, dim=-1)
    #print("HEYO")
    #for i in range(3):
    #    print("Swaps:", tensor2str(batch["labels"][i][~lmask[i]]))
    #    print("Preds:", tensor2str(pids[i][lmask[i]]))
    #    print("Label:", tensor2str(batch["labels"][i][lmask[i]]))
    #    print()

    loss = F.cross_entropy(
        flat[lmask.reshape(-1)],
        labels[lmask.reshape(-1)]
    )

    if "outp_tmask" in batch:
        tmask = batch["outp_tmask"].to(device)
    else:
        tmask = batch["outp_attn_mask"]
    trial = torch.ones_like(batch["labels"]).bool()
    pids = torch.argmax(logits, dim=-1)
    labels = batch["labels"]
    eq = pids[tmask]==labels[tmask]
    trial[tmask] = eq
    trial_acc = trial.sum(-1)==trial.shape[-1]
    trial_acc = trial_acc.float().mean()
    tok_acc = eq.float().mean()

    if verbose:
        labels = batch["labels"]
        inpts = batch["input_ids"]
        outs = torch.argmax(logits, dim=-1)#[perm[:2]]
        if pad_mask is None and "inpt_attn_mask" in batch:
            pmask = batch["inpt_attn_mask"]
        else:
            pmask = ~pad_mask[batch_indices]
        input_mask = pmask
        if "input_tmask" in batch:
            input_mask = pmask&(~batch["input_tmask"])
        if "outp_tmask" in batch:
            tmask = batch["outp_tmask"]
        else:
            tmask = batch["outp_attn_mask"]

        trg_pad_id =  tokenizer.pad_token_id
        trg_pad_tok = tokenizer.pad_token

        for i in range(min(2,len(outs))):
            idx_range = torch.arange(len(inpts[i]))
            src_swap = arglast(batch["src_swap_masks"][i])
            trg_swap = arglast(batch["trg_swap_masks"][i])
            print("Src Swap", int(src_swap), "- Trg Swap", int(trg_swap))
            print("Idx   :", tensor2str(idx_range))
            print("Src   :", tensor2str(batch["src_input_ids"][i]))
            print("Trg   :", tensor2str(inpts[i]))
            print("Preds :", tensor2str(outs[i]))
            print("Labels:", tensor2str(labels[i]))
            print("TrnLab:", tensor2str(labels[i][lmask[i]].long()))
            print("OuTmsk:", tensor2str(batch["outp_tmask"][i].long()))
            print("TrgSwp:", tensor2str(batch["trg_swap_masks"][i].long()))
            print("LosMsk:", tensor2str(lmask[i].long()))
            print()
            print("Inpts:", tensor2str(inpts[i][:trg_swap]))
            print("Gtrth:", tensor2str(labels[i][trg_swap:]))
            print("Preds:", tensor2str(outs[i][trg_swap:]))
            # Input Text
            input_text = tokenizer.decode(inpts[i][:trg_swap+1])
            if type(input_text)!=str:
                input_text = input_text[0]
            input_text = input_text.replace(trg_pad_tok, "")

            # Target Text
            target_text = tokenizer.decode(labels[i][trg_swap:])
            if type(target_text)!=str:
                target_text = target_text[0]
            target_text = target_text.replace(trg_pad_tok, "")

            # Generated Text
            generated_text = tokenizer.decode(outs[i][trg_swap:])
            if type(generated_text)!=str:
                generated_text = generated_text[0]
            generated_text = generated_text.replace(trg_pad_tok, "")

            if shuffle_targ_ids:
                print("Shuffled Input IDs")
            print("Input    :", input_text.replace("\n", "\\n")\
                .replace("<BOS>", "B")\
                .replace("<EOS>", "E")
            )
            print("Target   :", target_text.replace("\n", "\\n")\
                .replace("<BOS>", "B")\
                .replace("<EOS>", "E")
            )
            print("Generated:", generated_text.replace("\n", "\\n")\
                .replace("<BOS>", "B")\
                .replace("<EOS>", "E")
            )
            print()
            #print("TrgIds:", labels[i][tmask[i]])
            #print("GenIds:", outs[i][tmask[i]])
            #print()

    return loss, tok_acc, trial_acc

def get_embedding_name(model, layer=""):
    """
    This function serves to unify the layer naming amongst different
    model types.

    Args:
        model: torch Module
    """
    simplist_name = ""
    shortest_len = np.inf
    for name, modu in model.named_modules():
        if type(modu)==torch.nn.Embedding or "Embedding" in str(type(modu)):
            if name==layer: return name
            if len(name.split("."))<shortest_len:
                shortest_len = len(name.split("."))
                simplist_name = name
    return simplist_name

def main():
    arg_config = get_command_line_args(sys.argv)
    ##########################
    #    Default configuration
    ##########################
    defaults = {
        "save_root": "/data2/grantsrb/icml_mas/",
        "exp_name": "myexp",
        "conserve_memory": True,

        # Use two identical models by default (replace with real LLaMA repo names as needed)
        "model_names": [
            #"meta-llama/Llama-3.2-1B",
            "gpt2",
            "gpt2",
        ], #[, "gpt2"], #

        "dataset_names": [
            "task", "task"
        ],
        "n_train_samples": 10000, # sample counts only apply if using task generated
            # dataset
        "n_valid_samples": 1000,
        "dataset_kwargs": [
            {"name": "main", "split":"train", } for _ in range(2)
        ],
        "task_kwargs": [{} for _ in range(2)],
        "filter_by_correct": False,
        "filtered_dataset_paths": [
            "./data/filtered_gsm8k",  # where to save/load the filtered dataset
            "./data/filtered_gsm8k",  # where to save/load the filtered dataset
        ],
        "layers": [ # layers at which to attach the hooks
            "embeddings",
            "embeddings"
        ],  
        "cmodel_names": [
            "CountUpDown",
            "CountUpDown",
        ],
        "filter_names": [
            "default_filter",
            "default_filter",
        ],
        "swap_keys": [ ["full"], ["full"] ], # argue a list of
            # keys for each model.
        "incl_empty_varbs": False, # if true, includes an explicit
            # training of the extraneous information, encouraging
            # it to be a null operation.
        "mtx_types": ["RotationMatrix", "RotationMatrix"],
        "identity_init": False,
        "identity_rot": False,
        "mask_type":   "FixedMask", # BoundlessMask
        "n_units": None,
        "learnable_addition": False,
        "const_targ_inpt_id": False, # If true, will use the resp_id for all target input ids
        "fsr": False, # (Functionally sufficient representations) only applies if using fca. Discards the excess components. Equivalent to using a vector of 0s for all input embeddings

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

        "save_keys": ["mtx_types", "layers", "n_units","stepwise", "swap_keys"],
        "debugging": False,
    }
    config = {**defaults}
    config["git_hash"] = get_git_revision_hash()
    for k in arg_config: config[k] = arg_config[k]
    print("Config:")
    for k in sorted(list(config.keys())):
        print(k, config[k])

    config = config_prep(config) # general error catching
    config = fill_in_prompts_and_replacements(config)
    padding_sides = config["padding_sides"]

    save_folder = get_folder_from_path(config["model_names"][0])
    if not os.path.exists(save_folder):
        save_folder = os.path.join(
            config.get("save_root", "./"),
            config["model_names"][0],
        )
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    save_name = get_save_name(
        save_folder=save_folder,
        kwargs=arg_config,
        config=config)
    print("Saving to:", save_folder)

    jpath = os.path.join(save_folder, save_name + ".json")
    if not config.get("debugging", False):
        save_json(config, jpath)

    ##########################
    #    Load two models and tokenizers
    ##########################
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
    for mi,model_name in enumerate(config["model_names"]):
        model, tokenizer, model_config = get_model_and_tokenizer(
            model_name,
            padding_side=padding_sides[mi],
        )
        model.eval()
        model_configs.append(model_config)

        # Freeze model parameters so that only our rotation matrix is trained.
        for param in model.parameters():
            param.requires_grad = False
        print("Model", mi, "-", model_name)
        print(model)
        models.append(model)
        tokenizers.append(tokenizer)
        if config["layers"][mi] in {"embeddings", "inpt_identity"}:
            config["layers"][mi] = get_embedding_name(model)
            print("Decided Layer Name:", config["layers"][mi])

        if hasattr(model, "hf_device_map"):
            if config["layers"][mi] in model.hf_device_map:
                devices.append(model.hf_device_map[config["layers"][mi]])
            else:
                devices.append(model.hf_device_map[""])
        else:
            devices.append(poss_devices[mi])
            model.to(devices[-1])

        # Just collect a single step to determine the dimensionality of
        # the hooked layer
        with torch.no_grad():
            actvs = collect_activations(
                model,
                input_ids=torch.LongTensor([[0]]),
                layers=[config["layers"][mi]],
                batch_size=500,
                to_cpu=True,)
        m_sizes.append(actvs[config["layers"][mi]].shape[-1])

    ####################################################
    #    Load the datasets
    ####################################################
    print("Loading datasets...")
    datasets = { "train": [], "valid": [], }
    for mi in range(len(config["model_names"])):
        for k in datasets:
            n_samples = config[f"n_{k}_samples"]
            dkwargs = {**config["dataset_kwargs"][mi]}
            dkwargs["split"] = k
            dkwargs["data_path"] = config.get(
                f"{k}_data_paths",
                ["./data/multiobj.json", "./data/multiobj.json"]
            )[mi]
            tconfig = model_configs[mi].get("task_config", None)
            if tconfig: tconfig["unk_p"] = 0
            dataset = get_dataset(
                config["dataset_names"][mi],
                n_samples=n_samples,
                task_type=model_configs[mi].get("task_type", None),
                task_config=tconfig,
                **dkwargs)
            datasets[k].append(dataset)
        print("Model", mi)
        print(datasets["train"][mi]["text"][0])
        print(datasets["train"][mi]["task_mask"][0])
    print("Pre Dataset:", datasets["train"][0])

    ####################################################
    #    Tokenize the filtered dataset for autoregressive training.
    #    Here we form an input by concatenating the question and
    #    answer (with a newline and “Answer:” marker).
    ####################################################
    tokenized_datasets = {k: [] for k in datasets}
    infos = []
    for mi,tokenizer in enumerate(tokenizers):
        for k in tokenized_datasets:
            kwrgs = {**config}
            kwrgs["dataset_name"] = kwrgs["dataset_names"][mi]
            kwrgs["replacements"] = kwrgs["replacements"][mi]
            kwrgs["prompt"] = kwrgs["prompts"][mi]
            tokenized_datasets[k].append(
                tokenize_dataset(
                    dataset=datasets[k][mi],
                    tokenizer=tokenizer,
                    config=kwrgs,
                )
            )
            if "info" in model_configs[mi]:
                info = model_configs[mi]["info"]
                if type(info["pad_token_id"])==str:
                    info = add_token_ids_to_info(info=info, tokenizer=tokenizer)
            else:
                info = model_configs[mi].get(
                    "info",
                    make_tokenized_info(
                        replacements=kwrgs["replacements"],
                        tokenizer=tokenizer,
                        config=config)
                )
        infos.append(info)
    config["infos"] = infos
    print("Tok Dataset:", tokenized_datasets["train"][0])
    print("Cmodls:", config["cmodels"])

    ####################################################
    #    Make/Get Intervention Data
    ####################################################
    intrv_datasets = {k: dict() for k in tokenized_datasets }
    n_subspaces = 0
    print("Info:")
    print("1:", config["infos"][0])
    print()
    try:
        print("2:", config["infos"][1])
        print()
    except: pass
    for k in tokenized_datasets:
        for tidx in range(len(tokenized_datasets[k])):
            for sidx in range(len(tokenized_datasets[k])):
                incl_empty = config.get("incl_empty_varbs", False)
                skeys = config["swap_keys"][sidx] + incl_empty*[]
                tkeys = config["swap_keys"][tidx] + incl_empty*[]
                n_varbs = len(skeys)
                z = enumerate(zip(skeys,tkeys))
                for vidx,(src_swap_keys, trg_swap_keys) in z:
                    print(f"Making intrv data - Src{sidx} - Trg{tidx} - Var{vidx}")
                    print("Sample Src:", tokenized_datasets[k][sidx]["input_ids"][0])
                    print("Sample Trg:", tokenized_datasets[k][tidx]["input_ids"][0])
                    intrv_data = make_intrv_data_from_seqs(
                        trg_data=tokenized_datasets[k][tidx],
                        src_data=tokenized_datasets[k][sidx],
                        src_swap_keys=src_swap_keys,
                        trg_swap_keys=trg_swap_keys,
                        src_cmodel=config["cmodels"][sidx],
                        src_info=config["infos"][sidx],
                        src_filter=config["filters"][sidx],
                        trg_cmodel=config["cmodels"][tidx],
                        trg_info=config["infos"][tidx],
                        trg_filter=config["filters"][tidx],
                        stepwise=config.get("stepwise", False),
                    )
                    intrv_datasets[k][(sidx,tidx,vidx)] =\
                        Dataset.from_dict(intrv_data)
    tokenized_datasets = intrv_datasets

    # Create a DataLoader that iterates over indices of the filtered dataset.
    indices = list(range(len(datasets["train"][0])))
    train_loader = DataLoader(
        indices,
        batch_size=config["batch_size"],
        shuffle=True
    )

    indices = list(range(len(datasets["valid"][0])))
    valid_loader = DataLoader(
        indices,
        batch_size=config["eval_batch_size"],
        shuffle=True
    )

    ##########################
    #    Collect Source Activations
    ##########################
    with torch.no_grad():
        all_src_activations = {k:dict() for k in datasets}
        print("Collecting Activations")
        for k in all_src_activations:
            for model_pair in tokenized_datasets[k].keys():
                src_idx,trg_idx,varb_idx = model_pair
                src_model = models[src_idx]
                trg_model = models[trg_idx]
                startt = time.time()
                device = devices[src_idx]
                model_pair = (src_idx, trg_idx, 0) # include 0 for 0 varb idx
                print("Trg Model", trg_idx, config["model_names"][trg_idx])
                print("Src Model", src_idx, config["model_names"][src_idx])
                print("Device:", device)
                vbsize = config.get("eval_batch_size", 128)
                batch = collate_fn(
                    torch.arange(len(tokenized_datasets[k][model_pair])).long(),
                    tokenized_datasets[k][model_pair],
                    incl_src=True,
                    device="cpu")


                ### TODO:
                ##print("Varbl", varb_idx)
                ##for i in range(3):
                ##    indices = torch.arange(len(batch["input_ids"][i])).long()
                ##    print(tensor2str(indices))
                ##    for kk in batch:
                ##        print((kk+" "*(10-len(kk)))[:10], tensor2str(batch[kk][i].long()))
                ##    print()


                actvs = collect_activations(
                    src_model,
                    input_ids=batch["src_input_ids"],
                    attention_mask=batch["src_attention_mask"],
                    task_mask=batch["src_input_tmask"],
                    layers=[config["layers"][src_idx], "lm_head"],
                    tforce=True,
                    ret_pred_ids=True,
                    batch_size=vbsize,
                    to_cpu=True,
                    verbose=True,
                )

                all_src_activations[k][model_pair] =\
                    actvs[config["layers"][src_idx]].squeeze()

                pad_id = tokenizers[src_idx].pad_token_id
                pad_mask = batch["src_input_ids"]==pad_id 
                if "src_outp_tmask" in batch:
                    dword = config["replacements"][src_idx].get("done_word",None)
                    config["resp_id"] = tokenizers[src_idx](
                        config["replacements"][src_idx]["resp_word"])["input_ids"][-1]
                    try: config["resp_id"] = config["resp_id"][-1]
                    except: pass
                    eos_ids = [tokenizers[src_idx].eos_token_id]
                    if hasattr(tokenizers[src_idx], "eos_id"):
                        eos_ids.append(tokenizers[src_idx].eos_id)
                    try:
                        eos_ids.append(int(tokenizers[src_idx](dword)["input_ids"][-1]))
                    except: pass
                    try:
                        eos_ids.append(
                            int(tokenizers[src_idx](" "+dword)["input_ids"][-1]))
                    except: pass
                    eos_ids = torch.LongTensor(eos_ids)
                    in_eos_ids = torch.isin(batch["src_input_ids"].long(),eos_ids)
                    eos_and_tmask = get_mask_past_arglast(in_eos_ids, inclusive=True)
                    pad_mask = pad_mask|eos_and_tmask

                pred_ids = actvs["pred_ids"].squeeze()
                pred_ids[pad_mask] = pad_id

                if "src_outp_tmask" in batch:
                    tmask = batch["src_outp_tmask"].to(device)
                    flat_tmask = tmask.reshape(-1)
                else:
                    tmask = batch["src_outp_attn_mask"].to(device)
                    flat_tmask = tmask.reshape(-1)
                corrects = torch.ones_like(tmask)
                pids = pred_ids.to(device)[tmask]
                tids = batch["src_labels"] .to(device)[tmask]
                idx = pids==tids
                corrects[tmask] = idx
                corrects = corrects.float().sum(-1)==corrects.shape[-1]
                tokacc = (idx).float().mean().item()
                fullacc = corrects.float().mean().item()

                # Generated Text
                idx = 0
                input_text = tokenizers[src_idx].decode(batch["src_input_ids"][idx])
                if type(input_text)!=str:
                    input_text = input_text[0]
                input_text = input_text.replace(tokenizers[src_idx].pad_token, "")
                print("ExIds :", batch["src_input_ids"][idx][:10])
                print(
                    "ExInpt:", input_text.replace("\n", "\\n")\
                                         .replace("<BOS>", "B")\
                                         .replace("<EOS>", "E")
                )

                print(k.capitalize(), "TokAcc:", tokacc)
                print(k.capitalize(), "FullAcc:", fullacc)
                print("Exec Time:", time.time()-startt)
                print()

    ##########################
    #    Define the intervention object, optimizer, and plateau tracker
    ##########################
    config["n_subspaces"] = n_subspaces
    if config.get("mtx_kwargs", None) is None:
        mtx_kwarg_keys = {
            "rank", "identity_init", "bias", "mu",
            "sigma", "identity_rot", "orthogonal_map",
        }
        mtx_kwargs = dict()
        for key in mtx_kwarg_keys:
            if key in config:
                mtx_kwargs[key] = config[key]
        config["mtx_kwargs"] = [mtx_kwargs for _ in models]
    intrv_module = InterventionModule(
        sizes=m_sizes,
        **config,
    )
    intrv_module.eval()
    optimizer = torch.optim.Adam(
        intrv_module.parameters(),
        lr=config["lr"])
    plateau_tracker = PlateauTracker(**config)

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
    models = [model.eval() for model in models]
    optimizer.zero_grad()
    df_dict = {
        "global_step": [],
        "train_loss": [],
        "train_tok_acc": [],
        "train_trial_acc": [],
        "valid_loss": [],
        "valid_tok_acc": [],
        "valid_trial_acc": [],
        "src_idx": [],
        "trg_idx": [],
        "varb_idx": [],
    }
    end_training = False
    try:
        while global_step < config["num_training_steps"] and not end_training:
            for batch_indices in train_loader:
                # Forward passes. The hook functions will transform activations at the chosen layer.
                losses = dict()
                trial_accs = dict()
                tok_accs = dict()
                tot_loss = 0
                tot_tok = 0
                tot_trial = 0

                val_losses = dict()
                val_trial_accs = dict()
                val_tok_accs = dict()
                startt = time.time()
                for dirvar_tup in tokenized_datasets["train"]:
                    runtime = time.time()
                    (sidx,tidx,vidx) = dirvar_tup
                    accum = config.get("grad_accumulation_steps", 1)
                    if (sidx,tidx) in config["train_directions"]:
                        loss, tok_acc, trial_acc = forward_pass(
                            sidx=sidx,
                            tidx=tidx,
                            vidx=vidx,
                            model=models[tidx],
                            comms_dict=comms_dict,
                            batch_indices=batch_indices,
                            dataset=tokenized_datasets["train"][dirvar_tup],
                            src_activations=all_src_activations["train"][dirvar_tup],
                            device=devices[tidx],
                            config=config,
                        )
                        loss = loss/accum/(len(models)**2)
                        if config["conserve_memory"]:
                            n_tups = len(list(tokenized_datasets["train"].keys()))
                            (loss/float(n_tups)).backward()
                    else:
                        with torch.no_grad():
                            loss, tok_acc, trial_acc = forward_pass(
                                sidx=sidx,
                                tidx=tidx,
                                vidx=vidx,
                                model=models[tidx],
                                comms_dict=comms_dict,
                                batch_indices=batch_indices,
                                dataset=tokenized_datasets["train"][dirvar_tup],
                                src_activations=all_src_activations["train"][dirvar_tup],
                                device=devices[tidx],
                                config=config,
                            )
                            loss = loss/accum/(len(models)**2)
                    losses[dirvar_tup] = loss.item()
                    tot_loss += loss.to(devices[0])

                    tot_trial += trial_acc.item()/(len(models)**2)
                    tot_tok += tok_acc.item()/(len(models)**2)
                    trial_accs[dirvar_tup] = trial_acc.item()
                    tok_accs[dirvar_tup] = tok_acc.item()
                    print("Loss:", round(loss.item(), 5),
                        "- Time:", round(time.time()-runtime,5),
                        "- Step:", round(global_step),
                        end="                  \r"
                    )

                    # Print a sample generation every print_every steps.
                    if global_step % config["print_every"] == 0:
                        ####################################################
                        #### VALIDATION
                        ####################################################
                        print("\n\nSource Model", sidx, "- Target Model", tidx, "- Varbl:", vidx)
                        print("Validating...")
                        val_loss = 0
                        val_tok = 0
                        val_trial = 0
                        with torch.no_grad():
                            for val_indices in valid_loader:
                                vloss, vtok, vtrial = forward_pass(
                                    sidx=sidx,
                                    tidx=tidx,
                                    vidx=vidx,
                                    model=models[tidx],
                                    comms_dict=comms_dict,
                                    batch_indices=val_indices,
                                    dataset=tokenized_datasets["valid"][dirvar_tup],
                                    src_activations=all_src_activations["valid"][dirvar_tup],
                                    device=devices[tidx],
                                    tokenizer=tokenizers[tidx],
                                    config=config,
                                    verbose=True,
                                )
                                val_loss  += vloss.item() /len(valid_loader)
                                val_tok   += vtok.item()  /len(valid_loader)
                                val_trial += vtrial.item()/len(valid_loader)
                            val_losses[dirvar_tup] = val_loss
                            val_tok_accs[dirvar_tup] = val_tok
                            val_trial_accs[dirvar_tup] = val_trial

                if not config["conserve_memory"]:
                    tot_loss.backward()
                if global_step % accum==0:
                    optimizer.step()
                    optimizer.zero_grad()

                end_training = False
                if global_step % config["print_every"] == 0:
                    for vidx in range(n_varbs):
                        print("Varbl", vidx)
                        print("Mtx  Type:", config["mtx_types"][0])
                        print("Mask Type:", type(intrv_module.swap_mask).__name__,
                                "- FSR:", config["fsr"],
                                "- Const Inpt:", config["const_targ_inpt_id"],
                                "- Units:", intrv_module.swap_mask.n_units)
                        print()

                        print("Step:", global_step, "| Train Loss:", tot_loss.item())
                        print("Train Tok Acc:",  tot_tok)
                        s = "\tM1->M1: " + str(round(tok_accs[(0,0,vidx)], 5))
                        if len(models)>1:
                            s += "| M1->M2: " + str(round(tok_accs[(0,1,vidx)],5))
                            s += "\n\tM2->M1:" + str(round(tok_accs[(1,0,vidx)], 5))
                            s += "| M2->M2:" + str(round(tok_accs[(1,1,vidx)],5))
                        print(s)

                        print("Train Trial Acc:",tot_trial)
                        s = "\tM1->M1: " + str(round(trial_accs[(0,0,vidx)], 5))
                        if len(models)>1:
                            s += "| M1->M2: " + str(round(trial_accs[(0,1,vidx)],5))
                            s += "\n\tM2->M1:" + str(round(trial_accs[(1,0,vidx)], 5))
                            s += "| M2->M2:" + str(round(trial_accs[(1,1,vidx)],5))
                        print(s)
                        print()

                        print("Valid Tok Acc:")
                        s = "\tM1->M1: " + str(round(val_tok_accs[(0,0,vidx)], 5))
                        if len(models)>1:
                            s += "| M1->M2: " + str(round(val_tok_accs[(0,1,vidx)],5))
                            s += "\n\tM2->M1:" + str(round(val_tok_accs[(1,0,vidx)], 5))
                            s += "| M2->M2:" + str(round(val_tok_accs[(1,1,vidx)],5))
                        print(s)

                        print("Valid Trial Acc:")
                        s = "\tM1->M1: " + str(round(val_trial_accs[(0,0,vidx)], 5))
                        if len(models)>1:
                            s += "| M1->M2: " + str(round(val_trial_accs[(0,1,vidx)],5))
                            s += "\n\tM2->M1:" + str(round(val_trial_accs[(1,0,vidx)], 5))
                            s += "| M2->M2:" + str(round(val_trial_accs[(1,1,vidx)],5))
                        print(s)
                        print("Experiment:", os.path.join(save_folder, save_name))
                        print("M1:", config["model_names"][0])
                        if len(config["model_names"])>1:
                            print("M2:", config["model_names"][1])
                        print("Exec Time:", time.time()-startt)
                        print()

                    for (s,t,v) in tokenized_datasets["train"]:
                        tup = (s,t,v)
                        df_dict["global_step"].append(global_step)
                        df_dict["train_loss"].append(float(losses[tup]))
                        df_dict["train_tok_acc"].append(float(tok_accs[tup]))
                        df_dict["train_trial_acc"].append(float(trial_accs[tup]))
                        df_dict["valid_loss"].append(float(val_losses[tup]))
                        df_dict["valid_tok_acc"].append(float(val_tok_accs[tup]))
                        df_dict["valid_trial_acc"].append(float(val_trial_accs[tup]))
                        df_dict["src_idx"].append(s)
                        df_dict["trg_idx"].append(t)
                        df_dict["varb_idx"].append(v)
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
                if config.get("debugging", False):
                    print("Skipping saving due to debugging flag")
                elif end_training or global_step%svsteps:
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

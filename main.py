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
)
from utils import (
    collect_activations, device_fxn, get_command_line_args,
    default_to_list,
)
import seq_models as smods
from dl_utils.save_io import (
    get_save_name, load_checkpoint, get_folder_from_path, save_json, load_yaml,
)
from dl_utils.utils import get_git_revision_hash, get_mask_past_arglast
from dl_utils.tokenizer import Tokenizer
from interchange import InterventionModule

import pandas as pd # import after transformers

def fill_in_prompts_and_replacements(config, yaml_path="./constants.yaml"):
    consts = load_yaml(yaml_path)
    config["prompts"] = []
    config["replacements"] = []
    for model_name in config["model_names"]:
        print("Model Name:", model_name)
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

def gsm8k_is_correct_batch(model, tokenizer, examples, device, max_new_tokens=50):
    """
    Given a list of examples (each with a "question" and "answer"), tokenizes the questions,
    runs batch generation, and returns a list of booleans indicating whether the answer appears
    in the generated text for each example.
    """
    questions = examples["question"]
    answers = examples["answer"]
    # Tokenize the entire batch; we use padding so that each example is the same length.
    inputs = tokenizer(
        questions, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    # For each example, check whether its answer appears in the generated text.
    return [ans.split("####")[-1] in gen for ans, gen in zip(answers, generated_texts)]


def get_hook(comms_dict):
    def hook_fn(module, input, output):
        if "loop_count" not in comms_dict:
            comms_dict["loop_count"] = 0
        # output is assumed to be of shape (batch, seq_length, hidden_size)
        src_idx = comms_dict.get("src_idx",0)
        trg_idx = comms_dict.get("trg_idx",1)

        if hasattr(output,"hidden_states"):
            trg_actvs = output["hidden_states"]
        else:
            trg_actvs = output

        # Prep source vectors
        src_actvs = comms_dict["src_activations"]

        # Handle case where we have a specific swap mask
        if comms_dict.get("trg_swap_idxs", None) is not None:
            trg_swap_idxs = comms_dict["trg_swap_idxs"]
            src_swap_idxs = comms_dict["src_swap_idxs"]
            i = comms_dict["loop_count"]
            if len(src_actvs.shape)==2: # in contrast to len 3
                trg_swap_idxs = trg_swap_idxs[:,i]
                trg_swap_mask = trg_swap_idxs>-1
                src_swap_mask = (src_swap_idxs==trg_swap_idxs[:,None])
                src_swap_mask = src_swap_mask&(src_swap_idxs>0)
            else:
                trg_swap_mask = trg_swap_idxs>-1
                src_swap_mask = src_swap_idxs>-1
            placeholder = torch.empty_like(trg_actvs)
            placeholder[~trg_swap_mask] = trg_actvs[~trg_swap_mask]
            src_actvs = src_actvs[src_swap_mask]
            trg_actvs = trg_actvs[trg_swap_mask]

        comms_dict["loop_count"] += 1

        ## DEBUGGING
        #device = src_actvs.get_device()
        #p = torch.nn.Parameter(torch.ones_like(trg_actvs))
        #return output * p.to(device)
        ## DEBUGGING

        # Perform causal interchange
        #p = torch.nn.Parameter(torch.ones_like(src_actvs))
        #outs = src_actvs.to(device)*p
        intrv_module = comms_dict["intrv_module"]
        outs = intrv_module(
            target=trg_actvs,
            source=src_actvs,
            target_idx=trg_idx,
            source_idx=src_idx,)

        if comms_dict.get("trg_swap_idxs", None) is not None:
            placeholder[trg_swap_mask] = outs
            outs = placeholder

        if hasattr(output,"hidden_states"):
            output["hidden_states"] = outs
            return output
        else:
            return outs

    return hook_fn

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
        tokenizer = Tokenizer(
            words=set(),
            unk_token=None,
            word2id=mconfig.get("word2id",{}),
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
    return model, tokenizer

def forward_pass(
        sidx,
        tidx,
        model,
        batch_indices,
        dataset,
        comms_dict,
        src_activations,
        src_swap_idxs,
        device,
        tokenizer=None,
        pad_mask=None,
        task_mask=None,
        shuffle_targ_ids=False,
        verbose=False,
    ):
    ## Get batch
    batch = collate_fn( batch_indices, dataset, device=device)

    ## Set Comms Dict Values
    comms_dict["src_idx"] = sidx
    comms_dict["trg_idx"] = tidx
    comms_dict["loop_count"] = 0
    comms_dict["intrv_module"].to(device)
    comms_dict["intrv_module"].reset()
    comms_dict["src_activations"] =\
        src_activations[batch_indices].to(device)
    input_ids = batch["input_ids"]
    if "swap_idxs" in batch:
        ssm = src_swap_idxs[batch_indices].to(device)
        comms_dict["src_swap_idxs"] = ssm
        tsm = batch["swap_idxs"].to(device)
        comms_dict["trg_swap_idxs"] = tsm

        if shuffle_targ_ids:
            mask = tsm>-1
            # Shuffles the input ids
            msums = mask.long().sum(-1)
            perms = [torch.randperm(s).long() for s in msums]
            perm = [perms[i+1]+len(perms[i]) for i in range(len(perms)-1)]
            perm = torch.cat([perms[0]] + perm)
            input_ids[mask] = input_ids[mask][perm.to(device)]

    ## Run model
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],)

    # Calc Loss
    if "logits" in outputs:
        logits = outputs["logits"]
    else:
        logits = outputs.logits

    V = logits.shape[-1]
    flat = logits.reshape(-1,V)
    labels = batch["labels"].reshape(-1)
    tmask = batch["attention_mask"]

    loss = F.cross_entropy(
        flat[tmask.reshape(-1)],
        labels[tmask.reshape(-1)]
    )

    if "task_mask" in batch:
        tmask = batch["task_mask"].to(device)
    else:
        tmask = batch["attention_mask"]
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
        pmask = ~pad_mask[batch_indices]
        input_mask = pmask
        if task_mask is not None:
            tmask = task_mask[batch_indices]
            input_mask = pmask&(~tmask)
        else:
            tmask = pmask

        trg_pad_id =  tokenizer.pad_token_id
        trg_pad_tok = tokenizer.pad_token

        for i in range(min(2,len(outs))):
            # Input Text
            input_text = tokenizer.decode(inpts[i][input_mask[i]])
            if type(input_text)!=str:
                input_text = input_text[0]
            input_text = input_text.replace(trg_pad_tok, "")

            # Target Text
            target_text = tokenizer.decode(labels[i][tmask[i]])
            if type(target_text)!=str:
                target_text = target_text[0]
            target_text = target_text.replace(trg_pad_tok, "")

            # Generated Text
            generated_text = tokenizer.decode(outs[i][tmask[i]])
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
        "save_memory": True,
        # Use two identical models by default (replace with real LLaMA repo names as needed)
        "model_names": [
            #"meta-llama/Llama-3.2-1B",
            "gpt2",
            "gpt2",
        ], #[, "gpt2"], #
        "dataset_names": ["gsm8k", "gsm8k"],           # gsm8k dataset
        "dataset_kwargs": [
            {"name": "main", "split":"train", } for _ in range(2)
        ],
        "filter_by_correct": False,
        "padding_sides": ["left", "left"],
        "filtered_dataset_paths": [
            "./data/filtered_gsm8k",  # where to save/load the filtered dataset
            "./data/filtered_gsm8k",  # where to save/load the filtered dataset
        ],
        "layers": [ # layers at which to attach the hooks
            "model.embed_tokens",
            "transformer.wte"
        ],  
        "mtx_types": ["RotationMatrix", "RotationMatrix"],
        "identity_init": False,
        "identity_rot": False,
        "mask_type":   "FixedMask", # BoundlessMask
        "n_units": None,
        "learnable_addition": False,

        "num_training_steps": 50000,
        "print_every": 100,
        "batch_size": 32,
        "grad_accumulation_steps": 8,
        "lr": 1e-3,
        "max_length": 128,                 # max token length for our (toy) examples
        "eval_batch_size": 16,             # batch size for correctness evaluation

        "save_keys": ["mtx_types", "mask_type", "layers", "dataset_names"],
    }
    config = {**defaults}
    config["git_hash"] = get_git_revision_hash()
    for k in arg_config: config[k] = arg_config[k]
    print("Config:")
    for k in sorted(list(config.keys())):
        print(k, config[k])

    config["mtx_kwargs"] = [
        {**config} for _ in range(len(config["model_names"]))
    ]
    config["mask_kwargs"] = {**config}
    config = fill_in_prompts_and_replacements(config)

    config["padding_sides"] = default_to_list(
        config["padding_sides"],
        n_el=len(config["model_names"])
    )
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
    m_sizes = []
    devices = []
    for mi,model_name in enumerate(config["model_names"]):
        model, tokenizer = get_model_and_tokenizer(
            model_name,
            padding_side=padding_sides[mi],
        )
        model.eval()

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
    
    ##########################
    #    Load the dataset
    ##########################
    print("Loading datasets...")
    datasets = { "train": [], "valid": [], }
    for mi in range(len(config["dataset_names"])):
        for k in datasets:
            dkwargs = {**config["dataset_kwargs"][mi]}
            dkwargs["split"] = k
            dkwargs["data_path"] = config.get(
                f"{k}_data_paths",
                ["./data/multiobj.json", "./data/multiobj.json"]
            )[mi]
            dataset = get_dataset(config["dataset_names"][mi], **dkwargs)
            datasets[k].append(dataset)

    ##########################
    #    Tokenize the filtered dataset for autoregressive training.
    #    Here we form an input by concatenating the question and
    #    answer (with a newline and “Answer:” marker).
    ##########################
    tokenized_datasets = {k: [] for k in datasets}
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
        all_src_activations = {k:[] for k in datasets}
        all_src_swap_idxs =   {k:[] for k in datasets}
        all_src_task_masks =  {k:[] for k in datasets}
        all_src_pred_ids =    {k:[] for k in datasets}
        all_src_logits =      {k:[] for k in datasets}
        all_src_probs =       {k:[] for k in datasets}
        all_src_pad_masks =   {k:[] for k in datasets}
        print("Collecting Activations")
        for k in all_src_activations:
            for mi,model in enumerate(models):
                startt = time.time()
                device = devices[mi]
                print("Model", mi, config["model_names"][mi])
                print("Device:", device)
                vbsize = config.get("eval_batch_size", 128)
                batch = collate_fn(
                    torch.arange(len(tokenized_datasets[k][mi])).long(),
                    tokenized_datasets[k][mi],
                    device="cpu")

                actvs = collect_activations(
                    model,
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    layers=[config["layers"][mi], "lm_head"],
                    ret_pred_ids=True,
                    batch_size=vbsize,
                    to_cpu=True,
                    verbose=True,
                )

                all_src_activations[k].append(
                    actvs[config["layers"][mi]].squeeze())

                logits = actvs["lm_head"].squeeze()
                all_src_logits[k].append(logits)

                all_src_probs[k].append(torch.softmax(logits, dim=-1))

                pad_id = tokenizers[mi].pad_token_id
                all_src_pad_masks[k].append( batch["input_ids"]==pad_id )
                if "task_mask" in batch:
                    dword = config["replacements"][mi].get("done_word",None)
                    eos_ids = [tokenizers[mi].eos_token_id]
                    if hasattr(tokenizers[mi], "eos_id"):
                        eos_ids.append(tokenizers[mi].eos_id)
                    try:
                        eos_ids.append(int(tokenizers[mi](dword)["input_ids"][-1]))
                    except: pass
                    try:
                        eos_ids.append(
                            int(tokenizers[mi](" "+dword)["input_ids"][-1]))
                    except: pass
                    eos_ids = torch.LongTensor(eos_ids)
                    in_eos_ids = torch.isin(batch["input_ids"].long(),eos_ids)
                    eos_and_tmask = get_mask_past_arglast(in_eos_ids, inclusive=True)
                    all_src_pad_masks[k][-1] = all_src_pad_masks[k][-1]|eos_and_tmask

                all_src_pred_ids[k].append(actvs["pred_ids"].squeeze())
                all_src_pred_ids[k][-1][all_src_pad_masks[k][-1]] = pad_id

                if "swap_idxs" in batch:
                    all_src_swap_idxs[k].append(batch["swap_idxs"].cpu())
                if "task_mask" in batch:
                    all_src_task_masks[k].append(batch["task_mask"].cpu())

                if "task_mask" in batch:
                    tmask = batch["task_mask"].to(device)
                    flat_tmask = tmask.reshape(-1)
                else:
                    tmask = all_src_pad_masks[k][-1].to(device)
                    flat_tmask = tmask.reshape(-1)
                corrects = torch.ones_like(tmask)
                pids = all_src_pred_ids[k][-1].to(device)[tmask]
                tids = batch["labels"] .to(device)[tmask]
                idx = pids==tids
                corrects[tmask] = idx
                corrects = corrects.float().sum(-1)==corrects.shape[-1]
                tokacc = (idx).float().mean().item()
                fullacc = corrects.float().mean().item()

                # Generated Text
                idx = 0
                input_text = tokenizers[mi].decode(batch["input_ids"][idx])
                if type(input_text)!=str:
                    input_text = input_text[0]
                input_text = input_text.replace(tokenizers[mi].pad_token, "")
                print("ExIds :", batch["input_ids"][idx][:10])
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
    #    Define a single rotation matrix as a learnable parameter.
    #    (We then “force” it to be orthogonal after each optimizer step.)
    ##########################
    intrv_module = InterventionModule(
        sizes=m_sizes,
        **config,
    )
    intrv_module.eval()
    optimizer = torch.optim.Adam(
        intrv_module.parameters(),
        lr=config["lr"])
    
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
    }
    hook_fn_model1 = get_hook(comms_dict)
    hook_fn_model2 = get_hook(comms_dict)
    
    hook_module_model1 = get_hook_module(models[0], config["layers"][0])
    hook_module_model2 = get_hook_module(models[1], config["layers"][1])
    hook_handle1 = hook_module_model1.register_forward_hook(hook_fn_model1)
    hook_handle2 = hook_module_model2.register_forward_hook(hook_fn_model2)
    
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
    }
    while global_step < config["num_training_steps"]:
        for batch_indices in train_loader:
            # Forward passes. The hook functions will transform activations at the chosen layer.
            losses = []
            trial_accs = []
            tok_accs = []
            tot_loss = 0
            tot_tok = 0
            tot_trial = 0

            val_losses = []
            val_trial_accs = []
            val_tok_accs = []
            startt = time.time()
            for sidx in range(len(models)):
                losses.append([])
                trial_accs.append([])
                tok_accs.append([])
                val_losses.append([])
                val_trial_accs.append([])
                val_tok_accs.append([])
                for tidx in range(len(models)):
                    loss, tok_acc, trial_acc = forward_pass(
                        sidx=sidx,
                        tidx=tidx,
                        model=models[tidx],
                        comms_dict=comms_dict,
                        batch_indices=batch_indices,
                        dataset=tokenized_datasets["train"][tidx],
                        src_activations=all_src_activations["train"][sidx],
                        src_swap_idxs=all_src_swap_idxs["train"][sidx],
                        device=devices[tidx],
                        shuffle_targ_ids=config.get("shuffle_targ_ids", False),
                    )
                    accum = config.get("grad_accumulation_steps", 1)
                    loss = loss/accum/4.0
                    if config["save_memory"]:
                        loss.backward()
                    losses[-1].append(loss.item())
                    tot_loss += loss.to(devices[0])

                    tot_trial += trial_acc.item()/4.0
                    tot_tok += tok_acc.item()/4.0
                    trial_accs[-1].append(trial_acc.item())
                    tok_accs[-1].append(tok_acc.item())
                    print("Loss:", round(loss.item(), 5), end="                \r")

                    # Print a sample generation every print_every steps.
                    if global_step % config["print_every"] == 0:
                        ####################################################
                        #### VALIDATION
                        ####################################################
                        print("\n\nSource Model", sidx, "- Target Model", tidx)
                        print("Validating...")
                        val_loss = 0
                        val_tok = 0
                        val_trial = 0
                        with torch.no_grad():
                            for val_indices in valid_loader:
                                vloss, vtok, vtrial = forward_pass(
                                    sidx=sidx,
                                    tidx=tidx,
                                    model=models[tidx],
                                    comms_dict=comms_dict,
                                    batch_indices=val_indices,
                                    dataset=tokenized_datasets["valid"][tidx],
                                    src_activations=all_src_activations["valid"][sidx],
                                    src_swap_idxs=all_src_swap_idxs["valid"][sidx],
                                    device=devices[tidx],
                                    tokenizer=tokenizers[tidx],
                                    pad_mask=all_src_pad_masks["valid"][tidx],
                                    task_mask=all_src_task_masks["valid"][tidx],
                                    shuffle_targ_ids=config.get("shuffle_targ_ids", False),
                                    verbose=True,
                                )
                                val_loss  += vloss.item() /len(valid_loader)
                                val_tok   += vtok.item()  /len(valid_loader)
                                val_trial += vtrial.item()/len(valid_loader)
                            val_losses[-1].append(val_loss)
                            val_tok_accs[-1].append(val_tok)
                            val_trial_accs[-1].append(val_trial)

            if not config["save_memory"]:
                tot_loss.backward()
            if global_step % accum==0:
                optimizer.step()
                optimizer.zero_grad()

            if global_step % config["print_every"] == 0:
                print("Mtx  Type:", config["mtx_types"][0])
                print("Mask Type:", config["mask_type"],
                        "- Learn:", config["learnable_addition"],
                        "- Units:", intrv_module.swap_mask.n_units)
                print()

                print("Step:", global_step, "| Train Loss:", tot_loss.item())
                print("Train Tok Acc:",  tot_tok)
                print("\tM1->M1:", round(tok_accs[0][0], 5),
                      "| M1->M2:", round(tok_accs[0][1],5))
                print("\tM2->M1:", round(tok_accs[1][0], 5),
                      "| M2->M2:", round(tok_accs[1][1],5))
                print("Train Trial Acc:",tot_trial)
                print("\tM1->M1:", round(trial_accs[0][0],5),
                      "| M1->M2:", round(trial_accs[0][1], 5))
                print("\tM2->M1:", round(trial_accs[1][0],5),
                      "| M2->M2:", round(trial_accs[1][1],5))
                print()
                print("Valid Tok Acc:")
                print("\tM1->M1:", round(val_tok_accs[0][0], 5),
                      "| M1->M2:", round(val_tok_accs[0][1], 5))
                print("\tM2->M1:", round(val_tok_accs[1][0], 5),
                      "| M2->M2:", round(val_tok_accs[1][1], 5))
                print("Valid Trial Acc:")
                print("\tM1->M1:", round(val_trial_accs[0][0],5),
                      "| M1->M2:", round(val_trial_accs[0][1],5))
                print("\tM2->M1:", round(val_trial_accs[1][0],5),
                      "| M2->M2:", round(val_trial_accs[1][1],5))
                print("Experiment:", os.path.join(save_folder, save_name))
                print("M1:", config["model_names"][0])
                if len(config["model_names"])>1:
                    print("M2:", config["model_names"][1])
                print("Exec Time:", time.time()-startt)
                print()

                for s in range(len(models)):
                    for t in range(len(models)):
                        df_dict["global_step"].append(global_step)
                        df_dict["train_loss"].append(float(losses[s][t]))
                        df_dict["train_tok_acc"].append(float(tok_accs[s][t]))
                        df_dict["train_trial_acc"].append(float(trial_accs[s][t]))
                        df_dict["valid_loss"].append(float(val_losses[s][t]))
                        df_dict["valid_tok_acc"].append(float(val_tok_accs[s][t]))
                        df_dict["valid_trial_acc"].append(float(val_trial_accs[s][t]))
                        df_dict["src_idx"].append(s)
                        df_dict["trg_idx"].append(t)

            
            ### Save loss and state dict
            if global_step%config.get("save_every_steps", 100):
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

    ##########################
    # 9. Clean up: remove hooks.
    ##########################
    hook_handle1.remove()
    hook_handle2.remove()
    print("Training complete.")

if __name__ == "__main__":
    main()

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
    collate_fn, make_tokenized_info, add_token_ids_to_info,
    add_prompt, pad_data_dict, add_pad_masks, convert_to_tensors,
)
from utils import (
    collect_activations, device_fxn, get_command_line_args,
    default_to_list, tensor2str, get_len_hist,
)
import seq_models as smods
from dl_utils.save_io import (
    get_save_name, load_checkpoint, get_folder_from_path, save_json,
    load_yaml, get_config,
)
from dl_utils.utils import (
    get_git_revision_hash, get_mask_past_arglast, arglast, get_timestamp,
    analytical_linear_regression, 
)
from dl_utils.schedulers import PlateauTracker
from dl_utils.tokenizer import Tokenizer
from fca import perform_eigen_pca
from intrv_modules import InterventionModule
import filters
import causal_models
import constants as consts
from intrv_datas import make_intrv_data_from_seqs
from train import make_tokenizer_from_info
from hooks import get_stepwise_hook, get_indywise_hook, get_hook_module
from tasks import DEFAULT_INFOS

import pandas as pd # import after transformers to avoid versioning bug

def config_prep(config):
    # Catch plural errors:
    for k in [
        "cmodel_names", "swap_keys", "mtx_types", "model_names",
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

    # can assume different cmodels will default to appropriate parameters. This
    # reduces risk of error. Just make a new causal model for new interventions
    if type(config.get("cmodel_names",None))==str:
        cname = config["cmodel_names"]
        config["cmodel_names"] = [cname for _ in config["model_names"]]
    if config.get("cmodel_names", None) is None:
        mconfigs = [get_config(mname) for mname in config["model_names"]]
        mconfigs = [ mc if mc is not None else {} for mc in mconfigs ]
        t2c = consts.TASK2CMODEL
        cnames = [
            t2c.get(mc.get("task_type",None), "CountUpDown") for mc in mconfigs]
        config["cmodel_names"] = cnames
        print("Cmodel Names:", cnames)

    kwargs = { "hold_outs": [], }
    config["cmodels"] = [
        getattr(causal_models, cname)(**kwargs) for cname in config["cmodel_names"]
    ]
    print("Cmodels:", config["cmodels"])

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
    config["prompts"] = []
    config["replacements"] = []
    config["padding_sides"] = []
    for model_name in config["model_names"]:
        print("Model Name:", model_name)
        # Get padding side
        padding_side = consts.PADDING_SIDES.get(model_name, "right")
        config["padding_sides"].append(padding_side)

        # Get prompts
        prompt = consts.PROMPTS.get(model_name, "")
        if not prompt:
            for k in consts.PROMPTS:
                if k in model_name:
                    prompt = consts.PROMPTS[k]
        config["prompts"].append(prompt)
        print("Prompt:", prompt)

        # Get string replacement dict
        replacements = consts.REPLACEMENTS.get(
                model_name,
                None
            )
        if not replacements:
            replacements = {**consts.DEFAULT_REPLACEMENTS}
            for k in consts.REPLACEMENTS:
                if k in model_name:
                    replacements = {**replacements, **consts.REPLACEMENTS[k]}
        config["replacements"].append(replacements)
        print("Replacements:")
        for k,v in replacements.items():
            print(f"\t{k}: {v}")
        print()
    return config

def get_hf_config(model_name):
    """
    Currently a hacky solution to make project compatible with huggingface
    models.
    """
    return {
        "task_type": "MultiObject",
    }

def get_model_and_tokenizer(model_name, padding_side="left"):
    print(f"Loading model and tokenizer for {model_name}...")
    try:
        # Custom Models  
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
        # Huggingface Models
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
            tokenizer.pad_token_id = tokenizer(tokenizer.pad_token)["input_ids"][-1]
            tokenizer.pad_token = tokenizer.decode(tokenizer.pad_token_id)[-1]
        if not tokenizer.eos_token:
            tokenizer.eos_token = "EOS"
            tokenizer.eos_token_id = tokenizer(tokenizer.eos_token)["input_ids"][-1]
            tokenizer.eos_token = tokenizer.decode(tokenizer.eos_token_id)[-1]
        if not tokenizer.bos_token or tokenizer.bos_token==tokenizer.eos_token:
            tokenizer.bos_token = "BOS"
            tokenizer.bos_token_id = tokenizer(tokenizer.bos_token)["input_ids"][-1]
            tokenizer.bos_token = tokenizer.decode(tokenizer.bos_token_id)[-1]
        model = smods.HFTransformer(hf_model_type=model_name, device_map="auto")
        mconfig = get_hf_config(model_name)
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
        cl_latents=None,
        tokenizer=None,
        pad_mask=None,
        config=dict(),
        verbose=False,
        vidx=None,
        tforce=False,
        track_grad=True,
        cl_divergence=False,
    ):
    """
    Args:
        src_activations: torch tensor (B,S,D)
        cl_latents: torch tensor (B,S,D)
            use the trg_swap_mask to isolate the cl latent targets
        cl_divergence: bool
            track the kl divergence between the intervened vectors
            and the counterfactual latents.
    """
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
    comms_dict["intrv_vecs"] = None
    comms_dict["src_activations"] =\
        src_activations[batch_indices].to(device)
    input_ids = batch["input_ids"].clone()

    mask = None
    if "trg_swap_masks" in batch:
        comms_dict["trg_swap_masks"] = batch["trg_swap_masks"]
        comms_dict["src_swap_masks"] = batch["src_swap_masks"]
        mask = batch["trg_swap_masks"]>=0
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
    prev_grad_state = torch.is_grad_enabled()
    torch.set_grad_enabled(track_grad)
    try:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["inpt_attn_mask"],
            task_mask=batch["input_tmask"],
            tforce=tforce,
        )
    except:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["inpt_attn_mask"],
        )
    torch.set_grad_enabled(prev_grad_state)

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
        smask = batch["trg_swap_masks"]>=0
        smask = torch.roll(~smask, -1, dims=-1)
        smask[...,-1] = True
        lmask = lmask&(smask)
    #if "outp_tmask" in batch and config.get("stepwise", True):
    #    smask = batch["outp_tmask"]
    #    lmask = lmask&(smask)

    ### TODO
    #pids = torch.argmax(logits, dim=-1)
    #print("HEYO")
    #for i in range(3):
    #    print("Omask:",tensor2str(batch["outp_attn_mask"][i].long()))
    #    print("smask:",tensor2str(smask[i].long()))
    #    print("Label:", tensor2str(batch["labels"][i]))
    #    print("Inpts:", tensor2str(batch["input_ids"][i]))
    #    print("Swaps:", tensor2str(batch["labels"][i][batch["trg_swap_masks"][i]]))
    #    print("Preds:", tensor2str(pids[i][lmask[i]]))
    #    print("Label:", tensor2str(batch["labels"][i][lmask[i]]))
    #    print()

    ##################
    ## MAS LOSS
    ##################
    prev_grad_state = torch.is_grad_enabled()
    enable = track_grad and (sidx,tidx) in config["train_directions"]
    torch.set_grad_enabled(enable)
    loss = F.cross_entropy(
        flat[lmask.reshape(-1)],
        labels[lmask.reshape(-1)]
    )
    torch.set_grad_enabled(prev_grad_state)

    ##################
    ## CL LOSS
    ##################
    cl_loss = torch.zeros(1).to(device)
    cl_div = 0
    cl_sdx = 0 # Amount that the max value of the intervened latents 
        # exceeds the max value of the cl latents measured in standard 
        # deviations
    if cl_latents is not None and "intrv_vecs" in comms_dict:
        cl_latents = cl_latents[batch_indices].to(device)
        prev_grad_state = torch.is_grad_enabled()
        enable = track_grad and (sidx,tidx) in config["cl_directions"]
        torch.set_grad_enabled(enable)
        cl_loss = cl_loss_fxn(
            intrv_vecs=torch.stack(comms_dict["intrv_vecs"],dim=1),
            cl_latents=cl_latents,
            swap_mask=batch["trg_swap_masks"]>=0,
        )
        torch.set_grad_enabled(prev_grad_state)
        if cl_divergence:
            cl_div, cl_sdx = cl_kl_divergence(
                intrv_vecs=torch.stack(comms_dict["intrv_vecs"],dim=1),
                cl_vecs=cl_latents,
                swap_mask=batch["trg_swap_masks"]>=0,
                laplace=1,
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
            omask = batch["outp_attn_mask"]
        else:
            pmask = ~pad_mask[batch_indices]
            omask = batch["outp_attn_mask"]
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
            print("InptPd:", tensor2str(pmask[i].long()))
            print("OutpPd:", tensor2str(omask[i].long()))
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
            #target_text = target_text.replace(trg_pad_tok, "")

            # Generated Text
            generated_text = tokenizer.decode(outs[i][trg_swap:])
            if type(generated_text)!=str:
                generated_text = generated_text[0]
            #generated_text = generated_text.replace(trg_pad_tok, "")

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

    if cl_divergence:
        return loss, cl_loss, tok_acc, trial_acc, cl_div, cl_sdx
    return loss, cl_loss, tok_acc, trial_acc

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
        if name==layer: return name
        elif layer in name:
            if len(name.split("."))<shortest_len:
                shortest_len = len(name.split("."))
                simplist_name = name
        elif type(modu)==torch.nn.Embedding or "Embedding" in str(type(modu)):
            if name==layer: return name
            if len(name.split("."))<shortest_len:
                shortest_len = len(name.split("."))
                simplist_name = name
    return simplist_name

def mse_loss(x,y, *args, **kwargs):
    return torch.nn.functional.mse_loss(x,y,reduction="none").mean(-1)

def cosine_loss(x,y, targs=None, *args, **kwargs):
    if targs is None:
        device = device_fxn(x.get_device())
        targs = torch.ones(len(x)).to(device)
    return torch.nn.functional.cosine_embedding_loss(
        x,y,targs,reduction="none"
    )

def cos_and_mse_loss(x,y,targs=None,*args,**kwargs):
    return (cosine_loss(x,y,targs) + mse_loss(x,y))/2.

def get_loss_fxn(fxn_name):
    if "mse" in fxn_name:
        print("Using mse aux loss")
        return mse_loss
    if "cosine" in fxn_name or "cos" in fxn_name:
        print("Using cosine aux loss")
        return cosine_loss
    if "both" in fxn_name:
        return cos_and_mse_loss
    print("Invalid aux function name")
    raise NotImplemented

def np_kl_divergence(p, q):
    prob_logs = p*np.log(p/(q+1e-10)+1e-10)
    prob_logs[prob_logs!=prob_logs] = 100
    return prob_logs.sum()

def cl_kl_divergence(intrv_vecs, cl_vecs, swap_mask, laplace=1):
    """
    Args:
        intrv_vecs: torch tensor (B,S,D)
            the intervened vectors
        cl_vecs: torch tensor (B,S,D)
            the target vectors
        swap_mask: torch tensor (B,S)
            a mask where trues denote positions to use for the cl loss.
    Returns
        cl_divergence: torch tensor (1,)
            the counterfactual latent kl divergence
        cl_sdx: torch tensor (1,)
            the amount that the maximum extreme of the intrvened latents
            exceeds that of the corresponding extreme from the natural
            distribution measured in standard deviations.
    """
    intrv_vecs = intrv_vecs[swap_mask]
    cl_vecs = cl_vecs[swap_mask]
    if type(intrv_vecs) is torch.Tensor:
        intrv_vecs = intrv_vecs.cpu().numpy()
    if type(cl_vecs) is torch.Tensor:
        cl_vecs = cl_vecs.cpu().numpy()
    cl_sdx = max(
        intrv_vecs.max() - cl_vecs.max(),
        cl_vecs.min() - intrv_vecs.min(),
    )/cl_vecs.std()
    rang = [
        min(intrv_vecs.min(), cl_vecs.min()),
        max(intrv_vecs.max(), cl_vecs.max())
    ]
    cl_hist, bins = np.histogram(cl_vecs, range=rang, density=False)
    cl_hist += laplace
    cl_density = cl_hist/cl_hist.sum()
    intrv_hist, _ = np.histogram(intrv_vecs, bins=bins, density=False)
    intrv_hist += laplace
    intrv_density = intrv_hist/intrv_hist.sum()

    return np.mean([
        np_kl_divergence(intrv_density, cl_density),
        np_kl_divergence(cl_density, intrv_density),
    ]), cl_sdx

def cl_loss_fxn(intrv_vecs, cl_latents, swap_mask, loss_type="both"):
    """
    Args:
        intrv_vecs: torch tensor (B,S,D)
            the intervened vectors
        cl_latents: torch tensor (B,S,D)
            the target vectors
        swap_mask: torch tensor (B,S)
            a mask where trues denote positions to use for the cl loss.
        loss_type: str {"mse", "cos", "both"}
            optionally pick the loss function
    Returns
        cl_loss: torch tensor (1,)
            the counterfactual latent loss
    """
    preds = intrv_vecs[swap_mask]
    labls = cl_latents[swap_mask]
    loss_fxn = get_loss_fxn(loss_type)
    return loss_fxn(preds,labls).mean()

def get_cl_latents(
    model,
    swap_mask,
    input_ids,
    idxs,
    layer,
    device,
    bsize=500,
    tmask=None,
):
    """
    A helper function for collecting the counterfactual
    latents.

    Args:
        model: torch module
        swap_mask: bool tensor (B,S)
        input_ids: long tensor (B,S)
        idxs: torch Long tensor (N,2)
            a tensor in which the first column denotes
            the row index and the second column denotes
            the column index from which to take the
            latent vectors. N should be equal to the sum
            of all entries in the swap mask.
    """
    model.eval()
    model.to(device)
    outputs = collect_activations(
        model,
        input_ids=input_ids,
        layers=[layer],
        tforce=True,
        ret_pred_ids=True,
        batch_size=bsize,
        to_cpu=True,
        verbose=True,
    )

    # Quick Accuracy Check
    pred_ids = outputs["pred_ids"]
    corrects = torch.ones_like(pred_ids[:,:-1]).float()
    tmask = tmask.clone().bool() if tmask is not None else torch.ones_like(input_ids).bool()
    pids = pred_ids[:,:-1]
    oids = input_ids[:,1:]
    tmask = tmask[:,1:] # remove the last token from the mask
    acc = pids[tmask].long()==oids[tmask].long()
    corrects[tmask] = acc.float()
    corrects = corrects.sum(-1)==corrects.shape[-1]
    print()
    print("CL Token Acc:", acc.float().mean().item())
    print("CL Trial Acc:", corrects.float().mean().item())
    print()

    actvs = outputs[layer]
    shape = tuple([*swap_mask.shape, actvs.shape[-1]])
    assert swap_mask.long().sum()==len(idxs)
    cl_latents = torch.empty(shape)
    cl_latents[swap_mask] = actvs[idxs[:,0],idxs[:,1]]
    return cl_latents


def main():
    arg_config, command_keys = get_command_line_args(sys.argv)
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
        "cmodel_names": None,
        "filter_names": [
            "default_filter",
            "default_filter",
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
    print("Save Name:", save_name)

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
            config["layers"][mi] = get_embedding_name(
                model, config["layers"][mi])
            print("Decided Layer Name:", config["layers"][mi])

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
            tconfig = model_configs[mi].get("task_config", {})
            if tconfig is None: tconfig = {}
            if "max_count" in config:
                tconfig["max_count"] = config["max_count"]
            if tconfig: tconfig["unk_p"] = 0
            # The dataset consists of text (and task masks if applicable)
            # Will eventually allow vector representations as well
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

            # The tokenized dataset has replaced text specified in the
            # replacements dict, has not prepended a prompt or a bos token,
            # and has converted the text into tokens and then token ids.
            # None of the tokenized data is padded, and non are tensors.
            # fields include: "token_ids", "inpt_attn_mask", "task_mask"
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
                #print("Making Info!")
                #ttype = model_configs[mi].get("task_type", "MultiObject")
                #info = DEFAULT_INFOS[ttype]
                info = make_tokenized_info(
                    replacements=kwrgs["replacements"],
                    tokenizer=tokenizer,
                    config=config
                )
        infos.append(info)
    config["infos"] = infos
    print("Tok Dataset:", tokenized_datasets["train"][0])
    print("Cmodels:", config["cmodels"])
    for i in range(len(tokenized_datasets["train"])):
        print(i,"Example:")
        print("Seq   :", tokenized_datasets["train"][i]["input_ids"][0])
        print("TMask :", tokenized_datasets["train"][i]["task_mask"][0])
        print("Decode:", tokenizers[i].decode(tokenized_datasets["train"][i]["input_ids"][0]))
        print()

    ####################################################
    #    Make/Get Intervention Data
    ####################################################
    intrv_datasets = {k: dict() for k in tokenized_datasets }
    print("Info:")
    print("1:", config["infos"][0])
    print()
    try:
        print("2:", config["infos"][1])
        print()
    except: pass

    subspace_maps = { # maps between indices and string keys for each model
        "subspace2varbs": [],
        "varb2subspaces": [],
    }
    config["subspace_maps"] = subspace_maps
    for k in tokenized_datasets:
        for tidx in range(len(tokenized_datasets[k])):
            subspace_maps["subspace2varbs"].append(
                {i: key for i,key in enumerate(config["swap_keys"][tidx])}
            )
            subspace_maps["varb2subspaces"].append(
                {key: i for i,key in enumerate(config["swap_keys"][tidx])}
            )
            for sidx in range(len(tokenized_datasets[k])):
                n_varbs = len(config["swap_keys"][sidx])
                z = enumerate(zip(
                    config["swap_keys"][sidx],
                    config["swap_keys"][tidx],
                ))
                for vidx,(src_swap_key, trg_swap_key) in z:
                    if src_swap_key=="" and trg_swap_key=="" and tidx!=sidx:
                        # Only want to include empty training on within model
                        # interventions
                        continue
                    print(f"Making intrv data - Src{sidx} - Trg{tidx} - Var{vidx}")
                    print(sidx, "Info:", infos[sidx])
                    print(tidx, "Info:", infos[tidx])
                    print("Sample Src:", tokenized_datasets[k][sidx]["input_ids"][0])
                    print("SSampl Src:", tokenized_datasets[k][sidx]["input_ids"][0][1:])
                    print("Decode Src:", tokenizers[sidx].decode(tokenized_datasets[k][sidx]["input_ids"][0]))
                    print("Sample Trg:", tokenized_datasets[k][tidx]["input_ids"][0])
                    print("Sample Tsk:", [int(t) for t in tokenized_datasets[k][tidx]["task_mask"][0]])
                    ttype1 = model_configs[sidx].get("task_type", "MultiObject")
                    ttype2 = model_configs[tidx].get("task_type", "MultiObject")
                    sk1 = config["swap_keys"][sidx]=="full"
                    sk2 = config["swap_keys"][tidx]=="full"
                    usdfc = ttype1==ttype2 and sk1 and sk2
                    intrv_data = make_intrv_data_from_seqs(
                        trg_data=tokenized_datasets[k][tidx],
                        src_data=tokenized_datasets[k][sidx],
                        src_swap_keys=src_swap_key,
                        trg_swap_keys=trg_swap_key,
                        src_cmodel=config["cmodels"][sidx],
                        src_info=config["infos"][sidx],
                        src_filter=config["filters"][sidx],
                        trg_cmodel=config["cmodels"][tidx],
                        trg_info=config["infos"][tidx],
                        trg_filter=config["filters"][tidx],
                        stepwise=config.get("stepwise", False),
                        use_cl=(sidx,tidx) in config["cl_directions"],
                        use_src_data_for_cl=usdfc,
                        tokenizer=tokenizers[tidx],
                        ret_src_labels=True,
                    )
                    intrv_data = add_prompt(
                        intrv_data,
                        src_tokenizer=tokenizers[sidx],
                        trg_tokenizer=tokenizers[tidx],
                        src_prompt=config["prompts"][sidx],
                        trg_prompt=config["prompts"][tidx],
                        src_replacements=config["replacements"][sidx],
                        trg_replacements=config["replacements"][tidx],
                    )
                    intrv_data = pad_data_dict(
                        intrv_data,
                        src_pad_id=infos[sidx]["pad_token_id"],
                        trg_pad_id=infos[tidx]["pad_token_id"],
                        src_pad_side=padding_sides[sidx],
                        trg_pad_side=padding_sides[tidx],
                    )
                    intrv_data = add_pad_masks(
                        intrv_data,
                        src_info=infos[sidx],
                        trg_info=infos[tidx],
                    )
                    print("Post Src:", intrv_data["src_input_ids"][0])
                    print("Post Decode Src:", tokenizers[sidx].decode(intrv_data["src_input_ids"][0]))
                    print("Post Trg:", intrv_data["trg_input_ids"][0])
                    print("Post Decode Trg:", tokenizers[tidx].decode(intrv_data["trg_input_ids"][0]))
                    print()
                    print()
                    intrv_data = convert_to_tensors(intrv_data)
                    intrv_datasets[k][(sidx,tidx,vidx)] = Dataset.from_dict(intrv_data)
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
        cl_latents = {k:dict() for k in datasets}
        print("Collecting Activations")
        for k in all_src_activations:
            for dirvar_tup in tokenized_datasets[k].keys():
                src_idx,trg_idx,varb_idx = dirvar_tup
                src_model = models[src_idx].eval()
                trg_model = models[trg_idx].eval()
                startt = time.time()
                device = devices[src_idx]
                print("Trg Model", trg_idx, config["model_names"][trg_idx])
                print("Src Model", src_idx, config["model_names"][src_idx])
                print("Device:", device)
                vbsize = config.get("eval_batch_size", 128)
                batch = collate_fn(
                    torch.arange(len(tokenized_datasets[k][dirvar_tup])).long(),
                    tokenized_datasets[k][dirvar_tup],
                    incl_src=True,
                    device="cpu")


                #### TODO:
                #if varb_idx>0:
                #    print("Varbl", varb_idx)
                #    for i in range(3):
                #        indices = torch.arange(len(batch["input_ids"][i])).long()
                #        print(tensor2str(indices))
                #        for kk in batch:
                #            try:
                #                print((kk+" "*(10-len(kk)))[:10], tensor2str(batch[kk][i].long()))
                #            except:
                #                print("Failed to print", kk)
                #        print()
                #    assert False


                actvs = collect_activations(
                    src_model,
                    input_ids=batch["src_input_ids"],
                    attention_mask=batch["src_attention_mask"],
                    task_mask=batch["src_input_tmask"],
                    layers=[config["layers"][src_idx]],
                    tforce=False,
                    ret_pred_ids=True,
                    batch_size=vbsize,
                    to_cpu=True,
                    verbose=True,
                )

                src_actvs = actvs[config["layers"][src_idx]].squeeze()
                all_src_activations[k][dirvar_tup] = src_actvs

                ## Collect cl latents by generating them from cl sequences
                ## paired with cl indices to pick out the correct latents.
                cl_latents[k][dirvar_tup] = None
                if (src_idx,trg_idx) in config["cl_directions"]:
                    cl_latents[k][dirvar_tup] = get_cl_latents(
                        model=trg_model,
                        device=devices[trg_idx],
                        swap_mask=batch["trg_swap_masks"]>=0,
                        input_ids=batch["cl_input_ids"],
                        tmask=batch.get("cl_task_masks", None),
                        idxs=batch["cl_idxs"],
                        layer=config["layers"][trg_idx],
                    )

                pred_ids = actvs["pred_ids"].squeeze()

                print("Inpt:", batch["src_input_ids"].shape)
                print("InptTmask:", batch["src_input_tmask"].shape)
                if "src_outp_tmask" in batch:
                    tmask = batch["src_outp_tmask"].to(device)
                    print("OutpTmask:", batch["src_outp_tmask"].shape)
                else:
                    tmask = batch["src_outp_attn_mask"].to(device)
                    print("OutpAttn:", batch["src_outp_attn_mask"].shape)
                corrects = torch.ones_like(tmask)
                pids = pred_ids.to(device)[tmask]
                tids = batch["src_labels"] .to(device)[tmask]
                idx = pids==tids
                corrects[tmask] = idx
                corrects = corrects.long().sum(-1)==corrects.shape[1]
                tokacc = (idx).float().mean().item()
                fullacc = corrects.float().mean().item()

                tmask = batch["src_outp_tmask"]
                sidx = src_idx
                tidx = trg_idx
                #print(sidx,tidx,"Preds:")
                #for i in range(3):
                #    if i < len(pred_ids):
                #        print("\tRawPredIds:", pred_ids[i])
                #        print("\tRawLabeIds:", batch["src_labels"][i])
                #        print("\tTskPredIds:", pred_ids[i][tmask[i].bool()])
                #        print("\tTskLabeIds:", batch["src_labels"][i][tmask[i].bool()])
                #        print("\tRawPreds:",  tokenizers[sidx].decode(pred_ids[i]))
                #        print()
                #        print("\tRawLabels:", tokenizers[sidx].decode(batch["src_labels"][i]))
                #        print()
                #        print("\tTskPreds:",  tokenizers[sidx].decode(pred_ids[i][tmask[i].bool()]))
                #        print()
                #        print("\tTskLabels:", tokenizers[sidx].decode(batch["src_labels"][i][tmask[i].bool()]))
                #        print()
                #        print("\tCorrects:", corrects[i])
                #        print("----")

                # Generated Text
                idx = 0
                input_text = tokenizers[src_idx].decode(batch["src_input_ids"][idx])
                if type(input_text)!=str: input_text = input_text[0]
                input_text = input_text.replace(tokenizers[src_idx].pad_token, "")
                print("InpIds:", batch["src_input_ids"][idx][:10])
                print("PrdIds:", pred_ids[idx][:10])
                print(
                    "Inpt:", input_text.replace("\n", "\\n")\
                                         .replace("<BOS>", "B")\
                                         .replace("<EOS>", "E")
                )
                pred_text = tokenizers[src_idx].decode(pred_ids[idx])
                if type(pred_text)!=str: pred_text = pred_text[0]
                print(
                    "Pred:", pred_text.replace("\n", "\\n")\
                                         .replace("<BOS>", "B")\
                                         .replace("<EOS>", "E")
                )

                print(k.capitalize(), "TokAcc:", tokacc)
                print(k.capitalize(), "FullAcc:", fullacc)
                print("Exec Time:", time.time()-startt)
                print()

    ##########################
    #    Test Linear Decodability of Features
    ##########################
    if "src_labels" in intrv_data:
        print("Linear Decoding Results:")
        for dirvar_tup in all_src_activations["train"].keys():
            sidx,tidx,vidx = dirvar_tup
            swap_key = config["swap_keys"][sidx][vidx]
            if swap_key=="full": continue

            # Training
            actvs = all_src_activations["train"][dirvar_tup]
            actvs = actvs.reshape(-1, actvs.shape[-1])
            labels = torch.tensor(
                tokenized_datasets["train"][dirvar_tup]["src_labels"]
            )[...,:-1]#.reshape(-1).float()
            tids = torch.tensor(
                tokenized_datasets["train"][dirvar_tup]["src_input_ids"]
            )[...,:-1]#.reshape(-1)
            mask = (labels>-1)&(tids!=infos[sidx]["eos_token_id"])
            mask = mask&(tids!=infos[sidx]["trig_token_ids"][0])
            labels = labels.reshape(-1).float()
            mask = mask.reshape(-1)

            # Validation
            vactvs = all_src_activations["valid"][dirvar_tup]
            vactvs = vactvs.reshape(-1, vactvs.shape[-1])
            vlabels = torch.tensor(
                tokenized_datasets["valid"][dirvar_tup]["src_labels"]
            )[...,:-1]
            tids = torch.tensor(
                tokenized_datasets["valid"][dirvar_tup]["src_input_ids"]
            )[...,:-1]
            vmask = (vlabels>-1)&(tids!=infos[sidx]["eos_token_id"])
            vmask = vmask&(tids!=infos[sidx]["trig_token_ids"][0])
            vlabels = vlabels.reshape(-1).float()
            vmask = vmask.reshape(-1)

            # Linear Regression
            _, verr, vpreds, vlabs = analytical_linear_regression(
                X=actvs[mask],
                y=labels[mask],
                X_val=vactvs[vmask],
                y_val=vlabels[vmask],
                ret_preds_and_labels=True)
            corrects = torch.round(vpreds)==vlabs
            acc = corrects.float().mean()
            print("Model:", sidx, "- Var:", config["swap_keys"][sidx][vidx])
            print("\tErr:", verr, "- Acc:", acc.item())
            print("\tValues:", sorted(list(set(labels[mask].cpu().tolist()))))
        print()

    ##########################
    #    Define the intervention object, optimizer, and plateau tracker
    ##########################
    if config.get("mtx_kwargs", None) is None:
        mtx_kwarg_keys = {
            "rank", "identity_init", "bias", "mu",
            "sigma", "identity_rot", "orthogonal_map",
            "nonlin_align_fn",
        }
        mtx_kwargs = dict()
        for key in mtx_kwarg_keys:
            if key in config:
                mtx_kwargs[key] = config[key]
        config["mtx_kwargs"] = [mtx_kwargs for _ in models]
    config["sizes"] = m_sizes
    intrv_module = InterventionModule(
        **config,
    )
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
            all_actvs = []
            for dirvar_tup in all_src_activations["train"].keys():
                sidx,tidx,vidx = dirvar_tup
                if sidx!=midx:
                    continue

                actvs = all_src_activations["train"][dirvar_tup]
                if config["nonlin_align_fn"]!="identity":
                    actvs = intrv_module.rot_mtxs[midx].nonlin_fwd(actvs)
                all_actvs.append(actvs.reshape(-1, actvs.shape[-1]))
            actvs = torch.cat(all_actvs,dim=0)

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
                pca_matrix = torch.tensor(ret_dict["components"], device=devices[midx])
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
                tot_loss = 0
                tot_tok = 0
                tot_trial = 0

                val_losses = dict()
                val_trial_accs = dict()
                val_tok_accs = dict()
                val_cl_loss = dict()
                val_cl_div = dict()
                val_cl_sdx = dict() # Amount that the max value of the
                    # cl_latents exceeds the max value and the natural 
                    # latents max value measured in standard deviations 
                    # from the natural mean
                
                # Reset the tracking dictionaries for this step.
                tracking_dicts = [
                    losses, trial_accs, tok_accs,
                    val_losses, val_trial_accs, val_tok_accs,
                    val_cl_loss, val_cl_div, val_cl_sdx,
                ]
                for tidx in range(len(models)):
                    for sidx in range(len(models)):
                        for vidx in range(len(config["swap_keys"][sidx])):
                            dirvar_tup = (sidx, tidx, vidx)
                            for d in tracking_dicts:
                                d[dirvar_tup] = 0

                startt = time.time()
                accum = config.get("grad_accumulation_steps", 1)
                n_varbs = len(config["swap_keys"][0])
                for dirvar_tup in tokenized_datasets["train"]:
                    runtime = time.time()
                    (sidx,tidx,vidx) = dirvar_tup
                    track_train = (sidx,tidx) in config["train_directions"]
                    track_cl = (sidx,tidx) in config["cl_directions"]
                    track_grad = track_train or track_cl
                    loss, cl_loss, tok_acc, trial_acc = forward_pass(
                        sidx=sidx,
                        tidx=tidx,
                        vidx=vidx,
                        model=models[tidx],
                        comms_dict=comms_dict,
                        batch_indices=batch_indices,
                        dataset=tokenized_datasets["train"][dirvar_tup],
                        src_activations=all_src_activations["train"][dirvar_tup],
                        cl_latents=cl_latents["train"][dirvar_tup],
                        device=devices[tidx],
                        config=config,
                        tforce=True,
                        track_grad=track_grad,
                    )
                    cl_loss = cl_loss/accum/(len(models)**2)/n_varbs
                    loss = loss/accum/(len(models)**2)/n_varbs
                    combo_loss = torch.zeros_like(loss)
                    if track_train: combo_loss = loss
                    if track_cl:
                        eps = config.get("cl_eps",1)
                        combo_loss = combo_loss + eps*cl_loss

                    if config["conserve_memory"] and track_grad:
                        n_tups = len(list(tokenized_datasets["train"].keys()))
                        (combo_loss/float(n_tups)).backward()

                    losses[dirvar_tup] = loss.item()
                    tot_loss += combo_loss.to(devices[0])

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
                        for val_indices in valid_loader:
                            vloss, vcl_loss, vtok, vtrial, vcl_div, vcl_sdx = forward_pass(
                                sidx=sidx,
                                tidx=tidx,
                                vidx=vidx,
                                model=models[tidx],
                                comms_dict=comms_dict,
                                batch_indices=val_indices,
                                dataset=tokenized_datasets["valid"][dirvar_tup],
                                src_activations=all_src_activations["valid"][dirvar_tup],
                                cl_latents=cl_latents["valid"][dirvar_tup],
                                device=devices[tidx],
                                tokenizer=tokenizers[tidx],
                                config=config,
                                verbose=True,
                                tforce=False,
                                track_grad=False,
                                cl_divergence=True,
                            )
                            val_loss  += vloss.item() /len(valid_loader)
                            val_tok   += vtok.item()  /len(valid_loader)
                            val_trial += vtrial.item()/len(valid_loader)
                        val_losses[dirvar_tup] = val_loss
                        val_tok_accs[dirvar_tup] = val_tok
                        val_trial_accs[dirvar_tup] = val_trial
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
                    print("CauslModl:", config["cmodel_names"])
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
                    print()
                    for vidx in range(n_varbs):
                        print("Varbl", vidx, config["swap_keys"][sidx][vidx])

                        print("Step:", global_step, "| Train Loss:", tot_loss.item())
                        print("Train Tok Acc:",  tot_tok)
                        s = "\tM1->M1: " + str(round(tok_accs[(0,0,vidx)], 5))
                        if len(models)>1:
                            s += " | M1->M2: " + str(round(tok_accs[(0,1,vidx)],5))
                            s += "\n\tM2->M1: " + str(round(tok_accs[(1,0,vidx)], 5))
                            s += " | M2->M2: " + str(round(tok_accs[(1,1,vidx)],5))
                        print(s)

                        print("Train Trial Acc:",tot_trial)
                        s = "\tM1->M1: " + str(round(trial_accs[(0,0,vidx)], 5))
                        if len(models)>1:
                            s += " | M1->M2: " + str(round(trial_accs[(0,1,vidx)],5))
                            s += "\n\tM2->M1: " + str(round(trial_accs[(1,0,vidx)], 5))
                            s += " | M2->M2: " + str(round(trial_accs[(1,1,vidx)],5))
                        print(s)
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

                    print()
                    print("Layers:", config["layers"])
                    print("CauslModl:", config["cmodel_names"])
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
                    print()
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

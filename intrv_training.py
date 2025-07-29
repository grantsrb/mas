import torch
import numpy as np
from transformers import AutoTokenizer
import torch.nn.functional as F

from datas import collate_fn 
from utils import (
    collect_activations, device_fxn, tensor2str,
)
import seq_models as smods
from dl_utils.save_io import ( load_checkpoint, )
from dl_utils.utils import ( arglast, )
from dl_utils.tokenizer import Tokenizer
import constants as consts
from intrv_datas import make_intrv_data_from_seqs
from train import make_tokenizer_from_info

import pandas as pd # import after transformers to avoid versioning bug

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
        cl_vectors=None,
        tokenizer=None,
        pad_mask=None,
        config=dict(),
        verbose=False,
        vidx=None,
        tforce=False,
        track_grad=True,
        cl_divergence=False,
        baseline_losses=None,
        baseline_accs=None,
    ):
    """
    Args:
        src_activations: torch tensor (B,S,D)
        cl_vectors: torch tensor (B,S,D)
            use the trg_swap_mask to isolate the cl latent targets
        cl_divergence: bool
            track the kl divergence between the intervened vectors
            and the counterfactual latents.
        loss_divisor: tensor (B,)
            optionally argue losss to divide the losses by
        trial_divisor: tensor (B,)
            optionally argue trial accs to divide the trial acc by
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
    comms_dict["src_activations"] = src_activations[torch.LongTensor(batch_indices)]
    comms_dict["src_activations"] = comms_dict["src_activations"].to(device)
    input_ids = batch["input_ids"].clone()

    swap_mask = None
    if "trg_swap_masks" in batch:
        comms_dict["trg_swap_masks"] = batch["trg_swap_masks"]
        comms_dict["src_swap_masks"] = batch["src_swap_masks"]
        swap_mask = batch["trg_swap_masks"]>=0
    if swap_mask is not None and config.get("stepwise", True):
        if const_targ_inpt_id:
            print("Using Const Targ Ids")
            resp_id = config.get("resp_id", 6)
            input_ids[swap_mask] = int(resp_id)
        elif shuffle_targ_ids:
            print("Shuffling Targ Ids")
            # Shuffles the input ids
            msums = swap_mask.long().sum(-1)
            perms = [torch.randperm(s).long() for s in msums]
            perm = [perms[i+1]+len(perms[i]) for i in range(len(perms)-1)]
            perm = torch.cat([perms[0]] + perm)
            input_ids[swap_mask] = input_ids[swap_mask][perm.to(device)]
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

    #print("logits", logits.shape)
    #for k,v in batch.items():
    #    try:
    #        print(k,v.shape)
    #    except: print("fail:", k)

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

    #### TODO
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
        labels[lmask.reshape(-1)],
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
    if cl_vectors is not None and "intrv_vecs" in comms_dict:
        cl_vectors = cl_vectors[batch_indices].to(device)
        prev_grad_state = torch.is_grad_enabled()
        enable = track_grad and (sidx,tidx) in config["cl_directions"]
        torch.set_grad_enabled(enable)
        try:
            intrv_vecs = comms_dict["intrv_vecs"]
            if type(intrv_vecs)!=torch.Tensor:
                intrv_vecs = torch.stack(comms_dict["intrv_vecs"],dim=1)
            cl_loss = cl_loss_fxn(
                intrv_vecs=intrv_vecs,
                cl_vectors=cl_vectors,
                swap_mask=batch["trg_swap_masks"]>=0,
                loss_type=config.get("cl_loss_type", "both"),
            )
            if cl_divergence:
                cl_div, cl_sdx = cl_kl_divergence(
                    intrv_vecs=intrv_vecs,
                    cl_vecs=cl_vectors,
                    swap_mask=batch["trg_swap_masks"]>=0,
                    laplace=1,
                )
        except:
            print("Error in cl loss calculations")
        torch.set_grad_enabled(prev_grad_state)

    if "outp_tmask" in batch:
        tmask = batch["outp_tmask"].to(device)
    else:
        tmask = batch["outp_attn_mask"]
    trial = torch.zeros_like(batch["labels"]).bool()
    pids = torch.argmax(logits, dim=-1)
    labels = batch["labels"]
    eq = pids[tmask]==labels[tmask]
    trial[tmask] = eq
    trial = trial.sum(-1)
    trial_acc = trial==tmask.float().sum(-1)
    trial_acc = trial_acc.float().mean()
    tok_acc = eq.float().mean()

    prop_acc, prop_loss = torch.zeros(1).float(), torch.zeros(1).float()
    if baseline_accs is not None and baseline_losses is not None:
        base = baseline_accs[batch_indices].mean()
        prop_acc = (base-tok_acc)/base
    
        base = baseline_losses[batch_indices].mean()
        with torch.no_grad():
            prop_loss = (loss-base)/base

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
            #print("OuTmsk:", tensor2str(batch["outp_tmask"][i].long()))
            #print("TrgSwp:", tensor2str(batch["trg_swap_masks"][i].long()))
            #print("LosMsk:", tensor2str(lmask[i].long()))
            #print("InptPd:", tensor2str(pmask[i].long()))
            #print("OutpPd:", tensor2str(omask[i].long()))
            print()
            print("Inpts:", tensor2str(inpts[i][:trg_swap]))
            print("Gtrth:", tensor2str(labels[i][trg_swap-1:]))
            print("Preds:", tensor2str(outs[i][trg_swap-1:]))
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

    if baseline_accs is not None and baseline_losses is not None:
        if cl_divergence:
            return loss, cl_loss, tok_acc, trial_acc, cl_div, cl_sdx, prop_loss,prop_acc
        return loss, cl_loss, tok_acc, trial_acc, prop_loss,prop_acc
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
        return mse_loss
    if "cosine" in fxn_name or "cos" in fxn_name:
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

def cl_loss_fxn(intrv_vecs, cl_vectors, swap_mask, loss_type="both"):
    """
    Args:
        intrv_vecs: torch tensor (B,S,D)
            the intervened vectors
        cl_vectors: torch tensor (B,S,D)
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
    labls = cl_vectors[swap_mask]
    loss_fxn = get_loss_fxn(loss_type)
    return loss_fxn(preds,labls).mean()

def get_cl_vectors(
    model,
    trg_swap_mask,
    input_ids,
    layer,
    output_ids=None,
    device=None,
    idxs=None,
    bsize=500,
    tmask=None,
    idx_mask=None,
    preserve_dims=False,
):
    """
    A helper function for collecting the counterfactual
    latents.

    Args:
        model: torch module
        trg_swap_mask: bool tensor (B,S)
            the non-negative elements of the target swap mask
        input_ids: long tensor (B,S2)
            the input ids to produce the cl vectors
        idxs: torch Long tensor (N,2)
            a tensor in which the first column denotes
            the row index and the second column denotes
            the column index from which to take the
            latent vectors. N should be equal to the sum
            of all entries in the swap mask.
        idx_mask: torch bool tensor (B,S)
            optionally can argue a binary mask with 1s as the indexs
            for which we wish to harvest the counterfactual latents
            from the model's activations.
    """
    assert idxs is not None or idx_mask is not None
    assert not (idxs is None and idx_mask is None)
    model.eval()
    if device is not None: model.to(device)
    with torch.no_grad():
        outputs = collect_activations(
            model,
            input_ids=input_ids,
            layers=[layer],
            tforce=True,
            ret_pred_ids=True,
            batch_size=bsize,
            to_cpu=True,
            verbose=True,
            preserve_dims=preserve_dims,
        )

    # Quick Accuracy Check
    pred_ids = outputs["pred_ids"]
    corrects = torch.ones_like(pred_ids[:,:-1]).float()
    tmask = tmask.clone().bool() if tmask is not None else torch.ones_like(input_ids).bool()
    pids = pred_ids[:,:-1]
    if output_ids is not None:
        oids = output_ids[:,:-1]
    else:
        oids = input_ids[:,1:]
    tmask = tmask[:,1:] # remove the first token from the mask
    acc = pids[tmask].long()==oids[tmask].long()
    corrects[tmask] = acc.float()
    corrects = corrects.sum(-1)==corrects.shape[-1]
    print("CL Token Acc:", acc.float().mean().item())
    print("CL Trial Acc:", corrects.float().mean().item())
    print()

    actvs = outputs[layer][:,:-1]
    shape = tuple([*trg_swap_mask.shape, actvs.shape[-1]])
    cl_vectors = torch.empty(shape)
    if idxs is not None:
        assert trg_swap_mask.long().sum()==len(idxs)
        cl_vectors[trg_swap_mask] = actvs[idxs[:,0],idxs[:,1]]
    else:
        assert trg_swap_mask.long().sum()==idx_mask.long().sum()
        cl_vectors[trg_swap_mask.bool()] = actvs[idx_mask.bool()]
    return cl_vectors

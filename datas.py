import os
from utils import (
    load_json, default_to_list, extract_ids
)
from dl_utils.utils import (
    pad_to, arglast, get_mask_past_idx,
)
from utils import run_cmodel_to_completion
from datasets import load_dataset, Dataset
import pyarrow as pa
import pyarrow.dataset as ds
import torch
from constants import DEFAULT_REPLACEMENTS

import tasks

def gsm8k_tokenize_training(example, tokenizer, config, prompt=""):
    text = prompt + example["question"] + "\nAnswer:" + example["answer"]
    return tokenizer(text, truncation=True, padding="max_length", max_length=config["max_length"])

def numequiv_tokenize_training(example, tokenizer, config):
    text = example["text"]
    return tokenizer(text, truncation=True, padding="max_length", max_length=config["max_length"])[0]

def get_task_generated_dataset(
        n_samples,
        task_type,
        task_config=dict(),
    ):
    task = getattr(tasks, task_type)(**task_config)
    samps, tmasks, _ = task.generate_samples(n_samples)
    samps = [" ".join(samp) for samp in samps]
    return {"text": samps, "task_mask": tmasks}

def get_dataset(
        dataset_name,
        n_samples=10000,
        data_path=None,
        task_type=None,
        task_config=None,
        **kwargs):
    """
    Returns sequences that do not have a bos token, but do have some sort
    of eos indication due to the task.
    """
    if dataset_name=="gsm8k":
        return load_dataset(dataset_name, **kwargs)
    elif dataset_name=="task":
        d = get_task_generated_dataset(
            n_samples=n_samples,
            task_type=task_type,
            task_config=task_config,
        )
    ###elif dataset_name=="num_equivalence":
    ###    if kwargs.get("split", "train")=="train":
    ###        if not data_path:
    ###            data_path = "./data/multiobj_systematic_10000.json"
    ###        path = os.path.abspath(os.path.expanduser(data_path))
    ###    else:
    ###        if not data_path:
    ###            data_path = "./data/multiobj_systematic_1000.json"
    ###        path = os.path.abspath(os.path.expanduser(data_path))
    ###    d = load_json(path) #[{"text": t} for t in load_text(file_name=path)]
    return Dataset.from_dict(d)

def generate_token_ids_from_cmodel(n_samples, cmodel, info):
    """
    Uses a causal model to generate a dataset.

    Args:
        n_samples: int
        cmodel: CausalModel
        info: dict
    """
    samples = []
    task_masks = []
    meta_data = []
    for i in range(n_samples):
        varbs = cmodel.init_varbs
        inpt_token = info["bos_token_id"]
        seq, varbs_list, tmask = run_cmodel_to_completion(
            cmodel=cmodel,
            inpt_token=inpt_token,
            varbs=varbs,
            info=info,
            end_tokens={info.get("eos_token_id", None)},
        )
        samples.append( seq )
        task_masks.append( tmask )
        meta_data.append(varbs_list[-1])
    return samples, task_masks, meta_data

def extend_example(ex, seq_len, pad_side="left"):
    for k in ex:
        t = type(ex[k][0])
        ex[k] = pad_to( 
            arr=ex[k],
            tot_len=seq_len,
            fill_val=t(0),
            side=pad_side,
        )
    return ex

def ensure_equal_length(dsets, pad_sides="left"):
    pad_sides = default_to_list(pad_sides, n_el=len(dsets))
    len1 = len(dsets[0]["input_ids"][0])
    len2 = len(dsets[1]["input_ids"][0])
    if len1 > len2:
        idx = 1
        seq_len = len1
    elif len2 > len1:
        idx = 0
        seq_len = len2
    else:
        return dsets

    dset = dsets[idx].map(
        lambda ex: extend_example(ex, seq_len=seq_len, pad_side=pad_sides[idx]),
        batched=False,
    )
    dsets[idx] = dset
    return dsets

def get_swap_idxs(token_ids, replace_dict, tokenizer):
    """
    Determines the swap indices when using a prompt and
    a multi-token trigger. Returns a long tensor with ordered
    indices according to their swap position. The non-swap
    value is -1.

    Args:
        token_ids: torch tensor (B,S)
        replace_dict: dict
            'demo_word0': str
            'demo_word1': str
            'demo_word2': str
            'trig_word': str
            'resp_word': str
            'done_word': str
            'user_word': str
            'asst_word': str
    """
    mask_past_arglast = torch.ones_like(token_ids).bool()
    if "user_word" in replace_dict and replace_dict["user_word"]:
        user_word = replace_dict.get("user_word", "User:")
        user_ids = []
        uids = extract_ids( string=user_word, tokenizer=tokenizer, )
        user_ids += [int(u) for u in uids]
        uids = extract_ids( string=" " + user_word, tokenizer=tokenizer, )
        user_ids += [int(u) for u in uids]
        mask = torch.isin(token_ids, torch.LongTensor(user_ids))
        mask = mask[:,:-1]&mask[:,1:] 
        uidxs = arglast(mask, dim=-1)+1
        mask_past_arglast = get_mask_past_idx(
            shape=token_ids.shape,
            idx=uidxs,
            inclusive=False,
        )
    mask_keys = [
        "demo_word0", "demo_word1", "demo_word2",
        "trig_word", "resp_word", 
    ]
    mask_vals = []
    for k in mask_keys:
        k_ids = []
        text = replace_dict[k]
        k_ids =[int(kid) for kid in extract_ids(string=text, tokenizer=tokenizer)]+\
               [int(kid) for kid in extract_ids(string=" "+text,tokenizer=tokenizer)]
        mask_vals += k_ids
    mask_vals = torch.LongTensor(list(set(mask_vals)))
    isin = torch.isin(token_ids, mask_vals)
    mask = mask_past_arglast & isin
    idxs = -torch.ones_like(mask).long()
    idxs[mask] = torch.cat([torch.arange(s.sum()).long() for s in mask.long()])
    #try:
    #    print("Toks:", tokenizer.decode(token_ids[0][-20:]))
    #    print("MPAL:", mask_past_arglast[0])
    #    print("Isin:", isin[0])
    #    print("Mask:", mask[0])
    #    print("mask vals:", [tokenizer.decode(m) for m in mask_vals])
    #except: pass
    return idxs

def collate_fn(batch_indices, tokenized_dataset, device=0, incl_src=False):
    """
    A simple collate function that “batches” the tokenized examples.

    Attention masks use 1 to denote not padding tokens
    """
    batch = tokenized_dataset.select(batch_indices)
    d = {
        "input_ids":      torch.tensor(batch["trg_input_ids"])[...,:-1],
        "inpt_attn_mask": torch.tensor(batch["trg_inpt_attn_masks"])[...,:-1],
        "outp_attn_mask": torch.tensor(batch["trg_outp_attn_masks"])[...,1:],
        "labels":         torch.tensor(batch["trg_input_ids"])[...,1:],
        "src_input_ids":  torch.tensor(batch["src_input_ids"])[...,:-1],
    }
    if incl_src:
        d = {
          **d,
          "src_attention_mask": torch.tensor(batch["src_inpt_attn_masks"])[...,:-1],
          "src_outp_attn_mask": torch.tensor(batch["src_outp_attn_masks"])[...,1:],
          "src_labels":         torch.tensor(batch["src_input_ids"])[...,1:],
        }
    try:
        d["input_tmask"] = torch.tensor(batch["trg_task_masks"])[...,:-1].bool()
        d["outp_tmask"] = torch.tensor(batch["trg_task_masks"])[...,1:].bool()
        if incl_src:
            d["src_input_tmask"] = torch.tensor(
                batch["src_task_masks"])[...,:-1].bool()
            d["src_outp_tmask"] = torch.tensor(
                batch["src_task_masks"])[...,1:].bool()
    except: pass
    try:
        d["trg_swap_masks"] = torch.tensor(batch["trg_swap_masks"])[...,:-1]
        d["src_swap_masks"] = torch.tensor(batch["src_swap_masks"])[...,:-1]
    except: pass
    try:
        d["trg_swap_idxs"] = torch.LongTensor(batch["trg_swap_idxs"])
        d["src_swap_idxs"] = torch.LongTensor(batch["src_swap_idxs"])
    except: pass
    try:
        d["cl_idxs"] = torch.LongTensor(batch["cl_idxs"])
        d["cl_input_ids"] = torch.LongTensor(batch["cl_input_ids"])
    except: pass
    # In a standard LM objective the labels are the input_ids (shifted internally
    # by the model), but we don't do that
    return {k:v.to(device) for k,v in d.items()}

def replace_text(text, replacement_dict=DEFAULT_REPLACEMENTS):
    for k,v in replacement_dict.items():
        text = text.replace(k, v)
    return text

def get_max_length(text, tokenizer):
    toks = tokenizer(text, return_tensors="pt")["input_ids"]
    return toks.shape[-1] + 20*2 + 2

def add_token_ids_to_info(info, tokenizer):
    keys = list(info.keys())
    for k in keys:
        if not info[k]: continue
        if "tokens" in k and not "_id" in k:
            key_id = k[:-1] + "_ids"
            try:
                token_ids = [
                    int(tokenizer(tok)["input_ids"][-1]) for tok in info[k]
                ]
            except:
                token_ids = [
                    int(tokenizer.word2id[tok]) for tok in info[k]
                ]
            info[key_id] = token_ids
        elif "token" in k and not "_id" in k:
            key_id = k + "_id"
            try:
                info[key_id] = int(tokenizer(info[k])["input_ids"][-1])
            except:
                info[key_id] = int(tokenizer.word2id[info[k]])
    return info

def make_tokenized_info(replacements, tokenizer, config):
    """
    This func adds token ids to the info that only contains
    token strings. Any key that has the word "token" in it will
    be converted.

    Args:
        kwrgs: dict
            str: str
        tokenizer: Tokenizer
        config: dict
    Returns:
        info: dict
    """
    info = dict()
    info["bos_token"] = tokenizer.bos_token
    info["eos_token"] = tokenizer.eos_token
    info["pad_token"] = tokenizer.pad_token
    for k in replacements:
        info_key = "_tokens"
        if "demo" in k:
            info_key = "demo" + info_key
        elif "resp" in k:
            info_key = "resp" + info_key
        elif "done" in k:
            info_key = "eos_token"
            info[info_key] = replacements[k]
            continue
        elif "trig" in k:
            info_key = "trig" + info_key
        else:
            continue
        if info_key not in info:
            info[info_key] = []
        if replacements[k]:
            info[info_key].append(replacements[k])
    
    info = add_token_ids_to_info(info)
    return info

def add_prompt(
    data_dict,
    src_tokenizer,
    trg_tokenizer,
    src_prompt,
    trg_prompt,
    src_replacements,
    trg_replacements,
):
    """
    Prepends the prompt to the input ids and appropriately adjusts the
    indices and masks in the data_dict.

    Args:
        data_dict: dict
        src_tokenizer: Tokenizer
        trg_tokenizer: Tokenizer
        src_prompt: str
        trg_prompt: str
        src_replacements: dict
        trg_replacements: dict
    """
    src_prompt = replace_text(
        text=src_prompt, replacement_dict=src_replacements)
    trg_prompt = replace_text(
        text=trg_prompt, replacement_dict=trg_replacements)
    prompts = [src_prompt, trg_prompt]
    tokenizers = [src_tokenizer, trg_tokenizer]
    keys = ["src", "trg"]
    for prompt,tokenizer,key in zip(prompts,tokenizers,keys):
        if len(prompt)==0: continue
        ids = tokenizer(
            prompt,
            padding=None,
            return_tensors=False,
            truncation=False,
        )["input_ids"][0]
        el = len(ids)

        for k in data_dict:
            if key in k:
                if "input_ids" in k:
                    data_dict[k] = map(lambda x: [*ids] + x, data_dict[k])
                elif "idxs" in k:
                    data_dict[k] = map(lambda x: x+el, data_dict[k])
                elif "attention" in k or "attn" in k:
                    mask = [1 for _ in range(el)]
                    data_dict[k] = map(lambda x: mask + x, data_dict[k])
                elif "swap" in k:
                    mask = [-1 for _ in range(el)]
                    data_dict[k] = map(
                        lambda x: mask + [xx+el for xx in x],
                        data_dict[k],
                    )
                elif "task" in k:
                    mask = [0 for _ in range(el)]
                    data_dict[k] = map(lambda x: mask + x, data_dict[k])
    return data_dict

def tokenize_dataset(dataset, tokenizer, config):
    """
    Replaces text specified in the replacements dict, prepends a prompt,
    and tokenizes the text.
    """
    reps = config.get("replacements", DEFAULT_REPLACEMENTS)
    text = dataset.map(
        lambda ex: {
            "text": replace_text(
                text=ex["text"],
                replacement_dict=reps
            )
        },
        batched=False,
    )
    text = text["text"]
    add_eos = tokenizer.eos_token != reps["E"]
    if add_eos: 
        for i,t in enumerate(text):
            text[i] = text[i] + tokenizer.eos_token


    print("Text Sample:", text[0])
    print("Tokenizing...")
    tok_dict = tokenizer(
        text,
        padding=None,
        return_tensors=False,
        truncation=False,
    )
    bos_id = tokenizer.bos_token_id
    token_ids = tok_dict["input_ids"]
    token_ids = [[t for t in seq if t!=bos_id] for seq in token_ids]
    attn_masks = [[1 for _ in seq[:-1]]+[0] for seq in token_ids]
    task_masks = [tmask for tmask in dataset["task_mask"]]
    return Dataset.from_dict({
        "input_ids": token_ids,
        "inpt_attn_mask": attn_masks,
        "task_mask": task_masks,
    })

    #idx = tok_dict["input_ids"]==bos_id
    #dupls = idx.long().sum(-1)>1
    #idxs = torch.argmax(idx.long(), dim=-1)[dupls]
    #arng = torch.arange(len(idx)).long()[dupls]
    #tok_dict["input_ids"][arng, idxs] = tokenizer.pad_token_id
    #tok_dict["inpt_attn_mask"] = tok_dict["input_ids"]!=tokenizer.pad_token_id

    #try:
    #    print()
    #    print("Swaps:")
    #    for i in range(3):
    #        print("Full:", tok_dict["input_ids"][i][-20:])
    #        print("Swap:", tok_dict["input_ids"][i][swap_idxs[i]])
    #        print("Full:", [tokenizer.decode(t) for t in tok_dict["input_ids"][i][-20:]])
    #        print("Swap:", tokenizer.decode(tok_dict["input_ids"][i][swap_idxs[i]]))
    #        print()
    #    print()
    #except: pass

    #if "task_mask" in dataset.column_names:
    #    max_length = tok_dict["input_ids"].shape[-1]
    #    tmasks = []
    #    for i,tmask in enumerate(dataset["task_mask"]):
    #        # two 0s for annoying HF BOS business...
    #        bos_zeros = [0,0] if dupls[i] else [0]
    #        eos_zero = [0] if add_eos else []
    #        tmask = bos_zeros + tmask + eos_zero
    #        tmasks.append(pad_to(
    #            arr=tmask,
    #            tot_len=max_length,
    #            fill_val=0,
    #            side=tokenizer.padding_side,
    #        ))
    #    tok_dict["task_mask"] = torch.BoolTensor(tmasks)
    #    eos_ids = [tokenizer.eos_token_id]
    #    dword = reps["done_word"]
    #    try:
    #        eos_ids.append(int(tokenizer(dword)["input_ids"][-1]))
    #    except: pass
    #    try:
    #        eos_ids.append(
    #            int(tokenizer(" "+dword)["input_ids"][-1]))
    #    except: pass
    #    eos_ids = torch.LongTensor(eos_ids)
    #    in_eos_ids = torch.isin(tok_dict["input_ids"].long(),eos_ids)
    #    eos_and_tmask = in_eos_ids&tok_dict["task_mask"]
    #    tok_dict["inpt_attn_mask"] = tok_dict["inpt_attn_mask"]&~eos_and_tmask

    #    # Quick Tests
    #    tmask = tok_dict["task_mask"][0]
    #    assert torch.isin(tok_dict["input_ids"][0][tmask], eos_ids).float().sum()<=1

def pad_data_dict(
    data_dict,
    src_pad_id,
    trg_pad_id,
    src_pad_side,
    trg_pad_side,
):
    """
    Utility function for padding the data dict. Operates in place.
    """
    max_len = int(max(
        np.max([len(s) for s in data_dict["trg_input_ids"]]),
        np.max([len(s) for s in data_dict["src_input_ids"]]),
    ))
    for k in data_dict:
        if "src" in k:
            left = int(src_pad_side=="left")
            pad_id = src_pad_id
            off_key = "src_pad_ids"
        elif "trg" in k:
            left = int(trg_pad_side=="left")
            pad_id = trg_pad_id
            off_key = "trg_pad_ids"
        else:
            print("Skipping", k, "in padding")
            continue

        for i in range(len(data_dict["trg_input_ids"])):
            offset = max_len - len(data_dict[off_key][i])
            if "idx" in k and left:
                data_dict[k][i] += offset
            elif "mask" in k:
                mask = [False for _ in range(offset)]
            elif "input_id" in k:
                mask = [pad_id for _ in range(offset)]
            data_dict[k][i] = left*mask + data_dict[k][i] + mask*(1-left)

    return data_dict
        
def add_pad_masks(data_dict, src_info, trg_info):
    """
    Adds the keys to the data_dict:
        - src_inpt_attn_masks
        - src_outp_attn_masks
        - trg_inpt_attn_masks
        - trg_outp_attn_masks
    """
    for k,info in zip(["src","trg"], [src_info,trg_info]):
        pad_id = info["pad_token_id"]
        bos_id = info["bos_token_id"]
        eos_id = info["eos_token_id"]
        inpt_ids = torch.LongTensor(data_dict[k+"_input_ids"])
        attn_mask = (inpt_ids!=pad_id)
        eos_mask = torch.ones_like(inpt_ids)
        rows = torch.arange(len(eos_mask)).long()
        eos_mask[rows, arglast(inpt_ids==eos_id, axis=-1)] = 1
        data_dict[k+"_inpt_attn_masks"] = attn_mask&~eos_mask.bool()
        data_dict[k+"_outp_attn_masks"] = attn_mask&(inpt_ids!=bos_id)
    return data_dict

if __name__=="__main__":
    text = "hey there foo"
    assert replace_text(text, {"foo": "shoo"})=="hey there shoo"

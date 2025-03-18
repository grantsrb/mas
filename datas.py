import os
from utils import (
    load_json, default_to_list, extract_ids
)
from dl_utils.utils import (
    pad_to, arglast, get_mask_past_idx,
)
from datasets import load_dataset, Dataset
import pyarrow as pa
import pyarrow.dataset as ds
import torch

def gsm8k_tokenize_training(example, tokenizer, config, prompt=""):
    text = prompt + example["question"] + "\nAnswer:" + example["answer"]
    return tokenizer(text, truncation=True, padding="max_length", max_length=config["max_length"])

def numequiv_tokenize_training(example, tokenizer, config):
    text = example["text"]
    return tokenizer(text, truncation=True, padding="max_length", max_length=config["max_length"])[0]

def get_dataset(dataset_name, data_path=None, **kwargs):
    if dataset_name=="gsm8k":
        return load_dataset(dataset_name, **kwargs)
    elif dataset_name=="num_equivalence":
        if kwargs.get("split", "train")=="train":
            if not data_path:
                data_path = "./data/multiobj_systematic_10000.json"
            path = os.path.abspath(os.path.expanduser(data_path))
        else:
            if not data_path:
                data_path = "./data/multiobj_systematic_1000.json"
            path = os.path.abspath(os.path.expanduser(data_path))
        d = load_json(path) #[{"text": t} for t in load_text(file_name=path)]
        dset = Dataset.from_dict(d)
        return dset

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
        "trig_word", "resp_word", "done_word",
    ]
    mask_vals = []
    for k in mask_keys:
        k_ids = []
        text = replace_dict[k]
        k_ids = [int(kid) for kid in extract_ids(string=text, tokenizer=tokenizer)]+\
                [int(kid) for kid in extract_ids(string=" "+text, tokenizer=tokenizer)]
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

def collate_fn(batch_indices, tokenized_dataset, device=0):
    """
    A simple collate function that “batches” the tokenized examples.
    """
    batch = tokenized_dataset.select(batch_indices)
    d = {
        "input_ids":      torch.tensor(batch["input_ids"])[...,:-1],
        "attention_mask": torch.tensor(batch["attention_mask"])[...,:-1],
        "labels":         torch.tensor(batch["input_ids"])[...,1:],
    }
    try:
        d["task_mask"] = torch.tensor(batch["task_mask"])[...,1:].bool()
    except: pass
    try:
        d["swap_idxs"] = torch.tensor(batch["swap_idxs"])[...,:-1]
    except: pass
    # In a standard LM objective the labels are the input_ids (shifted internally by the model)
    return {k:v.to(device) for k,v in d.items()}

default_replacement_dict = {
    "demo_word0": "D0",
    "demo_word1": "D1",
    "demo_word2": "D2",
    "asst_word": "",
    "user_word": "",
    "trig_word": "T",
    "resp_word": "R",
    "done_word": "<EOS>",
}

def replace_text(text, replacement_dict=default_replacement_dict):
    for k,v in replacement_dict.items():
        text = text.replace(k, v)
    return text

def get_max_length(text, tokenizer):
    toks = tokenizer(text, return_tensors="pt")["input_ids"]
    return toks.shape[-1] + 20*2

def tokenize_dataset(dataset, tokenizer, config):
    prompt = config.get("prompt", "")
    if config["dataset_names"]=="gsm8k":
        raise NotImplemented
        # Needs swap mask and ideally task mask
        tokenized = dataset.map(
            lambda ex: gsm8k_tokenize_training(
                ex, tokenizer, config=config, prompt=prompt,
            ),
            batched=False)
    else:
        bos = tokenizer.bos_token
        reps = config.get("replacements", default_replacement_dict)
        prompt = replace_text(text=prompt, replacement_dict=reps)
        text = dataset.map(
            lambda ex: {
                "text": replace_text(
                    text=bos+" "+prompt+ex["text"],
                    replacement_dict=reps
                )
            },
            batched=False,
        )
        text = text["text"]
        add_eos = tokenizer.eos_token != reps["done_word"]
        for i,t in enumerate(text):
            if add_eos: text[i] = text[i] + tokenizer.eos_token


        max_length = get_max_length(text[0], tokenizer)
        print("Tokenizing")
        try:
            tok_dict = tokenizer(
                text,
                padding="max_length",
                return_tensors="pt",
                max_length=max_length,
                add_bos=False,
            )
        except:
            tok_dict = tokenizer(
                text,
                padding="max_length",
                return_tensors="pt",
                max_length=max_length,
            )
        idx = tok_dict["input_ids"]==tokenizer.bos_token_id
        dupls = idx.long().sum(-1)>1
        idxs = torch.argmax(idx.long(), dim=-1)[dupls]
        arng = torch.arange(len(idx)).long()[dupls]
        tok_dict["input_ids"][arng, idxs] = tokenizer.pad_token_id
        tok_dict["attention_mask"] = tok_dict["input_ids"]!=tokenizer.pad_token_id

        swap_idxs = get_swap_idxs(
            token_ids=tok_dict["input_ids"],
            replace_dict=reps,
            tokenizer=tokenizer)
        tok_dict["swap_idxs"] = swap_idxs

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

        if "task_mask" in dataset.column_names:
            max_len = tok_dict["input_ids"].shape[-1]
            tmasks = []
            for i,tmask in enumerate(dataset["task_mask"]):
                # two 0s for annoying HF BOS business...
                bos_zeros = [0,0] if dupls[i] else [0]
                eos_zero = [0] if add_eos else []
                tmask = bos_zeros + tmask + eos_zero
                tmasks.append(pad_to(
                    arr=tmask,
                    tot_len=max_len,
                    fill_val=0,
                    side=tokenizer.padding_side,
                ))
            tok_dict["task_mask"] = tmasks

            # Quick Tests
            assert len(swap_idxs)==len(tmasks) and len(swap_idxs[0])==len(tmasks[0])
            tmask = torch.BoolTensor(tok_dict["task_mask"][0])
            eos_ids = torch.LongTensor([
                tokenizer.convert_tokens_to_ids(reps["done_word"]),
                tokenizer.convert_tokens_to_ids(" "+reps["done_word"]),
            ])
            assert torch.isin(tok_dict["input_ids"][0][tmask], eos_ids).float().sum()<=1
        tokenized = Dataset.from_dict(tok_dict)
    return tokenized

if __name__=="__main__":
    text = "hey there foo"
    assert replace_text(text, {"foo": "shoo"})=="hey there shoo"

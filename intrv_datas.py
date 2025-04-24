"""
This module contains the code to create intervention
samples from an existing dataset, causal model, and index filter.
"""
import copy
import pandas as pd
import numpy as np
import torch
from datasets import Dataset

from dl_utils.utils import pad_to

def pad_seqs(data, max_len):
    for k in data:
        if "mask" in k or "seq" in k:
            data[k] = pad_to(
                data[k],
                tot_len=max_len,
                fill_val=0,
            )
    return data

def collect_varbs(seq, cmodel, varbs=None, info=None):
    """
    Collected the variables from the causal model after each input
    token.
    
    Args:
        seq: list of tokenized strings
        cmodel: CausalModel
            accepts a token and a dict of variables
        varbs: (optional) dict of variables or None
            optionally argue variables to argue to the causal model
            with the first input token.
        info: (optional) dict
            some cmodels require additional info which can be specified
            in the info dict
    Returns:
        outp_tokens: list of tokens
            the output produced by the causal model after each
            input token.
        varb_list: list of dicts
            the variables produced by the causal model after each
            input token.
    """
    if varbs is None:
        varbs = cmodel.init_varbs
    varb_list = []
    outp_token_ids = []
    for inpt_token in seq:
        varb_list.append(copy.deepcopy(varbs))
        outp, varbs = cmodel(token_id=inpt_token, varbs=varbs, info=info)
        outp_token_ids.append(outp)
    return outp_token_ids, varb_list

def run_to_completion(
        cmodel,
        inpt_token,
        varbs=None,
        info=None,
        end_tokens={}
    ):
    """
    Runs a causal model until it produces None or a token contained
    in the end_tokens dict.

    Args:
        cmodel: CausalModel
            accepts a token and a dict of variables
        inpt_token: tokenized string
        varbs: (optional) dict of variables or None
            optionally argue variables to argue to the causal model
            with the first input token.
        info: (optional) dict
            some cmodels require additional info which can be specified
            in the info dict
        end_tokens: set
            optionally specify tokens that indicate the model is finished.
            None is assumed to be an end condition for all cmodels
    Returns:
        outp_tokens: list of tokens
            the output produced by the causal model after each
            input token.
        varb_list: list of dicts
            the input variables used by the causal model with each
            input token.
    """
    return run_for_n_steps(
        cmodel=cmodel,
        inpt_token=inpt_token,
        varbs=varbs,
        info=info,
        end_tokens=end_tokens,
        n_steps=None,
    )

def run_for_n_steps(
        cmodel,
        inpt_token,
        n_steps=None,
        varbs=None,
        info=None,
        end_tokens=set(),
    ):
    """
    Runs a causal model until it prduces None or a token contained in the end_tokens
    dict.

    Args:
        cmodel: CausalModel
            accepts a token and a dict of variables
        inpt_token: tokenized string
        n_steps: (optional) int or None
            if int, will run the causal model for only that many steps or until
            an end condition is met. If None, will run the causal model until an
            end condition.
        varbs: (optional) dict of variables or None
            optionally argue variables to argue to the causal model
            with the first input token.
        info: (optional) dict
            some cmodels require additional info which can be specified
            in the info dict
        end_tokens: set
            optionally specify tokens that indicate the model is finished.
            None is assumed to be an end condition for all cmodels
    Returns:
        outp_tokens: list of tokens
            the output produced by the causal model after each
            input token.
        varb_list: list of dicts
            the input variables used by the causal model with each
            input token.
    """
    if end_tokens is None: end_tokens = set()
    outp_token_ids = []
    if not varbs:
        varbs = cmodel.init_varbs
    varb_list = []
    token = inpt_token
    step = 0
    while n_steps is None or step < n_steps:
        step += 1
        varb_list.append(copy.deepcopy(varbs))
        token, varbs = cmodel(
            token_id=token, varbs=varbs, info=info)
        outp_token_ids.append(token)
        if token in end_tokens or token is None:
            break
    return outp_token_ids, varb_list

def get_varbs_at_idx(seq, cmodel, idx, info=None):
    """
    Utility function to get the variables at a particular index
    in the seq. Finds the variables corresponding to the inputs
    of the index. So, an idx of 0 would return the init_varbs.

    Args:
        seq: list of str
        cmodel: python_function()
            causal model
        idx: int
            inclusive index to collect varbs used as the input to
            this index in the sequence
        info: dict
            extra information for the causal model
    Returns:
        varbs: dict
    """
    _, varb_list = collect_varbs(
        seq=seq[:idx+1], cmodel=cmodel, info=info)
    return varb_list[-1]

def make_df_from_seqs(seqs, cmodel, info=None):
    """
    Constructs a dataframe from the sequences and the causal model.

    Args:
        seqs: torch tensor
        cmodel: CausalModel
            accepts a token id and a dict of variables
        info: (optional) dict
            some cmodels require additional info which can be specified
            in the info dict
    Returns:
        df: pd DataFrame
            cols:
                'sample_idx': int
                    the index of the data sample sequence
                'step_idx': int
                    the index of the step in the sequence
                'inpt_token_id': int
                'outp_token_id': int
                '<varb_key1>': object
                    the input variable value corresponding to
                    <varb1>
    """
    df_dict = {
        "sample_idx": [],
        "step_idx": [],
        "inpt_token_id": [],
        "outp_token_id": [],
        **{k: [] for k in cmodel.init_varbs},
    }
    for sample_idx,seq in enumerate(seqs):
        varbs = cmodel.init_varbs
        for step_idx, inpt_token in enumerate(seq):
            df_dict["sample_idx"].append(sample_idx)
            df_dict["step_idx"].append(step_idx)
            df_dict["inpt_token_id"].append(int(inpt_token))
            for k in varbs:
                df_dict[k].append(varbs[k])
            outp_token, varbs = cmodel(
                token_id=int(inpt_token), varbs=varbs, info=info)
            df_dict["outp_token_id"].append(outp_token)
    df = pd.DataFrame(df_dict)
    return df

def sample_swaps(df, filter, info=None, stepwise=False):
    """
    Returns a swap mask and swap variables for each sample in the
    dataframe. Optionally specify the number of indices to sample,
    and specify whether you would like the mask to be contiguous
    from the start. 

    Args:
        df: pd DataFrame
            a dataframe containing the data
        filter: python_function(df_row, info)
            a python function that takes a dataframe row and an info
            dict and returns filtered indices (indices that we would
            like to sample from)
        info: (optional) dict or None
            optionally specify info for the filter to use in making
            its decisions.
        stepwise: bool
            if true, will create a contiguous swap mask from the
            first token in the sequence to the maximum sampled
            intervention index.
    Returns:
        swap_masks: list of bools
            returns a list of masks in which true denotes that the
            input index should be swapped
        swap_idxs: list of ints
            the last index of the swaps for each sample
        swap_varbs: list of dicts
            a snapshot of the variables at the last swap index
        max_len: int
            the maximum sequence length
    """
    df["is_valid"] = df.apply(
        lambda x: filter(x, info=info), axis=1)
    df["max_step"] = df.groupby(["sample_idx"])["step_idx"]\
        .transform("max")
    samples = df.loc[df["is_valid"]]\
            .groupby(["sample_idx"])\
            .sample(1)
    swap_masks = []
    swap_idxs = [] # the maximum swap index for each sample
    swap_varbs = [] # the input variables at the swap
    for sample in samples:
        swap_idx = int(sample["step_idx"])
        swap = np.zeros(int(sample["max_step"]), dtype=bool)
        if stepwise:
            swap[:swap_idx] = True
        else:
            swap[swap_idx] = True
        swap_masks.append(swap)
        swap_idxs.append(swap_idx)
        swap_varbs.append(dict(sample))
    return swap_masks, swap_idxs, swap_varbs

def make_counterfactual_seqs(
        trg_seqs,
        trg_swap_keys,
        src_swap_keys,
        trg_swap_idxs,
        src_swap_varbs,
        trg_swap_varbs,
        trg_task_masks,
        trg_cmodel,
        trg_info=None,
        *args, **kwargs,
    ):
    """
    This function does a lot of the heavy lifting for
    creating the intervention data.

    Args:
        trg_seqs: list of lists of tokenized strings
            the intial target sequences
        trg_swap_keys: list of str
            the name of the variable(s) to swap
        src_swap_keys: list of str
            the name of the variable(s) to swap
        trg_swap_idxs: list of bools
            a list of indexes corresponding to the intervention
            index in the target sequence for each sequence pair
        src_swap_varbs: list of dicts
            a list of input varbs corresponding to the intervention
            index in the source sequence for each sequence pair
        trg_swap_varbs: list of dicts
            a list of input varbs corresponding to the intervention
            index in the target sequence for each sequence pair
        trg_task_masks: list of lists of bools
            the task masks for the target sequences
        trg_cmodel: python_function(token, varbs, info)
            accepts a token and a dict of variables
        trg_info: dict
            optionally specify a dict of information to be used
            with the trg_cmodel
    """
    if "full" in trg_swap_keys:
        trg_swap_keys = list(trg_cmodel.init_varbs.keys())
        src_swap_keys = list(trg_cmodel.init_varbs.keys())
    elif type(trg_swap_keys)==str:
        trg_swap_keys = [trg_swap_keys]
        src_swap_keys = [src_swap_keys]
    z = zip(trg_seqs, trg_swap_idxs, trg_task_masks,
            src_swap_varbs, trg_swap_varbs)
    intrv_seqs = []
    intrv_varbs = []
    intrv_tmasks = []
    for seq_i, tup in enumerate(z):
        (trg_seq,trg_idx,trg_tmask,src_varbs,trg_varbs) = tup
        zkeys = zip(trg_swap_keys, src_swap_keys)
        trg_cmodel.queue_intervention(
            {tkey: src_varbs[skey] for tkey,skey in zkeys}
        )
        intrv_varbs = copy.deepcopy(trg_varbs)
        for trg_key,src_key in zip(trg_swap_keys, src_swap_keys):
            intrv_varbs[trg_key] = src_varbs[src_key]
        intrv_varbs.append(intrv_varbs)
        inpt_token = trg_varbs["inpt_token_id"]
        assert inpt_token==int(trg_seq[trg_idx])
        intrv_seq, intrv_tmask = run_to_completion(
            cmodel=trg_cmodel,
            inpt_token=inpt_token,
            varbs=trg_varbs,
            info=trg_info,
            end_tokens={trg_info.get("eos_token_id", None)},
        )
        intrv_seqs.append(
            trg_seq[:trg_idx+1] + intrv_seq
        )
        intrv_tmasks.append(
            trg_tmask[:trg_idx+1] + intrv_tmask
        )
    return intrv_seqs, intrv_varbs, intrv_tmasks

def make_intrv_data_from_seqs(
        trg_data,
        src_data,
        src_swap_keys,
        trg_swap_keys,
        src_cmodel,
        src_info,
        src_filter,
        trg_cmodel,
        trg_info,
        trg_filter,
        stepwise=False,
    ):
    """
    Constructs intervention data from the argued sequence pairs.
    Any referred to variables are always input variables instead
    of output variables. This makes the interventions easier.

    Args:
        trg_data: dict or ArrowDataset
            "input_ids": list of lists of token ids
            "task_mask": list of lists of bools
            "attention_mask": list of lists of bools
        src_seqs: dict or ArrowDataset
            "input_ids": list of lists of token ids
            "task_mask": list of lists of bools
            "attention_mask": list of lists of bools
        trg_task_masks: list of lists of bools
            the intial target sequence task masks
        src_task_masks: list of lists of bools
            the intial source sequence task masks
        trg_swap_keys: list of str
            keys used to determine which variables to replace in
            the target. If "full" is argued, uses all keys not
            specified in the ignore_keys in the CausalModel object.
            If empty list, will not perform an intervention.
        src_swap_keys: list of str
            keys used to determine which variables to take from the
            source.
        src_cmodel: python_function
        src_info: dict
        src_filter: python_function
        trg_cmodel: python_function
        trg_info: dict
        trg_filter: python_function
        stepwise: bool
    Returns:
        intrv_data: dict
            'src_seqs': list of lists of tokenized strings
                the intervention sample inputs for the source model
            'trg_seqs': list of lists of tokenized strings
                the intervention sample inputs for the target model
            'src_task_mask': list of lists of bools
            'trg_task_mask': list of lists of bools
                the task mask for the token inputs
            'src_swap_masks': list of lists of bools
                the swap mask for the source sequences
            'trg_swap_masks': list of lists of bools
                the swap mask for the target sequences
    """
    # 1. get the source variables and the swap indices
    src_seqs = src_data["input_ids"]
    src_task_masks = src_data["task_mask"]
    src_attn_masks = src_data["attention_mask"]
    src_df = make_df_from_seqs(
        seqs=src_seqs,
        cmodel=src_cmodel,
        info=src_info,
    )
    src_swap_masks, src_swap_idxs, src_swap_varbs = sample_swaps(
        df=src_df,
        filter=src_filter,
        info=src_info,
        stepwise=stepwise,
    )

    # 2. get the target variables and swap indices
    trg_seqs = trg_data["input_ids"]
    trg_task_masks = trg_data["task_mask"]
    trg_attn_masks = trg_data["attention_mask"]
    trg_df = make_df_from_seqs(
        seqs=trg_seqs,
        cmodel=trg_cmodel,
        info=trg_info,
    )
    if stepwise:
        trg_swap_masks = src_swap_masks
        trg_swap_idxs  = src_swap_idxs
        trg_swap_varbs = [
            get_varbs_at_idx(seq=tseq,cmodel=trg_cmodel, idx=tidx)\
                for tseq,tidx in zip(trg_seqs,trg_swap_idxs)
        ]
    else:
        trg_swap_masks, trg_swap_idxs, trg_swap_varbs = sample_swaps(
            df=trg_df,
            filter=trg_filter,
            info=trg_info,
            stepwise=stepwise,
        )

    # 3. Using the variables, seqs, and swap indices, create
    # intervention data.
    intrv_seqs, intrv_varbs, intrv_task_masks = make_counterfactual_seqs(
        trg_swap_keys=trg_swap_keys,
        src_swap_keys=src_swap_keys,
        trg_seqs=trg_seqs,
        trg_swap_idxs=trg_swap_idxs,
        trg_swap_varbs=trg_swap_varbs,
        trg_task_masks=trg_task_masks,
        src_swap_varbs=src_swap_varbs,
        trg_cmodel=trg_cmodel,
        trg_info=trg_info,
    )

    intrv_swap_masks = [
        pad_to(swap_mask, len(seq)) for swap_mask,seq in zip(trg_swap_masks, intrv_seqs)
    ]

    d = {
        "trg_seqs": intrv_seqs,
        "trg_varbs": intrv_varbs,
        "trg_swap_masks": intrv_swap_masks,
        "trg_task_masks": intrv_task_masks,
        "src_seqs": src_seqs,
        "src_varbs": src_swap_varbs,
        "src_swap_masks": src_swap_masks,
        "src_task_masks": src_task_masks,
    }
    max_len = int(np.max([len(seq) for seq in d["trg_seqs"]]))
    d = pad_seqs(d, max_len=max_len)
    for k in d: d[k] = torch.tensor(d[k])
    d["trg_attention_mask"] = d["trg_seqs"]!=trg_info["pad_token_id"]
    d["src_attention_mask"] = d["src_seqs"]!=src_info["pad_token_id"]
    return d
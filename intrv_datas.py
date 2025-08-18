"""
This module contains the code to create intervention
samples from an existing dataset, causal model, and index filter.
"""
import time
import copy
import pandas as pd
import numpy as np
import torch
from datasets import Dataset

from dl_utils.utils import (
    pad_to, get_nonzero_entries, get_mask_between, get_mask_past_idx,
    get_mask_past_arglast,
)
from utils import tensor2str, run_cmodel_to_completion

def pad_seqs(data, max_len, truncate=False):
    """
    Pads all of the sequences in data that have mask or seq
    or input_ids in their key name. Operates in place.

    args:
        data: dict
            all keys that contain "mask" or "seq" or "input_ids"
            are padded to the max_len arg
        max_len: int
        truncate: bool
            if true, will cut the sequences down to max_len
            if they exceed.
    Returns:
        data: dict
            the same data dict with padded sequences
    """
    for k in data:
        if "mask" in k or "seq" in k or "input_id" in k:
            for si,seq in enumerate(data[k]):
                data[k][si] = pad_to(
                    data[k][si],
                    tot_len=max_len,
                    fill_val=0,
                )
                if truncate:
                    data[k][si] = data[k][si][:max_len]
    return data

def collect_varbs(seq, cmodel, varbs=None, info=None, post_varbs=False):
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
        post_varbs: bool
            if true, will return the variables after each
            input token. Otherwise returns the variables
            before each token.
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
    task_mask = []
    for inpt_token in seq:
        if not post_varbs:
            varb_list.append(copy.deepcopy(varbs))
        outp, varbs, tmask = cmodel(
            token_id=inpt_token, varbs=varbs, info=info)
        if post_varbs:
            varb_list.append(copy.deepcopy(varbs))
        outp_token_ids.append(outp)
        task_mask.append(tmask)
    return outp_token_ids, varb_list, task_mask

def get_varbs_at_idx(seq, cmodel, idx, info=None, post_varbs=False):
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
        post_varbs: bool
            if true, will return the variables after each
            input token. Otherwise returns the variables
            before each token.
    Returns:
        varbs: dict
    """
    _, varb_list, _ = collect_varbs(
        seq=seq[:idx+1], cmodel=cmodel, info=info, post_varbs=post_varbs)
    return varb_list[-1]

def make_df_from_seqs(seqs, cmodel, info=None, post_varbs=False):
    """
    Constructs a dataframe from the sequences and the causal model.

    Args:
        seqs: torch tensor
        cmodel: CausalModel
            accepts a token id and a dict of variables
        info: (optional) dict
            some cmodels require additional info which can be specified
            in the info dict
        post_varbs: bool
            if true, will return the variables after each
            input token. Otherwise returns the variables
            before each token.
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
                    the variable value corresponding to <varb1> either
                    before or after processing by the causal model
                    depending on the value of `post_varbs`
    """
    df_dict = {
        "sample_idx": [],
        "step_idx": [],
        "inpt_token_id": [],
        "outp_token_id": [],
        "outp_tmask": [],
        **{k: [] for k in cmodel.init_varbs},
    }
    for sample_idx,seq in enumerate(seqs):
        varbs = cmodel.init_varbs
        for step_idx, inpt_token in enumerate(seq):
            df_dict["sample_idx"].append(sample_idx)
            df_dict["step_idx"].append(step_idx)
            df_dict["inpt_token_id"].append(int(inpt_token))
            if not post_varbs:
                for k in varbs:
                    df_dict[k].append(varbs[k])
            outp_token, varbs, tmask = cmodel(
                token_id=int(inpt_token), varbs=varbs, info=info)
            df_dict["outp_token_id"].append(outp_token)
            df_dict["outp_tmask"].append(tmask)
            if post_varbs:
                for k in varbs:
                    df_dict[k].append(varbs[k])
    df = pd.DataFrame(df_dict)
    return df

def sample_swaps(df, filter, info=None, stepwise=False):
    """
    Returns a swap mask and swap variables for each sample in the
    dataframe. Optionally specify whether you would like the mask
    to be contiguous from the start. 

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
        swap_masks: list of int lists
            returns a list of idx masks in which the value of the entry
            denotes the ordering of the swaps. The first swap is denoted
            by 0, the second by 1, and so on. -1 is the default value for
            positions that will not be swapped.
        swap_idxs: list of ints
            the last index of the swaps for each sample. This is equivalent
            to the maximum value along the last dimension of swap_masks.
        swap_varbs: list of lists of dicts
            a snapshot of the variables at each of the swap indexes
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
    for row_idx in range(len(samples)):
        sample = samples.iloc[row_idx]
        swap_idx = int(sample["step_idx"])
        swap_mask = [-1 for _ in range(int(sample["max_step"]+1))]
        if stepwise:
            swap_mask[:swap_idx+1] = [int(_) for _ in np.arange(swap_idx+1)]
            # Collect a list of varbs leading up to the swap idx
            samp_df = df.loc[df["sample_idx"]==int(sample["sample_idx"])]\
                .sort_values(by="step_idx")
            varbs = []
            for si in range(swap_idx+1):
                varbs.append(dict(samp_df.iloc[si]))
            swap_varbs.append(varbs)
        else:
            swap_mask[swap_idx] = 0
            swap_varbs.append([dict(sample)])
        swap_masks.append(list(swap_mask))
        swap_idxs.append(swap_idx)
    return swap_masks, swap_idxs, swap_varbs

def sample_cl_indices(
    df,
    varbs,
    keys=["obj_count", "phase", "count"],
    flatten=True,
    ignore_input_ids={},
    ignore_output_ids={},
):
    """
    A helper function for sampling indices from the data frame that have
    the argued variable makeup.

    Args:
        df: pd dataframe
        varbs: list of lists of dicts
            each dict is a dict of variables that should provide the
            desired makeup for the cl indices.
        flatten: bool
            if true, will flatten the returned list of lists to a
            single list
    Returns:
        cl_indices: list of lists of ints (or list of ints if flatten)
            a tuple of indices for each counterfactual latent. The
            index indicates the sample index and the sequence index
            for each sample.
    """
    if not keys: keys = list(varbs[0][0].keys())
    keys = [k for k in keys if k in varbs[0][0]]
    assert len(keys)>0
    cl_indices = []
    failures = []
    start_dfx = (~df["inpt_token_id"].isin(ignore_input_ids))&\
                (~df["outp_token_id"].isin(ignore_output_ids))
    for varb_list in varbs:
        indices = []
        for varb in varb_list: # Only length 1 if indywise
            dfx = start_dfx.copy()
            for k in keys:
                dfx = dfx&(df[k]==varb[k])
            samp = df.loc[dfx]
            if len(samp)>0:
                samp = samp.sample()
                indices.append([
                    int(samp.iloc[0]["sample_idx"]),
                    int(samp.iloc[0]["step_idx"]),
                ])
                failures.append(0)
            else:
                print("Failed to find CL Match!! Varbs")
                for k in keys:
                    print(k,varb[k])
                samp = df.loc[start_dfx].sample()
                indices.append([
                    int(samp.iloc[0]["sample_idx"]),
                    int(samp.iloc[0]["step_idx"]),
                ])
                failures.append(1)
                #print(df.head(30))
        if flatten:
            cl_indices += indices
        else:
            cl_indices.append(indices)
    return cl_indices, failures

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
        src_seqs=None,
        src_swap_idxs=None,
        stepwise=False,
        *args, **kwargs,
    ):
    """
    This function does a lot of the heavy lifting for
    creating the intervention data.

    Args:
        trg_seqs: list of lists of token ids
            the intial target sequences
        trg_swap_keys: list of str
            the name of the variable(s) to swap
        src_swap_keys: list of str
            the name of the variable(s) to swap
        trg_swap_idxs: list of bools
            a list of indexes corresponding to the intervention
            index in the target sequence for each sequence pair
        src_swap_varbs: list of lists of dicts
            a list of input varbs corresponding to the intervention
            index in the source sequence for each sequence pair
        trg_swap_varbs: list of lists of dicts
            a list of input varbs corresponding to the intervention
            index in the target sequence for each sequence pair
        trg_task_masks: list of lists of bools
            the task masks for the target sequences
        trg_cmodel: python_function(token, varbs, info)
            accepts a token and a dict of variables
        trg_info: dict
            optionally specify a dict of information to be used
            with the trg_cmodel
    Returns:
        intrv_seqs: list of lists of ints
        intrv_varbs_list: list of varb dicts
        intrv_tmasks: list of lists of bools
    """
    if "full" in trg_swap_keys:
        trg_swap_keys = list(trg_cmodel.init_varbs.keys())
        src_swap_keys = list(trg_cmodel.init_varbs.keys())
    elif type(trg_swap_keys)==str:
        trg_swap_keys = [trg_swap_keys]
        src_swap_keys = [src_swap_keys]
    if trg_swap_keys[0] is None or trg_swap_keys[0] in {"","null_varb"}:
        return trg_seqs, trg_swap_varbs, trg_task_masks

    zlist = [
        trg_seqs, trg_swap_idxs, trg_task_masks,
        src_swap_varbs, trg_swap_varbs,
    ]
    if src_seqs is not None:
        zlist += [src_seqs,src_swap_idxs]
    else:
        print("No Src Seqs")
        zlist += [trg_seqs, trg_swap_idxs]
    z = zip(*zlist)
    intrv_seqs = []
    intrv_varbs_list = []
    intrv_tmasks = []
    pad_id =  trg_info["pad_token_id"]
    eos_id =  trg_info["eos_token_id"]
    fill_id = trg_info["demo_token_ids"][-1]
    for seq_i, tup in enumerate(z):
        (trg_seq,trg_idx,trg_tmask,src_varbs,trg_varbs, src_seq, src_idx) = tup
        src_varbs = src_varbs[-1]
        trg_varbs = trg_varbs[-1]
        zkeys = zip(trg_swap_keys, src_swap_keys)
        intrv_varbs = {tkey: src_varbs[skey] for tkey,skey in zkeys}
        trg_cmodel.queue_intervention(intrv_varbs)
        if seq_i%500==0:
            print()
            print("Src Varbs:", src_varbs)
            print("Trg Varbs:", trg_varbs)
            print("Intrv Varbs:", trg_cmodel.swap_varbs)
            print()
        intrv_varbs = {**copy.deepcopy(trg_varbs), **intrv_varbs}
        intrv_varbs_list.append(intrv_varbs)
        if trg_idx<len(trg_seq):
            inpt_token = int(trg_seq[trg_idx])
        else: inpt_token = fill_id
        if inpt_token==pad_id: inpt_token = fill_id

        try:
            intrv_seq, intrv_tmask, _ = run_cmodel_to_completion(
                cmodel=trg_cmodel,
                inpt_token=inpt_token,
                varbs=trg_varbs, # The intervention is queued in the cmodel
                info=trg_info,
                end_tokens={trg_info.get("eos_token_id", None)},
            )
        except:
            print("Failed:")
            print("Src Varbs:", src_varbs)
            print("Trg Varbs:", trg_varbs)
            print("Intrv Varbs:", trg_cmodel.swap_varbs)
            assert False
        if stepwise:
            preseq = [t if t!=pad_id and t!=eos_id else fill_id for t in trg_seq[:trg_idx+1]]
            if len(preseq)<trg_idx+1:
                preseq += [fill_id for _ in range(trg_idx+1-len(preseq))]
            seq = preseq + intrv_seq
            tmask = [0 for t in preseq] + intrv_tmask
        else:
            preseq = trg_seq[:trg_idx+1]
            seq = preseq + intrv_seq
            tmask = trg_tmask[:trg_idx+1] + intrv_tmask
        intrv_seqs.append( seq )
        intrv_tmasks.append( tmask )
        if seq_i%500==0:
            print("Samp:", seq_i)
            print("\tSrc :", tensor2str(torch.tensor(src_seq[:src_idx+1]),n=5))
            print("\tSOut:", tensor2str(torch.tensor(src_seq[src_idx+1:]),n=5))
            print("\tInpt:", tensor2str(torch.tensor(preseq[:trg_idx+1]),n=5))
            print("\tOutp:", tensor2str(torch.tensor(intrv_seq),n=5))
    return intrv_seqs, intrv_varbs_list, intrv_tmasks

def get_labels(varb_df, key):
    """
    Returns the value of the variable at each index in the df based on
    step and ep indices.

    Args:
        varb_df: dataframe
            the variables at each step in the sequences. Needs keys:
            "step_idx" and "sample_idx"
        key: str
            the key to extract
    """
    varb_df = varb_df.sort_values(by=["sample_idx", "step_idx"])
    sample_idxs = sorted(list(set(varb_df["sample_idx"])))
    labels = []
    for samp_idx in sample_idxs:
        samp = varb_df.loc[varb_df["sample_idx"]==samp_idx, key]
        labels.append( list(samp) )
    return labels

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
        sample_w_replacement=True,
        use_cl=False,
        use_src_data_for_cl=True,
        tokenizer=None,
        ret_src_labels=True,
        ret_varbs=False,
    ):
    """
    Constructs intervention data from the argued sequence pairs.
    Any referred to variables are always input variables instead
    of output variables. This makes the interventions easier.

    Args:
        trg_data: dict or ArrowDataset
            "input_ids": list of lists of token ids
            "task_mask": list of lists of bools
            "inpt_attn_mask": list of lists of bools
        src_seqs: dict or ArrowDataset
            "input_ids": list of lists of token ids
            "task_mask": list of lists of bools
            "inpt_attn_mask": list of lists of bools
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
        src_cmodel: CausalModel
        src_info: dict
        src_filter: python_function
        trg_cmodel: CausalModel
        trg_info: dict
        trg_filter: python_function
        stepwise: bool
        sample_w_replacement: bool
            if true, will sample the source sequences with replacement
        use_cl: bool
            if true, will collect cl indices and sequences for the
            cl loss.
        use_src_data_for_cl: bool
            if true, will provide the cl data from the src sequences
        ret_src_labels: bool
            if true, will return linear regression labels for the source
            data.
        ret_varbs: bool
            if true, will return the source and target swap variables
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
        source_df: dict
            optionally return the source dataframe
    """
    if type(src_swap_keys)==str: src_swap_keys = [src_swap_keys]
    if type(trg_swap_keys)==str: trg_swap_keys = [trg_swap_keys]
    if sample_w_replacement:
        indices = np.random.randint(0,len(src_data),len(src_data))
    else:
        indices = np.random.permutation(len(src_data))
    src_data = src_data.select(indices)

    # 1. get the source variables and the swap indices
    src_seqs = list(src_data["input_ids"])
    src_task_masks = list(src_data["task_mask"])
    src_attn_masks = list(src_data["inpt_attn_mask"])
    src_df = make_df_from_seqs(
        seqs=src_seqs,
        cmodel=src_cmodel,
        info=src_info,
        post_varbs=True,
    )
    src_swap_masks, src_swap_idxs, src_swap_varbs = sample_swaps(
        df=src_df,
        filter=src_filter,
        info=src_info,
        stepwise=stepwise,
    )
    ret_src_labels = ret_src_labels and src_swap_keys[0] not in {"full","","null_varb"}
    if ret_src_labels:
        src_labels = get_labels(varb_df=src_df, key=src_swap_keys[0])
    assert len(src_swap_masks[0])==len(src_seqs[0])

    # 2. get the target variables and swap indices
    trg_seqs =       list(trg_data["input_ids"])
    trg_task_masks = list(trg_data["task_mask"])
    trg_attn_masks = list(trg_data["inpt_attn_mask"])
    trg_df = make_df_from_seqs(
        seqs=trg_seqs,
        cmodel=trg_cmodel,
        info=trg_info,
        post_varbs=False,
    )

    if stepwise:
        trg_swap_masks = copy.deepcopy(src_swap_masks)
        trg_swap_idxs  = copy.deepcopy(src_swap_idxs)
        trg_swap_varbs = [
            [get_varbs_at_idx(
                seq=tseq,
                cmodel=trg_cmodel,
                idx=tidx,
                info=trg_info,
                post_varbs=False,)]\
                for tseq,tidx in zip(trg_seqs,trg_swap_idxs)
        ]
    else:
        trg_swap_masks, trg_swap_idxs, trg_swap_varbs = sample_swaps(
            df=trg_df,
            filter=trg_filter,
            info=trg_info,
            stepwise=False,
        )

    # 3. Using the variables, seqs, and swap indices, create
    # intervention data.
    intrv_seqs, intrv_varbs, intrv_task_masks = make_counterfactual_seqs(
        trg_swap_keys=trg_swap_keys,
        src_swap_keys=src_swap_keys,
        trg_seqs=trg_seqs,
        src_seqs=src_seqs,
        trg_swap_idxs=trg_swap_idxs,
        src_swap_idxs=src_swap_idxs,
        trg_swap_varbs=trg_swap_varbs,
        trg_task_masks=trg_task_masks,
        src_swap_varbs=src_swap_varbs,
        trg_cmodel=trg_cmodel,
        trg_info=trg_info,
        stepwise=stepwise,
    )

    # Collect the counterfactual latent data if needed. cl_idxs are row,col
    # pairs that can be used to index into and isolate the counterfactual
    # latents produced using the cl_seqs from the target model.
    cl_idxs = None
    cl_varbs = intrv_varbs
    if use_cl:
        if stepwise: raise NotImplemented
        if type(cl_varbs[0])!=list:
            cl_varbs = [[v] for v in cl_varbs]  # Make it a list of lists
        if use_src_data_for_cl:
            print("Using Src Data For CL...")
            cl_idxs = get_nonzero_entries(src_swap_masks)
            cl_seqs = copy.deepcopy(src_seqs)
            cl_tmasks = copy.deepcopy(src_task_masks)
            failures = [0 for _ in range(len(cl_idxs))]
        else:
            print("Sampling New CL Indices...")
            cl_df = make_df_from_seqs(
                seqs=trg_seqs,
                cmodel=trg_cmodel,
                info=trg_info,
                post_varbs=True,
            )
            # cl_varbs: one varb dict wrapped in a list for each row
            cl_idxs, failures = sample_cl_indices(
                df=cl_df,
                varbs=cl_varbs,
                ignore_input_ids={
                    trg_info["bos_token_id"],
                    trg_info["eos_token_id"],
                    *trg_info["trig_token_ids"],
                },
                ignore_output_ids={trg_info["pad_token_id"]},
            )
            cl_seqs = copy.deepcopy(trg_seqs)
            cl_tmasks = copy.deepcopy(trg_task_masks)

        cl_idxs = torch.tensor(cl_idxs).long()
        # TODO make use of failures
        failures = torch.tensor(failures).bool()
        #cl_idx_mask = [] # marks the columns that are valid for each row
        #for row,seq in enumerate(cl_seqs):
        #    cols = cl_idxs[cl_idxs[:,0]==row, -1].tolist()
        #    cl_idx_mask.append(
        #        [1 if col in cols else 0 for col in range(len(seq))]
        #    )
        assert len(cl_idxs)==np.sum([np.sum(np.asarray(s)>=0) for s in src_swap_masks])
            

    if tokenizer is not None:
        print("Outids :", intrv_seqs[0])
        print("Outputs:", tokenizer.decode(intrv_seqs[0]))

    intrv_swap_masks = [
        pad_to(msk, len(seq), fill_val=-1)[:len(seq)] for msk,seq\
                                in zip(trg_swap_masks, intrv_seqs)
    ]
    d = {
        "trg_input_ids": intrv_seqs,
        "trg_swap_masks": intrv_swap_masks,
        "trg_task_masks": intrv_task_masks,
        "src_input_ids": src_seqs,
        "src_swap_masks": src_swap_masks,
        "src_task_masks": src_task_masks,
        "trg_swap_idxs": trg_swap_idxs,
        "src_swap_idxs": src_swap_idxs,
    }
    if ret_src_labels:
        d["src_labels"] = src_labels

    if cl_idxs is not None:
        d["cl_idxs"] = cl_idxs
        #d["cl_idx_masks"] = cl_idx_mask
        d["cl_input_ids"] = cl_seqs
        d["cl_task_masks"] = cl_tmasks
        d["cl_failures"] = failures
        #assert len(cl_idx_mask[0])==len(cl_seqs[0])
    if ret_varbs:
        varbs = {
            "src_swap_varbs": src_swap_varbs,
            "trg_swap_varbs": trg_swap_varbs,
            "cl_varbs": cl_varbs,
        }
        return d, varbs
    return d

def make_intrv_data_from_src_data(
    src_input_ids,
    trg_input_ids,
    trg_prompt,
    src_prompt,
    src_actvs,
    trg_actvs,
    trg_tokenizer,
    src_tokenizer,
    src_output_ids=None,
    trg_output_ids=None,
    stepwise=True,
    ret_cl_data=True,
    min_cfct_len=10,
    min_intrv_idx=10,
    shuffle=True,
    null_varb=False,
    n_samples=None,
    as_tensors=False,
):
    """
    This function creates intervention data saved source data.

    Function Steps:
    - shuffles target sequences from available sequences without replacement
      if shuffle is true
    - makes tokenized text
    - samples src and trg intervention indices and swap masks (same if
      stepwise is true and prompt lengths are the same)
    - replaces trg sequences with src sequences after the intervention
      indices
    - returns original src sequences with ~swap_mask indices as cl data
      if returning cl data

    Args:
        src_input_ids: tensor (B,S1)
            the source model input ids that we wish to use for
            interventions
        trg_input_ids: tensor (B,S2)
            the target model input ids that we wish to use for
            interventions
        shuffle: bool
            if false, will not mix up target and src data.
        null_varb: bool
            if true, will will not transfer from source sequences into
            targ sequences. useful for generating null interventions
            when training with empty varbs.
        trg_prompt: str
            the target prompt will be prepended to the text
        src_prompt: str
            the source prompt will be prepended to the text
        src_actvs: tensor (B,S,D)
            the latent activations from the source model. S includes
            the prompt tokens
        trg_tokenizer: Tokenizer
            the tokenizer for the target model
        src_tokenizer: Tokenizer
            the tokenizer for the source model
        src_output_ids: tensor (B,S1)
            optionally argue labels for the source model.
        trg_output_ids: tensor (B,S2)
            optionally argue labels for the target model.
        stepwise: bool
            if true, will assume stepwise interventions
        ret_cl_data: bool
            if true, will collect cl indices and sequences to be used by
            the target model to collect cl vectors.
        min_cfct_len: int
            the minimum counterfactual len. this ensures that there are
            some predictions following the interchange.
        min_intrv_idx: int
            the minimum index to sample for the intervention. this ensures
            that there is some buildup before the interchange.
        n_samples: int
            only applies if shuffle is True.
            optionally specify the number of samples to generate. If this
            is not None and greater than 0, will sample pairs with
            replacement from both target and source text until specified
            number of samples is met. Otherwise will simply shuffle the
            target samples and pair without replacement.
        as_tensors: bool
            if true, will return values in data_dict as tensors rather
            than lists
    Returns:
        data_dict: dict
            "trg_input_ids":  tensor or list - shape (B,S1)
            "src_input_ids":  tensor or list - shape (B,S1)
            "trg_task_masks": tensor or list - shape (B,S1)
            "src_task_masks": tensor or list - shape (B,S1)
            "trg_swap_masks": tensor or list - shape (B,S1)
                returns a list of idx masks in which the value of the entry
                denotes the ordering of the swaps. The first swap is denoted
                by 0, the second by 1, and so on. -1 is the default value for
                positions that will not be swapped.
            "src_swap_masks": tensor or list - shape (B,S2)
                returns a list of idx masks in which the value of the entry
                denotes the ordering of the swaps. The first swap is denoted
                by 0, the second by 1, and so on. -1 is the default value for
                positions that will not be swapped.
            "trg_swap_idxs":  tensor or list - shape (B,)
            "src_swap_idxs":  tensor or list - shape (B,)
            "cl_input_ids":   tensor or list - shape (B,S1)
            "cl_idx_masks":        tensor or list - shape (L,2)
                this is a 2d tensor containing row,col indices that can
                be used to select the counterfactual latents from the
                target model when it is run on the cl_input_ids.
            "trg_output_ids":  tensor or list - shape (B,S1)
                if output_ids is not none, will return output ids as
                well as input ids
            "src_output_ids":  tensor or list - shape (B,S1)
                if output_ids is not none, will return output ids as
                well as input ids
    """
    if not stepwise: raise NotImplemented

    ##############################################
    ### Make tokenized text
    ##############################################

    startt = time.time()
    print("Tokenizing Src...")
    # Src tokens
    src_prompt_len = 0
    if src_prompt:
        # Still needs a lot of work and planning before we can do prompt
        # based experiments. need to tokenize prompt and prepend to ids
        raise NotImplemented
    src_sample_len = src_actvs.shape[1]
    src_max_length = src_sample_len + src_prompt_len
    print("Exec Time:", time.time() - startt)

    startt = time.time()
    print("Tokenizing Trg...")
    # Trg Tokens
    trg_prompt_len = 0
    if trg_prompt:
        raise NotImplemented
    trg_sample_len = trg_actvs.shape[1]
    trg_max_length = trg_sample_len + trg_prompt_len
    print("Exec Time:", time.time() - startt)

    # CL Tokens
    if ret_cl_data:
        startt = time.time()
        print("Tokenizing CL...")
        if null_varb:
            cl_input_ids = trg_input_ids.clone()
            cl_output_ids = trg_output_ids.clone()
            cl_max_length = trg_max_length
        else:
            cl_input_ids = src_input_ids.clone()
            cl_output_ids = src_output_ids.clone()
            cl_max_length = src_max_length
        print("Exec Time:", time.time() - startt)

    ##############################################
    ### Sample Sequence Pairs
    ##############################################
    startt = time.time()
    if shuffle:
        print("Shuffling Samples...")
        tot_src_samps = len(src_input_ids)
        tot_trg_samps = len(trg_input_ids)
        if not n_samples:
            src_samp_idxs = torch.randperm(tot_src_samps).long()
            trg_samp_idxs = torch.randperm(tot_trg_samps).long()
        else:
            src_samp_idxs = torch.randint(0,tot_src_samps, (n_samples,))
            trg_samp_idxs = torch.randint(0,tot_trg_samps, (n_samples,))
    else:
        print("Not Shuffling Samples...")
        n = min(len(src_input_ids),len(trg_input_ids))
        if n_samples: n = min(n, n_samples)
        src_samp_idxs = torch.arange(n).long()
        trg_samp_idxs = torch.arange(n).long()

    src_actvs = src_actvs[src_samp_idxs]
    src_input_ids = src_input_ids[src_samp_idxs]
    trg_input_ids = trg_input_ids[trg_samp_idxs]
    if src_output_ids is not None:
        src_output_ids = src_output_ids[src_samp_idxs]
        trg_output_ids = trg_output_ids[trg_samp_idxs]
    if ret_cl_data:
        cl_input_ids = cl_input_ids[src_samp_idxs]
        if src_output_ids is not None:
            cl_output_ids = cl_output_ids[src_samp_idxs]
    print("Exec Time:", time.time() - startt)

    ##############################################
    ### Sample Intervention Indices
    # Sample src and trg intervention indices and swap masks (src and trg
    #   swap masks are same if stepwise is true and prompt lengths are
    #   the same and the tokenizers are the same)
    ##############################################
    startt = time.time()
    print("Making Intervention Masks...")
    sample_len = src_sample_len
    if stepwise:
        sample_len = min(src_sample_len, trg_sample_len)
    min_intrv_idx = min(min_intrv_idx, sample_len-min_cfct_len-1)
    intrv_idxs = torch.randint(
        min_intrv_idx, sample_len-min_cfct_len, (len(src_actvs),))
    if stepwise: # STEPWISE
        trg_intrv_idxs = trg_prompt_len + intrv_idxs
        src_intrv_idxs = src_prompt_len + intrv_idxs

        trg_swap_mask = get_mask_between(
            shape=trg_input_ids.shape,
            startx=torch.zeros(len(trg_intrv_idxs))+trg_prompt_len,
            endx=trg_intrv_idxs,
            inclusive=True,
        ).long()
        num_idxs = torch.cat([torch.arange(row.sum()) for row in trg_swap_mask])
        bool_idx = trg_swap_mask.bool()
        trg_swap_mask[bool_idx] = num_idxs
        trg_swap_mask[~bool_idx] = -1

        src_swap_mask = get_mask_between(
            shape=src_input_ids.shape,
            startx=torch.zeros(len(src_intrv_idxs))+src_prompt_len,
            endx=src_intrv_idxs,
            inclusive=True,
        ).long()
        bool_idx = src_swap_mask.bool()
        src_swap_mask[bool_idx] = num_idxs
        src_swap_mask[~bool_idx] = -1

        # print("Trg  idx:", trg_intrv_idxs[0])
        # print("Trg  Example:", trg_swap_mask[0])
        # print("Ssrc idx:", src_intrv_idxs[0])
        # print("Ssrc Example:", src_swap_mask[0])

    else: # INDYWISE
        src_intrv_idxs = intrv_idxs + src_prompt_len
        trg_intrv_idxs = torch.randint(
            trg_prompt_len+1,
            sample_len-min_cfct_len,
            (len(src_actvs),))
        row_idx = torch.arange(len(trg_intrv_idxs)).long()
        trg_swap_mask = torch.zeros_like(trg_input_ids)-1
        trg_swap_mask[row_idx,trg_intrv_idxs] = 0
        src_swap_mask = torch.zeros(src_actvs.shape[:2]).long()-1
        src_swap_mask[row_idx,src_intrv_idxs] = 0
    print("Exec Time:", time.time() - startt)

    startt = time.time()
    print("Making Task Masks...")
    trg_task_mask = get_mask_past_arglast(trg_swap_mask, inclusive=False)
    src_task_mask = get_mask_past_arglast(src_swap_mask, inclusive=False)
    print("Exec Time:", time.time() - startt)

    # Replaces trg sequences with src sequences after the intervention
    #   indices
    startt = time.time()
    trg_len = trg_swap_mask.shape[1]-trg_prompt_len
    src_len = src_swap_mask.shape[1]-src_prompt_len
    if not null_varb:
        print("Replacing elements with counterfactuals...")
        endx = torch.zeros(len(trg_intrv_idxs))+min(trg_len,src_len)
        trg_replace_mask = get_mask_between(
            shape=trg_swap_mask.shape,
            startx=trg_intrv_idxs+1,
            endx=endx+trg_prompt_len,
        )
        src_replace_mask = get_mask_between(
            shape=src_swap_mask.shape,
            startx=src_intrv_idxs+1,
            endx=endx+src_prompt_len,
        )
        #print("EXAMPLE----------------")
        #print("Trg  idx:", trg_intrv_idxs[0])
        #print("Ssrc idx:", src_intrv_idxs[0])

        #print("indices   :", tensor2str(torch.arange(len(trg_input_ids[0]))))
        #print("pre Trg Ex:", tensor2str(trg_input_ids[0]))
        #print("pre Src Ex:", tensor2str(src_input_ids[0]))

        trg_input_ids[trg_replace_mask] = src_input_ids[src_replace_mask]

        #print("Trg Exampl:", tensor2str(trg_input_ids[0]))
        #print("Src Exampl:", tensor2str(src_input_ids[0]))

        if src_output_ids is not None:
            trg_replace_mask = get_mask_between(
                shape=trg_swap_mask.shape,
                startx=trg_intrv_idxs,
                endx=endx+trg_prompt_len,
            )
            src_replace_mask = get_mask_between(
                shape=src_swap_mask.shape,
                startx=src_intrv_idxs,
                endx=endx+src_prompt_len,
            )
            trg_output_ids[trg_replace_mask] = src_output_ids[src_replace_mask]
    print("Exec Time:", time.time() - startt)

    if trg_len>src_len:
        startt = time.time()
        print("Padding...")
        # UNTESTED
        print("Using different trg and src lengths is untested!!")
        pad_mask = get_mask_past_idx(
            shape=trg_swap_mask.shape,
            idx=endx+trg_prompt_len,
            inclusive=True
        )
        trg_input_ids[pad_mask] = trg_tokenizer.pad_token_id
        print("Exec Time:", time.time() - startt)

    d = {
        "src_input_ids": src_input_ids,
        "trg_input_ids": trg_input_ids,
        "src_task_masks": src_task_mask,
        "trg_task_masks": trg_task_mask,
        "src_swap_masks": src_swap_mask,
        "trg_swap_masks": trg_swap_mask,
        "src_swap_idxs": src_intrv_idxs,
        "trg_swap_idxs": trg_intrv_idxs,
        "src_actvs": src_actvs,
    }

    if src_output_ids is not None:
        d["src_output_ids"] = src_output_ids
        d["trg_output_ids"] = trg_output_ids

    # Returns original src sequences with ~swap_mask indices as cl data
    #   if returning cl data
    if ret_cl_data:
        startt = time.time()
        print("Making CL Data...")
        if not stepwise: raise NotImplemented
        cl_mask = get_mask_between(
            shape=cl_input_ids.shape,
            startx=torch.zeros(len(intrv_idxs))+src_prompt_len,
            endx=intrv_idxs+src_prompt_len,
            inclusive=True,
        )
        cl_task_mask = get_mask_past_arglast(cl_mask, inclusive=True)
        d["cl_input_ids"] = cl_input_ids
        d["cl_task_masks"] = cl_task_mask
        d["cl_idx_masks"] = cl_mask
        assert torch.all(cl_mask.bool()==(trg_swap_mask>=0))
        if src_output_ids is not None:
            d["cl_output_ids"] = cl_output_ids
        print("Exec Time:", time.time() - startt)
    if not as_tensors:
        startt = time.time()
        print("Converting to lists...")
        d = {k: v.tolist() for k,v in d.items()}
        print("Exec Time:", time.time() - startt)
    return d


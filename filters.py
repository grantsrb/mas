"""
This package contains methods to filter the dataset(s) for valid
intervention indices. Typically the filters will take in a
pandas row and operate on the values in the row.

filter: python_function(df_row, info)
    a python function that takes a dataframe row and an info
    dict and returns filtered indices (indices that we would
    like to sample from)
"""
default_info = {
    "pad_token": "<PAD>",
    "bos_token": "<BOS>",
    "eos_token": "<EOS>",
    "pad_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,
}

def default_filter(df_row, info=None):
    if info is None: info = default_info
    bad_token_ids = {
        info.get("pad_token_id", 0),
        info.get("bos_token_id", 1),
        info.get("eos_token_id", 2),
        *info.get("trig_token_ids", [7]),
    }
    return df_row.inpt_token_id not in bad_token_ids

default_excl_vals = {
    "count": { 2,4,9,14,17 },
}
def excl_varb_vals_filter(
        df_row,
        excl_varb_vals=default_excl_vals,
        info=None
):
    if info is None: info = default_info
    bad_token_ids = {
        info.get("pad_token_id", 0),
        info.get("bos_token_id", 1),
        info.get("eos_token_id", 2),
        *info.get("trig_token_ids", [7]),
    }
    dfx = df_row.inpt_token_id not in bad_token_ids
    for key in excl_varb_vals:
        if key in df_row:
            dfx = dfx and not( int(df_row[key]) in excl_varb_vals[key] )
    return dfx

def keep_varb_vals_filter(
        df_row,
        keep_varb_vals=default_excl_vals,
        info=None
):
    if info is None: info = default_info
    bad_token_ids = {
        info.get("pad_token_id", 0),
        info.get("bos_token_id", 1),
        info.get("eos_token_id", 2),
        *info.get("trig_token_ids", [7]),
    }
    dfx = df_row.inpt_token_id not in bad_token_ids
    for key in keep_varb_vals:
        if key in df_row:
            dfx = dfx and ( int( df_row[key] ) in keep_varb_vals[key])
    return dfx

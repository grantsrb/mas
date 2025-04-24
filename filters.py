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
    "eos_token": "<EOS>",
    "bos_token": "<BOS>",
}

def default_filter(df_row, info=None):
    if info is None: info = default_info
    bad_tokens = {info.get("eos_token", "<EOS>"), info.get("bos_token", "<BOS>")}
    return dict(df_row)["outp_token"] not in bad_tokens
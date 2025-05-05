import numpy as np
import torch
import pandas as pd
import os
from torch.optim.lr_scheduler import _LRScheduler
from dl_utils.utils import (
    arglast, get_mask_past_id, get_mask_between, pad_to,
)
from dl_utils.training import read_command_line_args
import dl_utils.save_io as savio
from utils import collect_activations
import seq_models as seq_mods
import copy
from tqdm import tqdm
import scipy.stats as stats
import tasks

DEVICES = { -1: "cpu", **{i:i for i in range(10)} }

def device_fxn(device):
    if device<0: return "cpu"
    return device

def get_l2_rdm(X):
    """
    X: (B,D)
    """
    return (X-X[:,None]).norm(2,dim=-1) # BxB

def cos_sim(X, X2):
    X = X/X.norm(2,dim=-1)[:,None]
    X2 = X2/X2.norm(2,dim=-1)[:,None]
    return torch.matmul(X,X2.T)

def get_cos_rdm(X, norm_along_neurons=False):
    """
    X: (B,D)
    """
    if norm_along_neurons:
        std = torch.sqrt( ((X-X.mean(0))**2).mean(0) )
        X = (X-X.mean(0))/(std)
    return 1-cos_sim(X,X) # BxB

def get_cor_rdm(X):
    """
    X: (B,D)
    """
    return 1-mtx_cor(X.T,X.T,zscore=True) # BxB

def compute_HSIC(K, L) -> float:
    """
    Compute the Hilbert-Schmidt Independence Criterion (HSIC) given two kernel matrices K and L.

    Args:
    K: tensor (N,N)
        Kernel matrix for variable X 
    L: tensor (N,N)
        Kernel matrix for variable Y

    Returns:
        float: The HSIC value.
    """
    n = K.shape[0]  # Assume K and L are square matrices of the same size
    assert K.shape == L.shape, "K and L must have the same dimensions"

    # Centering matrix
    H = torch.eye(n) - (1/n) * torch.ones((n, n))

    # Compute HSIC
    HSIC_value = torch.trace(K @ H @ L @ H) / ((n - 1) ** 2)
    return HSIC_value

def compute_CKA(K, L):
    """
    Args:
    K: tensor (N,N)
        Kernel matrix for variable X 
    L: tensor (N,N)
        Kernel matrix for variable Y
    """
    KL = compute_HSIC(K,L)
    KK = compute_HSIC(K,K)
    LL = compute_HSIC(L,L)
    return KL/torch.sqrt(KK*LL)

def get_cka(X, X2, batch_size=None, n_runs=1, sim_metric="cosine", prenorm=True):
    """
    X: torch tensor (B,N)
    X2: torch tensor (B,M)
    sim_metric: str
        "l2" or "cosine"
    prenorm: bool
        if true will normalize each neuron over the B dim
    """
    cors = []
    device = device_fxn(X.get_device())
    for run in tqdm(range(n_runs)):
        if batch_size is not None:
            perm = torch.randperm(len(X)).long().to(device)[:batch_size]
        else:
            perm = torch.arange(len(X)).long().to(device)
        x1 = X[perm]
        x2 = X2[perm]
        if prenorm:
            x1 = (x1-x1.mean(0))/x1.std(0)
            x2 = (x2-x2.mean(0))/x2.std(0)
        if sim_metric in {"cos", "cosine"}:
            mtx1 = get_cos_rdm(x1) # BxB
            mtx2 = get_cos_rdm(x2) # BxB
        elif sim_metric == "cor":
            mtx1 = get_cor_rdm(x1)
            mtx2 = get_cor_rdm(x2)
        else:
            mtx1 = get_l2_rdm(x1)
            mtx2 = get_l2_rdm(x2)
        cka = compute_CKA(mtx1, mtx2).cpu().data.numpy()
        cors.append(cka)
    return np.mean(cors)

def get_rsa(X,X2, batch_size=None, n_runs=1, sim_metric="cosine", cor_type="spearmanr", prenorm=False):
    """
    X: torch tensor (B,N)
    X2: torch tensor (B,M)
    sim_metric: str
        "l2" or "cosine"
    prenorm: bool
        if true will normalize each neuron over the B dim
    """
    cors = []
    device = device_fxn(X.get_device())
    for run in tqdm(range(n_runs)):
        if batch_size is not None:
            perm = torch.randperm(len(X)).long().to(device)[:batch_size]
        else:
            perm = torch.arange(len(X)).long().to(device)
        x1 = X[perm]
        x2 = X2[perm]
        if prenorm:
            x1 = (x1-x1.mean(0))/x1.std(0)
            x2 = (x2-x2.mean(0))/x2.std(0)
        if sim_metric in {"cos", "cosine"}:
            mtx1 = get_cos_rdm(x1) # BxB
            mtx2 = get_cos_rdm(x2) # BxB
        elif sim_metric == "cor":
            mtx1 = get_cor_rdm(x1)
            mtx2 = get_cor_rdm(x2)
        else:
            mtx1 = get_l2_rdm(x1)
            mtx2 = get_l2_rdm(x2)
        lower_tri = torch.tril(torch.ones_like(mtx1)).bool()
        cor = getattr(stats, cor_type)(
            mtx1[lower_tri].cpu().data.numpy(),
            mtx2[lower_tri].cpu().data.numpy()
        ).correlation
        cors.append(cor)
    return np.mean(cors)

def expl_var(fx, x):
    """
    Assumes fx and x are 2d matrices
    
    Args:
        fx: tensor (B,D)
            These are the reconstructed/predicted representations.
            Batch is first dim, features are second dim
        x: tensor (B,D)
            These are the original representations.
            Batch is first dim, features are second dim
    """
    og_mean = x.mean(0)
    og_var = (((x-og_mean)**2).sum(-1)).sum()
    #lamp_var = (((x)**2).sum(-1)).sum()
    
    pred_var = (((fx-x)**2).sum(-1)).sum()
    return 1-pred_var/og_var #, 1-pred_var/lamp_var
    
default_sim_data_dict = {
    "cka_cos": [],
    "cka_l2": [],
    "rsa_raw_l2_spr": [],
    "rsa_raw_cos_spr": [],
    "rsa_raw_l2_prs": [],
    "rsa_raw_cos_prs": [],
    "rsa_nrm_l2_spr": [],
    "rsa_nrm_cos_spr": [],
    "rsa_nrm_l2_prs": [],
    "rsa_nrm_cos_prs": [],
}

def compute_similarities(X, X2, n_runs=1, *args, **kwargs):
    sims = copy.deepcopy(default_sim_data_dict)

    sims["cka_cos"] = get_cka(X, X2, n_runs=n_runs, sim_metric="cosine", prenorm=True)
    sims["cka_l2"] = get_cka(X, X2, n_runs=n_runs, sim_metric="l2", prenorm=True)
    sims["rsa_raw_cos_spr"] = get_rsa(X, X2, n_runs=n_runs, sim_metric="cosine", cor_type="spearmanr", prenorm=False)
    sims["rsa_raw_l2_spr"] =  get_rsa(X, X2, n_runs=n_runs, sim_metric="l2",     cor_type="spearmanr", prenorm=False)
    sims["rsa_raw_cos_prs"] = get_rsa(X, X2, n_runs=n_runs, sim_metric="cosine", cor_type="pearsonr",  prenorm=False)
    sims["rsa_raw_l2_prs"] =  get_rsa(X, X2, n_runs=n_runs, sim_metric="l2",     cor_type="pearsonr",  prenorm=False)
    sims["rsa_nrm_cos_spr"] = get_rsa(X, X2, n_runs=n_runs, sim_metric="cosine", cor_type="spearmanr", prenorm=True)
    sims["rsa_nrm_l2_spr"] =  get_rsa(X, X2, n_runs=n_runs, sim_metric="l2",     cor_type="spearmanr", prenorm=True)
    sims["rsa_nrm_cos_prs"] = get_rsa(X, X2, n_runs=n_runs, sim_metric="cosine", cor_type="pearsonr",  prenorm=True)
    sims["rsa_nrm_l2_prs"] =  get_rsa(X, X2, n_runs=n_runs, sim_metric="l2",     cor_type="pearsonr",  prenorm=True)
    return sims

def initialize_sim_data_dict():
    data = copy.deepcopy(default_sim_data_dict)
    data["model_folder1"] = []
    data["model_folder2"] = []
    data["layer1"] = []
    data["layer2"] = []
    return data

def get_dataset(config, **kwargs):
    """
    Returns:
        seqs: list of lists of str
            the string tokens
    """
    n_samples = config.get("n_samples_per_count", 15)
    config["unk_p"] = 0
    config["min_count"] = config.get("min_count", 1)
    config["max_count"] = config.get("max_count", 20)
    config["task_type"] = config.get("task_type", "MultiObject")
    task = getattr(tasks, config["task_type"])(**config)

    samps = []
    tmasks = []
    varbs_list = []

    max_len = 0
    for obj_count in range(config["min_count"], config["max_count"]+1):
        for si in range(n_samples):
            samp, tmask, varbs = task.generate_sample(obj_count=obj_count)
            # Add BOS token
            samps.append([ "B" ] + samp)
            max_len = max(max_len, len(samps[-1]))
    samps = [pad_to(samp, max_len, fill_val="P") for samp in samps]
    return samps

def get_model_and_config(model_folder):
    checkpt = savio.load_checkpoint(model_folder)
    config = checkpt["config"]
    model = getattr(seq_mods,config["model_type"])(**config)
    temp = seq_mods.LossWrapper(model=model, config=config)
    temp.load_state_dict(checkpt["state_dict"])
    model = temp.model
    return model, config

def tokenize_input_data(input_data, mconfig):
    """
    Args:
        input_data: list of lists of str tokens
        mconfig: dict
    """
    info = mconfig["info"]
    default_word2id = {
        info["pad_token"]: 0,
        info["bos_token"]: 1,
        info["eos_token"]: 2,
        "D": 3,
        info.get("demo_tokens", ["D0"])[0]: 3,
        info.get("demo_tokens", ["D0","D1"])[1]: 4,
        info.get("demo_tokens", ["D0","D1","D2"])[2]: 5,
        info.get("resp_tokens", ["R"])[0]: 6,
        info.get("trig_tokens", ["T"])[0]: 7,
        info.get("void_token", "U"): 8
    }
    demo_toks = ["D0", "D1", "D2"]
    resp_toks = ["R"]
    word2id = mconfig.get("word2id", default_word2id)
    pad_id =  word2id[info["pad_token"]]
    demo_id = word2id[info["demo_tokens"][0]]
    resp_id = word2id[info["resp_tokens"][0]]
    input_ids = []
    for seq in input_data:
        ids = []
        for tok in seq:
            if tok in word2id:
                ids.append(word2id[tok])
            elif tok in demo_toks:
                ids.append(demo_id)
            elif tok in resp_toks:
                ids.append(resp_id)
            else:
                print("Failed to convert", tok, "to id")
                ids.append(word2id.get("U", pad_id))
        input_ids.append(ids)
    return torch.LongTensor(input_ids)


def get_actvs(model, layer, input_ids):
    return collect_activations(
        model=model,
        input_ids=input_ids,
        layers=layer,
    )[layer]

if __name__=="__main__":
    model_folders, _, config = read_command_line_args()
    layers = config.get("layers", ["identities.0", "identities.0"])
    if type(layers)==str: layers = [layers for _ in range(2)]
    assert type(layers)==list and len(layers)==2
    overwrite = config.get("overwrite", True)

    # List of equal len lists of token str (including padding tokens)
    input_data = get_dataset(config=config)
    for model_folder1 in model_folders:
        save_path = os.path.join( model_folder1, "rsa_cka_sims.csv" )

        sim_data = initialize_sim_data_dict() # Dict
        old_df = pd.DataFrame(sim_data) if not os.path.exists(save_path)\
            else pd.read_csv(save_path)
        prev_comp_folders = set(old_df["model_folder2"])

        model1, mconfig1 = get_model_and_config(model_folder1)
        # Dict of tensors: input_ids, pad_mask
        mf1_data = tokenize_input_data(input_data, mconfig=mconfig1)
        # tensor of activations from the argued layer. default identities.0
        mf1_actvs = get_actvs(model=model1, layer=layers[0], input_ids=mf1_data)

        for model_folder2 in model_folders:
            if not overwrite and model_folder2 in prev_comp_folders:
                print("Skipping", model_folder2, "dup to previous record")

            model2, mconfig2 = get_model_and_config(model_folder2)
            mf2_data = tokenize_input_data(input_data, mconfig=mconfig2)
            mf2_actvs = get_actvs(
                model=model2, layer=layers[1], input_ids=mf2_data)

            sims = compute_similarities(X=mf1_actvs, X2=mf2_actvs, **config)
            sim_data["model_folder1"].append(model_folder1)
            sim_data["model_folder2"].append(model_folder2)
            sim_data["layer1"].append(layers[0])
            sim_data["layer2"].append(layers[1])
            for k in sims:
                sim_data[k].append(sims[k])

        new_df = pd.DataFrame(sim_data)
        if overwrite:
            old_df = old_df.loc[~old_df["model_folder2"].isin(set(new_df["model_folder2"]))]
        df = pd.concat([old_df, new_df])
        df.to_csv(save_path)
    print()
    print("Succeeded!")


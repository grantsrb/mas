import numpy as np
import torch
import copy
from tqdm import tqdm
import scipy.stats as stats

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
    device = K.device
    n = K.shape[0]  # Assume K and L are square matrices of the same size
    assert K.shape == L.shape, "K and L must have the same dimensions"

    # Centering matrix
    H = torch.eye(n) - (1/n) * torch.ones((n, n))
    H = H.to(device)

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

def get_cka(
        X, X2,
        batch_size=None,
        n_runs=1,
        sim_metric="cosine",
        prenorm=True,
        to_numpy=True,
        to_cpu=False,
        verbose=True,
):
    """
    X: torch tensor (B,N)
    X2: torch tensor (B,M)
    sim_metric: str
        "l2" or "cosine"
    prenorm: bool
        if true will normalize each neuron over the B dim
    to_numpy: bool
        if true, the CKA is returned as a numpy array
    to_cpu: bool
        if true, the CKA is returned on the cpu
    """
    ckas = []
    device = device_fxn(X.get_device())
    if verbose:
        pbar = tqdm(range(n_runs))
    else:
        pbar = range(n_runs)
    for run in pbar:
        if batch_size is not None:
            perm = torch.randperm(len(X)).long().to(device)[:batch_size]
            x1 = X[perm]
            x2 = X2[perm]
        else:
            x1 = X
            x2 = X2

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
        cka = compute_CKA(mtx1, mtx2)
        if to_cpu:
            cka = cka.cpu()
        ckas.append(cka)
    if to_numpy:
        return np.mean([c.cpu().data.numpy() for c in ckas])
    return torch.mean(ckas)

def get_rsa(
        X,X2,
        batch_size=None,
        n_runs=1,
        sim_metric="cosine",
        cor_type="spearmanr",
        prenorm=False,
):
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

def compute_similarities(
        X, X2,
        n_runs=10,
        sample_size=1000,
        *args,
        **kwargs
):
    """
    Args:
        X: torch tensor
        X2: torch tensor
        n_runs: int
            number of times to run and average over each metric
        sample_size: int
            optionally argue a sample size to use a uniformly sampled
            sub sample from the activation matrices.
    """
    sims = copy.deepcopy(default_sim_data_dict)

    print("Computing CKAs")
    sims["cka_cos"] = get_cka(X, X2, n_runs=n_runs,
        sim_metric="cosine", prenorm=True, batch_size=sample_size)
    sims["cka_l2"] = get_cka(X, X2, n_runs=n_runs,
        sim_metric="l2", prenorm=True, batch_size=sample_size)
    print("Computing RSAs")
    sims["rsa_raw_cos_spr"] = get_rsa(X, X2, n_runs=n_runs,
        sim_metric="cosine", cor_type="spearmanr",
        prenorm=False, batch_size=sample_size)
    sims["rsa_raw_l2_spr"] =  get_rsa(X, X2, n_runs=n_runs,
        sim_metric="l2", cor_type="spearmanr",
        prenorm=False, batch_size=sample_size)
    sims["rsa_raw_cos_prs"] = get_rsa(X, X2, n_runs=n_runs,
        sim_metric="cosine", cor_type="pearsonr",  
        prenorm=False, batch_size=sample_size)
    sims["rsa_raw_l2_prs"] =  get_rsa(X, X2, n_runs=n_runs,
        sim_metric="l2", cor_type="pearsonr",  
        prenorm=False, batch_size=sample_size)
    sims["rsa_nrm_cos_spr"] = get_rsa(X, X2, n_runs=n_runs,
        sim_metric="cosine", cor_type="spearmanr",
        prenorm=True, batch_size=sample_size)
    sims["rsa_nrm_l2_spr"] =  get_rsa(X, X2, n_runs=n_runs,
        sim_metric="l2", cor_type="spearmanr",
        prenorm=True, batch_size=sample_size)
    sims["rsa_nrm_cos_prs"] = get_rsa(X, X2, n_runs=n_runs,
        sim_metric="cosine", cor_type="pearsonr",
        prenorm=True, batch_size=sample_size)
    sims["rsa_nrm_l2_prs"] =  get_rsa(X, X2, n_runs=n_runs,
        sim_metric="l2",     cor_type="pearsonr",
        prenorm=True, batch_size=sample_size)
    return sims

def get_mean(x, axis=None, batch_size=1000):
    """
    Returns mean of x along argued axis. Used for reducing memory
    footprint on large datasets.

    x: ndarray or torch tensor
    axis: int
    batch_size: int
        size of increment when calculating mean
    """
    cumu_sum = 0
    if axis is None:
        for i in range(0,len(x), batch_size):
            cumu_sum = cumu_sum + x[i:i+batch_size].sum()
        return cumu_sum/x.numel()
    else:
        for i in range(0,len(x), batch_size):
            cumu_sum = cumu_sum + x[i:i+batch_size].sum(axis)
        return cumu_sum/x.shape[axis]

def get_std(x, axis=None, batch_size=1000, mean=None):
    """
    Returns std of x along argued axis. Used for reducing memory
    footprint on large datasets. Does not use n-1 correction.

    x: ndarray or torch tensor
    axis: int
    batch_size: int
        size of increment when calculating mean
    mean: int or ndarray or torch tensor
        The mean to be used in calculating the std. If None, mean is
        automatically calculated. If ndarray or torch tensor, must
        match datatype of x.
    """
    if type(x) == type(np.array([])):
        sqrt = np.sqrt
    else:
        sqrt = torch.sqrt
    if mean is None:
        mean = get_mean(x,axis,batch_size)
    cumu_sum = 0
    if axis is None:
        for i in range(0,len(x), batch_size):
            cumu_sum = cumu_sum + ((x[i:i+batch_size]-mean)**2).sum()
        return sqrt(cumu_sum/x.numel())
    else:
        for i in range(0,len(x), batch_size):
            cumu_sum=cumu_sum+((x[i:i+batch_size]-mean)**2).sum(axis)
        return sqrt(cumu_sum/x.shape[axis])

def pearsonr(x,y):
    """
    Calculates the pearson correlation coefficient along the 0th dimension.
    This gives same results as scipy's version but allows you to calculate
    the coefficient over much larger data sizes. Additionally allows
    calculation for torch tensors.

    Inputs:
        x: ndarray or torch tensor (T, ...)
            the dimension that will be averaged must be the first.
            dimensionality and type must match that of y
        y: ndarray or torch tensor (T, ...)
            the dimension that will be averaged must be the first.
            dimensionality and type must match that of x

    Returns:
        pearsonr: ndarray or torch tensor (...)
            shape will be the same as input but without the first
            dimension. As such, the correlations are calculated
            between cells in the same spatial location.

    """
    shape = None if len(x.shape) == 1 else x.shape[1:]
    assert type(x) == type(y)
    x = x.reshape(len(x), -1)
    y = y.reshape(len(y), -1)
    try:
        mux = x.mean(0)
        muy = y.mean(0)
        # STD calculation ensures same calculation is performed for
        # ndarrays and torch tensors. Torch tensor .std() uses n-1 
        # correction
        if isinstance(x, np.ndarray):
            sqrt = np.sqrt
        else:
            sqrt = torch.sqrt
        sigx = sqrt((x**2).mean(0)-mux**2)+1e-7
        sigy = sqrt((y**2).mean(0)-muy**2)+1e-7
    except MemoryError as e:
        mux = get_mean(x,axis=0)
        muy = get_mean(y,axis=0)
        sigx = get_std(x,mean=mux,axis=0)+1e-7
        sigy = get_std(y,mean=muy,axis=0)+1e-7
    x = x-mux
    y = y-muy
    numer = (x*y).mean(0)
    denom = sigx*sigy
    r = numer/denom
    if shape is not None:
        r = r.reshape(shape)
    return r

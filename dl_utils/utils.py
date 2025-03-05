import numpy as np
import torch
import os
import sys
import subprocess
from datetime import datetime
try:
    import cv2
    from tqdm import tqdm
except:
    pass

def device_fxn(device):
    if type(device)==str: return device
    if device<0: return "cpu"
    return device

def try_key(d, key, val):
    """
    d: dict
    key: str
    val: object
        the default value if the key does not exist in d
    """
    if key in d:
        return d[key]
    return val

def get_datetime_str():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

def resize2Square(img, size):
    """
    resizes image to a square with the argued size. Preserves the aspect
    ratio.

    img: ndarray (H,W, optional C)
    size: int
    """
    h, w = img.shape[:2]
    c = None if len(img.shape) < 3 else img.shape[2]
    if h == w: 
        return cv2.resize(img, (size, size), cv2.INTER_AREA)
    if h > w: 
        dif = h
    else:
        dif = w
    interpolation = cv2.INTER_AREA if dif > size else\
                    cv2.INTER_CUBIC
    x_pos = int((dif - w)/2.)
    y_pos = int((dif - h)/2.)
    if c is None:
      mask = np.zeros((dif, dif), dtype=img.dtype)
      mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
      mask = np.zeros((dif, dif, c), dtype=img.dtype)
      mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
    return cv2.resize(mask, (size, size), interpolation)

def rand_sample(arr, n_samples=1):
    """
    Randomly samples a single element from the argued array.

    arr: sequence of some sort
    """
    if not isinstance(arr,list): arr = list(arr)
    if len(arr) == 0: print("len 0:", arr)
    samples = []
    perm = np.random.permutation(len(arr))
    for i in range(n_samples):
        samples.append(arr[perm[i]])
    if len(samples) == 1: return samples[0]
    return samples

def get_max_key(d):
    """
    Returns key corresponding to maxium value

    d: dict
        keys: object
        vals: int or float
    """
    max_v = -np.inf
    max_k = None
    for k,v in d.items():
        if v > max_v:
            max_v = v
            max_k = k
    return max_k

def update_shape(shape, kernel=3, padding=0, stride=1, op="conv"):
    """
    Calculates the new shape of the tensor following a convolution or
    deconvolution

    shape: list-like or int
        the height/width of the activations
    kernel: int or list-like
        size of the kernel
    padding: list-like or int
    stride: list-like or int
    op: str
        'conv' or 'deconv'
    """
    if type(shape) == type(int()):
        shape = np.asarray([shape])
    else:
        shape = np.asarray(shape)
    if type(kernel) == type(int()):
        kernel = np.asarray([kernel for i in range(len(shape))])
    else:
        kernel = np.asarray(kernel)
    if type(padding) == type(int()):
        padding = np.asarray([padding for i in range(len(shape))])
    else:
        padding = np.asarray(padding)
    if type(stride) == type(int()):
        stride = np.asarray([stride for i in range(len(shape))])
    else:
        stride = np.asarray(stride)

    if op == "conv":
        shape = (shape - kernel + 2*padding)/stride + 1
    elif op == "deconv" or op == "conv_transpose":
        shape = (shape - 1)*stride + kernel - 2*padding
    if len(shape) == 1:
        return int(shape[0])
    return [int(s) for s in shape]

def top_k_acc(preds, labels, k=5, as_tensor=False):
    """
    Returns the top_n accuracy for the argued predictions and labels

    Args:
        preds: torch float tensor (B, L)
            the logits or probabilities
        labels: torch long tensor (B,)
            the correct labels
        k: int
            the k to use for top k
        as_tensor: bool
            if true, returns result as a tensor
    Returns:
        top_n: float or tensor

    """
    ps = preds.reshape(-1,preds.shape[-1])
    args = torch.topk(ps,k,largest=True,sorted=False,dim=-1).indices
    acc = (args==labels.reshape(-1)[:,None]).float().sum(-1).mean()
    if as_tensor:
        return acc
    return acc.item()

def pad_list(arr, tot_len, fill_val=0, side="right"):
    """
    Pads the argued list to the goal length. Operates in place.

    Args:
        arr: list
        tot_len: int
            the length of the resulting array
        fill_val: object
            the value to use for the padding
        side: str
            pad on the left side or the right
    Returns:
        arr: list
            the padded list
    """
    n_pad = tot_len - len(arr)
    if n_pad<=0: return arr
    if side=="right":
        for _ in range(n_pad):
            arr.append(fill_val)
    else:
        padding = [fill_val for _ in range(n_pad)]
        arr = padding + arr
    return arr

def pad_list_to(arr, tot_len, fill_val=0, side="right"):
    """
    Pads the argued array to the goal length. Operates in place.

    Args:
        arr: list
        tot_len: int
            the length to pad to
        fill_val: int
            the symbol to use for the padding
        side: str
            pad on the left side or the right
    Returns:
        arr: list
            the padded list
    """
    n_pad = tot_len - len(arr)
    if n_pad<=0: return arr
    if side=="right":
        for i in range(n_pad):
            arr.append(fill_val)
    else:
        padding = [fill_val for _ in range(n_pad)]
        arr = padding + arr
    return arr

def pad_to(arr, tot_len, fill_val=0, side="right", dim=-1):
    """
    Pads the argued list to the goal length along a single dimension.

    Args:
        arr: list or ndarray or torch tensor
            cannot take a mixture of datatypes. If list is argued,
            must be 1 dimensional list. Cannot take 2d list.
        tot_len: int
            the length of the resulting array
        fill_val: object
            the value to use for the padding
        side: str
            pad on the left side or the right
    Returns:
        arr: list
            the padded list
    """
    if type(arr)==list:
        return pad_list_to(
            arr,
            tot_len=tot_len,
            fill_val=fill_val,
            side=side,
        )
    if dim<0: dim = len(arr.shape) + dim
    n_pad = tot_len - arr.shape[dim]
    if n_pad<=0: return arr
    tup = (0,n_pad) if side=="right" else (n_pad, 0)
    if type(arr)==type(np.zeros((1,))):
        pad_tups = [
            (0,0) if i!= dim else tup for i in range(len(arr.shape))
        ]
        arr = np.pad(arr, pad_tups, constant_values=fill_val)
    elif type(arr)==type(torch.zeros(1)):
        pad_tup = [0 for _ in range(2*len(arr.shape))]
        # PyTorch decided to make things complicated by reversing the
        # order that the tuple refers to
        pad_tup[-2*(dim+1)+int(side=="right")] = n_pad
        arr = torch.nn.functional.pad(
            arr, tuple(pad_tup), value=fill_val
        )
    return arr

def num2base(n, b):
    """
    Converts a number to a new base returning a string.
    (Taken from Stack Overflow)

    Args:
        n: int
            the number that is currently in base 10 that you would
            like to convert to another base b
        b: int
            the new number base
    Returns:
        numerals: list of ints
            the numerals of the argued number in the new base
    """

    if n == 0:
        return [0]
    numerals = []
    while n:
        numerals.append(int(n % b))
        n //= b
    return numerals[::-1]

def get_one_hot(ids, L):
    """
    Ignores ids that are less than 0.

    Args:
        ids: torch long tensor (..., N)
        L: int
            the length of the one-hot vector
    Returns:
        one_hots: torch long tensor (..., N, L)
    """
    to_list = False
    if type(ids)==list:
        to_list = True
        ids = torch.LongTensor(ids)
    ignores = ids<0
    if torch.any(ignores):
        ids[ignores] = 0
    shape = [*ids.shape, L]
    device = ids.get_device()
    if device<0: device = "cpu"
    one_hots = torch.zeros( shape, device=device )
    one_hots.scatter_(
        dim=-1,
        index=ids[...,None],
        src=torch.ones_like(one_hots)
    )
    if torch.any(ignores):
        one_hots[ignores] = 0
    if to_list:
        one_hots = one_hots.tolist()
    return one_hots

def get_one_hots(*args, **kwargs):
    return get_one_hot(*args, **kwargs)

def get_mask_past_id(src, id_, incl_id=False):
    """
    Returns a mask in which ones denote all spaces after the first
    occurance of the argued `id_`

    Args:
        src: long tensor  (B,S)
        id_: int
        incl_id: bool
            optionally include the first occurance of the id in the
            mask.
    Returns:
        mask: bool tensor (B,S)
            true values denote indexes past or including the first
            occurance of the `id_` along the last dimension
    """
    return get_mask_past_ids(src, ids=id_)

def get_mask_past_ids(src, ids, incl_id=False, last_occurence=False):
    """
    Returns a mask in which ones denote all spaces after the first
    occurance of any of the values within `ids`.

    Args:
        src: long tensor  (B,S)
        ids: sequence of ints or int or long tensor (M,)
        incl_id: bool
            optionally include the first occurance of the id in the
            mask.
        last_occurence: bool
            if true, will return the mask past the last occurence of the
            argued id. Otherwise will return the mask after the first
            occurence reading from left to right. Defaults to false
            because this is less efficient.
    Returns:
        mask: bool tensor (B,S)
            true values denote indexes past (or including) the first
            occurance of a value within `ids` along the last dimension
    """
    if type(ids)==int:
        ids = torch.LongTensor([ids])
    elif type(ids)==list or type(ids)==set:
        ids = torch.LongTensor([*ids])
    device = device_fxn(src.get_device())
    ids = ids.to(device)
    B,S = src.shape
    is_id = torch.isin(src, ids).long()
    if last_occurence: id_idxs = arglast(is_id, dim=-1)
    else: id_idxs = torch.argmax(is_id, dim=-1)
    # if ids does not appear, then default idx is past last idx
    id_idxs[torch.sum(is_id,dim=-1)==0] = src.shape[-1]
    arange = torch.arange(S)[None].repeat((B,1)).long()
    if incl_id:
        mask = arange.to(device)>=id_idxs[:,None]
    else:
        mask = arange.to(device)>id_idxs[:,None]
    return mask

def get_causal_mask_like(inpt: torch.Tensor):
    """
    Generates an upper-triangular matrix of True, with Falses on
    diag and lower triangle like the argued inpt. Thus, this
    generates a causal mask where True denotes the unattended tokens.

    Args:
        inpt: tensor (...,S,L)
    Returns:
        BoolTensor: (1,S,L)
    """
    if len(inpt.shape)==2:
        S = inpt.shape[-1]
        L = S
        mask = generate_square_subsequent_mask(S)
    else:
        S,L = inpt.shape[-2], inpt.shape[-1]
        mask = generate_square_subsequent_mask(max(S,L))
        mask = mask[:S,:L]
    device = inpt.get_device()
    if device<0: device = "cpu"
    return mask.to(device)[None]

def generate_square_subsequent_mask(
        sz: int,
        device=torch.device(torch._C._get_default_device()),
        dtype="bool"):
    """
    Generates an upper-triangular matrix of True, with Falses on
    diag and lower triangle. Thus, False represents attended indices.

    Args:
        sz: int
            the size of the square mask
        device: int or str or None
        dtype: str ("bool" or "float")
    Returns:
        BoolTensor (sz,sz)
            False values in lower left including the diagonal
    """
    if dtype=="float" or dtype==float:
        mask = torch.triu(
            torch.full(
                (sz, sz),
                float('-inf'),
                device=device
            ).float(),
            diagonal=1,
        )
    else:
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
    return mask.to(device)


def generate_ktoken_causal_mask(
        sz: int,
        k: int=5,
        device=torch.device(torch._C._get_default_device()),
        dtype="bool"):
    """
    Generates an upper-triangular matrix of True, with Falses on
    diag and lower triangle. Thus, False represents attended indices.

    Args:
        sz: int
            the size of the square mask
        k: int
            the number of tokens to include in the attention window.
            This does not include the self-attention, so, the mask
            actually has k+1 entries along each row.
        device: int or str or None
        dtype: str ("bool" or "float")
    Returns:
        BoolTensor (sz,sz)
            False values in lower left including the diagonal
    """
    mask = torch.triu(torch.ones(sz, sz), diagonal=1).long()
    submask = 1-torch.triu(torch.ones(sz, sz), diagonal=-k).long()
    mask = (mask + submask).bool()
    if dtype=="float" or dtype==float:
        temp = mask
        mask = torch.full((sz,sz), float("-inf"),device=device).float()
        mask[~temp] = 0
    return mask.to(device)


def arglast(mask, dim=None, axis=-1):
    """
    This function finds the index of the last max value along a given
    dimension. torch.flip creates a copy of the tensor, so it's
    actually not as efficient as using numpy's np.flip which only
    returns a view.

    Args:
        mask: bool (B,N)
        dim: int
    Returns:
        the index of the last true value along the dimension
    """
    if dim is None: dim = axis
    if type(mask)==type(np.zeros(1)):
        argmaxs = np.argmax(np.flip(mask, axis=dim), axis=dim)
    else:
        argmaxs = torch.argmax(torch.flip(mask.float(), dims=(dim,)), dim=dim)
    return mask.shape[dim] - argmaxs - 1

def padmask2attnmask(pad_mask):
    """
    Converts a padding mask into an attention mask to be argued to
    huggingface's attention_mask. Does so by doing an outer product
    of the row vectors with themselves. The result allows you to
    combine masks with more flexibility.

    IMPORTANT!!!!!!!! The mask must be made such that true values
        denote non-padding, i.e. do-attend-to tokens. Be very
            careful here! This function will not work if the mask is
            inverted.

    Args:
        pad_mask: Tensor (B,S)
            true values denote non-padding, false is padding. Be very
            careful here! This function will not work if the mask is
            inverted.
    Returns:
        attn_mask: Tensor (B,S,S)
    """
    B,S = pad_mask.shape
    reps = (1,S,1)
    return pad_mask[:,None].repeat(reps)
    #return torch.einsum("bs,bj->bsj", pad_mask, pad_mask)

def get_causal_mask(sz: int):
    """
    Generates an upper-triangular matrix of True, with Falses on
    diag and lower triangle.

    Returns:
        BoolTensor (sz,sz)
            True values are masked out (non-attended) values
    """

    return generate_square_subsequent_mask(sz)

def get_causal_cross_mask(step_masks):
    """
    This function uses the high level step indices to build a mask
    to prevent the different modalities from looking ahead in time
    while allowing different numbers of single modal sub steps for
    a given global multi modal time step. To make a cross mask for
    more than 2 modalities, use this function for every possible
    combination and stitch the masks together.

    Args:
        step_masks: list of long tensors [(B,S1), (B,S2)]
            a list of length 2 of tensors that denote the global,
            multi-modal time step of the individual mode.
    Returns:
        cross_mask: bool tensor (B,S1,S2)
            a cross mask to align modalities temporally. The length
            of the list is determined by the number of elements
            in `seq_lens`
    """
    device = step_masks[0].get_device()
    if device<0: device = "cpu"
    for smask in step_masks:
        smask[smask<0] = torch.max(smask)+1
    shape = [*step_masks[0].shape, step_masks[1].shape[-1]]
    cross_mask = torch.zeros(shape).to(device)
    cross_mask = cross_mask + step_masks[0][..., None]
    cross_mask = cross_mask - step_masks[1][:,None]
    cross_mask[cross_mask<=0] = -1
    cross_mask[cross_mask>0] = 0
    cross_mask[cross_mask<0] = 1
    return cross_mask.bool()

def get_full_cross_mask(step_masks):
    """
    Constructs a causal cross mask by stitching different types of
    masks together. The full mask consists of a standard causal mask
    for attending to positions intra-modality (within modality) and a
    causal cross mask for attending inter-modality (outside of modality).

    Mask: [mode1 causal mask,   cross causal mask1 ]
          [cross causal mask2, mode2 causal mask ]

    Args:
        step_masks: list of long tensors [(B,S1), (B,S2)]
            a list of length 2 of tensors that denote the global,
            multi-modal time step of the individual mode.
    Returns:
        cross_mask: bool tensor (B,S1+S2,S1+S2)
            a causal cross attention mask. true values mean padding,
            non-attended locations. Does not allow modality x to
            attend to current timestep of modality y and visa-versa.
    """
    # TODO: Allow longer sequence to attend to shorter sequence at
    #   current global timestep and allow shorter sequence to attend
    #   to first substep of longer sequence at current timestep
    device = step_masks[0].get_device()
    if device<0: device = "cpu"
    cross_mask1 = get_causal_cross_mask(step_masks)
    mode1_mask = get_causal_mask(step_masks[0].shape[-1]).to(device)
    mode2_mask = get_causal_mask(step_masks[1].shape[-1]).to(device)
    cross_mask2 = torch.flip(torch.rot90(
        cross_mask1,k=1,dims=(1,2)
    ),dims=(-1,))
    cross_mask = torch.cat([
        torch.cat([
            mode1_mask[None].repeat((len(cross_mask1),1,1)),
            cross_mask1
        ],dim=-1),
        torch.cat([ 
            cross_mask2, mode2_mask[None].repeat((len(cross_mask1),1,1))
        ],dim=-1)
    ],dim=1)
    return cross_mask

def get_mask_past_idx(shape, idx, inclusive=False):
    """
    Returns a binary mask past the argued indices in the idx vector
    along the last axis.

    Args:
        shape: tuple of ints
            the shape of the resulting mask
        idx: long tensor (B,)
            the indices that mark the start of the mask
    Returns:
        mask: bool tensor (shape)
            ones are at indices after the argued idx along the last dim
    """
    device = idx.get_device()
    if device<0: device = "cpu"
    try:
        arr = torch.arange(shape[-1])
    except:
        if hasattr(shape,"shape"): # probably argued a tensor on accident
            shape = shape.shape
            arr = torch.arange(shape[-1])
        else: assert False
    reps = []
    for _ in range(len(shape)-1):
        arr = arr[None]
    reps = tuple(list(shape[:-1])+[1])
    arr = arr.repeat(reps).to(device)
    if inclusive: return arr>=idx[:,None]
    return arr>idx[:,None]

def get_mask_between(shape, startx, endx):
    """
    Returns a binary mask that ranges from the start indices to the
    end indices along the last axis. Excludes the indices argued in endx.

    Args:
        shape: tuple of ints
            the shape of the resulting mask
        startx: tensor (B,)
            the starting indices. should have the same length as the
            non-final dimensions of shape
        endx: tensor (B,)
            the ending indices. same shape as startx
    Returns:
        mask: bool tensor (shape)
    """
    arr = torch.arange(shape[-1])
    reps = []
    for _ in range(len(shape)-1):
        arr = arr[None]
    reps = tuple(list(shape[:-1])+[1])
    arr = arr.repeat(reps)
    return (arr>=startx[:,None])&(arr<endx[:,None])

def package_versions(globals_dict=None, verbose=False):
    """
    Finds the versions of all packages used in this script

    Args:
        globals_dict: dict
            just argue `globals()`
    """
    if globals_dict is None: globals_dict = globals()
    packages = dict()
    modules = list(set(sys.modules) & set(globals_dict))
    if verbose:
        print("Packages:")
    for module_name in modules:
        module = sys.modules[module_name]
        try:
            v = getattr(module, '__version__', 'unknown')
            packages[module_name] = v
            if verbose:
                print("\t", module_name, v)
        except:
            packages[module_name] = "unknown"
    return packages

def get_git_revision_hash():
    """
    Finds the current git hash
    """
    return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).decode('ascii').strip()

def mtx_cor(X, Y, batch_size=500, to_numpy=False, zscore=True, device=None):
    """
    Creates a correlation matrix for X and Y using the GPU

    X: torch tensor or ndarray (T, C) or (T, C, H, W)
    Y: torch tensor or ndarray (T, K) or (T, K, H1, W1)
    batch_size: int
        batches the calculation if this is not None
    to_numpy: bool
        if true, returns matrix as ndarray
    zscore: bool
        if true, both X and Y are normalized over the T dimension
    device: int
        optionally argue a device to use for the matrix multiplications

    Returns:
        cor_mtx: (C,K) or (C*H*W, K*H1*W1)
            the correlation matrix
    """
    if len(X.shape) < 2:
        X = X[:,None]
    if len(Y.shape) < 2:
        Y = Y[:,None]
    if len(X.shape) > 2:
        X = X.reshape(len(X), -1)
    if len(Y.shape) > 2:
        Y = Y.reshape(len(Y), -1)
    if type(X) == type(np.array([])):
        to_numpy = True
        X = torch.FloatTensor(X)
        Y = torch.FloatTensor(Y)
    if device is None:
        device = X.get_device()
        if device<0: device = "cpu"
    if zscore:
        xmean = X.mean(0)
        xstd = torch.sqrt(((X-xmean)**2).mean(0))
        ymean = Y.mean(0)
        ystd = torch.sqrt(((Y-ymean)**2).mean(0))
        xstd[xstd<=0] = 1
        X = (X-xmean)/(xstd+1e-5)
        ystd[ystd<=0] = 1
        Y = (Y-ymean)/(ystd+1e-5)
    X = X.permute(1,0)

    with torch.no_grad():
        if batch_size is None:
            X = X.to(device)
            Y = Y.to(device)
            cor_mtx = torch.einsum("it,tj->ij", X, Y).detach().cpu()
        else:
            cor_mtx = []
            for i in range(0,len(X),batch_size): # loop over x neurons
                sub_mtx = []
                x = X[i:i+batch_size].to(device)

                # Loop over y neurons
                for j in range(0,Y.shape[1], batch_size):
                    y = Y[:,j:j+batch_size].to(device)
                    cor_block = torch.einsum("it,tj->ij",x,y)
                    cor_block = cor_block.detach().cpu()
                    sub_mtx.append(cor_block)
                cor_mtx.append(torch.cat(sub_mtx,dim=1))
            cor_mtx = torch.cat(cor_mtx, dim=0)
    cor_mtx = cor_mtx/len(Y)
    if to_numpy:
        return cor_mtx.numpy()
    return cor_mtx

def requires_grad(model, state):
    """
    Turns grad calculations on and off for all parameters in the model

    model: torch Module
    state: bool
        if true, gradient calculations are performed
        if false, gradient calculations are not
    """
    for p in model.parameters():
        try:
            p.requires_grad = state
        except:
            pass

def get_hook(key, layer_dict=None, to_numpy=True, to_cpu=False):
    """
    Returns a hook function that can be used to collect gradients
    or activations in the backward or forward pass respectively of
    a torch Module.

    key: str
        name of layer of interest
    layer_dict: dict
        Can be empty
        keys: str
            names of model layers of interest
        vals: NA
    to_numpy: bool
        if true, the gradients/activations are returned as ndarrays.
        otherwise they are returned as torch tensors
    """
    if layer_dict is None: layer_dict = dict()
    if to_numpy:
        def hook(module, inp, out):
            layer_dict[key] = out.detach().cpu().numpy()
    elif to_cpu:
        def hook(module, inp, out):
            layer_dict[key] = out.cpu()
    else:
        def hook(module, inp, out):
            layer_dict[key] = out
    return hook, layer_dict

def inspect(model, X, insp_keys=set(), batch_size=500, to_numpy=True,
                                                       to_cpu=True,
                                                       no_grad=False,
                                                       verbose=False):
    """
    Get the response from the argued layers in the model as np arrays.
    If model is on cpu, operations are performed on cpu. Put model on
    gpu if you want operations to be performed on gpu.

    Args:
        model - torch Module or torch gpu Module
        X - ndarray or FloatTensor (T,C,H,W)
        insp_keys - set of str
            name of layers activations to collect. if empty set, only
            the final output is returned.
        to_numpy - bool
            if true, activations will all be ndarrays. Otherwise torch
            tensors
        to_cpu - bool
            if true, torch tensors will be on the cpu.
            only effective if to_numpy is false.
        no_grad: bool
            if true, gradients will not be calculated. if false, has
            no impact on function.

    returns: 
        layer_outs: dict of np arrays or torch cpu tensors
            Each inspection key will have a key in this dict with a key
            of the activations at the inspected layer. Also, the output
            layer will be included under the name "outputs".
            Keys:
                "outputs": default key for output layer
                "<insp_key_i>": output at layer <insp_key_i>
    """
    layer_outs = dict()
    handles = []
    insp_keys_copy = set()
    for key, mod in model.named_modules():
        if key in insp_keys:
            insp_keys_copy.add(key)
            hook, _ = get_hook(
                key,
                layer_dict=layer_outs,
                to_numpy=to_numpy,
                to_cpu=to_cpu)
            handle = mod.register_forward_hook(hook)
            handles.append(handle)
    set_insp_keys = set(insp_keys)
    if len(set_insp_keys-insp_keys_copy) > 0 and "outputs" not in set_insp_keys:
        print("Insp keys:", set_insp_keys-insp_keys_copy, "not found")
    insp_keys = insp_keys_copy
    if not isinstance(X,torch.Tensor):
        X = torch.FloatTensor(X)

    # prev_grad_state is used to ensure we do not mess with an outer
    # "with torch.no_grad():" statement
    prev_grad_state = torch.is_grad_enabled() 
    if to_numpy or no_grad:
        # Turns off all gradient calculations. When returning numpy
        # arrays, the computation graph is inaccessible, as such we
        # do not need to calculate it.
        torch.set_grad_enabled(False)

    device = device_fxn(next(model.parameters()).get_device())
    try:
        if batch_size is None or batch_size > len(X):
            preds = model(X.to(device))
            if to_numpy:
                layer_outs['outputs'] = preds.detach().cpu().numpy()
            else:
                layer_outs['outputs'] = preds.cpu()
        else:
            batched_outs = {key:[] for key in insp_keys}
            outputs = []
            rnge = range(0,len(X), batch_size)
            if verbose:
                rnge = tqdm(rnge)
            for batch in rnge:
                x = X[batch:batch+batch_size]
                preds = model(x.to(device)).cpu()
                if to_numpy: preds = preds.detach().numpy()
                outputs.append(preds)
                for k in layer_outs.keys():
                    batched_outs[k].append(layer_outs[k])
                    layer_outs[k] = None
            batched_outs['outputs'] = outputs
            if to_numpy:
                layer_outs = {k:np.concatenate(v,axis=0) for k,v in\
                                               batched_outs.items()}
            else:
                layer_outs = {k:torch.cat(v,dim=0) for k,v in\
                                         batched_outs.items()}
    except RuntimeError as e:
        print("Runtime error. Check your batch size and try using",
                "inspect with torch.no_grad() enabled")
        raise RuntimeError(str(e))

        
    # If we turned off the grad state, this will turn it back on.
    # Otherwise leaves it the same.
    torch.set_grad_enabled(prev_grad_state) 

    # This for loop ensures we do not create a memory leak when
    # using hooks
    for i in range(len(handles)):
        handles[i].remove()
    del handles
    return layer_outs

def integrated_gradient(
        model, X, layer,
        intg_shape=None,
        output_units=None,
        alpha_steps=10,
        batch_size=500,
        y=None,
        lossfxn=None,
        to_numpy=False,
        verbose=False):
    """
    Returns the integrated gradient for a particular stimulus at the
    argued layer. This function always operates with the model in
    eval mode due to the need for a deterministic model. If the model
    is argued in train mode, it is set to eval mode for this function
    and returned to train mode at the end of the function. As such,
    this note is largely irrelavant, but will hopefully satisfy the
    curious or anxious ;)

    Inputs:
        model: PyTorch Module
        X: Input stimuli ndarray or torch FloatTensor (T,D,H,W)
        layer: str layer name
        intg_shape: None or tuple of ints
            the shape of the integrated gradient ignoring the batch dim.
        output_units: int or list of ints or None
            the indices of the output units of interest. if None, uses
            sum of all output units as function to differentiate.
        alpha_steps: int, integration steps
        batch_size: step size when performing computations on GPU
        y: torch FloatTensor or ndarray (T,N)
            if None, ignored
        lossfxn: some differentiable function
            if None, ignored
    Outputs:
        intg_grad: ndarray or FloatTensor (T, D) or (T, *intg_shape)
            integrated gradient
        gc_activs: ndarray or FloatTensor (T,N)
            activation of the final layer of the model
    """
    raise NotImplemented # UNTESTED CODE AT THIS POINT

    # Handle Gradient Settings
    # Model gradient unnecessary for integrated gradient
    requires_grad(model, False)

    # Save current grad calculation state
    prev_grad_state = torch.is_grad_enabled()
    torch.set_grad_enabled(True) # Enable grad calculations
    prev_train_state = model.training
    model.eval()

    if intg_shape is None or output_units is None:
        output = inspect(model, X[:1], insp_keys=[layer],
                                           batch_size=None,
                                           to_cpu=True,
                                           no_grad=True,
                                           verbose=False)
        if intg_shape is None: intg_shape = output[layer].shape[1:]
        if output_units is None:
            n_units = output["output"].reshape(1,-1).shape[-1]
            output_units = list(range(n_units))
    intg_grad = torch.zeros(len(X), *intg_shape)
    if isinstance(output_units,int): output_units = [output_units]
    out_logits = None


    if batch_size is None: batch_size = len(X)
    if not isinstance(X, torch.Tensor): X = torch.FloatTensor(X)
    X.requires_grad = True
    idxs = torch.arange(len(X)).long()
    n_loops = int(np.ceil(len(X)/batch_size))
    for batch in range(n_loops):
        prev_response = None
        linspace = torch.linspace(0,1,alpha_steps)
        if verbose:
            print("Calculating for batch {}/{}".format(batch, len(X)))
            linspace = tqdm(linspace)
        idx = idxs[batch*batch_size:(batch+1)*batch_size]
        for alpha in linspace:
            x = alpha*X[idx]
            # Response is dict of activations. response[layer] has
            # shape intg_grad.shape
            response = inspect(model, x, insp_keys=[layer],
                                           batch_size=None,
                                           to_numpy=False,
                                           to_cpu=False,
                                           no_grad=False,
                                           verbose=False)
            if prev_response is not None:
                ins = response[layer]
                outs = response['outputs'][:,output_units]
                if lossfxn is not None and y is not None:
                    truth = y[idx][:,output_units]
                    outs = lossfxn(outs,truth)
                grad = torch.autograd.grad(outs.sum(), ins)[0]
                grad = grad.data.detach().cpu()
                grad = grad.reshape(len(grad), *intg_grad.shape[1:])
                # At intermediate layers, we don't know if the 0 alpha
                # point has a response of 0, so we must subtract the
                # zero alpha response to find the appropriate baseline.
                # We also sum over each intermediate step because there
                # might be a nonlinear response curve at intermediate
                # layers.
                act = (response[layer].data.cpu()-prev_response[layer])
                act = act.reshape(grad.shape)
                intg_grad[idx] += grad*act
                if alpha == 1:
                    if out_logits is None:
                      out_logits = torch.zeros(len(X),len(output_units))
                    outs = response['outputs'][:,output_units]
                    out_logits[idx] = outs.data.cpu()
            prev_response = {k:v.data.cpu() for k,v in response.items()}
    del response
    del grad

    # Return to previous gradient calculation state
    requires_grad(model, True)
    # return to previous grad calculation state and training state
    torch.set_grad_enabled(prev_grad_state)
    if prev_train_state: model.train()
    if to_numpy:
        ndgrad = intg_grad.data.cpu().numpy()
        ndactivs = out_logits.data.cpu().numpy()
        return ndgrad, ndactivs
    return intg_grad, out_logits

def rolling_window(array, window, time_axis=0):
    """
    Make an ndarray with a rolling window of the last dimension

    Parameters
    ----------
    array : array_like
        Array to add rolling window to

    window : int
        Size of rolling window

    time_axis : int, optional
        The axis of the temporal dimension, either 0 or -1
        (Default: 0)

    Returns
    -------
    Array that is a view of the original array with a added dimension
    of size w.

    Examples
    --------
    >>> x=np.arange(10).reshape((2,5))
    >>> rolling_window(x, 3)
    array([[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
           [[5, 6, 7], [6, 7, 8], [7, 8, 9]]])

    Calculate rolling mean of last dimension:

    >>> np.mean(rolling_window(x, 3), -1)
    array([[ 1.,  2.,  3.],
           [ 6.,  7.,  8.]])
    """
    if time_axis == 0:
        array = array.T
    elif time_axis==-1 or time_axis==len(array.shape)-1:
        pass
    else:
        raise ValueError('Time axis must be 0 (first dimension) or -1\
                                                              (last)')

    assert window >= 1, "`window` must be at least 1."
    assert window < array.shape[-1], "`window` is too long."

    # with strides
    shape = array.shape[:-1] + (array.shape[-1] - window, window)
    strides = array.strides + (array.strides[-1],)
    arr = np.lib.stride_tricks.as_strided(array, shape=shape,
                                             strides=strides)

    if time_axis == 0:
        return np.rollaxis(arr.T, 1, 0)
    else:
        return arr


if __name__=="__main__":
    shape = (5,6)
    sz = 10
    k = 4
    for i in range(3):
        window = k+i
        mask = generate_ktoken_causal_mask(
            sz=sz, k=window, dtype="float"
        )
        print("sz:", sz)
        print("k:", window)
        print(mask)
        print(mask.long())
        print()

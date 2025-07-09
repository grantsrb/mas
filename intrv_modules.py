import math
import torch
import copy

from fca import FunctionalComponentAnalysis
from utils import device_fxn
from dl_utils.torch_modules import IdentityModule, InvTanh, InvSigmoid

class RankRotationMatrix(torch.nn.Module):
    def __init__(self,
            size,
            rank=None,
            identity_init=False,
            bias=False,
            mu=0,
            sigma=1,
            identity_rot=False,
            orthogonal_map=None,
            nonlin_align_fn=None,
            **kwargs):
        """
        size: int
            the height and width of the rotation matrix
        rank: int
            the rank of the rotation matrix
        identity_init: bool
            if true, will initialize the rotation matrix to the identity
            matrix.
        bias: bool
            if true, will include a shifting term in the rotation matrix
        mu: float or FloatTensor (size,)
            Used to center each feature dim of the activations.
        sigma: float or FloatTensor (size,)
            Used to scale each feature dim of the activations.
        identity_rot: bool
            if true, will always reset the rotation matrix to the
            identity. Used for debugging.
        nonlin_align_fn: callable
            inverse of a function to apply to the input before the rotation matrix.
        """
        super().__init__()
        if rank is None or not rank: rank = size
        self.rank = rank
        self.identity_rot = identity_rot
        self.identity_init = identity_init
        self.set_nonlin_fn(nonlin_align_fn)

        if type(mu)==float or type(mu)==int:
            self.mu = mu
        else:
            self.register_buffer("mu", mu)
        if type(sigma)==float or type(sigma)==int:
            self.sigma = sigma
        else:
            self.register_buffer("sigma", sigma)

        lin = torch.nn.Linear(size, size, bias=False)
        if identity_init:
            lin.weight.data = torch.eye(
                size,dtype=lin.weight.data.dtype)

        # Shifting parameters
        if bias:
            self.bias = torch.nn.Parameter(
                torch.zeros(size,dtype=lin.weight.data.dtype))
        else:
            self.bias = 0
        if self.identity_rot:
            self.rot_module = lin
        else:
            # Orthogonal parameterization ensures that the weight is always
            # orthogonal
            self.rot_module = torch.nn.utils.parametrizations.orthogonal(
                lin, orthogonal_map=orthogonal_map)

    @property
    def weight(self):
        if self.identity_rot:
            self.rot_module.weight.data = torch.eye(
              self.size,
              dtype=self.rot_module.weight.data.dtype,
              device=device_fxn(self.rot_module.weight.get_device()),
            )
        return self.rot_module.weight[:,:self.rank]

    @property
    def weight_inv(self):
        if self.identity_rot:
            self.rot_module.weight.data = torch.eye(
              self.size,
              dtype=self.rot_module.weight.data.dtype,
              device=device_fxn(self.rot_module.weight.get_device()),
            )
        return self.rot_module.weight[:,:self.rank].T

    @property
    def size(self):
        return self.rot_module.weight.shape[0]

    @property
    def shape(self):
        return self.weight.shape

    def reset(self):
        pass

    def get_condition(self, p=None):
        return torch.linalg.cond(self.weight, p=p)

    def set_nonlin_fn(self, nonlin_align_fn):
        """
        Sets the non-linear function to apply to the input before the
        rotation matrix. Actually uses the inverse first!!
        """
        if nonlin_align_fn is None or nonlin_align_fn=="identity":
            self.nonlin_fwd = IdentityModule()
            self.nonlin_inv = IdentityModule()
        elif nonlin_align_fn=="tanh":
            self.nonlin_fwd = InvTanh()
            self.nonlin_inv = torch.nn.Tanh()
        elif nonlin_align_fn=="sigmoid":
            self.nonlin_fwd = InvSigmoid()
            self.nonlin_inv = torch.nn.Sigmoid()
        else:
            raise ValueError("nonlin_align_fn must be identity, tanh, or sigmoid, got: {}".format(nonlin_align_fn))

    def rot_forward(self, h):
        h = self.nonlin_fwd(h)
        h = (h-self.mu)/self.sigma
        return torch.matmul(h+self.bias, self.weight)

    def rot_inv(self, h):
        h = torch.matmul(h, self.weight_inv)-self.bias
        h = h*self.sigma + self.mu
        h = self.nonlin_inv(h)
        return h

    def forward(self, h, inverse=False):
        if inverse: return self.rot_inv(h)
        return self.rot_forward(h)

class RotationMatrix(RankRotationMatrix):
    def __init__(self, size, *args, **kwargs):
        """
        size: int
            the height and width of the rotation matrix
        rank: int
            the number of dims to slice the rotation matrix
        identity_init: bool
            if true, will initialize the rotation matrix to the identity
            matrix.
        bias: bool
            if true, will include a shifting term in the rotation matrix
        mu: float or FloatTensor (size,)
            Used to center each feature dim of the activations.
        sigma: float or FloatTensor (size,)
            Used to scale each feature dim of the activations.
        identity_rot: bool
            if true, will always reset the rotation matrix to the
            identity. Used for debugging.
        """
        kwargs["rank"] = size
        super().__init__(size=size, *args, **kwargs)

class FCARotationMatrix(torch.nn.Module):
    def __init__(self, 
            size,
            rank=None,
            identity_init=False,
            bias=False,
            mu=None,
            sigma=None,
            identity_rot=False,
            **kwargs):
        """
        size: int
            the height and width of the rotation matrix
        rank: int
            the rank of the rotation matrix
        identity_init: bool
            if true, will initialize the rotation matrix to the identity
            matrix.
        bias: bool
            if true, will include a shifting term in the rotation matrix
        mu: FloatTensor (size,)
            Used to center each feature dim of the activations.
        sigma: FloatTensor (size,)
            Used to scale each feature dim of the activations.
        identity_rot: bool
            if true, will always reset the rotation matrix to the
            identity. Used for debugging.
        """
        super().__init__()
        self.rot_module = FunctionalComponentAnalysis(
            size=size,
            means=mu,
            stds=sigma,
            init_rank=rank,
        )
        self.rot_module.set_fixed(True)

    @property
    def weight_inv(self):
        return self.weight.T

    @property
    def weight(self):
        if self.identity_rot:
            return torch.eye(
              self.size, device=self.rot_module.get_device(),).float()
        return self.rot_module.weight

    @property
    def size(self):
        return self.rot_module.size

    @property
    def shape(self):
        return self.rot_module.weight.shape

    def reset(self):
        self.rot_module.reset_fixed_weight()

    def get_condition(self, p=None):
        return torch.ones(1)

    def rot_forward(self, h):
        return self.rot_module(h)

    def rot_inv(self, h):
        return self.rot_module(h, inverse=True)

    def forward(self, h, inverse=False):
        return self.rot_module(h, inverse=inverse)

class PositiveSymmetricDefiniteMatrix(torch.nn.Module):
    def __init__(self, size, identity_init=False, *args, **kwargs):
        super().__init__()
        self.eps = 1e-1
        self.size = size
        self.core_mtx = torch.nn.Parameter(torch.randn(size,size)/math.sqrt(size))
        if identity_init:
            self.core_mtx.data = torch.eye(size)

    def get_psd_mtx(self):
        return torch.mm(self.core_mtx, self.core_mtx.T) +\
            self.eps*torch.eye(
                self.core_mtx.shape[-1],
                device=device_fxn(self.core_mtx.get_device()),
            )

    @property
    def weight(self):
        return self.get_psd_mtx()

    def inv(self):
        """
        Computes the inverse of a positive symmetric-definite matrix using Cholesky
        decomposition.
        """
        L = torch.linalg.cholesky(self.weight)
        return torch.cholesky_inverse(L)

class SymmetricDefiniteMatrix(PositiveSymmetricDefiniteMatrix):
    """
    Similar to a PSD matrix, but learns signs to multiply rows of the
    PSD matrix to allow it to be negative
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.signs = torch.nn.Parameter(0.01*torch.randn(self.size))

    @property
    def weight(self):
        psd = self.get_psd_mtx()
        signs = torch.nn.functional.tanh(self.signs)
        signs = signs + self.eps*torch.sign(signs) # offset to ensure nonzero
        return psd*signs

    def inv(self):
        """
        Computes the inverse of a positive symmetric-definite matrix using Cholesky
        decomposition.
        """
        return torch.linalg.inv(self.weight)


class PSDRotationMatrix(RotationMatrix):
    """
    Creates a Positive Symmetric Definite matrix
    """
    def __init__(self,
            size,
            identity_init=False,
            **kwargs):
        """
        size: int
            the height and width of the rotation matrix
        identity_init: bool
            if true, will initialize the rotation matrix to the identity
            matrix.
        bias: bool
            if true, will include a shifting term in the rotation matrix
        """
        super().__init__(size=size, **kwargs)
        self.rot_module = PositiveSymmetricDefiniteMatrix(
            size=size,
            identity_init=identity_init)

    def rot_inv(self, h):
        h = torch.matmul(h, self.rot_module.inv())
        h = h*self.sigma + self.mu
        return self.nonlin_inv(h)

class SDRotationMatrix(PSDRotationMatrix):
    """
    Creates a Symmetric Definite rotation matrix
    """
    def __init__(self,
            size,
            identity_init=False,
            **kwargs):
        """
        size: int
            the height and width of the rotation matrix
        identity_init: bool
            if true, will initialize the rotation matrix to the identity
            matrix.
        bias: bool
            if true, will include a shifting term in the rotation matrix
        """
        super().__init__(size=size, **kwargs)
        self.rot_module = SymmetricDefiniteMatrix(
            size=size,
            identity_init=identity_init)


class RelaxedRotationMatrix(RankRotationMatrix):
    """
    This module is similar to the RotationMatrix, it will however relax
    the orthonormal constraint on the rotation matrix, constraining it to
    only be an invertible matrix. This is done by using a diagonal
    matrix with non-zero values before the rotation matrix.
    """
    def __init__(self, eps=0.01, rot_first=False, half_neg=True, *args, **kwargs):
        """
        eps: float
            a small value to ensure no division by 0
        rot_first: bool
            debugging tool. if true, will apply the orthogonal rotation
            matrix before the scaling matrix. if true, should make this
            module equivalent to the target class.
        half_neg: bool
            if true, will initialize half of the diagonal elements to negative
            values
        """
        super().__init__(*args, **kwargs)
        self.diag = torch.nn.Parameter(torch.ones(self.size).float())
        if half_neg:
            perm = torch.randperm(self.diag.shape[0]).long()
            self.diag.data[perm] = -self.diag.data[perm]
        self.eps = eps
        self.rot_first = rot_first
        assert kwargs.get("nonlin_align_fn") in {None, "identity"}

    @property
    def scale_mtx(self):
        return torch.diag(self.diag+self.eps*torch.sign(self.diag))

    def get_condition(self, p="fro"):
        if self.rot_first:
            m = torch.matmul(self.weight, self.scale_mtx[:self.rank])
        else:
            m = torch.matmul(self.scale_mtx, self.weight)
        return torch.linalg.cond(m,p=p)

    def diag_forward(self, h):
        diag = (self.diag+self.eps*torch.sign(self.diag))[:h.shape[-1]]
        return torch.matmul(h,torch.diag(diag))
        #return torch.matmul(h,torch.diag(self.diag+self.eps))

    def diag_inv(self, h):
        diag = (self.diag+self.eps*torch.sign(self.diag))[:h.shape[-1]]
        return torch.matmul(h,torch.diag(1/diag))
        #return torch.matmul(h,torch.diag(1/(self.diag+self.eps)))

    def rot_first_forward(self, h, inverse=False):
        if inverse: return self.rot_inv(self.diag_inv(h))-self.bias
        return self.diag_forward(self.rot_forward(h+self.bias))

    def scale_first_forward(self, h, inverse=False):
        if inverse: return self.diag_inv(self.rot_inv(h))-self.bias
        return self.rot_forward(self.diag_forward(h+self.bias))

    def forward(self, h, inverse=False):
        if self.rot_first:
            return self.rot_first_forward(h, inverse=inverse)
        else:
            return self.scale_first_forward(h, inverse=inverse)

    def unit_forward(self, h, inverse=False):
        if inverse: return self.rot_inv(h)-self.bias
        return self.rot_forward(h+self.bias)

class ScaledRotationMatrix(RankRotationMatrix):
    """
    This module is similar to the RotationMatrix, it will however apply
    a scaling before the initial rotation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scalar = torch.nn.Parameter(torch.ones(1).float())

    def forward(self, h, inverse=False):
        if inverse: return self.rot_inv(h)/self.scalar-self.bias
        return self.rot_forward(self.scalar*(h+self.bias))

    def unit_forward(self, h, inverse=False):
        if inverse: return self.rot_inv(h)-self.bias
        return self.rot_forward(h+self.bias)

class Mask(torch.nn.Module):
    def __init__(self, size, n_units=None, *args, **kwargs):
        """
        A base class for masks that will be used to swap neurons in the
        intervention module.
        """
        super().__init__()
        self.temperature = None
        self.size = size
        if n_units is None:
            n_units = [1]
        elif type(n_units)==int:
            n_units = [n_units]
        elif type(n_units)==list:
            n_units = [int(units) for units in n_units]
        self.n_units = n_units
        masks = []
        start = 0
        for i,units in enumerate(self.n_units):
            mask = torch.zeros(self.size).float()
            mask[start:start+units] = 1
            start += units
            masks.append(mask)
            if start>=self.size:
                break
        if len(masks)<len(self.n_units):
            self.n_units = self.n_units[:len(masks)]
        self.n_units.append(max(self.size-start,0))
        assert sum(self.n_units)==self.size
        mask = torch.zeros(self.size).float()
        if start<self.size: mask[start:] = 1
        masks.append(mask)
        self.register_buffer("masks", torch.vstack(masks))

    def get_boundary_mask(self, subspace=0):
        if subspace>=len(self.masks):
            return self.masks[-1]
        return self.masks[subspace]

    def forward(self, target, source, subspace=0):
        """
        target: torch tensor (B,H)
            the main vector that will receive new neurons for
            causal interchange
        source: torch tensor (B,H)
            the vector that will give neurons to create a
            causal interchange in the other sequence
        subspace: int
            the subspace to use for the intervention.
            
        Returns:
            target: torch tensor (B,H)
                the vector that received new neurons for
                a causal interchange
        """
        mask = self.get_boundary_mask(subspace)
        masked_trg = (1-mask)[:target.shape[-1]]*target
        masked_src = mask[:source.shape[-1]]*source
        if masked_trg.shape[-1]<=masked_src.shape[-1]:
            swapped = masked_trg + masked_src[...,:masked_trg.shape[-1]]
        else:
            swapped = masked_trg
            swapped[...,:masked_src.shape[-1]] += masked_src
        return swapped

class FixedMask(Mask):
    def __init__(self,
            size=None,
            n_units=1,
            custom_mask=None,
            *args, **kwargs):
        """
        1s in the early dims, 0s in the later dims.

        size: int
            the number of the hidden state vector
        n_units: int or list of ints
            the number of units to swap. if a list is argued, uses it to
            determine the number of subspaces and their sizes. Otherwise
            defaults to 2 subspaces, one of size n_units and the other
            of size size-n_units.
        """
        super().__init__(size=size, n_units=n_units, *args, **kwargs)
        if custom_mask is not None and len(custom_mask)>0:
            self.size = custom_mask.shape[-1]
            masks = [custom_mask.float()]
            masks.append(1-custom_mask.float())
            self.n_units = [mask.sum().item() for mask in masks]
            self.masks[:] = torch.vstack(masks)

class ZeroMask(Mask):
    def __init__(self,
            size=None,
            n_units=1,
            learnable_addition=False,
            *args, **kwargs):
        """
        This mask will not swap between argued vectors, but will rather
        zero out the masked dims and add a learned vector in the place
        of the zeros if learnable_addition is true.

        size: int
            the number of the hidden state vector
        n_units: int
            the number of units to swap
        learnable_addition: bool
            if true, will learn a vector to add into the zeroed dims
        """
        super().__init__(size=size, n_units=n_units, *args, **kwargs)
        self.learnable_add = learnable_addition
        add_size = self.size-self.n_units[0]
        if self.learnable_add and add_size>0:
            self.add_vec = torch.nn.Parameter(0.01*torch.randn(add_size))

    def forward(self, target, source, subspace=0):
        """
        target: torch tensor (B,H)
            the main vector that will receive new neurons for
            causal interchange
        source: torch tensor (B,H)
            the vector that will give neurons to create a
            causal interchange in the other sequence
            
        Returns:
            target: torch tensor (B,H)
                the vector that received new neurons for
                a causal interchange
        """
        mask = self.get_boundary_mask(subspace=subspace)
        masked_trg = torch.zeros_like(target)
        masked_src = mask[:source.shape[-1]]*source
        if masked_trg.shape[-1]<=masked_src.shape[-1]:
            swapped = masked_trg + masked_src[...,:masked_trg.shape[-1]]
        else:
            swapped = masked_trg
            swapped[...,:masked_src.shape[-1]] += masked_src
        return swapped

class BoundlessMask(FixedMask):
    def __init__(self,
                 size,
                 temperature=0.01,
                 full_boundary=False,
                 split_start=True,
                 *args, **kwargs):
        """
        size: int
            the size of the hidden state vector
        temperature: float
        full_boundary: bool
            if true, will create an individual parameter for each
            neuron in the swap mask. Ideally, you will want to anneal the
            temperature progressively over the course of training
            and add an L1 loss term on the mask to the overall loss
            term. 
        split_start: bool
            if true, will start the boundaries so that half of the swap
            mask is all zeros and half is ones. Otherwise starts all ones
        """
        raise NotImplementedError(
            "BoundlessMask is not implemented yet. Use FixedMask or ZeroMask instead."
        )
        super().__init__()
        self.size = size
        self.temperature = temperature
        self.split_start = split_start
        self.register_buffer("indices", torch.arange(self.size).float())
        if full_boundary:
            self.boundaries = torch.nn.Parameter(torch.ones(size))
            if self.split_start: self.boundaries.data[:size//2] *= -1
            self.get_boundary_mask = self.full_boundary_mask
        else:
            self.boundaries = torch.nn.Parameter(torch.FloatTensor([-1,size+1]))
            if self.split_start:
                self.boundaries.data[0] = (size+1)//2
            self.get_boundary_mask = self.edges_boundary_mask

    @property
    def mask(self):
        return self.get_boundary_mask()

    @property
    def n_units(self):
        return (self.mask>0.5).float().sum()

    def full_boundary_mask(self):
        return torch.sigmoid(self.boundaries/self.temperature)
                                 
    def edges_boundary_mask(self):
        boundary_x, boundary_y = self.boundaries
        return (torch.sigmoid((self.indices - boundary_x) / self.temperature) * \
            torch.sigmoid((boundary_y - self.indices) / self.temperature))**2

class InterventionModule(torch.nn.Module):
    def __init__(self,
            sizes,
            mtx_types=["RotationMatrix", "RotationMatrix"],
            mtx_kwargs=None,
            mask_type="FixedMask", 
            mask_kwargs=None,
            fsr=False,
            n_units=None,
            n_subspaces=None,
            *args, **kwargs):
        """
        Args:
            sizes: list of ints
                the sizes of the distributed vectors for each matrix.
            mtx_types: list of str
            mtx_kwargs: list of dicts
                the key word arguments to pass to each matrix instantiation
            mask_type: str
                the type of mask for doing the substitution
            mask_kargs: dict
                keyword arguments for the mask object
            fsr: bool
                (functionally sufficient representations)
                if true, will zero out the orthogonal complement of the
                rotation matrix.
            n_units: int or list of ints
                Determines the number of units to swap in the intervention.
                If a list of ints, will use to determine the size of each
                subspace for n_subspaces-1. The last subspace always takes
                the size of the remaining neurons (0 is a possible size).
            n_subspaces: int or None
                the number of subspaces to include in the intervention
                mask. Defaults to 2. Assumes the same size for each
                subspace other than the last one. If n_units is a list of
                ints, will use the corresponding sizes for each subspace
                until the last subspace which will default to the remaining
                number of dimensions.
        """
        super().__init__()
        self.sizes = sizes
        if type(self.sizes)==int:
            self.sizes = [self.sizes]
        self.fsr = fsr
        if n_units is not None:
            if mask_kwargs is None:
                mask_kwargs = {}
            if n_units is not None and type(n_units)==int:
                n_units = [n_units for _ in range(n_subspaces)]
            mask_kwargs["n_units"] = n_units
        # TODO
        print("Using {} subspaces with sizes: {}".format(n_subspaces,n_units))

        # Rotation Matrices
        if type(mtx_types)==str:
            mtx_types = [mtx_types]
        if len(mtx_types)<len(self.sizes):
            mtx_types = [mtx_types[0] for _ in self.sizes]
        if mtx_kwargs is None:
            mtx_kwargs = [{} for _ in self.sizes]
        elif type(mtx_kwargs)==dict:
            mtx_kwargs = [mtx_kwargs for _ in self.sizes]
        self.do_reversal = False
        max_rank = min([size for size in self.sizes])
        default_rank = max_rank//2
        rank = None
        for i,d in enumerate(mtx_kwargs):
            d = copy.deepcopy(d)
            d["size"] = self.sizes[i]
            if mtx_types[i] in {"RankRotationMatrix", "FCARotationMatrix"}:
                assert n_subspaces is not None and n_subspaces==1
                d["rank"] = d.get("rank",
                    d.get("n_units",
                        mask_kwargs.get("n_units", [default_rank])[0]
                    )
                )
                if d["rank"] is None: d["rank"] = default_rank
                elif d["rank"]>max_rank:
                    print("Reducing Interchange Rank to", max_rank)
                    d["rank"] = max_rank
                rank = d["rank"] if rank is None else min(rank, d["rank"])
        for i,d in enumerate(mtx_kwargs):
            d = copy.deepcopy(d)
            d["size"] = self.sizes[i]
            d["rank"] = rank
            if self.fsr:
                mask_type = "ZeroMask"
            mtx_kwargs[i] = d
        self.rot_mtxs = torch.nn.ModuleList([
            globals()[t](**kwrg) for t,kwrg in zip(mtx_types, mtx_kwargs)
        ])

        # Swap Mask
        size = max(self.sizes)
        if mask_kwargs is None:
            mask_kwargs = dict()
        n_units = mask_kwargs.get("n_units", None)
        if rank is not None: n_units = [rank for _ in range(n_subspaces)]
        elif n_units is None: n_units = [default_rank for _ in range(n_subspaces)]
        elif type(n_units)==int: n_units = [n_units for _ in range(n_subspaces)]
        n_units = [min(units,max_rank) for units in n_units]
        mask_kwargs["n_units"] = n_units
        mask_kwargs["size"] = size
        self.swap_mask = globals()[mask_type](**mask_kwargs)

    def reset(self):
        for mtx in self.rot_mtxs:
            if hasattr(mtx, "reset"):
                mtx.reset()

    def forward(self,
            target,
            source,
            target_idx=0,
            source_idx=-1,
            varb_idx=None,
        ):
        """
        target: torch tensor (B,H)
            the vector that will receive new neurons
        source: torch tensor (B,H)
            the vector that will give neurons
        target_idx: int
            the index of the target rotation matrix
        source_idx: int
            the index of the source rotation matrix
        varb_idx: None or int or torch long tensor
            indicates the subspace to use for the
            intervention.

        Returns:
            new_h: torch tensor (B,H)
                the causally interchanged vector
        """
        if varb_idx is None: varb_idx = 0

        trg_mtx = self.rot_mtxs[target_idx]
        src_mtx = self.rot_mtxs[source_idx]
        if not self.fsr and type(trg_mtx)==FCARotationMatrix and type(src_mtx)==FCARotationMatrix:
            new_h = target + trg_mtx(src_mtx(source) - trg_mtx(target), inverse=True)
        else:
            rot_trg_h = trg_mtx(target)
            rot_src_h = src_mtx(source)

            rot_swapped = self.swap_mask(
                target=rot_trg_h,
                source=rot_src_h,
                subspace=varb_idx,
            )

            new_h = trg_mtx(rot_swapped, inverse=True)
        return new_h
    
    
def load_intrv_module(path, ret_config=False):
    """
    This is a helper function to load saved InterventionModule checkpoint.
    
    Args:
        path: str
        ret_config: bool
            if true, will return the configuration dict alongside the
            das module.
    Returns:
        intrv_modu: InterventionModule
            can access the rotation modules using `intrv_modu.rot_mtxs`
    """
    checkpt = torch.load(
        path,
        map_location=torch.device("cpu"),
        weights_only=False)
    intrv_modu = InterventionModule(**checkpt["config"])
    intrv_modu.load_state_dict(checkpt["state_dict"])
    if ret_config:
        return das_modu, checkpt["config"]
    return das_modu

if __name__=="__main__":
    seq_len = 10
    n_neurons=2
    proj_size = 100
    identity_init = False
    identity_rot = False

    x = torch.Tensor([[1,0,0]])
    y = torch.Tensor([[0,0,0]])
    size = 3
    #size = [x.shape[-1], y.shape[-1],]
    #size = [y.shape[-1], x.shape[-1], ]
    intr_modu = InterventionModule(
            sizes=size,
            mtx_types=["RankRotationMatrix", "RankRotationMatrix"],
            mtx_kwargs={
                "rank": n_neurons,
                "proj_size": proj_size,
                "identity_init": identity_init,
                "identity_rot": identity_rot,
            },
            mask_type="FixedMask", 
            mask_kwargs=None,)

    print("x:", x)
    with torch.no_grad():
        #rot_x = intr_modu(x,y)
        rot_x = intr_modu(x,x)
    print("rot_x", rot_x)
    print()
    print("x:", x)
    print("y:", y)
    with torch.no_grad():
        #rot_x = intr_modu(x,y)
        rot_x = intr_modu(x,y)
    print("rot_x:", rot_x)


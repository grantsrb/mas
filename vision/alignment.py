import copy
import math
import torch
import torch.nn.functional as F

from torch_modules import (
    IdentityModule, InvTanh, InvSigmoid,
    PositiveSymmetricDefiniteMatrix, SymmetricDefiniteMatrix,
    ReversibleResnet, InvertibleBatchNorm1d,
)

def device_fxn(device):
    if device<0: return "cpu"
    return device

class RotationMatrix(torch.nn.Module):
    def __init__(self,
            size,
            identity_init=False,
            bias=False,
            mu=0,
            sigma=1,
            identity_rot=False,
            orthogonal_map=None,
            nonlin_align_fn=None,
            normalize=False,
            batch_norm=False,
            post_batch_norm=False,
            dtype=None,
            **kwargs):
        """
        size: int
            the height and width of the rotation matrix
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
        normalize: bool
            if true, will learn normalization parameters for the
            activations before the rotation matrix.
        batch_norm: bool
            if true, will learn batch normalization parameters for the
            activations before the rotation matrix.
        post_batch_norm: bool
            if true, will apply a batch normalization after the rotation
            matrix.
        """
        super().__init__()
        self.identity_rot = identity_rot
        self.identity_init = identity_init
        if dtype is None:
            dtype = torch.float32
        self.dtype = dtype
        self.set_nonlin_fn(nonlin_align_fn)

        if normalize:
            self.mu = torch.nn.Parameter(torch.zeros(size, dtype=torch.float))
            self.sigma = torch.nn.Parameter(torch.ones(size, dtype=torch.float))
        else:
            if type(mu)==float or type(mu)==int:
                mu = torch.ones(1)*mu
            if type(sigma)==float or type(sigma)==int:
                sigma = torch.ones(1)*sigma
            self.register_buffer("mu", mu)
            self.register_buffer("sigma", sigma)
        if batch_norm:
            self.bn = InvertibleBatchNorm1d(size)
        else:
            self.bn = IdentityModule()
        if post_batch_norm:
            self.post_bn = InvertibleBatchNorm1d(size)
        else:
            self.post_bn = IdentityModule()


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
        self.set_dtype(self.dtype)

    def set_dtype(self, dtype):
        self.dtype = dtype
        for p in self.parameters():
            p.data = p.data.to(dtype)

    @property
    def weight(self):
        if self.identity_rot:
            self.rot_module.weight.data = torch.eye(
              self.size,
              dtype=self.rot_module.weight.data.dtype,
              device=device_fxn(self.rot_module.weight.get_device()),
            )
        return self.rot_module.weight

    @property
    def weight_inv(self):
        if self.identity_rot:
            self.rot_module.weight.data = torch.eye(
              self.size,
              dtype=self.rot_module.weight.data.dtype,
              device=device_fxn(self.rot_module.weight.get_device()),
            )
        return self.rot_module.weight.T

    @property
    def size(self):
        return self.rot_module.weight.shape[0]

    @property
    def shape(self):
        return self.weight.shape

    def set_normalization_params(self, mu=None, sigma=None):
        """
        Sets the normalization parameters for the rotation matrix.
        If mu or sigma are None, will not set them.

        Args:
            mu: float or FloatTensor (size,)
                Used to center each feature dim of the activations.
            sigma: float or FloatTensor (size,)
                Used to scale each feature dim of the activations.
        """
        if mu is not None:
            if type(mu)==float or type(mu)==int:
                mu = torch.tensor([mu]*self.size)
            elif type(mu)==list:
                mu = torch.tensor(mu)
            elif not isinstance(mu, torch.Tensor):
                raise ValueError("mu must be a float, list, or torch tensor")
            if hasattr(self, "mu"): delattr(self, "mu")
            self.register_buffer("mu", mu)
        if sigma is not None:
            if type(sigma)==float or type(sigma)==int:
                sigma = torch.tensor([sigma]*self.size)
            elif type(sigma)==list:
                sigma = torch.tensor(sigma)
            elif not isinstance(sigma, torch.Tensor):
                raise ValueError("sigma must be a float, list, or torch tensor")
            if hasattr(self, "sigma"):
                delattr(self, "sigma")
            self.register_buffer("sigma", sigma)

    def reset(self):
        pass

    def get_condition(self, p=None):
        return torch.linalg.cond(self.weight, p=p)

    def set_nonlin_fn(self, nonlin_align_fn):
        """
        Sets the non-linear function to apply to the input before the
        rotation matrix. Actually uses the inverse first!!
        """
        self.nonlin_fn = nonlin_align_fn
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
        h = (h-self.mu)/torch.abs(self.sigma+1e-8)
        h = self.bn(h)
        h = torch.matmul(h+self.bias, self.weight)
        h = self.post_bn(h)
        return h

    def rot_inv(self, h):
        h = self.post_bn.inv(h)
        h = torch.matmul(h, self.weight_inv)-self.bias
        h = self.bn.inv(h)
        h = h*torch.abs(self.sigma+1e-8) + self.mu
        h = self.nonlin_inv(h)
        return h

    def forward(self, h, inverse=False):
        if inverse: return self.rot_inv(h)
        return self.rot_forward(h)

class InvertedRotationMatrixWrapper(torch.nn.Module):
    """
    Inverts a rotation matrix
    """
    def __init__(self, rotation_matrix):
        """
        Args:
            rotation_matrix: RotationMatrix
                the rotation matrix to invert
        """
        super().__init__()
        self.rotation_matrix = rotation_matrix

    @property
    def weight(self):
        return self.rotation_matrix.weight_inv

    @property
    def weight_inv(self):
        return self.rotation_matrix.weight

    @property
    def size(self):
        return self.rot_module.weight.shape[0]

    @property
    def shape(self):
        return self.weight.shape

    def set_normalization_params(self, *args, **kwargs):
        return self.rotation_matrix.set_normalization_params(*args, **kwargs)

    def reset(self):
        pass

    def get_condition(self, p=None):
        return torch.linalg.cond(self.weight, p=p)

    def set_nonlin_fn(self, *args, **kwargs):
        return self.rotation_matrix.set_nonlin_fn(*args, **kwargs)

    def rot_forward(self, *args, **kwargs):
        return self.rotation_matrix.rot_inv(*args, **kwargs)

    def rot_inv(self, *args, **kwargs):
        return self.rotation_matrix.rot_forward(*args, **kwargs)

    def forward(self, h, inverse=False):
        return self.rotation_matrix(h, inverse=not inverse)

class FCARotationMatrix(torch.nn.Module):
    def __init__(self, 
            size,
            rank=None,
            identity_init=False,
            bias=False,
            mu=None,
            sigma=None,
            identity_rot=False,
            dtype=None,
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
        if dtype is None:
            dtype = torch.float32
        self.dtype = dtype
        self.set_dtype(self.dtype)
    
    def set_dtype(self, dtype):
        self.dtype = dtype
        for p in self.parameters():
            p.data = p.data.to(dtype)

    def set_normalization_params(self, mu=None, sigma=None):
        """
        Sets the normalization parameters for the rotation matrix.
        If mu or sigma are None, will not set them.
        """
        if mu is not None:
            if type(mu)==float or type(mu)==int:
                mu = torch.tensor([mu]*self.rot_module.size)
            elif type(mu)==list:
                mu = torch.tensor(mu)
            elif not isinstance(mu, torch.Tensor):
                raise ValueError("mu must be a float, list, or torch tensor")
            self.rot_module.set_means(means=mu)
        if sigma is not None:
            if type(sigma)==float or type(sigma)==int:
                sigma = torch.tensor([sigma]*self.rot_module.size)
            elif type(sigma)==list:
                sigma = torch.tensor(sigma)
            elif not isinstance(sigma, torch.Tensor):
                raise ValueError("sigma must be a float, list, or torch tensor")
            self.rot_module.set_stds(stds=sigma)

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

    @property
    def weight_inv(self):
        if self.identity_rot:
            return torch.eye(
              self.size,
              dtype=self.rot_module.weight.data.dtype,
              device=device_fxn(self.rot_module.weight.get_device()),
            )
        return self.rot_module.inv()

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

class LinearMatrix(RotationMatrix):
    """
    Creates a linear matrix
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
        self.rot_module = torch.nn.Linear(size, size, bias=False)
        if identity_init:
            self.rot_module.weight.data = torch.eye(
              size,
              dtype=self.rot_module.weight.data.dtype,
              device=device_fxn(self.rot_module.weight.get_device()),
            )
        else:
            self.rot_module.weight.data = torch.randn(size, size)/math.sqrt(size)

    @property
    def weight_inv(self):
        if self.identity_rot:
            return torch.eye(
              self.size,
              dtype=self.rot_module.weight.data.dtype,
              device=device_fxn(self.rot_module.weight.get_device()),
            )
        return torch.linalg.pinv(self.rot_module.weight)

class LowRankTransformation(torch.nn.Module):
    def __init__(self, original_dimensions=10, added_dimensions=10, transformation_type="zeros"):
        super().__init__()
        self.transformation_type = transformation_type
        self.og_transformation_type = transformation_type
        self.original_dimensions = original_dimensions
        if self.transformation_type == "dummy":
            added_dimensions = original_dimensions
        self.added_dimensions = added_dimensions
        D = original_dimensions+added_dimensions
        self.rotation_matrix = RotationMatrix(size=D, bias=False)
        for p in self.rotation_matrix.parameters():
            p.requires_grad = False
        self.ablation = False

    def set_ablation(self, ablation):
        if not ablation:
            self.transformation_type = self.og_transformation_type
        else:
            self.transformation_type = ablation

    def forward(self, x, inverse=False):
        if inverse:
            return self.rotation_matrix(x, inverse=True)[:, :self.original_dimensions]
        device = device_fxn(x.get_device())
        if self.transformation_type == "noise":
            x = torch.cat([x, torch.randn(x.shape[0], self.added_dimensions).to(device)], dim=-1)
        elif self.transformation_type == "dummy":
            x = torch.cat([x, x.data.clone().detach()], dim=-1)
        else: # zeros
            x = torch.cat([x, torch.zeros(x.shape[0], self.added_dimensions).to(device)], dim=-1)
        return self.rotation_matrix(x)

class RevResnetRotation(torch.nn.Module):
    """
    Creates a rotation module that uses reversible resnets to perform
    the 'rotation'
    """
    def __init__(self,
            size,
            n_layers=3,
            mu=0,
            sigma=1,
            nonlin_align_fn=None,
            normalize=False,
            batch_norm=False,
            dtype=None,
            **kwargs):
        """
        size: int
            the height and width of the rotation matrix
        n_layers: int
            the number of residual layers
        mu: float or FloatTensor (size,)
            Used to center each feature dim of the activations.
        sigma: float or FloatTensor (size,)
            Used to scale each feature dim of the activations.
        normalize: bool
            if true, will learn normalization parameters for the
            activations before the rotation matrix.
        batch_norm: bool
            if true, will learn batch normalization parameters for the
            activations before the rotation matrix.
        """
        super().__init__()
        if dtype is None:
            dtype = torch.float32
        self.dtype = dtype
        self.set_dtype(self.dtype)

        self.set_nonlin_fn(nonlin_align_fn)

        if normalize:
            self.mu = torch.nn.Parameter(torch.zeros(size, dtype=torch.float))
            self.sigma = torch.nn.Parameter(torch.ones(size, dtype=torch.float))
        else:
            if type(mu)==float or type(mu)==int:
                mu = torch.ones(1)*mu
            if type(sigma)==float or type(sigma)==int:
                sigma = torch.ones(1)*sigma
            self.register_buffer("mu", mu)
            self.register_buffer("sigma", sigma)
        if batch_norm:
            self.bn = InvertibleBatchNorm1d(size)
        else:
            self.bn = IdentityModule()

        self.rot_module = ReversibleResnet(
            size=size,
            n_layers=n_layers,
        )
        self.set_dtype(self.dtype)
    
    def set_dtype(self, dtype):
        self.dtype = dtype
        for p in self.parameters():
            p.data = p.data.to(dtype)

    @property
    def size(self):
        return self.rot_module.size

    def set_normalization_params(self, mu=None, sigma=None):
        """
        Sets the normalization parameters for the rotation matrix.
        If mu or sigma are None, will not set them.

        Args:
            mu: float or FloatTensor (size,)
                Used to center each feature dim of the activations.
            sigma: float or FloatTensor (size,)
                Used to scale each feature dim of the activations.
        """
        if mu is not None:
            if type(mu)==float or type(mu)==int:
                mu = torch.tensor([mu]*self.size)
            elif type(mu)==list:
                mu = torch.tensor(mu)
            elif not isinstance(mu, torch.Tensor):
                raise ValueError("mu must be a float, list, or torch tensor")
            if hasattr(self, "mu"): delattr(self, "mu")
            self.register_buffer("mu", mu)
        if sigma is not None:
            if type(sigma)==float or type(sigma)==int:
                sigma = torch.tensor([sigma]*self.size)
            elif type(sigma)==list:
                sigma = torch.tensor(sigma)
            elif not isinstance(sigma, torch.Tensor):
                raise ValueError("sigma must be a float, list, or torch tensor")
            if hasattr(self, "sigma"):
                delattr(self, "sigma")
            self.register_buffer("sigma", sigma)

    def reset(self):
        pass

    def get_condition(self, p=None):
        return 0

    def set_nonlin_fn(self, nonlin_align_fn):
        """
        Sets the non-linear function to apply to the input before the
        rotation matrix. Actually uses the inverse first!!
        """
        self.nonlin_fn = nonlin_align_fn
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
        h = (h-self.mu)/torch.abs(self.sigma+1e-5)
        h = self.bn(h)
        return self.rot_module(h)

    def rot_inv(self, h):
        h = self.rot_module.inv(h)
        h = self.bn.inv(h)
        h = h*torch.abs(self.sigma+1e-5) + self.mu
        h = self.nonlin_inv(h)
        return h

    def forward(self, h, inverse=False):
        if inverse: return self.rot_inv(h)
        return self.rot_forward(h)


class ScaledRotationMatrix(RotationMatrix):
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
    def __init__(self, size, subspace_sizes, dtype=None, *args, **kwargs):
        """
        A base class for masks that will be used to swap neurons in the
        alignment module.

        Args:
            size: int
                the dimension of the model
            subspace_sizes: int or list of ints
                the number of dimensions for each subspace of the mask.
                If a list is argued, uses it to determine the number of
                subspaces and their sizes. Otherwise, defaults to 2 subspaces,
                one of size subspace_sizes and the other
                of size size-subspace_sizes.
            dtype: torch.dtype
                the dtype to use for the mask. If None, will use float32
        """
        super().__init__()
        self.temperature = None
        self.size = size
        self.subspace_sizes = subspace_sizes
        if dtype is None:
            dtype = torch.float32
        self.dtype = dtype
        if type(self.subspace_sizes)==int:
            self.subspace_sizes = [self.subspace_sizes]
        elif type(self.subspace_sizes)==list:
            self.subspace_sizes = [int(units) for units in self.subspace_sizes]

        masks = []
        start = 0
        for si in range(len(self.subspace_sizes)):
            mask = torch.zeros(self.size).float()
            end = start+self.subspace_sizes[si]
            if end>self.size:
                self.subspace_sizes[si] = self.size-start
            mask[start:start+self.subspace_sizes[si]] = 1
            start += self.subspace_sizes[si]
            masks.append(mask)
            if start>=self.size:
                self.subspace_sizes = self.subspace_sizes[:si+1]
                break
        self.subspace_sizes.append(max(self.size-start,0))
        assert sum(self.subspace_sizes)==self.size
        mask = torch.zeros(self.size).float()
        if start<self.size:
            mask[start:] = 1
        masks.append(mask)
        self.register_buffer("masks", torch.vstack(masks).to(dtype))

    def set_dtype(self, dtype):
        self.dtype = dtype
        for p in self.parameters():
            p.data = p.data.to(dtype)
        if hasattr(self, "masks"):
            self.masks.data = self.masks.data.to(dtype)

    @property
    def n_subspaces(self):
        return len(self.masks)

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
            subspace_sizes=1,
            custom_mask=None,
            *args, **kwargs):
        """
        1s in the early dims, 0s in the later dims.

        size: int
            the number of the hidden state vector
        subspace_sizes: int or list of ints
            the number of units to swap. if a list is argued, uses it to
            determine the number of subspaces and their sizes. Otherwise
            defaults to 2 subspaces, one of size subspace_sizes and the other
            of size size-subspace_sizes.
        """
        super().__init__(size=size, subspace_sizes=subspace_sizes, *args, **kwargs)
        if custom_mask is not None and len(custom_mask)>0:
            self.size = custom_mask.shape[-1]
            masks = [custom_mask.float()]
            masks.append(1-custom_mask.float())
            self.subspace_sizes = [mask.sum().item() for mask in masks]
            self.masks[:] = torch.vstack(masks)

class ZeroMask(Mask):
    def __init__(self,
            size=None,
            subspace_sizes=1,
            learnable_addition=False,
            *args, **kwargs):
        """
        This mask will not swap between argued vectors, but will rather
        zero out the masked dims and add a learned vector in the place
        of the zeros if learnable_addition is true.

        size: int
            the number of the hidden state vector
        subspace_sizes: int
            the number of units to swap
        learnable_addition: bool
            if true, will learn a vector to add into the zeroed dims
        """
        super().__init__(size=size, subspace_sizes=subspace_sizes, *args, **kwargs)
        self.learnable_add = learnable_addition
        add_size = self.size-self.subspace_sizes[0]
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

class AlignmentModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def set_normalization_params(self, midx, mu=None, sigma=None):
        """
        Sets the normalization parameters for the rotation matrices.
        If mu or sigma are None, will not set them.

        Args:
            midx: int
                the index of the rotation matrix to set the parameters for
            mu: float or FloatTensor (size,)
                Used to center each feature dim of the activations.
            sigma: float or FloatTensor (size,)
                Used to scale each feature dim of the activations.
        """
        self.rot_mtxs[midx].set_normalization_params(mu=mu, sigma=sigma)

    def solve_and_set_rotation_matrix(self, midx, target_mtx, verbose=False):
        """
        Solves for the orthogonal parameter of the rotation matrix to
        be equal to the target matrix.

        Args:
            midx: int
                the index of the rotation matrix to set
            target_mtx: torch tensor (size,size)
                the target orthogonal matrix to set the orthogonalized
                rotation matrix to.
        """
        if isinstance(self.rot_mtxs[midx], FCARotationMatrix):
            raise NotImplementedError(
                "FCARotationMatrix does not support solving for orthogonal parameters yet."
            )
            self.rot_mtxs[midx].rot_module.set_initialization_vecs(
                target_mtx=target_mtx,)
        if verbose:
            print("Solving for rotation matrix initialization...")
        rot_module = self.rot_mtxs[midx].rot_module
        rot_module = solve_for_orthogonal_param(
            rot_module=rot_module,
            target_mtx=target_mtx,
            lr=1e-2,
            tol=1e-6,
            max_iter=1500,
            max_restarts=20,
            verbose=verbose,
        )
        self.rot_mtxs[midx].rot_module = rot_module

    def reset(self):
        for mtx in self.rot_mtxs:
            if hasattr(mtx, "reset"):
                mtx.reset()


class MASAlignment(AlignmentModule):
    def __init__(self,
            model_dims,
            mtx_type="orthogonal",
            subspace_sizes=None,
            mtx_kwargs=None,
            mask_type="FixedMask", 
            mask_kwargs=None,
            dtype=None,
            *args, **kwargs):
        """
        Args:
            model_dims: list of ints
                the sizes of the distributed vectors for each matrix. The
                length of this list defines the number of models to align.
            mtx_type: str
                options are: "orthogonal", "symmetric_definite",
                "positive_symmetric_definite", and "revresnet".
            mtx_kwargs: dict
                the key word arguments to pass to each matrix instantiation
            mask_type: str
                the type of mask for doing the substitution
            mask_kargs: dict
                keyword arguments for the mask object
            subspace_sizes: int or list of ints
                Determines the number of units to swap in the intervention.
                If a list of ints, will use to determine the size of each
                subspace for n_subspaces-1. The last subspace always takes
                the size of the remaining neurons (0 is a possible size).
            dtype: torch.dtype
                the dtype to use for the alignment module. If None, will use
                float32
        """
        super().__init__()
        self.model_dims = model_dims
        self.dtype = dtype
        if self.dtype is None:
            self.dtype = torch.float32
        self.n_models = len(self.model_dims)
        if mask_kwargs is None:
            mask_kwargs = {}
        if subspace_sizes is None:
            subspace_sizes = [int(max(self.model_dims))]
        if type(subspace_sizes)==int:
            subspace_sizes = [subspace_sizes]

        print(f"Using subspace sizes: {subspace_sizes}")

        # Make the swap masks
        mask_kwargs["subspace_sizes"] = subspace_sizes
        mask_kwargs["size"] = max(self.model_dims)
        mask_kwargs["dtype"] = self.dtype
        self.swap_mask = globals()[mask_type](**mask_kwargs)

        # Make the rotation matrices
        self.rot_mtxs = torch.nn.ModuleList([])
        if mtx_type=="orthogonal":
            mtx_class = RotationMatrix
        elif mtx_type=="linear":
            mtx_class = LinearMatrix
        elif mtx_type=="symmetric_definite":
            mtx_class = SDRotationMatrix
        elif mtx_type=="positive_symmetric_definite":
            mtx_class = PSDRotationMatrix
        elif mtx_type=="revresnet":
            mtx_class = RevResnetRotation
        else:
            raise ValueError(f"Invalid mtx_type: {mtx_type}")
        if mtx_kwargs is None:
            mtx_kwargs = {**kwargs}
        for si,size in enumerate(self.model_dims):
            if type(mtx_kwargs)==list:
                mkwargs = mtx_kwargs[si]
            else:
                mkwargs = mtx_kwargs
            mkwargs["dtype"] = self.dtype
            self.rot_mtxs.append(mtx_class(size=size, **mkwargs))

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
            indicates the subspace to use for the intervention.

        Returns:
            new_h: torch tensor (B,H)
                the causally interchanged vector
        """
        if varb_idx is None: varb_idx = 0
        og_dtype = target.dtype
        target = target.to(self.dtype)
        source = source.to(self.dtype)

        trg_mtx = self.rot_mtxs[target_idx]
        src_mtx = self.rot_mtxs[source_idx]

        rot_src_h = src_mtx(source)
        rot_trg_h = trg_mtx(target)

        rot_swapped = self.swap_mask(
            source=rot_src_h,
            target=rot_trg_h,
            subspace=varb_idx,
        )

        new_h = trg_mtx(rot_swapped, inverse=True)
        return new_h.to(og_dtype)
    
    
class ModelStitch(AlignmentModule):
    def __init__(self,
            model_dims,
            mtx_type="linear",
            mtx_kwargs=None,
            dtype=None,
            same_matrix=False,
            *args, **kwargs):
        """
        Args:
            model_dims: list of ints
                the sizes of the distributed vectors for each matrix. The
                length of this list defines the number of models to align.
            mtx_type: str
                options are: "orthogonal", "linear", "revresnet", "symmetric_definite",
                "positive_symmetric_definite".
            mtx_kwargs: dict
                the key word arguments to pass to each matrix instantiation
            same_matrix: bool
                if true, will use the same matrix for both models, but
                will invert it for the second model. Must only
                have two models and cannot use linear matrices.
            dtype: torch.dtype
                the dtype to use for the alignment module. If None, will use
                float32
        """
        super().__init__()
        self.model_dims = model_dims
        self.dtype = dtype
        self.same_matrix = same_matrix
        if self.dtype is None:
            self.dtype = torch.float32
        self.n_models = len(self.model_dims)

        # Make the swap masks
        self.swap_mask = FixedMask(
            subspace_sizes=[max(self.model_dims)],
            size=max(self.model_dims),
            dtype=self.dtype,
        )

        # Make the rotation matrices
        self.rot_mtxs = torch.nn.ModuleList([])
        if mtx_type=="orthogonal":
            mtx_class = RotationMatrix
        elif mtx_type=="linear":
            assert self.n_models==2, "Linear matrices must have two models."
            mtx_class = LinearMatrix
        elif mtx_type=="symmetric_definite":
            mtx_class = SDRotationMatrix
        elif mtx_type=="positive_symmetric_definite":
            mtx_class = PSDRotationMatrix
        elif mtx_type=="revresnet":
            mtx_class = RevResnetRotation
        else:
            raise ValueError(f"Invalid mtx_type: {mtx_type}")
        if mtx_kwargs is None:
            mtx_kwargs = {**kwargs}
        for si,size in enumerate(self.model_dims):
            if type(mtx_kwargs)==list:
                mkwargs = mtx_kwargs[si]
            else:
                mkwargs = mtx_kwargs
            mkwargs["dtype"] = self.dtype
            if self.same_matrix and si==1:
                self.rot_mtxs.append(
                    InvertedRotationMatrixWrapper(self.rot_mtxs[0]))
            else:
                self.rot_mtxs.append(mtx_class(size=size, **mkwargs))

    def forward(self,
            target,
            source,
            target_idx=0,
            source_idx=-1,
            *args, **kwargs
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

        Returns:
            new_h: torch tensor (B,H)
                the causally interchanged vector
        """
        og_dtype = source.dtype
        source = source.to(self.dtype)

        src_mtx = self.rot_mtxs[source_idx]

        if self.same_matrix and source_idx==1:
            rot_src_h = src_mtx(source, inverse=True)
        else:
            rot_src_h = src_mtx(source)

        return rot_src_h.to(og_dtype)
    
def solve_for_orthogonal_param(
        rot_module,
        target_mtx,
        lr=1e-2,
        tol=1e-6,
        max_iter=1500,
        max_restarts=20,
        verbose=False
):
    """
    Optimizes the underlying parameter of an object created from
    torch.nn.utils.parametrizations.orthogonal or an equivalent rot_module
    to have a weight matrix that is as close as possible to the target_matrix.

    Args:
        rot_module: ParametrizedLinear or PositiveSymmetricDefiniteMatrix (M,M)
            The orthogonalized matrix (e.g., from orthogonal parametrization).
            Use torch.nn.utils.parametrizations.orthogonal to create it.
            Can also be a PositiveSymmetricDefiniteMatrix or
            SymmetricDefiniteMatrix.
        U: torch.Tensor (M,M)
            The target orthogonal matrix.
        lr (float): Learning rate.
        tol (float): Loss tolerance for early stopping.
        max_iter (int): Maximum number of gradient steps.
        max_restarts (int): Maximum number of restarts if convergence is not reached.
        verbose (bool): If True, prints progress.

    Returns:
        rot_module: ParametrizedLinear (M,M)
            The optimized orthogonal matrix (same shape as U).
    """
    device = next(rot_module.parameters()).get_device()
    if device < 0: device = "cpu"
    target_mtx = target_mtx.to(device)
    mus_and_stds = dict()
    with torch.no_grad():
        for p in rot_module.parameters():
            mus_and_stds[p] = (p.mean().item(), p.std().item())

    best_params = None
    best_loss = float("inf")
    loss = torch.tensor(float("inf"), device=device)
    n_restarts = 0
    while n_restarts<=max_restarts and loss.item() > tol:
        n_restarts += 1
        loss = torch.tensor(float("inf"), device=device)
        optimizer = torch.optim.Adam(rot_module.parameters(), lr=lr)
        for i in range(max_iter):
            optimizer.zero_grad()
            Q = rot_module.weight
            loss = F.mse_loss(Q, target_mtx)
            if loss.item() < tol:
                if verbose:
                    print(f"Converged at iter {i}, loss={loss.item():.2e}")
                break
            loss.backward()
            optimizer.step()
            if verbose and i % 1000 == 0:
                print(f"Iter {i}, loss={loss.item():.4e}")
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_params = {name: param.data.clone() for name, param in rot_module.named_parameters()}

        if n_restarts <= max_restarts and loss.item() > tol:
            if verbose:
                print(f"Failed to converge with loss={loss.item():.2e}, attempting restart...")
            new_module = torch.nn.utils.parametrizations.orthogonal(
                torch.nn.Linear(
                    in_features=rot_module.weight.shape[0],
                    out_features=rot_module.weight.shape[1],
                    bias=False,
                )
            )
            new_module.to(device)
            if str(type(rot_module))==str(type(new_module)):
                rot_module = new_module
            else:
                with torch.no_grad():
                    for param in rot_module.parameters():
                        param.data = mus_and_stds[param][0] +\
                            mus_and_stds[param][1] * torch.randn_like(param.data)
    if verbose:
        print(f"Best loss: {best_loss:.4e} after {n_restarts} restarts")
    if best_params is not None:
        for name, param in rot_module.named_parameters():
            param.data = best_params[name]

    return rot_module

def load_alignment(alignment, path):
    try:
        sd = torch.load(path)
        alignment.load_state_dict(sd)
    except:
        alignment.rot_mtxs[0].mu = sd["rot_mtxs.0.mu"]
        alignment.rot_mtxs[1].mu = sd["rot_mtxs.1.mu"]
        alignment.load_state_dict(sd)
    return alignment

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


import math
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel

def device_fxn(device):
    if device<0: return "cpu"
    return device

class IdentityModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        return x

    def inv(self, x: Tensor, *args, **kwargs) -> Tensor:
        return x

class InvTanh(nn.Module):
    """
    Inverse tanh activation function.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Arguments:
            x: Tensor
        """
        return torch.atanh(x)

class InvSigmoid(nn.Module):
    """
    Inverse sigmoid activation function.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Arguments:
            x: Tensor
        """
        return torch.log(x/(1-x))

class PositiveSymmetricDefiniteMatrix(torch.nn.Module):
    def __init__(self, size, identity_init=False, *args, **kwargs):
        super().__init__()
        self.eps = 1e-1
        self.size = size
        self.core_mtx = torch.nn.Parameter(
            torch.randn(size,size)/math.sqrt(size))
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

class ReversibleBlock(nn.Module):
    def __init__(self, fn_F, fn_G):
        super().__init__()
        self.F = fn_F
        self.G = fn_G

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        y1 = x1 + self.F(x2)
        y2 = x2 + self.G(y1)
        return torch.cat([y1, y2], dim=1)

    def inv(self, y):
        y1, y2 = y.chunk(2, dim=1)
        x2 = y2 - self.G(y1)
        x1 = y1 - self.F(x2)
        return torch.cat([x1, x2], dim=1)

class SimpleFn(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
            nn.ReLU(inplace=True),
            nn.Linear(size, size),
            nn.BatchNorm1d(size)
        )

    def forward(self, x):
        return self.net(x)

class ReversibleResnet(nn.Module):
    def __init__(self, size, n_layers):
        super().__init__()
        self.size = size
        self.rev_blocks = nn.ModuleList()
        for _ in range(n_layers):
            F_block = SimpleFn(size//2)
            G_block = SimpleFn(size//2)
            self.rev_blocks.append(ReversibleBlock(F_block, G_block))

    def forward(self, x):
        fx = x
        for block in self.rev_blocks:
            fx = block(fx)
        return fx

    def inv(self, x):
        fx = x
        for block in reversed(self.rev_blocks):
            fx = block.inv(fx)
        return fx

class InvertibleBatchNorm1d(nn.Module):
    def __init__(self, size, momentum=0.999):
        """
        momentum - float (0 <= momentum < 1)
            this is the exponentially moving average factor for
            updating the running mean and std. 0 uses the mean and std
            of the current activations. 1 uses the mean and std of the
            activations at the last forward pass.
        """
        super().__init__()
        self.size = size
        self.momentum = momentum
        self.scale = torch.nn.Parameter(torch.ones(size))
        self.bias = torch.nn.Parameter(torch.zeros(size))
        self.running_mean = torch.nn.Parameter(torch.zeros(size))
        self.running_std = torch.nn.Parameter(torch.ones(size))
        self.last_mean = torch.zeros(size)
        self.last_std = torch.ones(size)

    def forward(self, x):
        if self.training:
            self.last_mean = x.mean(dim=0)
            self.last_std = x.std(dim=0)
            self.running_mean.data = self.momentum * self.running_mean.data\
                + (1 - self.momentum) * self.last_mean
            self.running_std.data = self.momentum * self.running_std.data\
                + (1 - self.momentum) * self.last_std
        else:
            self.last_mean = self.running_mean.data
            self.last_std = self.running_std.data
        x = (x - self.last_mean) / (self.last_std + 1e-5)
        x = x * (self.scale+torch.sign(self.scale)*1e-5) + self.bias
        return x

    def inv(self, x):
        x = (x - self.bias) / (self.scale+torch.sign(self.scale)*1e-5)
        x = x * (self.last_std+1e-5) + self.last_mean
        return x

class BackboneWithLinearHead(nn.Module):
    def __init__(
            self,
            backbone: AutoModel,
            hidden_dim: int,
            num_classes: int = 10,
            freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        if freeze_backbone:
            for name,p in self.backbone.named_parameters():
                if "pooler" in name:
                    continue
                p.requires_grad = False
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.classifier.weight.data = torch.randn_like(self.classifier.weight.data)\
            * (2/hidden_dim)**0.5

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values)
        outputs = outputs["pooler_output"]
        return self.classifier(outputs.flatten(1))

class ViTWithLinearHead(BackboneWithLinearHead):
    def forward(self, pixel_values):
        """
        pixel_values: (B, 3, H, W) already preprocessed by AutoImageProcessor
        Uses the CLS token representation for classification.
        """
        outputs = self.backbone(pixel_values=pixel_values)
        # For ViT models, outputs.last_hidden_state[:, 0] is the [CLS] token
        cls = outputs.last_hidden_state[:, 0]
        return self.classifier(cls)

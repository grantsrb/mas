import math
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

try:
    from .utils import (
        padmask2attnmask,
        get_causal_cross_mask,
        get_causal_mask,
        get_full_cross_mask,
        device_fxn,
    )
except:
    from utils import (
        padmask2attnmask,
        get_causal_cross_mask,
        get_causal_mask,
        get_full_cross_mask,
        device_fxn,
    )

try:
    from transformers.modeling_attn_mask_utils import (
        _prepare_4d_causal_attention_mask
    )
except:
    print("Failed to import causal attention mask util")

try:
    from transformers import LlamaModel
    from transformers.modeling_outputs import BaseModelOutputWithPast
except:
    class LlamaModel(nn.Module):
        """
        Hacky way to avoid needing import
        """
        def __init__(self,):
            super().__init__()

    class BaseModelOutputWithPast:
        def __init__(self, 
            last_hidden_state=None,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            loss=None,
            ):
            """
            Hacky way to avoid needing import
            """
            self.last_hidden_state = last_hidden_state
            self.past_key_values = past_key_values
            self.hidden_states = hidden_states
            self.loss = loss

from torch import Tensor
from typing import List, Optional, Tuple, Union

DEVICES = {
    -1: "cpu", **{i:i for i in range(10)}
}

class MLPLayer(nn.Module):
    def __init__(self,
            inpt_size,
            outp_size,
            noise=0,
            drop_p=0,
            bnorm=False,
            lnorm=False,
            scaleshift=False,
            actv_fxn="ReLU"):
        """
        Defines a single layer for an MLP

        Args:
            inpt_size: int
                the dimension of the inputs
            outp_size: int
                the dimension of the final output
            noise: float
                the std of added noise before the relue at each layer.
            drop_p: float
                the probability of dropping a node
            bnorm: bool
                if true, batchnorm is included before each relu layer
            lnorm: bool
                if true, layer norm is included before each relu layer
            scaleshift: bool
                if true, a ScaleShift layer is added after the activation
                function
        """
        super().__init__()

        block = [  ]
        if bnorm: block.append( nn.BatchNorm1d(inpt_size) )
        if lnorm: block.append( nn.LayerNorm(inpt_size) )
        if scaleshift: block.append( ScaleShift((inpt_size,)) )
        block.append( nn.Linear(inpt_size, outp_size) )
        if noise:
            block.append( GaussianNoise(noise) )
        if drop_p:
            block.append( nn.Dropout(drop_p) )
        if actv_fxn:
            block.append( getattr(nn, actv_fxn)() )
        self.net = nn.Sequential(*block)

    def forward(self, x, *args, **kwargs):
        return self.net(x)

class MLP(nn.Module):
    def __init__(
            self,
            inpt_size,
            outp_size,
            n_layers=2,
            h_sizes=None,
            noise=0,
            drop_p=0,
            bnorm=False,
            lnorm=False,
            scaleshift=False,
            actv_fxn="ReLU",
            ):
        """
        Defines a simple fully connected Sequential module

        Args:
            inpt_size: int
                the dimension of the inputs
            outp_size: int
                the dimension of the final output
            n_layers: int
                the number of layers for the fc net
            h_sizes: int or list or None
                the dimensionality of the hidden layers. if an int is
                argued, all hidden sizes take on this value. If a list
                is argued, the sizes correspond to the layers. If
                None is argued, the sizes will progressively grow to
                be the selected size.
            noise: float
                the std of added noise before the relue at each layer.
            drop_p: float
                the probability of dropping a node
            bnorm: bool
                if true, batchnorm is included before each relu layer
            lnorm: bool
                if true, layer norm is included before each relu layer
            scaleshift: bool
                if true, a ScaleShift layer is added after the activation
                function
        """
        super().__init__()
        if h_sizes is None:
            div = n_layers
            m = min(outp_size, inpt_size)
            if outp_size<inpt_size:
                diff = inpt_size-outp_size
                h_sizes = [
                    m + int(i/div*diff) for i in reversed(range(n_layers))
                ]
            else:
                diff = outp_size-inpt_size
                h_sizes = [
                    m + int(i/div*diff) for i in range(1,n_layers+1)
                ]
        elif type(h_sizes)==int:
            h_sizes = [ h_sizes ]

        size = h_sizes[0] if n_layers > 1 else outp_size
        block = [  ]
        kwargs = {
            "inpt_size": inpt_size,
            "outp_size": size,
            "noise": noise,
            "drop_p": drop_p,
            "actv_fxn": actv_fxn,
            "bnorm": False,
            "lnorm": False,
            "scaleshift": False,
        }
        for i in range(n_layers):
            if i+1 == n_layers:
                size = outp_size
                kwargs["noise"] = 0
                kwargs["drop_p"] = 0
                kwargs["actv_fxn"] = None
            else:
                size = h_sizes[i]
            kwargs["outp_size"] = size
            block.append(
                MLPLayer( **kwargs )
            )
            kwargs["inpt_size"] = size
            kwargs["bnorm"] = bnorm
            kwargs["lnorm"] = lnorm
            kwargs["scaleshift"] = scaleshift
        self.net = nn.Sequential(*block)

    def forward(self, x, *args, **kwargs):
        return self.net(x)

class CoreModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @property
    def is_cuda(self):
        try:
            return next(self.parameters()).is_cuda
        except:
            return False

    def get_device(self):
        return device_fxn(next(self.parameters()).get_device())

    def sample_with_temperature(self, logits, temperature):
        """
        Args:
            logits: torch float tensor (..., L)
            temperature: float or None
                a value to increase the sampling entropy. ignored if
                0 or None
        Returns:
            samples: torch Long Tensor (...,)
        """
        if not temperature: return torch.argmax(logits, dim=-1)
        ps = torch.nn.functional.softmax( logits/temperature, dim=-1 )
        og_shape = ps.shape
        ps = ps.reshape(-1, ps.shape[-1])
        samp = torch.multinomial(ps, num_samples=1)[...,0]
        return samp.reshape(og_shape[:-1])


class Flatten(nn.Module):
    """
    Reshapes the activations to be of shape (B,-1) where B
    is the batch size
    """
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)

class Reshape(nn.Module):
    """
    Reshapes the activations to be of shape (B, *shape) where B
    is the batch size.
    """
    def __init__(self, shape):
        """
        shape: tuple of ints
            do not include batch size
        """
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(len(x), *self.shape)

    def extra_repr(self):
        return "shape={}".format(self.shape)

class Transpose(nn.Module):
    """
    Transposes the argued axes. Do include the batch dimension in
    your argument
    """
    def __init__(self, axes, *args):
        """
        axes: tuple of ints
            do include the batch dimension
        """
        super().__init__()
        if type(axes)==int: axes = [axes] 
        else: axes = [*axes]

        if len(args) > 0:
            axes = axes + [*args]
        self.axes = axes
    
    def forward(self, x, *args, **kwargs):
        """
        x: torch tensor
        """
        return x.permute(self.axes)

class GaussianNoise(nn.Module):
    def __init__(self, std=0.1, trainable=False, adapt=False,
                                               momentum=.95):
        """
        std - float
            the standard deviation of the noise to add to the layer.
            if adapt is true, this is used as the proportional value to
            set the std to based of the std of the activations.
            gauss_std = activ_std*std
        trainable - bool
            If trainable is set to True, then the std is turned into
            a learned parameter. Cannot be set to true if adapt is True
        adapt - bool
            adapts the gaussian std to a proportion of the
            std of the received activations. Cannot be set to True if
            trainable is True
        momentum - float (0 <= momentum < 1)
            this is the exponentially moving average factor for
            updating the activ_std. 0 uses the std of the current
            activations.
        """
        super(GaussianNoise, self).__init__()
        self.trainable = trainable
        self.adapt = adapt
        assert not (self.trainable and self.adapt)
        self.std = std
        self.sigma = nn.Parameter(torch.ones(1)*std,
                            requires_grad=trainable)
        self.running_std = 1
        self.momentum = momentum if adapt else None

    def forward(self, x):
        if not self.training or self.std == 0:
            return x
        if self.adapt:
            xstd = x.std().item()
            self.running_std = self.momentum*self.running_std +\
                                          (1-self.momentum)*xstd
            self.sigma.data[0] = self.std*self.running_std
        noise = self.sigma*torch.randn_like(x)
        return x + noise

    def extra_repr(self):
        s = 'std={}, trainable={}, adapt={}, momentum={}'
        return s.format(self.std, self.trainable,
                        self.adapt, self.momentum)

class PositionalEncoding(nn.Module):
    """
    Taken from pytorch tutorial. A simple positonal encoding taken from
    vaswani et al.
    """
    def __init__(self, d_model, dropout= 0.1, max_len= 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        Returns:
            enc: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:,:x.size(1)]
        return self.dropout(x)

class ScaleShift(nn.Module):
    """
    Scales and shifts the activations by a learnable amount.
    """
    def __init__(self, shape, scale=True, shift=True):
        """
        shape: tuple (depth, height, width) or (length,)
            shape of the incoming activations discluding the
            batch dimension
        scale: bool
            include multiplicative parameter
        shift: bool
            include summing parameter
        """
        super(ScaleShift, self).__init__()
        self.shape = shape
        self.scale = scale
        self.shift = shift
        self.scale_param = nn.Parameter(torch.ones(shape).float(),
                                              requires_grad=scale)
        self.shift_param= nn.Parameter(torch.zeros(shape).float(),
                                              requires_grad=shift)
    def forward(self, x):
        return x*self.scale_param + self.shift_param

    def extra_repr(self):
        s = 'shape={}, scale={}, shift={}'
        return s.format(self.shape, self.scale, self.shift)

class NullOp(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

class ContainedLSTM(nn.Module):
    """
    Contained lstms handle all recurrent vectors for you. You simply
    pass an input sequence to the forward function with the number of
    outputs you would like. It returns the outputs as a tensor (B,N,H).
    It also resets the h and c vectors at the beginning of each forward
    pass.
    """
    def __init__(self, inpt_size, h_size, lnorm=True, *args, **kwargs):
        super().__init__()
        self.inpt_size = inpt_size
        self.h_size = h_size
        self.lstm = nn.LSTMCell(self.inpt_size, self.h_size)
        self.lnorm = lnorm
        if self.lnorm:
            self.lnorm_h = nn.LayerNorm(self.h_size)
        self.register_buffer('h', torch.zeros(1,self.h_size))
        self.register_buffer('c', torch.zeros(1,self.h_size))

    def forward(self, x, mask=None):
        """
        Args:
            x: torch tensor (B, S, I)
            mask: torch bool tensor (B,S)
                a boolean tensor where true denotes that the end of the
                sequence has been reached. These inputs are not
                included.
        Returns:
            fx: torch tensor (B, H)
        """
        h = self.h.repeat(len(x), 1)
        c = self.c.repeat(len(x), 1)
        output = torch.zeros_like(h)
        for i in range(x.shape[1]):
            if self.lnorm:
                h = self.lnorm_h(h)
            h, c = self.lstm(x[:,i], (h,c))
            if mask is not None:
                output[~mask[:,i]] = h[~mask[:,i]]
            else: output = h
        return output

class GenerativeLSTM(nn.Module):
    """
    This module handles all recurrent vectors for you. You simply
    pass the input in to the forward function with the number of
    outputs you would like. It returns the outputs as a tensor (B,N,H).
    It also resets the h and c vectors at the beginning of each forward
    pass.
    """
    def __init__(self, inpt_size, h_size, lnorm=True, *args, **kwargs):
        super().__init__()
        self.inpt_size = inpt_size
        self.h_size = h_size
        self.lstm = nn.LSTMCell(self.inpt_size, self.h_size)
        self.lnorm = lnorm
        if self.lnorm:
            self.lnorm_h = nn.LayerNorm(self.h_size)
            self.lnorm_c = nn.LayerNorm(self.h_size)
        self.register_buffer('h', torch.zeros(1,self.h_size))
        self.register_buffer('c', torch.zeros(1,self.h_size))

    def forward(self, x, n):
        """
        Args:
            x: torch tensor (B, I)
            n: int
                the number of recurrent loops
        Returns:
            fx: torch tensor (B, N, H)
        """
        h = self.h.repeat(len(x), 1)
        c = self.c.repeat(len(x), 1)
        outpts = []
        for _ in range(n):
            if self.lnorm:
                h,c = self.lnorm_h(h), self.lnorm_c(c)
            h, c = self.lstm(x, (h,c))
            outpts.append(h)
        return torch.stack(outpts, dim=1)

class SplitPathwayRecurrence(nn.Module):
    """
    This module uses k parallel recurrent cells for n layers in place
    of a single recurrent model for n layers.

              rnn_0 -> linear_00 -> ... -> linear_0n 
            /                                        \
    inpt -> - rnn_1 -> linear_10 -> ... -> linear_1n  - cat -> out
            ..........................................
            \                                        /
              rnn_k -> linear_k0 -> ... -> linear_kn
    """
    def __init__(self,
                 inpt_size,
                 d_model,
                 rnn_type="RNNCell",
                 n_linears=0,
                 n_paths=2,
                 *args, **kwargs):
        """
        inpt_size: int
        d_model: int
        rnn_type: str
            the type of rnn cell to use
        n_linears: int
            the number of subsequent linear layers per pathway (n in the
            diagram)
        n_paths: int
            the number of parallel pathways (k in the diagram)
        """
        super().__init__()
        self.inpt_size = inpt_size
        self.d_model = d_model
        self.n_linears = n_linears
        self.n_paths = n_paths
        assert self.d_model % self.n_paths == 0
        self.h_size = d_model//self.n_paths

        if hasattr(torch.nn, rnn_type):
            rnn_type = getattr(torch.nn, rnn_type)
        else:
            rnn_type = globals()[rnn_type]

        self.inpt_proj = nn.Linear(inpt_size, d_model)
        self.h_proj = nn.Linear(d_model, d_model)
        self.rnns = nn.ModuleList([])
        self.linears = nn.ModuleList([])
        rng = range(self.n_linears)
        for k in range(self.n_paths):
            self.rnns.append(rnn_type(self.h_size,self.h_size))
            linears = [ nn.Linear(self.h_size,self.h_size) for _ in rng ]
            self.linears.append( nn.ModuleList(linears) )

    def run_path(self, path_idx, x, h, *args, **kwargs):
        """
        path_idx: int
            the index of the rnn and weights
        x: torch tensor (B, H)
            batch by h_size dimension
        h: torch tensor (B, H)
            the recurrent state
        """
        new_h = self.rnns[path_idx](x,h)
        if len(self.linears[path_idx])>0: # assume new_h is a tensor
            for linear in self.linears[path_idx]:
                new_h = linear(new_h)
        return new_h

    def forward(self, x, hs, *args, **kwargs):
        """
        x: torch tensor (B, I)
            batch by inpt dimension
        hs: tuple or list of tensors
            there should be n_paths hs. each h is a tensor
            of shape (B, H).
        """
        B,I = x.shape
        fx = self.inpt_proj(x).reshape(B, self.n_paths, self.h_size)
        if type(hs)==type(torch.ones(0)):
            hs = self.h_proj(hs).reshape(B,self.n_paths,self.h_size)
        rets = [
          self.run_path(i, fx[:,i], hs[:,i]) for i in range(self.n_paths)
        ]
        if type(rets[0])==tuple:
            raise NotImplemented
            cats = [[] for _ in range(len(rets[0]))]
            for r in rets:
                for i in range(len(r)):
                    cats[i].append(r[i])
            rets = tuple([torch.cat(c,dim=-1) for c in cats])
        else:
            rets = torch.cat(rets, dim=-1)
        return rets

class SplitRNN(SplitPathwayRecurrence):
    pass

class SplitGRU(SplitPathwayRecurrence):
    def __init__(self,
                inpt_size,
                d_model,
                rnn_type="GRUCell",
                *args, **kwargs):
        super().__init__(inpt_size, d_model, rnn_type=rnn_type, **kwargs)

class SplitLSTM(SplitPathwayRecurrence):
    def __init__(self,
                inpt_size,
                d_model,
                rnn_type="LSTMCell",
                *args, **kwargs):
        super().__init__(inpt_size, d_model, rnn_type=rnn_type, **kwargs)

class CrossAttention(nn.Module):
    """
    Builds off the pytorch multihead attention module to combine multiple
    different modalities symetrically into a single multi-head attention.
    """
    def __init__(self,
                 embed_dim,
                 num_heads,
                 num_modes=2,
                 dropout=0.,
                 bias=True,
                 add_bias_kv=False,
                 add_zero_attn=True,
                 kdim=None,
                 vdim=None,
                 batch_first=True,
                 device=None,
                 dtype=None,
                 *args, **kwargs) -> None:
        """
        Args:
            embed_dim: int
                Total dimension of the model.
            num_heads: int
                Number of parallel attention heads. Note that embed_dim
                will be split across num_heads (i.e. each head will have
                dimension embed_dim // num_heads).
            num_modes: int
                the number of modalities to be combined into the
                self-attention.
            dropout: float
                Dropout probability on attn_output_weights. Default:
                0.0 (no dropout).
            bias: bool
                If specified, adds bias to input / output projection
                layers. Default: True.
            add_bias_kv: bool
                If specified, adds bias to the key and value sequences
                at dim=0. Default: False.
            add_zero_attn: bool
                If specified, adds a new batch of zeros to the key and
                value sequences at dim=1. Default: False.
            kdim: int
                Total number of features for keys. Default: None
                (uses kdim=embed_dim).
            vdim: int
                Total number of features for values. Default: None
                (uses vdim=embed_dim).
            batch_first: bool
                If True, then the input and output tensors are provided
                as (batch, seq, feature). Default: False (seq, batch,
                feature).
            device: int or str
            dtype: str
        """
        super().__init__(*args, **kwargs)
        self.mh_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )
        self.num_modes = num_modes
        self.mode_encodings = nn.Embedding(self.num_modes, embed_dim)
        torch.nn.init.kaiming_uniform_(
            self.mode_encodings.weight,
            mode='fan_in',
            nonlinearity='leaky_relu'
        )

    def forward(self,
                queries,
                keys,
                values,
                key_padding_masks=None,
                step_masks=None,
                need_weights=True,
                is_causal=True,
                average_attn_weights=True,
                tforce=True,
                *args, **kwargs):
        """
        Args:
            queries: (List of Tensors)
                One entry for each modality.
                Query embeddings of shape (L,Eq)(L,Eq​) for unbatched
                input, (L,N,Eq)(L,N,Eq​) when batch_first=False or
                (N,L,Eq)(N,L,Eq​) when batch_first=True, where LL is the
                target sequence length, NN is the batch size, and EqEq​
                is the query embedding dimension embed_dim. Queries are
                compared against key-value pairs to produce the output.
                See “Attention Is All You Need” for more details.
            keys: (List of Tensors)
                One entry for each modality.
                Key embeddings of shape (S,Ek)(S,Ek​) for unbatched input,
                (S,N,Ek)(S,N,Ek​) when batch_first=False or
                (N,S,Ek)(N,S,Ek​) when batch_first=True, where SS is the
                source sequence length, NN is the batch size, and EkEk​
                is the key embedding dimension kdim. See “Attention Is
                All You Need” for more details.
            values: (List of Tensors)
                One entry for each modality.
                Value embeddings of shape (S,Ev)(S,Ev​) for unbatched
                input, (S,N,Ev)(S,N,Ev​) when batch_first=False or
                (N,S,Ev)(N,S,Ev​) when batch_first=True, where SS is the
                source sequence length, NN is the batch size, and EvEv​
                is the value embedding dimension vdim. See “Attention Is
                All You Need” for more details.
            key_padding_masks: (Optional[List of Tensors])
                One entry for each modality.
                If specified, a mask of shape (N,S)(N,S) indicating
                which elements within key to ignore for the purpose of
                attention (i.e. treat as “padding”). For unbatched query,
                shape should be (S)(S). Binary and float masks are
                supported. For a binary mask, a True value indicates that
                the corresponding key value will be ignored for the
                purpose of attention. For a float mask, it will be
                directly added to the corresponding key value.
            need_weights: (bool)
                If specified, returns attn_output_weights in addition
                to attn_outputs. Set need_weights=False to use the
                optimized scaled_dot_product_attention and achieve the
                best performance for MHA. Default: True.
            step_masks: (Optional[List of LongTensors])
                One entry for each modality. A list of 2D masks denoting
                step of the information relative to the other modalities.
                This allows you to use causal masking based on the step
                of an environment instead of the step of each embedding,
                preventing attention to positions that are at a future
                state of the environment. Must be of shape [(NN,S1), ...,
                (NN,Sk)], where NN is the batch size, S1 is the sequence
                length of the first modality and Sk is the sequence
                length of the kth modality. Only Long type masks
                are supported.
            average_attn_weights: (bool)
                If true, indicates that the returned attn_weights should
                be averaged across heads. Otherwise, attn_weights are
                provided separately per head. Note that this flag only
                has an effect when need_weights=True. Default: True
                (i.e. average weights across heads)
            is_causal: (bool)
                If true, applies a causal mask within each modality.
            tforce: bool
                If true, will use all queries. If false,
                will use only the last embedding of each modality as
                the queries (saving computation).
        """
        cross_mask = get_full_cross_mask(step_masks) # (B,S1+S2,S1+S2)
        pad_mask = torch.cat(key_padding_masks, dim=-1)# (B,S1+S2)
        pad_mask = padmask2attnmask(pad_mask) # (B,S1+S2,S1+S2)
        attn_mask = ~(cross_mask|pad_mask)

        if not tforce:
            # only take the latest queries and the corresponding masks
            running_sum = 0
            idxs = []
            # TODO: Need to transpose to index into correct axis
            for q in queries:
                running_sum += q.shape[1]
                idxs.append(running_sum-1)
            idxs = torch.LongTensor(idxs,device=self.get_device())
            cross_mask = cross_mask[idxs]
            queries = [ q[:,-1:] for q in queries ]

        queries = [
          q + self.mode_encodings.weight[i] for i,q in enumerate(queries)
        ]
        keys = [
          k + self.mode_encodings.weight[i] for i,k in enumerate(keys)
        ]
        values = [
          v + self.mode_encodings.weight[i] for i,v in enumerate(values)
        ]
        query = torch.cat(queries, dim=1)
        key =   torch.cat(keys, dim=1)
        value = torch.cat(values, dim=1)
        attn_out, attn_weights = self.mh_attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=pad_mask,
            attn_mask=cross_mask,
        )
        if need_weights:
            return attn_out, attn_weights
        return attn_out

class CrossAttentionPrep(nn.Module):
    """
    This module preps the incoming sequence to be used by a standard
    transformer by applying modality specific embeddings and building
    the cross-modal attention mask.
    """
    def __init__(self,
                 embed_dim,
                 num_modes=2,
                 dtype=None,
                 *args, **kwargs) -> None:
        """
        Args:
            embed_dim: int
                Total dimension of the model.
            num_modes: int
                the number of modalities to be combined into the
                self-attention.
            device: int or str
            dtype: str
        """
        super().__init__(*args, **kwargs)
        self.num_modes = num_modes
        self.mode_encodings = nn.Parameter(
            0.01*torch.randn(self.num_modes, embed_dim)
        )
        torch.nn.init.kaiming_uniform_(
            self.mode_encodings,
            mode='fan_in',
            nonlinearity='leaky_relu'
        )

    def forward(self,
                inpt_list,
                pad_masks=None,
                step_masks=None,
                n_steps=None,
                *args, **kwargs):
        """
        Args:
            inpt_list: list of torch FloatTensors [(B,S1,E), (B,S2,E)]
                a list of the embedding/latent vectors.
            pad_masks: list of torch BoolTensors [(B,S1), (B,S2)]
                A list of the pad masks. A True value indicates that
                the corresponding key value will be ignored for the
                purpose of attention. True means padding.
            step_masks: list of torch LongTensors [(B,S1), (B,S2)]
                One entry for each modality. Currently only 2 modalities
                are supported. It should be a list of masks denoting
                step of the information relative to the other modalities.
                This allows you to use causal masking based on the step
                of an environment instead of the step of each embedding,
                preventing attention to positions that are at a future
                state of the environment. B is the batch size, S1 is
                the sequence length of the first modality and S2 is the
                sequence length of the 2nd modality. Only Long type masks
                are supported.
        Returns:
            inpts:
                the concatenated inputs
            cross_mask: torch bool tensor (B,S1+S2,S1+S2)
                true means unattended indices
            pad_mask: torch bool tensor (B,S1+S2)
                true means unattended, padding indices
        """
        ## TODO: QUESTION DECISION FOR MODE ENCODINGS. Cannot use freedom
        ## forward if using mode encodings with current setup.
        #inpt_list = [
        #  inpt+self.mode_encodings[i] for i,inpt in enumerate(inpt_list)
        #]

        if inpt_list is not None:
            inpts = torch.cat(inpt_list, dim=1) # (B,S1+S2,E)
        else: inpts = None

        # cross mask assumes true is padding
        # TODO: Need to be careful about step mask because we don't know
        # when the environment steps when we're predicting a multi-step
        # chunk of text, and this influences whether the text sees
        # the appropriate vision inputs.
        if n_steps and n_steps>0:
            step_masks[1] = torch.nn.functional.pad(
                step_masks[1],
                (0,n_steps, 0,n_steps),
                value=torch.max(step_masks[1])+1
            )
        # returned cross mask has true as padded, non-atteneded idxs
        cross_mask = get_full_cross_mask(step_masks) # (B,S1+S2,S1+S2)
        pad_mask = torch.cat(pad_masks, dim=-1)# (B,S1+S2)
        #pad_mask = padmask2attnmask(pad_mask) # (B,S1+S2,S1+S2)
        #attn_mask = cross_mask|pad_mask
        return inpts, cross_mask, pad_mask

class FlexibleLlamaModel(LlamaModel):
    """
    Overrides the forward function for more flexible attention inputs.
    Allows the model to use non-causal attention if desired. Does not
    allow flash attention when using non-cauasal attention.
    """
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if output_attentions is None:
            output_attentions = self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None\
                else self.config.output_hidden_states
        )
        if use_cache is None:
            use_cache = self.config.use_cache

        if return_dict is None:
            return_dict = self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds "
                "at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds"
            )

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            if input_ids is not None: device = input_ids.device
            else: device = inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        flash = getattr(self.config, "_flash_attn_2_enabled", False)
        if len(attention_mask.shape)==2 and flash:
            # 2d mask is passed through the layers
            if not (attention_mask is not None and 0 in attention_mask):
                attention_mask = None
        elif len(attention_mask.shape)==2:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length
            )
        elif len(attention_mask.shape)==3:
            #n_heads = self.config.num_attention_heads
            #attention_mask = attention_mask[:,None].repeat(1,n_heads,1,1)
            attention_mask = attention_mask[:,None]

        # embed positions
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache: use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if past_key_values is None:
                past_key_value = None
            else:
                past_key_value = past_key_values[idx]

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (
                    layer_outputs[2 if output_attentions else 1],
                )

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            el = [
              hidden_states,next_cache,all_hidden_states,all_self_attns
            ]
            return tuple( v for v in el if v is not None )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class RotaryEmbedding(nn.Module):
    """
    Code for this module was slightly modified from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/rope/__init__.py
    """
    def __init__(self, d: int, base: int=10000):
        """
        Args:
            d: int
                the dimensionality of the projected queries or keys.
                Must be divisible by 2.
            base: int
        """
        super().__init__()
        self.d = d
        assert self.d%2==0
        self.base = base
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, s: int, device=-1):
        """
        Args:
            x: torch Tensor (B,NHeads,Seq,D)
            offset: int
                an value to effectively increase the position of x in
                the sequence. This is helpful for using past_key_values
        Returns:
            None
        """
        if self.cos_cached is not None and s <= self.cos_cached.shape[0]:
            self.cos_cached = self.cos_cached.to(device)
            self.sin_cached = self.sin_cached.to(device)
            return

        denom = self.base ** (torch.arange(0, self.d, 2).float() / self.d)
        theta = 1. / denom.to(device)

        seq_idx = torch.arange(s, device=device).float()

        idx_theta = torch.einsum("n,d->nd", seq_idx, theta)

        idx_theta = torch.cat([idx_theta,idx_theta],dim=1)

        self.cos_cached = idx_theta.cos()
        self.sin_cached = idx_theta.sin()

    def neg_half(self, x:torch.Tensor):
        """
        Args:
            x: torch Tensor (B,NHeads,Seq,D)
        """
        d_2 = self.d//2
        return torch.cat([-x[...,d_2:], x[...,:d_2]], dim=-1)

    def forward(self, x: torch.tensor, offset=0, position_ids=None):
        """
        Args:
            x: torch Tensor (B,NHeads,Seq,D)
            offset: int
                the amount to offset the positional encodings by
            position_ids: None or torch LongTensor (B,Seq)
                optionally specify the positional indices for each token
                respectively. The argued offset will be added to the
                position_ids
        Returns:
            x_rope: torch Tensor (B,NHeads,Seq,D)
        """
        B,N,S,D = x.shape
        device = device_fxn(x.get_device())
        s = S+offset if position_ids is None else torch.max(position_ids)+1
        self._build_cache(s=s, device=device)
        x_pass = None
        x_rope = x
        if self.d<x.shape[-1]:
            x_rope, x_pass = x[..., :self.d], x[...,self.d:]
        neg_half_x = self.neg_half(x_rope)
        # If you got an error here, you probably need a different sized
        # rotary dimension. Try arguing a power of 2 for d_model and
        # use one or an even number of attention heads.
        if position_ids is None:
            x_rope = (x_rope*self.cos_cached[offset:S+offset])
            x_rope = x_rope + (neg_half_x*self.sin_cached[offset:S+offset])
        else:
            x_rope = (x_rope*self.cos_cached[position_ids])
            x_rope = x_rope + (neg_half_x*self.sin_cached[position_ids])
        if x_pass is not None:
            x_rope = torch.cat([x_rope, x_pass],dim=-1)
        return x_rope

class MultiHeadAttention(nn.Module):
    def __init__(self, 
            d_model,
            nhead,
            kdim=None,
            vdim=None,
            dropout=0.1,
            bias=True,
            batch_first=True,
            *args, **kwargs):
        """
        A multiheaded self-attention module that uses pytorch's
        F.scaled_dot_product_attention function.

        Args:
            d_model: int
            nhead: int
            dropout: float
            bias: bool
            batch_first: bool
            kdim: int
                the incoming dimensionality of the keys
            vdim: int
                the incoming dimensionality of the values
        """
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first
        self.dropout = dropout

        self.proj_dim = d_model//nhead

        if kdim is None:
            kdim = d_model
        self.kdim = kdim
        if vdim is None:
            vdim = d_model
        self.vdim = vdim

        if not batch_first:
            raise NotImplemented

        self.q_proj = nn.Linear(
            self.d_model, self.proj_dim*nhead, bias=bias)
        self.k_proj = nn.Linear(
            self.kdim, self.proj_dim*nhead, bias=bias)
        self.v_proj = nn.Linear(
            self.vdim, self.proj_dim*nhead, bias=bias)
        self.out_proj = nn.Linear(
            self.proj_dim*nhead, self.d_model, bias=bias)
        self.init_weights()

        self.sdp_attn = ScaledDotProductAttn()

        self.q_identity = IdentityModule()
        self.k_identity = IdentityModule()
        self.v_identity = IdentityModule()

    def init_weights(self,):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def emb_fxn(self, q, k, *args, **kwargs):
        """
        Helpful abstraction function for relative/rotary encodings
        """
        return q, k

    def forward(self,
            q,k,v,
            attn_mask=None,
            is_causal=False,
            past_key_value=None,
            use_cache=False,
            output_attentions=False,
            position_ids=None,
            *args,**kwargs):
        """
        Args:
            q: torch float tensor (B,L,D)
            k: torch float tensor (B,S,K)
            v: torch float tensor (B,S,V)
            attn_mask: torch bool tensor (B,L,S)
                true values denote attended values. Ideally do not
                argue a float tensor, but if you do, the tensor will
                be added to the attention score. Thus -inf should be
                at unattended locations.
            is_causal: bool
                optionally apply a causal mask without arguing a mask.
                will error if mask is not None and this is true.
            past_key_values: optional, tuple of tensors
                the indices will refer to the following (key or value,
                batch, head, seq, size).
            use_cache: bool
                if true, will return new past_key_values
            output_attentions: bool
                if true, will return the unscaled attention weights.
            position_ids: None or LongTensor (B,S)
                optionally argue the position ids for the positional
                encodings.
        Returns:
            ret_dict:
                "output": torch tensor (B,L,D)
                    the output of the multihead attention operation
                "past_key_value": tuple of tensors
                    the 0 index refers to the key calculations after
                    the projection before the rotary embedding. It will
                    have shape (B,N,T,P) where T is the sequence dim.
                    The 1 index is similar but refers to the past values.
                "attentions": (B,N,L,S)
                    the unscaled attention weights
        """
        ret_dict = dict()
        N,P = self.nhead, self.proj_dim
        B,L,D = q.shape
        B,S,K = k.shape
        B,S,V = v.shape

        q = self.q_proj(q).reshape(B,L,N,P).permute(0,2,1,3)
        k = self.k_proj(k).reshape(B,S,N,P).permute(0,2,1,3)
        v = self.v_proj(v).reshape(B,S,N,P).permute(0,2,1,3)
        v = self.v_identity(v)
        k = self.k_identity(k)
        q = self.q_identity(q)

        if past_key_value is not None:
            # Assumes past_key_value = k or v; (B,N,S1,P)
            k = torch.cat([past_key_value[0],k],dim=-2)
            v = torch.cat([past_key_value[1],v],dim=-2)
            B,N,S,P = k.shape
            B,N,S,P = v.shape
        if use_cache:
            ret_dict["past_key_value"] = (k,v)

        if attn_mask is not None:
            if attn_mask.dtype!=torch.bool:
                attn_mask = attn_mask==0
            if len(attn_mask.shape)==3:
                if attn_mask.shape[0]!=B:
                    attn_mask = attn_mask.reshape(B,N,L,S)
                else:
                    attn_mask = attn_mask[:,None].repeat((1,N,1,1))


        q,k = self.emb_fxn(q,k, position_ids=position_ids)

        attn_out = F.scaled_dot_product_attention(
            query=q,key=k,value=v,attn_mask=attn_mask,is_causal=is_causal)
        if output_attentions:
            # TODO
            #scale = math.sqrt(k.shape[-1])
            #weights = torch.einsum("bnlp,bnsp->bnls", q,k)/scale
            #if attn_mask is not None:
            #    weights = weights.masked_fill_(~attn_mask,float(-math.inf))
            #ret_dict["attentions"] = torch.softmax(weights, dim=-1)
            ret = self.sdp_attn(q,k,v,attn_mask)
            for k in ret:
                ret_dict[k] = ret[k]
            if "attn_out" in ret and ret["attn_out"] is not None:
                attn_out = ret["attn_out"]
                del ret_dict["attn_out"]
            #try:
            #    assert torch.isclose(ret["attn_out"], attn_out)
            #except:
            #    print("sdp not close!!")
            #    print("sdp attn:", ret["attn_out"][0,0])
            #    print("torch attn:", attn_out[0,0])
            #    print("sdp attn:", ret["attn_out"].shape)
            #    print("torch attn:", attn_out[0,0].shape)

        ret_dict["output"] = self.out_proj(
            attn_out.permute(0,2,1,3).reshape(B,L,N*P))
        return ret_dict

class ScaledDotProductAttn(nn.Module):
    """
    This can be useful for causal interventions in the attention
    mechanism of transformers.
    """
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, mask):
        """
        Args:
            q: tensor (B,N,L,D)
            k: tensor (B,N,S,D)
            v: tensor (B,N,S,P)
        Returns:
            attentions: tensor (B,N,L,S)
            attn_out: None or tensor (B,N,L,P)
                if a tensor is returned, it overwrites the attn_out
                in the MultiHeadAttention forward function.
        """
        # = inpt_tuple
        scale = math.sqrt(k.shape[-1])
        strens = torch.einsum("bnld,bnsd->bnls", q,k)/scale
        if mask is not None:
            strens = strens.masked_fill_(~mask,float(-math.inf))
        attns = torch.softmax(strens, dim=-1)

        ret_dict = {
            "attentions": attns,
        }

        # Can optionally return the attn_out and it will overwrite
        # the attention out in the MultiHeadAttention module forward
        # function.
        #exp_strens = torch.exp(strens)
        #stren_vals = torch.einsum("bnls,bnsp->bnlsp", exp_strens, v)
        #attn_out = torch.einsum("bnls,bnsp->bnlp",attns,v)
        #ret_dict["attn_out"] = attn_out

        return ret_dict

#class RotaryEmbeddingWrapper(nn.Module):
#    """
#    Wrapper to make interventions easier
#    """
#    def __init__(self, rotary_emb, *args, **kwargs):
#        super().__init__()
#        self.rotary_emb = rotary_emb
#
#    def forward(self, q, k, position_ids=None, *args, **kwargs):
#        """
#        q: torch tensor (B,NHead,Length,P)
#        k: torch tensor (B,NHead,S,P)
#        position_ids: torch LongTensor (S)
#        """
#        offset = k.shape[-2]-1 if k.shape[-2]!=q.shape[-2] else 0
#        if position_ids is not None:
#            position_ids = position_ids.long()
#            qpids = position_ids[-q.shape[-2]:]
#            q = self.rotary_emb(q, offset=0, position_ids=qpids)
#        else:
#            q = self.rotary_emb(q, offset=offset)
#        k = self.rotary_emb(k, position_ids=position_ids)
#        return q, k
#
#class RotaryAttention(MultiHeadAttention):
#    def __init__(self, rot_dim=None, *args, **kwargs):
#        """
#        A multiheaded self-attention module that uses rotary encodings.
#
#        Args:
#            see MultiHeadAttention for args and kwargs
#
#            rot_dim: int
#                the number of dimensions to use for the rotary encodings.
#                Must be divisible by 2 and must be less than or equal to
#                d_model//n_heads.
#        """
#        super().__init__(*args, **kwargs)
#        if rot_dim is None: rot_dim = self.proj_dim
#        self.rotary_emb = RotaryEmbedding(d=rot_dim)
#        self.rotary_emb_wrapper = RotaryEmbeddingWrapper(self.rotary_emb)
#
#    def emb_fxn(self, q, k, position_ids=None, *args, **kwargs):
#        """
#        q: torch tensor (B,NHead,Length,P)
#        k: torch tensor (B,NHead,S,P)
#        position_ids: torch LongTensor (S)
#        """
#        return self.rotary_emb_wrapper(
#            q=q, k=k, position_ids=position_ids, *args, **kwargs)



class RotaryAttention(MultiHeadAttention):
    def __init__(self, rot_dim=None, *args, **kwargs):
        """
        A multiheaded self-attention module that uses rotary encodings.

        Args:
            see MultiHeadAttention for args and kwargs

            rot_dim: int
                the number of dimensions to use for the rotary encodings.
                Must be divisible by 2 and must be less than or equal to
                d_model//n_heads.
        """
        super().__init__(*args, **kwargs)
        if rot_dim is None: rot_dim = self.proj_dim
        self.rotary_emb = RotaryEmbedding(d=rot_dim)

    def emb_fxn(self, q, k, position_ids=None, *args, **kwargs):
        """
        q: torch tensor (B,NHead,Length,P)
        k: torch tensor (B,NHead,S,P)
        position_ids: torch LongTensor (S)
        """
        offset = k.shape[-2]-1 if k.shape[-2]!=q.shape[-2] else 0
        if position_ids is not None:
            position_ids = position_ids.long()
            qpids = position_ids[-q.shape[-2]:]
            q = self.rotary_emb(q, offset=0, position_ids=qpids)
        else:
            q = self.rotary_emb(q, offset=offset)
        k = self.rotary_emb(k, position_ids=position_ids)
        return q, k


class SimpleEncoderLayer(nn.Module):
    """
    A custom transformer encoder layer.
    """
    def __init__(self,
            d_model,
            nhead=4,
            dim_feedforward=2048,
            dropout=0.1,
            activation=F.relu,
            batch_first=True,
            norm_first=True,
            bias=True,
            layer_norm_eps=1e-5,
            llama=False,
            device=None,
            dtype=None,
            *args, **kwargs):
        """
        Args:
            d_model: int
            nhead: int
            dropout: float
            bias: bool
            batch_first: bool
        """
        super().__init__()
        self.llama = llama
        factory_kwargs = {"device": device, "dtype": dtype}
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
            **factory_kwargs,)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(
            d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(
            dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        #if self.llama:
        #    self.norm1 = nn.RMSNorm( d_model, **factory_kwargs)
        #    self.norm2 = nn.RMSNorm( d_model, **factory_kwargs)
        #    print("Using llama!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #else:
        self.norm1 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation
        self.kv_identity = IdentityModule()

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    # self-attention block
    def _ma_block(self, q: Tensor, kv: Tensor,
                  attn_mask: Optional[Tensor],
                  is_causal: bool = False,
                  past_key_value=None,
                  use_cache=False,
                  output_attentions=False,
                  position_ids=None) -> Tensor:
        """
        Args:
            q: torch tensor (B,L,D)
            kv: torch tensor (B,S,D)
            mask: torch bool tensor (B,L,S)
                true values denote attended values. Ideally do not
                argue a float tensor, but if you do, the tensor will
                be added to the attention score. Thus -inf should be
                at unattended locations.
            is_causal: bool
                optionally apply a causal mask without arguing a mask.
                will error if mask is not None and this is true.
            past_key_values: optional, tuple of tensors
                the indices will refer to the following (key or value,
                batch, head, seq, size).
            use_cache: bool
                if true, will return new past_key_values
            output_attentions: bool
                if true, will return the unscaled attention weights
            position_ids: None or LongTensor (B,S)
                optionally argue the position ids for the positional
                encodings.
        """
        kv = self.kv_identity(kv)
        ret_dict = self.self_attn(q, kv, kv,
                           attn_mask=attn_mask,
                           is_causal=is_causal,
                           past_key_value=past_key_value,
                           use_cache=use_cache,
                           output_attentions=output_attentions,
                           position_ids=position_ids)
        ret_dict["output"] = self.dropout1(ret_dict["output"])
        return ret_dict

    def forward(self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False,
            past_key_value: tuple=None,
            use_cache: bool=False,
            output_attentions=False,
            position_ids=None,
            *args, **kwargs) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: torch tensor (B,S,E)
            src_mask: the mask for the src sequence (optional).
                true/1s mean do attend
            src_key_padding_mask: the mask for the src keys per batch (optional).
                true/1s mean do attend
            is_causal: If specified, applies a causal mask as ``src mask``.
                Default: ``False``.
                Warning:
                ``is_causal`` provides a hint that ``src_mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            past_key_value: tensor (B,S,...) or None
                if using caching, can argue a tensor here. It should
                be the hidden states fed into this layer from the
                previous step.
            use_cache: bool
                if true, will return the intermediate key_value
                computations.
            output_attentions: bool
                if true, will return the unscaled attention weights
            position_ids: None or LongTensor (B,S)
                optionally argue the position ids for the positional
                encodings.
        Returns:
            ret_dict:
                "output": torch tensor (B,L,D)
                    the output of the multihead attention operation
                "past_key_value": tuple of tensors
                    the 0 index refers to the key calculations after
                    the projection before the rotary embedding. It will
                    have shape (B,N,T,P) where T is the sequence dim.
                    The 1 index is similar but refers to the past values.
                "attentions": (B,N,L,S)
                    the unscaled attention weights
        """
        ret_dict = dict()
        if src_mask is not None:
            #src_mask = ~src_mask
            if src_mask.shape[1]!=src.shape[1]:
                src_mask = src_mask[:,-src.shape[1]:]
        if src_key_padding_mask is not None:
            raise NotImplemented
            src_key_padding_mask = ~src_key_padding_mask

        x = src
        if self.llama:
            # Llama Architecture
            norm_x = self.norm1(x)
            attn_ret = self._ma_block(
                q=norm_x,
                kv=norm_x,
                attn_mask=src_mask,
                is_causal=is_causal,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
                position_ids=position_ids,)
            x = x + attn_ret["output"]
            x = x + self._ff_block(self.norm2(x))
        elif self.norm_first:
            x = self.norm1(x)
            attn_ret = self._ma_block(
                q=x,
                kv=x,
                attn_mask=src_mask,
                is_causal=is_causal,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
                position_ids=position_ids,)
            x = self.norm2(x + attn_ret["output"])
            x = x + self._ff_block(x)
        else:
            attn_ret = self._ma_block(
                q=x,
                kv=x,
                attn_mask=src_mask,
                is_causal=is_causal,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
                position_ids=position_ids,)
            x = self.norm1(x + attn_ret["output"])
            x = self.norm2(x + self._ff_block(x))
        if use_cache:
            ret_dict["past_key_value"] = attn_ret["past_key_value"]
        if output_attentions:
            ret_dict["attentions"] = attn_ret["attentions"]
        ret_dict["hidden_states"] = x
        return ret_dict


class RotaryEncoderLayer(SimpleEncoderLayer):
    """
    A custom rotary transformer encoder layer.
    """
    def __init__(self,
            *args,
            rot_dim=None,
            **kwargs):
        """
        Args:
            d_model: int
            nhead: int
            dropout: float
            bias: bool
            batch_first: bool
            rot_dim: int
                kdim must be divisible by rot_dim
        """
        super().__init__(*args, **kwargs)
        self.self_attn = RotaryAttention(
            rot_dim=rot_dim,
            **kwargs,)


class PKVEncoderLayer(nn.TransformerEncoderLayer):
    """
    A custom transformer encoder layer using PyTorch's MultiHeadAttention
    module. The purpose of building a custom module is for ease caching
    intermediate computations while maintaining a great degree of
    flexibility over the architecture.
    """
    def _ma_block(self, q: Tensor, kv: Tensor,
                  attn_mask: Optional[Tensor],
                  key_padding_mask: Optional[Tensor],
                  is_causal: bool = False) -> Tensor:
        x = self.self_attn(q, kv, kv,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False,
                           is_causal=is_causal)[0]
        return self.dropout1(x)

    def forward(self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False,
            past_key_value: tuple=None,
            use_cache: bool=False,) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: torch tensor (B,S,E)
            src_mask: the mask for the src sequence (optional).
                true/1s mean do attend
            src_key_padding_mask: the mask for the src keys per batch (optional).
                true/1s mean do attend
            is_causal: If specified, applies a causal mask as ``src mask``.
                Default: ``False``.
                Warning:
                ``is_causal`` provides a hint that ``src_mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            past_key_value: tensor (B,S,...) or None
                if using caching, can argue a tensor here. It should
                be the hidden states fed into this layer from the
                previous step.
            use_cache: bool
                if true, will return the intermediate key_value
                computations.
        Returns:
            fx: torch Tensor
        """
        ret_dict = dict()
        if src_mask is not None:
            src_mask = ~src_mask
            if src_mask.shape[1]!=src.shape[1]:
                src_mask = src_mask[:,-src.shape[1]:]
        if src_key_padding_mask is not None:
            src_key_padding_mask = ~src_key_padding_mask

        if past_key_value is None:
            if use_cache: ret_dict["past_key_value"] = [src.clone()]
            hidden_states = super().forward(
                src=src,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,)
            ret_dict["hidden_states"] = hidden_states
        else:
            full_seq = torch.cat(past_key_value+[src],dim=1)
            if use_cache:
                ret_dict["past_key_value"] = [full_seq.clone()]

            if self.norm_first:
                full_seq = self.norm1(full_seq)

            x = full_seq[:,-src.shape[1]:]


            #if src_mask is not None:
            #    src_mask = ~src_mask
            #    if src_mask.shape[1]!=x.shape[1]:
            #        src_mask = src_mask[:,-x.shape[1]:]
            #if src_key_padding_mask is not None:
            #    src_key_padding_mask = ~src_key_padding_mask

            x = x + self._ma_block(
                q=x,
                kv=full_seq,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                is_causal=is_causal)

            if not self.norm_first:
                x = self.norm1(x)
            else:
                x = self.norm2(x)
            x = x + self._ff_block(x)
            if not self.norm_first:
                x = self.norm2(x)

            ret_dict["hidden_states"] = x
        return ret_dict

def print_tensor(t, n_tab=0):
    if len(t.shape)==2:
        el = t.tolist()
        for e in el:
            print("\t"*n_tab, e)
    else:
        for tt in t:
            print_tensor(tt, n_tab=n_tab+1)
            print()

class IdentityModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        return x

class IdentityPositionalEncoding(nn.Module):
    def __init__(self,
                 drop_p:float=0,
                 *args, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = self.dropout( x )
        return x

class PositionalEncoding(nn.Module):
    def __init__(self,
                 posenc_drop_p:float=0,
                 drop_p:float=0.1,
                 max_len:int=1000):
        super().__init__()
        self.posenc_dropout = nn.Dropout(p=posenc_drop_p)
        self.dropout = nn.Dropout(p=drop_p)
        self.arange = np.arange(max_len).astype("int")

    def rand_forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        n = np.random.randint(x.size(1), self.pe.shape[0]+1)
        idxs = torch.sort(torch.randperm(n)[:x.size(1)]).values.long()
        x = self.dropout( x + self.posenc_dropout(self.pe[idxs]) )
        return x

    def skip_rand_forward(
            self,
            x: Tensor,
            mask: Tensor,
            *args,
            **kwargs
        ) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
            mask: Tensor, shape ``[batch_size, seq_len]``
                pad mask. true values represent padding/blotching
        """
        if mask is None: return self.rand_forward(x)
        # pe: N, E
        n = np.random.randint(x.size(1), self.pe.shape[0]+1)
        idxs = torch.sort(torch.randperm(n)[:x.size(1)]).values.long()
        pe = self.posenc_dropout(self.pe[idxs])

        sums = (~mask).float().sum(-1)
        idxs = torch.cat([torch.arange(s) for s in sums], axis=0).long()
        fx = torch.zeros_like(x)
        fx[~mask] += pe[idxs]
        fx = x + fx

        return self.dropout( fx )

    def vanil_forward(self,
                      x: Tensor,
                      pids: Tensor=None,
                      *args, **kwargs) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
            pids: LongTensor (B,S)
        """
        if pids is not None:
            shape = [s for s in pids.shape] + [self.pe.shape[-1]]
            posencs = self.pe[pids.reshape(-1)].reshape(shape)
        else:
            posencs = self.pe
        x = self.dropout( x + self.posenc_dropout(posencs[:x.size(1)]) )
        return x

    def skip_vanil_forward(
            self,
            x: Tensor,
            mask: Tensor,
            *args,
            **kwargs
        ) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
            mask: Tensor, shape ``[batch_size, seq_len]``
                pad mask. true values represent padding/blotching
        """
        if mask is None: return self.vanil_forward(x)
        pe = self.posenc_dropout(self.pe[:x.size(1)])

        sums = torch.sum((~mask).float(), -1)
        idxs = torch.cat([torch.arange(s) for s in sums], axis=0).long()
        fx = torch.zeros_like(x)
        fx[~mask] += pe[idxs]
        fx = x + fx

        return self.dropout( fx )

class RandPositionalEncoding(PositionalEncoding):
    def __init__(self,
                 d_model:int,
                 posenc_drop_p:float=0,
                 drop_p:float=0.1,
                 max_len:int=1000,
                 learnable:bool=False,
                 pad_pos_skip:bool=False):
        super().__init__(posenc_drop_p, drop_p, max_len=max_len)
        self.pad_pos_skip = pad_pos_skip

        pe = 0.1*math.sqrt(max_len/d_model)*torch.randn(max_len,d_model)
        if learnable: self.pe = torch.nn.Parameter(pe)
        else: self.register_buffer('pe', pe)

        if pad_pos_skip:
            self.forward = self.skip_rand_forward
        else:
            self.forward = self.rand_forward

class SinPositionalEncoding(PositionalEncoding):
    def __init__(self,
                 d_model:int,
                 posenc_drop_p:float=0,
                 drop_p:float=0.1,
                 max_len:int=1000,
                 learnable:bool=False,
                 pad_pos_skip:bool=False):
        super().__init__(posenc_drop_p, drop_p, max_len=max_len)
        self.pad_pos_skip = pad_pos_skip

        position = torch.arange(max_len).unsqueeze(1)
        scale = (-math.log(10000.0) / d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2) * scale)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        if learnable: self.pe = torch.nn.Parameter(pe)
        else: self.register_buffer('pe', pe)

        if pad_pos_skip:
            self.forward = self.skip_vanil_forward
        else:
            self.forward = self.vanil_forward


class RandSinPositionalEncoding(SinPositionalEncoding):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.pad_pos_skip:
            self.forward = self.skip_rand_forward
        else:
            self.forward = self.rand_forward


if __name__=="__main__":
    mlp = MLP(
        inpt_size=10, outp_size=3, n_layers=4, h_sizes=None, lnorm=True,
        noise=5, drop_p=0.5,
        )
    print(mlp)

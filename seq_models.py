from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import dl_utils.seq_models as smods
import dl_utils.torch_modules as tmods
from dl_utils.utils import (
    generate_square_subsequent_mask, arglast, top_k_acc,
    update_shape, padmask2attnmask, pad_to, device_fxn,
    generate_ktoken_causal_mask, get_one_hot
)
import math

DEVICES = {
    -1: "cpu", **{i:i for i in range(10)}
}

class LSTM(smods.LSTM):
    def __init__(self,trigger_ids=7,linear_output=False,legacy_lstm=False,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.legacy = legacy_lstm
        if type(trigger_ids)==int:
            trigger_ids = [trigger_ids]

        inpt_size = self.d_model
        if not self.legacy:
            self.rnns = nn.ModuleList([])
            for _ in range(self.n_layers):
                print("Making new lstm cell", inpt_size, self.d_model//2)
                self.rnns.append(
                    nn.LSTMCell(inpt_size, self.d_model//2)
                )
                inpt_size = self.d_model//2
        d_hid = self.d_model*4
        modules = []
        modules.append(torch.nn.Linear( inpt_size, d_hid ))
        modules.append(torch.nn.GELU())
        if self.l_norm:
            modules.append(torch.nn.LayerNorm(d_hid))
        modules.append(torch.nn.Dropout(self.drop_p))
        self.decoder = torch.nn.Sequential( *modules )
        self.lm_head = torch.nn.Linear( d_hid, self.out_tokens )

        # We will use the identities to enable hooking into the rnn
        # state after the prediction is made.
        self.inpt_identity = tmods.IdentityModule()
        self.identities = nn.ModuleList([])
        for _ in range(self.n_layers):
            self.identities.append(tmods.IdentityModule())

        self.register_buffer("trigger_ids", torch.LongTensor(
            [tid for tid in trigger_ids]
        ))
        if linear_output:
            self.decoder = tmods.IdentityModule()
            self.lm_head = torch.nn.Linear(self.d_model,self.out_tokens)

    def step_core(self, inpt, h, layer, idx):
        """
        This function is purely meant to allow for easier DAS
        interventions
        """
        hcopies = [x.clone() for x in h]
        h = self.rnns[layer](inpt[idx], (h[0][idx],h[1][idx]))
        if self.h_norm or self.l_norm:
            h,c = h
            h = self.layer_norms[layer](h)
            h = (h,c)
        for hcop,og in zip(hcopies, h):
            hcop[idx] = og
        # This will allow us to easily intervene in the same way
        # as we do for the single state recurrent models.
        cat = self.identities[layer](torch.cat(hcopies,dim=-1))
        h = (cat[..., :h[0].shape[-1]], cat[..., h[0].shape[-1]:]) 
        return h

    def forward(self, inpts:torch.Tensor=None,
                      pad_mask:torch.BoolTensor=None,
                      task_mask:torch.BoolTensor=None,
                      n_steps:int=0,
                      temperature=None,
                      input_ids:torch.Tensor=None,
                      attention_mask:torch.Tensor=None,
                      inputs_embeds=None,
                      ret_gtruth=True,
                      *args, **kwargs):
        """
        Arguments:
            inpts: Tensor, shape ``[bsize, seq_len]``
            pad_mask: Tensor, shape ``[bsize, seq_len]``
                true means padding
            attention_mask: Tensor, shape ``[bsize, seq_len]``
                huggingface style padding mask where false means padding.
                True for positions that you want to attend to.
            task_mask: Tensor, shape ``[bsize, seq_len]``
                true means the tokens are a part of the prediction
                task and as such should not be teacher forced. Tokens at
                indices that are false will be fed into the model
                regardless of the model's predictions.
            n_steps: int
                the number of prediction steps if not using teacher
                forcing
            temperature: float
                a parameter to adjust the entropy of the
                token sampling. high temperature means high entropy
            inputs_embeds: None or Tensor, shape (B,S,D)
                optionally argue the embeddings directly instead of
                token ids.
            ret_gtruth: bool
                if true, will return the pred ids of the argued input
                instead of the predicted ids when the task mask is 0.
                The model's predicted logits are always returned. This
                is for ease of computing accuracy in random sequences.
        Returns:
            ret_dict: dict
                logits: Tensor of shape ``[bsize,seq_len+n_steps,n_tokens]``
                pred_ids: Tensor of shape ``[bsize,seq_len+n_steps,n_tokens]``
        """
        if input_ids is not None:
            inpts = input_ids
        input_ids = inpts
        if inputs_embeds is None:
            embs = self.embeddings(inpts)
        else: embs = inputs_embeds

        B,S,D = embs.shape
        logits = []
        pred_ids = []
        past_hs,past_cs = [],[]
        hs,cs = self.get_fresh_recurrent_vectors(B)
        if attention_mask is not None:
            pad_mask = ~attention_mask.bool()
        if pad_mask is None:
            pad_mask = torch.zeros_like(inpts).bool()
        # The task mask allows us to do teacher forcing up until
        # the prediction task begins. At each point that the task mask
        # is true, we do not teacher force.
        if task_mask is None:
            task_mask = torch.zeros_like(pad_mask).bool()

        # Loop through sequence
        for step in range(S+n_steps):
            if step<embs.shape[1]:
                pmask = pad_mask[:,step]
                tmask = task_mask[:,step]
                inpt = embs[:,step]
                # Insert predicted embeddings when tmask is true
                if torch.any(tmask) and step>0:
                    inpt[tmask] = self.embeddings( pred_ids[-1][tmask] )
            else:
                inpt = self.embeddings(pred_ids[-1])
                pmask = None
                tmask = None
            inpt = self.inpt_identity(inpt)

            ret_dict = self.step(
                inputs_embeds=inpt,
                pad_mask=pmask,
                hs=hs, cs=cs,
                temperature=temperature,
            )
            hs,cs = ret_dict["hs"], ret_dict["cs"]

            past_hs.append(hs)
            past_cs.append(cs)
            logits.append(ret_dict["logits"])
            new_preds = ret_dict["pred_ids"]
            if step<S-1 and ret_gtruth:
                new_ids = input_ids[:,step+1].data.clone()
                # If next step is not task pred, then we'll return the
                # prediction as the ground truth.
                tmask = task_mask[:,step+1] 
                new_preds[~tmask] = new_ids[~tmask]
            pred_ids.append(new_preds)
        return {
            "logits": torch.stack(logits, dim=1),
            "pred_ids": torch.stack(pred_ids,dim=1),
            "hs": hs, "cs": cs,
            "past_hs": past_hs,
            "past_cs": past_cs,
        }

class RNN(smods.RNN):
    def __init__(self,trigger_ids=7,linear_output=False,*args,**kwargs):
        super().__init__(*args, **kwargs)
        if type(trigger_ids)==int:
            trigger_ids = [trigger_ids]

        # We will use the identities to enable hooking into the rnn
        # state after the prediction is made.
        self.inpt_identity = tmods.IdentityModule()
        self.identities = nn.ModuleList([])
        for _ in range(self.n_layers):
            self.identities.append(tmods.IdentityModule())

        self.register_buffer("trigger_ids", torch.LongTensor(
            [tid for tid in trigger_ids]
        ))
        if linear_output:
            self.decoder = tmods.IdentityModule()
            self.lm_head = torch.nn.Linear(self.d_model,self.out_tokens)

    def step_core(self, inpt, h, layer, idx):
        hcopy = h.clone()
        h = self.rnns[layer](inpt[idx], h[idx])
        if self.h_norm or self.l_norm:
            h = self.layer_norms[layer](h)
        hcopy[idx] = h
        hcopy = self.identities[layer](hcopy)
        return hcopy

    def forward(self, inpts:torch.Tensor=None,
                      pad_mask:torch.Tensor=None,
                      task_mask:torch.Tensor=None,
                      n_steps:int=0,
                      temperature=None,
                      input_ids:torch.Tensor=None,
                      attention_mask:torch.Tensor=None,
                      inputs_embeds=None,
                      ret_gtruth=True,
                      *args, **kwargs):
        """
        Arguments:
            inpts: Tensor, shape ``[bsize, seq_len]``
            pad_mask: Tensor, shape ``[bsize, seq_len]``
                true means padding
            attention_mask: Tensor, shape ``[bsize, seq_len]``
                huggingface style padding mask where false means padding.
                True for positions that you want to attend to.
            task_mask: Tensor, shape ``[bsize, seq_len]``
                true means the tokens are a part of the prediction
                task, and, as such should not be teacher forced. Tokens
                at indices that are false will be fed into the model.
            n_steps: int
                the number of prediction steps if not using teacher
                forcing
            temperature: float
                a parameter to adjust the entropy of the
                token sampling. high temperature means high entropy
            inputs_embeds: None or Tensor, shape (B,S,D)
                optionally argue the embeddings directly instead of
                token ids.
            ret_gtruth: bool
                if true, will return the pred ids of the argued input
                instead of the predicted ids. The predicted logits are
                returned the same either way. This is for ease of
                computing accuracy in random sequences.
        Returns:
            ret_dict: dict
                logits: Tensor of shape ``[bsize,seq_len+n_steps,n_tokens]``
                pred_ids: Tensor of shape ``[bsize,seq_len+n_steps,n_tokens]``
        """
        if input_ids is not None:
            inpts = input_ids
        input_ids = inpts
        if inputs_embeds is None:
            embs = self.embeddings(inpts)
        else: embs = inputs_embeds

        B,S,D = embs.shape
        logits = []
        pred_ids = []
        past_hs = []
        hs = self.get_fresh_recurrent_vectors(B)
        if attention_mask is not None:
            pad_mask = ~attention_mask.bool()
        if pad_mask is None:
            pad_mask = torch.zeros_like(inpts).bool()
        # The task mask allows us to do selective teacher forcing.
        # Indices that are false will be teacher forced.
        if task_mask is None:
            task_mask = torch.zeros_like(pad_mask).bool()

        # Loop through sequence
        for step in range(S+n_steps):
            if step<embs.shape[1]:
                pmask = pad_mask[:,step]
                inpt = embs[:,step]
                tmask = task_mask[:,step]
                # Insert predicted embeddings when tmask is true
                if torch.any(tmask) and step>0:
                    inpt[tmask] = self.embeddings( pred_ids[-1][tmask] )
            else:
                inpt = self.embeddings(pred_ids[-1])
                tmask = None
            inpt = self.inpt_identity(inpt)
            ret_dict = self.step(
                inputs_embeds=inpt,
                pad_mask=pmask,
                hs=hs,
                temperature=temperature,
            )
            hs = [h for h in ret_dict["hs"]]
            past_hs.append(hs)
            logits.append(ret_dict["logits"])
            new_preds = ret_dict["pred_ids"]
            if ret_gtruth and step<S-1:
                new_ids = input_ids[:,step+1].data.clone()
                # If next step is not task pred, then we'll return the
                # prediction as the ground truth.
                tmask = task_mask[:,step+1] 
                new_preds[~tmask] = new_ids[~tmask]
            pred_ids.append(new_preds)
        return {
            "logits": torch.stack(logits, dim=1),
            "pred_ids": torch.stack(pred_ids,dim=1),
            "hs": hs,
            "past_hs": past_hs,
        }

class GRU(RNN):
    def __init__(self, trigger_ids=[3], rnn_type="GRUCell", *args, **kwargs):
        super().__init__(
            *args, trigger_ids=trigger_ids, rnn_type=rnn_type, **kwargs
        )

class LinearRNN(RNN):
    def __init__(self, trigger_ids=7, *args, **kwargs):
        super().__init__( *args, trigger_ids=trigger_ids, **kwargs)
        d_hid = self.d_model*4
        modules = []
        modules.append(torch.nn.Linear( self.d_model, d_hid ))
        if self.l_norm:
            modules.append(torch.nn.LayerNorm(d_hid))
        modules.append(torch.nn.Dropout(self.drop_p))
        self.decoder = torch.nn.Sequential( *modules )
        # lm_head has the final linear layer

class ToyRNNCell(torch.nn.Module):
    def __init__(self, inpt_size, d_model,*args, **kwargs):
        super().__init__()
        self.inpt_size = inpt_size
        self.d_model = d_model
        self.reset_parameters()

    def reset_parameters(self):
        self.inpt_linear = torch.nn.Linear(
            self.inpt_size,self.d_model, bias=False)
        self.weight_ih = self.inpt_linear.weight
        self.weight_ih.data = torch.eye(self.inpt_size)[:self.d_model]
        self.h_linear = torch.nn.Linear(
            self.d_model,self.d_model, bias=False)
        self.weight_hh = self.h_linear.weight
        self.weight_hh.data = torch.eye(self.d_model)
        self.bias_ih = torch.nn.Parameter(torch.zeros(1))
        self.bias_hh = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, h, *args, **kwargs):
        return self.inpt_linear(x) + self.h_linear(h) + self.bias_ih + self.bias_hh

class ToyRNN(RNN):
    """
    This is the base class for toy models.
    """
    def __init__(self,
            toy_scale=1,
            max_count=100,
            pad_id=0,
            bos_id=1,
            eos_id=2,
            resp_id=6,
            demo_ids=[3,4,5],
            trigger_ids=[7],
            *args,
            **kwargs):
        super().__init__( *args, **kwargs)
        self.toy_scale = toy_scale
        self.max_count = max_count+10
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.resp_id = resp_id
        self.demo_ids = demo_ids
        self.lm_head = None
        self.rnns[0] = None
        self.reset_parameters()

    def get_fresh_recurrent_vectors(self, *args, **kwargs):
        device = self.get_device()
        self.reset_parameters()
        self.to(device)
        ret = super().get_fresh_recurrent_vectors(*args, **kwargs)
        return ret

class Toy1dRNN(ToyRNN):
    """
    This model works by encoding the count as the magnitude of the state
    vector along the first dimension. It encodes the phase along the
    second dimension.
    """
    def __init__(self, toy_scale=1, max_count=100, *args, **kwargs):
        super().__init__( *args, **kwargs)

    def reset_parameters(self):
        bos_id   = self.bos_id
        eos_id   = self.eos_id
        resp_id  = self.resp_id
        demo_ids = self.demo_ids
        tids = self.trigger_ids
        tlen = len(self.trigger_ids)

        ts = self.toy_scale
        self.embeddings.weight.data.zero_()
        self.embeddings.weight.data[bos_id, -1] = ts
        self.embeddings.weight.data[resp_id, 0] = -ts
        for demo_id in demo_ids:
            self.embeddings.weight.data[demo_id,0] = ts
        for tid in tids:
            self.embeddings.weight.data[tid,-1] = -2*ts

        if self.rnns[0] is None:
            self.rnns[0] = ToyRNNCell(self.d_model,self.d_model)
        else:
            self.rnns[0].reset_parameters()
            self.rnns[0].weight_ih.data = torch.eye(self.d_model)
            self.rnns[0].weight_hh.data = torch.eye(self.d_model)
            self.rnns[0].bias_ih.data.zero_()
            self.rnns[0].bias_hh.data.zero_()

        if self.lm_head is None:
            self.decoder = tmods.IdentityModule()
            self.lm_head = torch.nn.Linear(
                self.d_model, self.out_tokens, bias=False)
        self.lm_head.weight.data.zero_()

        self.lm_head.weight.data[eos_id,-1] = -1.1
        self.lm_head.weight.data[resp_id,0] = 1
        self.lm_head.weight.data[resp_id,-1] = -1
        for demo_id in demo_ids:
            self.lm_head.weight.data[demo_id,0] = 1
            self.lm_head.weight.data[demo_id,-1] = 1
        if hasattr(self.lm_head, "bias") and self.lm_head.bias is not None:
            self.lm_head.bias.data.zero_()

class Toy2dRNN(Toy1dRNN):
    """
    This model uses the first dimension to encode the input count,
    the second dim for the response count, and the 3rd dim for the
    phase.
    """
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)
        self.lm_head = None
        self.reset_parameters()

    def reset_parameters(self):
        bos_id   = self.bos_id
        eos_id   = self.eos_id
        resp_id  = self.resp_id
        demo_ids = self.demo_ids
        tids = self.trigger_ids
        tlen = len(self.trigger_ids)

        ts = self.toy_scale

        self.embeddings.weight.data.zero_()

        self.embeddings.weight.data[bos_id,-1] = ts
        self.embeddings.weight.data[resp_id,1] = ts
        for demo_id in demo_ids:
            self.embeddings.weight.data[demo_id,0] = ts
        for tid in tids:
            self.embeddings.weight.data[tid,-1] = -2*ts

        if self.rnns[0] is None:
            self.rnns[0] = ToyRNNCell(self.d_model, self.d_model)
        else:
            self.rnns[0].weight_ih.data = torch.eye(self.d_model)
            self.rnns[0].weight_hh.data = torch.eye(self.d_model)
            self.rnns[0].bias_ih.data.zero_()
            self.rnns[0].bias_hh.data.zero_()

        if self.lm_head is None:
            self.decoder = tmods.IdentityModule()
            self.lm_head = torch.nn.Linear(
                self.d_model, self.out_tokens, bias=False)
        self.lm_head.weight.data.zero_()

        self.lm_head.weight.data[eos_id,0] = -1
        self.lm_head.weight.data[eos_id,1] = 1
        self.lm_head.weight.data[eos_id,-1] = -1.1
        self.lm_head.weight.data[resp_id,0] = 1
        self.lm_head.weight.data[resp_id,1] = -1
        self.lm_head.weight.data[resp_id,-1] = -1
        for demo_id in demo_ids:
            self.lm_head.weight.data[demo_id,-1] = self.max_count
        if hasattr(self.lm_head, "bias") and self.lm_head.bias is not None:
            self.lm_head.bias.data.zero_()

class ToyRotationCell(torch.nn.Module):
    """
    Rotates theta radians only the first two dims of vectors of size
    d_model
    """
    def __init__(self, inpt_size, d_model, theta=0.1, *args, **kwargs):
        """
        d_model: int
        theta: float
            rotation in radians
        """
        super().__init__()
        assert inpt_size==d_model
        self.inpt_size = inpt_size
        self.d_model = d_model
        self.theta = theta
        self.inpt_linear = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.inpt_linear is None:
            self.inpt_linear = torch.nn.Linear(
                self.inpt_size,self.d_model, bias=False)
            self.h_linear = torch.nn.Linear(
                self.d_model,self.d_model, bias=False)
        self.weight_ih = self.inpt_linear.weight
        self.weight_ih.data = torch.eye(self.inpt_size)[:self.d_model]

        self.weight_hh = self.h_linear.weight
        self.weight_hh.data = torch.eye(self.d_model)
        self.weight_hh.data[:2,:2] = torch.FloatTensor([
            [float(np.cos(self.theta)), -float(np.sin(self.theta))],
            [float(np.sin(self.theta)),  float(np.cos(self.theta))],
        ])
        self.bias_ih = torch.nn.Parameter(torch.zeros(1))
        self.bias_hh = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, h, *args, **kwargs):
        return self.h_linear(x*h)

class ToyRotationRNN(ToyRNN):
    """
    This model works by encoding the count as the rotation about the
    z axis of a d dimensional vector. It encodes the phase both as the
    sign of the y dimensino and the sign of the z the z dimension.
    """
    def __init__(self, *args, **kwargs):
        self.max_count = kwargs.get("max_count", 100)
        self.theta = float(np.pi)/(self.max_count+10)
        super().__init__( *args, **kwargs)

    def reset_parameters(self):
        bos_id   = self.bos_id
        eos_id   = self.eos_id
        resp_id  = self.resp_id
        demo_ids = self.demo_ids
        tids = self.trigger_ids
        tlen = len(self.trigger_ids)

        ts = self.toy_scale
        resp_emb = torch.FloatTensor([1, 1, 1]) # Response Token Embedding
        demo_emb = torch.FloatTensor([1, 1, 1]) # Demo Token Embedding
        trig_emb = torch.FloatTensor([1,-1,-1]) # Trigger Embedding
        elen = len(trig_emb)

        self.embeddings.weight.data.zero_()
        self.embeddings.weight.data[bos_id,:elen] = resp_emb
        self.embeddings.weight.data[resp_id,:elen] = resp_emb
        for did in demo_ids:
            self.embeddings.weight.data[did,:elen] = demo_emb
        for tid in tids:
            self.embeddings.weight.data[tid,:elen] = trig_emb

        if self.rnns[0] is None:
            self.rnns[0] = ToyRotationCell(
                self.d_model,self.d_model,self.theta)
        else:
            self.rnns[0].reset_parameters()

        if self.lm_head is None:
            self.decoder = tmods.IdentityModule()
            self.lm_head = torch.nn.Linear(
                self.d_model, self.out_tokens, bias=True)
        self.lm_head.weight.data.zero_()
        self.lm_head.bias.data.zero_()

        demo_out = torch.FloatTensor([0,   ts,   ts  ])
        resp_out = torch.FloatTensor([ts,  -ts,   0  ])
        eos_out  = torch.FloatTensor([0,   ts,  -ts  ])
        elen = len(demo_out)

        self.lm_head.weight.data[eos_id,:elen] = eos_out
        self.lm_head.bias.data[eos_id] = float(np.sin(self.theta))
        self.lm_head.weight.data[resp_id,:elen] = resp_out
        for did in demo_ids:
            self.lm_head.weight.data[did,:elen] = demo_out
        if hasattr(self.lm_head, "bias") and self.lm_head.bias is not None:
            self.lm_head.bias.data.zero_()

    def get_fresh_recurrent_vectors(self, *args, **kwargs):
        device = self.get_device()
        self.reset_parameters()
        self.to(device)
        ret = super().get_fresh_recurrent_vectors(*args, **kwargs)
        init_vec = torch.Tensor(
            [1,0,1] + [0 for _ in range(self.d_model-3)]).to(device)
        ret = [r+init_vec for r in ret]
        return ret

class ToyOrthoRotCell(torch.nn.Module):
    """
    Uses individual neurons to track quantity. Requires an extra neuron
    for tracking phase.
    """
    def __init__(self, inpt_size, d_model, *args, **kwargs):
        """
        inpt_size: int
        d_model: int
        """
        super().__init__()
        self.inpt_size = inpt_size
        self.d_model = d_model
        self.inpt_linear = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.inpt_linear is None:
            self.inpt_linear = torch.nn.Linear(
                self.inpt_size,self.d_model, bias=False)
            rand = torch.randn(self.d_model-1,self.d_model-1)
            orth = torch.linalg.svd(rand)[0]
            init_h_mtx = torch.eye(self.d_model)
            init_h_mtx[:-1,:-1] = orth
            self.register_buffer("init_h_mtx", init_h_mtx)
            self.h_linear = torch.nn.Linear(
                self.d_model,self.d_model, bias=False)

        self.weight_ih = self.inpt_linear.weight
        self.weight_ih.data = torch.eye(self.inpt_size)[:self.d_model]

        self.h_linear.weight.data = self.init_h_mtx
        self.weight_hh = self.h_linear.weight
        self.inv = torch.inverse(self.weight_hh)

        self.bias_ih = torch.nn.Parameter(torch.zeros(1))
        self.bias_hh = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, h, *args, **kwargs):
        outs = self.inpt_linear(x)
        not_trig = outs[...,-1]>=-1
        idx = (h[...,-1]>0)&not_trig
        outs[idx] = outs[idx] + torch.matmul(h[idx], self.weight_hh)
        idx = (h[...,-1]<0)&not_trig
        outs[idx] = outs[idx] + torch.matmul(h[idx], self.inv)
        is_trig = ~not_trig
        outs[is_trig] = outs[is_trig] + h[is_trig]
        return outs

class HackyDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        fx = x.clone()
        fx[...,1:-1] = torch.abs(x[...,1:-1])
        return fx

class ToyOrthoRotRNN(ToyRNN):
    """
    This model works by encoding the count as the rotation about the
    z axis of a d dimensional vector. It encodes the phase both as the
    sign of the y dimensino and the sign of the z the z dimension.
    """
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)

    def reset_parameters(self):
        bos_id   = self.bos_id
        eos_id   = self.eos_id
        resp_id  = self.resp_id
        demo_ids = self.demo_ids
        tids = self.trigger_ids
        tlen = len(self.trigger_ids)

        ts = self.toy_scale
        bos_emb = torch.zeros(self.d_model) # Trigger Embedding
        bos_emb[0] = 1
        bos_emb[-1] = 1
        trig_emb = torch.zeros(self.d_model) # Trigger Embedding
        trig_emb[-1] = -2

        self.embeddings.weight.data.zero_()
        self.embeddings.weight.data[bos_id] = bos_emb
        for tid in tids:
            self.embeddings.weight.data[tid] = trig_emb

        if self.rnns[0] is None:
            self.rnns[0] = ToyOrthoRotCell(
                self.d_model,self.d_model)
        else:
            self.rnns[0].reset_parameters()

        if self.lm_head is None:
            self.decoder = HackyDecoder()
            self.lm_head = torch.nn.Linear(
                self.d_model, self.out_tokens, bias=True)
        self.lm_head.weight.data.zero_()
        self.lm_head.bias.data.zero_()

        big_num = self.max_count*ts
        demo_out = torch.ones(self.d_model)
        resp_out = torch.ones(self.d_model)
        resp_out[0] = -1
        resp_out[-1] = -1
        eos_out = torch.zeros(self.d_model)
        eos_out[0] = 1
        eos_out[-1] = -1

        self.lm_head.weight.data[eos_id] = eos_out
        self.lm_head.weight.data[resp_id] = resp_out
        for did in demo_ids:
            self.lm_head.weight.data[did] = demo_out
        if hasattr(self.lm_head, "bias") and self.lm_head.bias is not None:
            self.lm_head.bias.data.zero_()


class Transformer(smods.Transformer):
    """
    Uses trigger to determine when to switch to freeform prediction.
    """
    def __init__(self, trigger_ids=[7], *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Trigger Ids:", trigger_ids)
        if type(trigger_ids)==int:
            trigger_ids = [trigger_ids]
        self.register_buffer("trigger_ids", torch.LongTensor(
            [tid for tid in trigger_ids]
        ))

    def freedom_fwd(self,
                    inpts:torch.Tensor=None,
                    mask:torch.Tensor=None,
                    pad_mask:torch.Tensor=None,
                    task_mask:torch.Tensor=None,
                    input_ids:torch.Tensor=None,
                    attention_mask:torch.Tensor=None,
                    n_steps:int=0,
                    temperature=None,
                    inputs_embeds=None,
                    past_key_values=None,
                    stop_ids=None,
                    ret_gtruth=True,
                    output_attentions=False,
                    position_ids=None,
                    *args, **kwargs):
        """
        Arguments:
            inpts: Tensor, shape ``[bsize, seq_len]``
            mask: Tensor, shape ``[seq_len, seq_len]``
                true means padding, or unattended locations
            pad_mask: Tensor, shape ``[bsize, seq_len]``
                true means padding
            attention_mask: Tensor, shape ``[bsize, seq_len]``
                huggingface style padding mask where false means padding.
                True for positions that you want to attend to.
            task_mask: Tensor, shape ``[bsize, seq_len]``
                true means the tokens are a part of the prediction
                task and as such should not be teacher forced. Tokens at
                indices that are false will be fed into the model
                regardless of the model's predictions.
            n_steps: int
                the number of prediction steps if not using teacher
                forcing
            temperature: float
                a parameter to adjust the entropy of the
                token sampling. high temperature means high entropy
            inputs_embeds: tensor (B,S,E)
                optionally argue embeddings instead of token ids
            past_key_values: tuple of tuple of tensors
                the output of a huggingface cache. used to speed up
                computations. See huggingface documentation for more
                details
            stop_ids: set of ints
                the prediction loop will terminate if the model produces
                a token that is contained within stop_ids. The resulting
                return sequence will be the sequence including the stop
                id
            ret_gtruth: bool
                if true, will return the pred ids of the argued input
                instead of the predicted ids where the task mask is false.
                The predicted logits are returned the same either way.
                This is for ease of computing accuracy in random sequences.
            position_ids: None or LongTensor (S,)
                optionally argue the position ids for the positional
                encodings.
        Returns:
            ret_dict: dict
                "pred_ids": torch LongTensor (B,S+NSteps)
                "logits": torch FloatTensor (B,S+NSteps,NTokens)
                "past_key_values": None or tuple of tuple of tensors
                "last_hidden_state": torch FloatTensor (B,S+NSteps,E)
        """
        if input_ids is not None:
            inpts = input_ids
        if inpts is None:
            raise NotImplemented
            B,S = inputs_embeds.shape[:2]
        else:
            B,S = inpts.shape

        n_loops = S + n_steps

        if stop_ids is not None:
            if type(stop_ids)==int: stop_ids = [stop_ids]
            if len(stop_ids)>0:
                stop_ids = torch.LongTensor(list(stop_ids))
                stop_ids = stop_ids.to(self.get_device())
            else: stop_ids = None

        pred_ids = torch.zeros(
            (B,n_loops), device=self.get_device()
        ).long()

        logits = torch.zeros(
            (B,n_loops,self.n_tokens),
            device=self.get_device()
        )

        # Need to ensure we use the appropriate input type between
        # the inpts ids and the input embeddings
        if inpts is None:
            raise NotImplemented # use ids only for now
            inpt_emb = inputs_embeds[:,:1]
            inpt = None
        else:
            inpt = inpts[:,:1]
            inpt_emb = None

        # Masks
        if attention_mask is not None:
            pad_mask = ~attention_mask.bool()
        if pad_mask is not None:
            pad_mask = ~(pad_mask.bool())
            if pad_mask.shape[-1]<n_loops:
                p = n_loops - pad_mask.shape[-1]
                pad_mask = torch.nn.functional.pad(
                    pad_mask, (0, p), value=True
                )
        # Custom attention mask
        if mask is not None:
            mask = ~(mask.bool())
            if mask.shape[-1]<n_loops:
                p = n_loops - mask.shape[-1]
                mask = torch.nn.functional.pad(
                    mask, (0, p, 0, p), value=True
                )
            # if pad_mask is not None:
            #     pad_mask = padmask2attnmask(pad_mask)

        # Positional indices
        if position_ids is not None and position_ids.shape[-1]<S+n_steps:
            device = device_fxn(position_ids.get_device())
            p = S+n_steps - position_ids.shape[-1]
            new_positions = torch.arange(p).to(device).long()
            new_positions += position_ids[-1]
            position_ids = torch.cat([position_ids,new_positions],dim=-1)

        # Need to ensure the padding mask is the full length of the
        # past_key_values if past_key_values is not none
        p_end = 1
        if past_key_values is not None and past_key_values[0] is not None:
            # pkv idxs: layer, k or v, values
            p_end = past_key_values[0][0].shape[1]

        h_states = []
        attentions = None
        pids = None

        if task_mask is None:
            task_mask = torch.zeros_like(pad_mask).bool()
        else:
            if task_mask.shape[-1]<n_loops:
                t = n_loops - task_mask.shape[-1]
                task_mask = torch.nn.functional.pad(
                    task_mask, (0, t), value=True
                )
        all_hidden_states = []
        for step in range(n_loops):
            # attn and pad masks are currently inverted, 1 means do attend
            attn_mask = None
            if pad_mask is not None:
                e = p_end+step
                attn_mask = pad_mask[:,:e]
            if mask is not None:
                e = p_end+step
                if attn_mask is not None:
                    attn_mask = mask[:,:e,:e]&padmask2attnmask(attn_mask)
                else:
                    attn_mask = mask[:,:e,:e]
            if position_ids is not None:
                pids = position_ids[:e]

            output = self.encoder(
                input_ids=inpt,
                attention_mask=attn_mask,
                use_cache=True,
                past_key_values=past_key_values,
                inputs_embeds=inpt_emb,
                output_attentions=output_attentions,
            )
            for h in range(len(output.hidden_states)):
                if h >= len(all_hidden_states):
                    all_hidden_states.append([])
                all_hidden_states[h].append(output.hidden_states[h])
            past_key_values = output.past_key_values

            if output_attentions:
                if attentions is None:
                    attentions = [[layer] for layer in output.attentions]
                else:
                    for layer in range(len(output.attentions)):
                        attentions[layer].append(output.attentions[layer])

            if len(h_states)==0:
                h_states.append(output.last_hidden_state)
            else:
                h_states.append(output.last_hidden_state[:,-1:])
            if not hasattr(output, "logits"):
                state = h_states[-1][:,-1]
                pred = self.lm_head(self.decoder(state))
            else: pred = output.logits[:,-1]
            logits[:,step] = pred
            argmaxs = self.sample_with_temperature( pred, temperature )
            pred_ids[:,step] = argmaxs.squeeze()

            if step < n_loops-1:
                inpt_emb = None
                inpt = pred_ids[:,step:step+1].clone()

                if step+1<inpts.shape[1]:
                    tmask = task_mask[:,step+1]
                    if torch.any(~tmask):
                        if ret_gtruth:
                            pred_ids[~tmask,step] = inpts[~tmask,step+1]
                        inpt[~tmask] = inpts[~tmask,step+1:step+2]
                if stop_ids is not None and torch.isin(inpt, stop_ids):
                    logits = logits[:,:step+1]
                    pred_ids = pred_ids[:,:step+1]
                    break
        ret_dict = {
            "hidden_states": [torch.cat(h,dim=1) for h in all_hidden_states],
            "logits": logits,
            "pred_ids":  pred_ids,
            "past_key_values": past_key_values,
            "last_hidden_state": torch.cat(h_states, dim=1),
        }
        if output_attentions:
            for layer in range(len(attentions)):
                npad = attentions[layer][-1].shape[-1]
                for step in range(len(attentions[layer])):
                    padded = pad_to(attentions[layer][step],npad,dim=-1)
                    attentions[layer][step] = padded
                attentions[layer] = torch.cat(attentions[layer],dim=2)
            ret_dict["attentions"] = attentions
        return ret_dict

class KWindowTransformer(Transformer):
    """
    Uses an attention mask that only allows attending to the last
    k tokens +1 for self.
    """
    def __init__(self, attn_window=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_window = attn_window

    def tforce_fwd(self,
                    inpts:torch.Tensor=None,
                    input_ids:torch.Tensor=None,
                    mask:torch.Tensor=None,
                    past_key_values=None,
                    *args, **kwargs):
        if input_ids is not None:
            inpts = input_ids
        if inpts is None:
            S = inputs_embeds.shape[1]
        else:
            S = inpts.shape[1]
        if past_key_values is not None:
            S += past_key_values[0][0].shape[2]
        mask = generate_ktoken_causal_mask(S,k=self.attn_window)[None]
        mask = mask.to(self.get_device())
        return super().tforce_fwd(
            inpts=inpts,
            mask=mask,
            past_key_values=past_key_values,
            *args, **kwargs)

    def freedom_fwd(self,
                    inpts:torch.Tensor=None,
                    input_ids:torch.Tensor=None,
                    mask:torch.Tensor=None,
                    n_steps:int=0,
                    past_key_values=None,
                    *args, **kwargs):
        if input_ids is not None:
            inpts = input_ids
        if inpts is None:
            S = inputs_embeds.shape[1]
        else:
            S = inpts.shape[1]
        if past_key_values is not None:
            S += past_key_values[0][0].shape[2]
        S += n_steps
        mask = generate_ktoken_causal_mask(S,k=self.attn_window)[None]
        mask = mask.to(self.get_device())
        return super().freedom_fwd(
            inpts=inpts,
            mask=mask,
            pad_mask=pad_mask,
            task_mask=task_mask,
            n_steps=n_steps,
            past_key_values=past_key_values,
            *args, **kwargs)

class LossWrapper(torch.nn.Module):
    """
    This class wraps the model to keep the loss calculations distributed
    on all GPUs. Otherwise one gpu is overloaded with computational
    costs.
    """
    def __init__(self,
                 model,
                 config,
                 pad_id=0,
                 bos_id=1,
                 eos_id=2,
                 tokenizer=None,
                 loss_fxn=torch.nn.functional.cross_entropy,
                 *args, **kwargs):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        if self.tokenizer:
            pad_id = getattr(self.tokenizer, "pad_id", pad_id)
            bos_id = getattr(self.tokenizer, "bos_id", bos_id)
            eos_id = getattr(self.tokenizer, "eos_id", eos_id)
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.config = config
        self.label_smoothing = self.config.get("label_smoothing", 0)
        self.loss_scale = 1./self.config.get("n_grad_loops",1)
        self.loss_fxn = loss_fxn

    def forward(self,
                data,
                tforce=True,
                no_grad=False,
                temperature=None,
                top_k=5,
                reduce_metrics=True,
                output_attentions=False,
                *args, **kwargs):
        """
        Args:
            data: dict
                "input_ids": LongTensor (B,S1)
                    the token indices of the input sequence. The CMP
                    token should be appended to the end of each sentence.
                "input_pad_mask": BoolTensor (B,S1)
                    attention mask for padding purposes. trues mean
                    padding.
                "output_ids": LongTensor (B,S2)
                    the token indices of the target sequence. An EOS
                    token should be appended to the end of each sentence
                "output_pad_mask": BoolTensor (B,S1)
                    attention mask for padding purposes. trues mean
                    padding.
                "task_mask": BoolTensor (B,S1) 
                    an additional mask that uses 0s to denote positions
                    that should always be teacher forced and 1s as
                    positions that should be predicted during inference.
            ret_preds: bool
                if true, will return the predictions
            tforce: bool
                determines whether model should use teacher forcing for
                predictions or not.
            incl_intl_prob: bool
                if true, will include the initial problem in the loss.
                if false, will exclude initial problem from the loss.
            temperature: float
                a temperature parameter for softmax sampling. Set to
                low number for high confidence sampling, high value
                for low confidence sampling
            no_grad: bool
                if true, this function will not call .backward() on
                the loss. If false, this function will still only call
                .backward if in training mode.
            top_k: int optional
                if argued, returns a calculation of the top_k accuracy
            reduce_metrics: bool
                if true, loss and acc will be averaged over all samples.
                if false, loss and acc will be returned as tensors for
                each token prediction
            output_attentions: bool
                if true, will return the attention weights
        Returns:
            ret_dict: dict (keys: str, vals: torch tensor)
                "loss": torch tensor (1,) or (B,)
                "acc": torch tensor (1,) or (B,)
                    the raw accuracy for the non-rmb task
                "preds": torch tensor (B,S,P)
                    the prediction logits. only returned if ret_preds is
                    true
        """
        ret_dict = dict()
        pad_id = self.pad_id
        bos_id = self.bos_id
        eos_id = self.eos_id
        if "input_pad_mask" not in data:
            inpt_pad_mask = (data["input_ids"]==pad_id)
            inpt_pad_mask = inpt_pad_mask|(data["input_ids"]==eos_id)
        else: inpt_pad_mask = data["input_pad_mask"].clone()
        if "output_pad_mask" not in data:
            out_pad_mask = data["output_ids"]==pad_id
            out_pad_mask = out_pad_mask==bos_id
        else: out_pad_mask = data["output_pad_mask"].clone()
        if self.config.get("concat",False):
            inpt_pad_mask = torch.zeros_like(data["input_ids"]).bool()
            out_pad_mask = torch.zeros_like(data["output_ids"]).bool()

        # The task mask is a tool to teacher force select portions of
        # the input. 0's and falses denote teacher forced inputs, 1s
        # denote portions that should not be teacher forced.
        if "task_mask" not in data:
            inpt_task_mask = torch.ones_like(inpt_pad_mask)
            out_task_mask = torch.ones_like(inpt_pad_mask)
        else:
            inpt_task_mask = data["task_mask"][:,:-1]
            out_task_mask = data["task_mask"][:,1:]

        # Predictions
        inpts = data["input_ids"]
        out_ids = data["output_ids"]
        tot_len = data["output_ids"].shape[-1]-inpts.shape[-1]

        device = self.model.get_device()
        if inpts.get_device()!=self.model.get_device():
            inpts = inpts.to(device)
            inpt_pad_mask = inpt_pad_mask.to(device)
            inpt_task_mask = inpt_task_mask.to(device)
            out_ids = out_ids.to(device)
            out_pad_mask = out_pad_mask.to(device)
            out_task_mask = out_task_mask.to(device)

        ret_dict = self.model(
            inpts,
            pad_mask=inpt_pad_mask,
            task_mask=inpt_task_mask, #ignored if tforce is true
            tforce=tforce,
            n_steps=tot_len,
            temperature=temperature,
        )

        ## Loss
        #################################
        inpt_mask = ~inpt_pad_mask.reshape(-1)
        out_mask =  ~out_pad_mask.reshape(-1)
        logits = ret_dict["logits"]
        ps = logits.reshape(
            -1, logits.shape[-1]
        )[inpt_mask]
        labels = out_ids.reshape(-1)[out_mask]
        try:
            loss = self.loss_scale*self.loss_fxn(
                ps,labels,
                reduction="none",
                label_smoothing=self.label_smoothing
            )
        except:
            self.print_data(
              data,inpt_pad_mask=inpt_pad_mask,out_pad_mask=out_pad_mask
            )
            assert False

        if not reduce_metrics:
            temp = torch.zeros_like(out_ids).float()
            temp[out_mask.reshape(out_ids.shape)] = loss
            loss = temp
        else:
            loss = loss.mean()
        ret_dict["loss"] = loss

        ## Acc
        #################################
        if "pred_ids" in ret_dict:
            pred_ids = ret_dict["pred_ids"]
            pred_ids = pred_ids.reshape(-1)
        else:
            pred_ids = torch.argmax(ps, dim=-1)
            ret_dict["pred_ids"] = torch.argmax(logits, dim=-1)
        acc = (pred_ids.to(device)[inpt_mask]==labels).float()
        if reduce_metrics: acc = acc.mean()
        else: 
            temp = torch.zeros_like(out_ids).float()
            temp[out_mask.reshape(out_ids.shape)] = acc.long()
            acc = temp
        ret_dict["acc"] = acc

        ## Corrects
        output_ids = data["output_ids"].to(device)
        pred_ids = pred_ids.reshape(output_ids.shape).to(device)
        tmask = out_task_mask
        corrects = torch.ones_like(output_ids).long()
        corrects[tmask] = (pred_ids[tmask]==output_ids[tmask]).long()
        if self.config.get("concat", False) and tforce:
            ret_dict["corrects"] = corrects.sum(-1)/corrects.shape[-1]
        else:
            ret_dict["corrects"] = corrects.sum(-1)==corrects.shape[-1]

        ret_dict["top_k"] = top_k_acc(
            logits, out_ids, top_k, as_tensor=True
        )
        return ret_dict
    
    def print_data(self, data, inpt_pad_mask, out_pad_mask):
        if not self.tokenizer: self.tokenizer = EmptyTokenizer()
        for i in range(len(data["input_ids"])):
            print()
            print("Full inpt:",
              self.tokenizer.decode(data["input_ids"][i]))
            print("Full Outpt:",
              self.tokenizer.decode(data["output_ids"][i]))
            print("dropped inpt:",
              self.tokenizer.decode(
                data["input_ids"][i].cpu()[inpt_pad_mask[i].cpu()]))
            print("dropped out:",
              self.tokenizer.decode(
                data["output_ids"][i].cpu()[out_pad_mask[i].cpu()]))
            print("post inpt:",
              self.tokenizer.decode(
                data["input_ids"][i].cpu()[~inpt_pad_mask[i].cpu()]))
            print("post out:",
              self.tokenizer.decode(
                data["output_ids"][i].cpu()[~out_pad_mask[i].cpu()]))

        idx = inpt_pad_mask.float().sum(-1).cpu()!=out_pad_mask.float().sum(-1).cpu()
        print()
        print()
        print()
        print()
        data = {k: v.cpu() for k,v in data}
        inpt_pad_mask = inpt_pad_mask.cpu()
        out_pad_mask = out_pad_mask.cpu()
        for i in range(idx.long().sum(-1)):
            print("Full inpt:",
              self.tokenizer.decode(data["input_ids"][idx][i]))
            print("Full Outpt:",
              self.tokenizer.decode(data["output_ids"][idx][i]))
            print("dropped inpt:",
              self.tokenizer.decode(
                data["input_ids"][idx][i][inpt_pad_mask[idx][i]]))
            print("dropped out:",
              self.tokenizer.decode(
                data["output_ids"][idx][i][out_pad_mask[idx][i]]))
            print("post inpt:",
              self.tokenizer.decode(
                data["input_ids"][idx][i][~inpt_pad_mask[idx][i]]))
            print("post out:",
              self.tokenizer.decode(
                data["output_ids"][idx][i][~out_pad_mask[idx][i]]))

class EmptyTokenizer:
    def __init__(self):
        pass
    def decode(self, x):
        return x

def make_model(config):
    config["model_type"] = config.get("model_type","GRU")
    config["n_layers"] = config.get("n_layers",1)
    model = globals()[config["model_type"]](**config)
    return LossWrapper(model=model, config=config, **config)


from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import dl_utils.torch_modules as tmods
from .utils import (
    generate_square_subsequent_mask, arglast, top_k_acc,
    update_shape, padmask2attnmask, pad_to
)
import math

try:
    from mamba_ssm import Mamba
    from mamba_ssm.utils.generation import InferenceParams
except:
    print("You must install mamba")

try:
    from transformers import (
        CONFIG_MAPPING,
        GPT2Config,
        AutoModelForCausalLM,
        OpenAIGPTConfig,
        GPTJConfig,
        LlamaConfig,
        TransfoXLConfig,
    )
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    from transformers.modeling_attn_mask_utils import (
        _prepare_4d_causal_attention_mask
    )
except:
    print("Failed to import causal attention mask util")

try:
    from transformers.modeling_outputs import BaseModelOutputWithPast
except:
    from .torch_modules import BaseModelOutputWithPast

DEVICES = {
    -1: "cpu", **{i:i for i in range(10)}
}

class SequenceModule(tmods.CoreModule):
    def __init__(self,
                n_tokens: int=None,
                out_tokens: int=None,
                d_model:int=128,
                n_heads:int=4,
                h_mult:int=4,
                h_norm:bool=False,
                l_norm:bool=False,
                n_layers:int=3,
                norm_first:bool=True,
                drop_p:float=0.5,
                max_posencs:int=1000,
                posenc_drop_p:float=None,
                learn_posencs:bool=True,
                actv_fxn:str="gelu",
                scale_attn_weights:bool=True,
                scale_by_inv_layer:bool=True,
                reorder_and_upcast:bool=False,
                hf_model_type:str="llama",
                pretrained:bool=False,
                init_range:float=0.1,
                seq_len:int=2048,
                #add_zero_attn:bool=False,
                *args, **kwargs):
        """
        n_tokens: int
            the number of tokens for the embedding layer. if None,
            no embedding layer is used.
        out_tokens: int
            Determines the size of the output predictions
        d_model: int
            the number of dimensions for the latent vectors
        n_heads: int
            the number of attention heads
        h_mult: int
            a multiplier to determine the hidden dimensionality of the
            feed forward networks in the model.
        h_norm: bool
            if true, will use a layer norm on the lstm hidden state
        l_norm: bool
            if true, will include layer norms where appropriate
        n_layers: int
            the number of transformer layers
        norm_first: bool
            if true, applies layer norm before the operations in the
            encoder layer (this seemed to be better in some paper I
            can't remember the name of)
        drop_p: float
            the dropout probability. 0 means no dropout.
        max_posencs: int
            the number of possible embeddings. If
        posenc_drop_p: float optional
            the dropout probability for positional encodings. 0 means
            no dropout. defaults to drop_p if none
        learn_posencs: bool
            determines whether or not gradients are backpropagated into
            the positional encodings.
        actv_fxn: str
            the transformer activation function
        hf_model_type: str
            the huggingface transformer base. only applies if using
            HFModel types. Specifies the hf model base type.
        scale_attn_weights: bool
            scale attention weights by dividing by sqrt(hidden_size)
        scale_by_inv_layer: bool
            scale attention weights by inverse layer index. see
            huggingface docs for details
        reorder_and_upcast: bool
            reorder and upcast attention. see huggingface docs for details
        pretrained: bool
            if true, will ignore model specs and use a pretrained
            huggingface model. only applies if using HF model types.
        init_range: float
            a lower and upper bound for uniform weight sampling for
            the embeddings and output dense layer.
        # Commented out largely due to poorer performance
        #add_zero_attn: bool
        #    if true, will allow the model to learn zero attention,
        #    essentially allowing for a residual connection to add
        #    nothing to the persistent representation.
        """
        super().__init__()
        self.n_tokens = n_tokens
        self.out_tokens = out_tokens
        if self.out_tokens is None: self.out_tokens = self.n_tokens
        self.d_model = d_model
        self.n_heads = n_heads
        self.h_mult = h_mult
        self.h_norm = h_norm
        self.l_norm = l_norm
        self.n_layers = n_layers
        self.drop_p = drop_p
        self.norm_first = norm_first
        self.max_posencs = max_posencs
        self.posenc_drop_p = posenc_drop_p
        if self.posenc_drop_p is None:
            self.posenc_drop_p = drop_p
        self.learn_posencs = learn_posencs
        self.actv_fxn = actv_fxn
        self.hf_model_type = hf_model_type
        self.scale_attn_weights = scale_attn_weights
        self.scale_by_inv_layer = scale_by_inv_layer
        self.reorder_and_upcast = reorder_and_upcast
        self.pretrained = pretrained
        self.init_range = init_range
        self.seq_len = seq_len
        #self.add_zero_attn = add_zero_attn

    def init_weights(self, init_range=0.1) -> None:
        pass
        #initrange = 0.1
        #if hasattr(self, "embeddings"):
        #    self.embeddings.weight.data.uniform_(-initrange, initrange)
        #self.lm_head.bias.data.zero_()
        #self.lm_head.weight.data.uniform_(-initrange, initrange)

    def get_config(self):
        """
        Finds the appropirate configuration when using Huggingface
        models.
        """
        n_tokens = self.n_tokens if self.n_tokens else self.out_tokens
        d_hid = self.h_mult*self.d_model
        config_kwargs = {
            "vocab_size": n_tokens,
            "hidden_size": self.d_model,
            "intermediate_size": d_hid,
            "num_hidden_layers": self.n_layers,
            "num_attention_heads": self.n_heads,
            "num_key_value_heads": self.n_heads,
            "hidden_act": self.actv_fxn,
            "n_positions": self.max_posencs,
            "rotary_dim": self.d_model//self.n_heads,
            "rope_theta": self.d_model//self.n_heads,
            "n_ctx": self.max_posencs,
            "n_embd": self.d_model,
            "n_head": self.n_heads,
            "n_inner": d_hid,
            "activation_function": self.actv_fxn,
            "resid_pdrop": self.drop_p,
            "embd_pdrop":  0,
            "attn_pdrop":  self.drop_p,
            "scale_attn_weights": self.scale_attn_weights,
            "scale_attn_by_inverse_layer_idx": self.scale_by_inv_layer,
            "tie_word_embeddings": False,
            "torch_dtype": "float32",
            "reorder_and_upcast_attn": self.reorder_and_upcast,
            "add_cross_attention": False,
            "max_position_embeddings": max(2048, self.seq_len),
        }
        if self.hf_model_type=="gpt2":
            config = GPT2Config()
        elif self.hf_model_type == "gptj":
            config = GPTJConfig()
        elif self.hf_model_type == "llama":
            config = LlamaConfig()
        elif self.hf_model_type == "transxl":
            config = TransfoXLConfig()
        config.update(config_kwargs)
        return config

    def forward(self, inpts:torch.Tensor=None,
                      mask:torch.Tensor=None,
                      pad_mask:torch.Tensor=None,
                      is_causal:bool=None,
                      tforce:bool=True,
                      n_steps:int=0,
                      temperature=None,
                      inputs_embeds:torch.Tensor=None,
                      input_ids=None,
                      attention_mask:torch.Tensor=None,
                      use_cache=False,
                      past_key_values=None,
                      stop_ids=None,
                      output_attentions=False,
                      *args, **kwargs):
        """
        Arguments:
            inpts: Tensor, shape ``[bsize, seq_len]``
            mask: Tensor, shape ``[seq_len, seq_len]``
                an optional attention mask. if transformer uses auto-
                regressive prediction, simply mark `is_causal` to true
                and leave this field equal to None.
            pad_mask: Tensor, shape ``[bsize, seq_len]``
                true means padding
            is_causal: bool
                If specified, applies a causal mask as mask (optional)
                and ignores attn_mask for computing scaled dot product
                attention.
            tforce: bool
                determines whether or not to teacherforce
            n_steps: int
                the number of prediction steps if not using teacher
                forcing
            temperature: float
                a parameter to adjust the entropy of the
                token sampling. high temperature means high entropy
            inputs_embeds: None or FloatTensor (B,S,E)
                optionally argue input embeddings instead of token ids.
            input_ids: torch LongTensor (B,S)
                optionally use different keyword
            attention_mask: bool tensor (B,S)
                pad mask except that 0s denote padding tokens
            past_key_values: tuple of tuple of tensors
                if use_cache is true, will return saved computations
                that can be argued on the next pass to save on
                computational complexity
            stop_ids: set of ints
                the prediction loop will terminate if the model produces
                a token that is contained within stop_ids. The resulting
                return sequence will be the sequence including the stop
                id
            output_attentions: bool
                if true, will return the unscaled attention weights
        Returns:
            ret_dict: dict
                if tforce:
                    "pred_ids": torch LongTensor (B,S)
                    "logits": torch FloatTensor (B,S,NTokens)
                else:
                    "pred_ids": torch LongTensor (B,S+NSteps)
                    "logits": torch FloatTensor (B,S+NSteps,NTokens)
                "past_key_values": None or tuple of tuple of tensors
        """
        if input_ids is not None:
            inpts = input_ids
        if attention_mask is not None:
            pad_mask = ~attention_mask
        if pad_mask is None:
            if past_key_values is None or past_key_values[0] is None:
                if inpts is not None:
                    if len(inpts.shape)==2:
                        pad_mask = torch.zeros_like(inpts).bool()
                    else:
                        pad_mask = torch.zeros(
                            inpts.shape[:2], device=self.get_device()
                        ).bool()
                else:
                    pad_mask = torch.zeros(
                        inputs_embeds.shape[:2]
                    ).bool().to( self.get_device() )

        ##### TODO: DELETME
        ##Right now adding a custom mask for debugging purposes
        if mask is None:
            if inpts is None:
                S = inputs_embeds.shape[1]
            else:
                S = inpts.shape[1]
            if past_key_values is not None:
                S += past_key_values[0][0].shape[2]
            if not tforce:
                S += n_steps
            mask = generate_square_subsequent_mask(S)[None]
            mask = mask.to(self.get_device())
        ####pad_mask = None
        #### TODO: END DELETE

        if tforce:
            ret_dict = self.tforce_fwd(
                inpts=inpts,
                mask=mask,
                pad_mask=pad_mask,
                is_causal=is_causal,
                use_cache=use_cache,
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                temperature=temperature,
                output_attentions=output_attentions,
                *args, **kwargs,
            )
        else:
            ret_dict = self.freedom_fwd(
                inpts=inpts,
                mask=mask,
                pad_mask=pad_mask,
                is_causal=is_causal,
                n_steps=n_steps,
                temperature=temperature,
                use_cache=use_cache,
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                stop_ids=stop_ids,
                output_attentions=output_attentions,
                *args, **kwargs,
            )
        return ret_dict

    def generate(self, *args, **kwargs):
        if "max_new_tokens" in kwargs:
            kwargs["n_steps"] = kwargs["max_new_tokens"]
        return self.freedom_fwd(*args, **kwargs)

class RNN(SequenceModule):
    def __init__(self, rnn_type="RNNCell", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = "RNN"
        if hasattr(torch.nn, rnn_type):
            self.rnn_type = getattr(torch.nn, rnn_type)
        else:
            self.rnn_type = getattr(tmods, rnn_type)

        if self.n_tokens:
            self.embeddings = torch.nn.Embedding(
                self.n_tokens, self.d_model
            )
        if self.l_norm:
            self.pre_norm = torch.nn.LayerNorm(self.d_model)
        self.rnns = torch.nn.ModuleList([])
        if self.l_norm or self.h_norm:
            self.layer_norms = torch.nn.ModuleList([])
        for i in range(self.n_layers):
            if self.l_norm or self.h_norm:
                self.layer_norms.append(torch.nn.LayerNorm(self.d_model))
            if hasattr(torch.nn, rnn_type):
                rnn = self.rnn_type(self.d_model, self.d_model)
            else:
                rwargs = {
                    **kwargs,
                    "inpt_size": self.d_model,
                    "d_model": self.d_model}
                rnn = self.rnn_type(**rwargs)
            self.rnns.append(rnn)
        d_hid = self.d_model*4
        modules = []
        modules.append(torch.nn.Linear( self.d_model, d_hid ))
        modules.append(torch.nn.GELU())
        if self.l_norm:
            modules.append(torch.nn.LayerNorm(d_hid))
        modules.append(torch.nn.Dropout(self.drop_p))
        self.decoder = torch.nn.Sequential( *modules )
        self.lm_head = torch.nn.Linear( d_hid, self.out_tokens )

    def get_fresh_recurrent_vectors(self, batch_size=1):
        """
        Args:
            batch_size: int
        Returns:
            hs: list of tensors with shape (B, H)
                the length of the list is determined by the number of lstms
                in the model.
            cs: list of tensors with shape (B, H)
                the length of the list is determined by the number of lstms
                in the model.
        """
        n = self.n_layers
        hs = [torch.zeros(batch_size,rnn.hidden_size) for rnn in self.rnns]
        d = self.get_device()
        return [h.to(d) for h in hs]
    
    def step(self,
             inpts=None,
             pad_mask=None,
             hs=None,
             temperature=None,
             prev_logits=None,
             inputs_embeds=None,
             *args, **kwargs):
        """
        Arguments:
            inpts: Tensor, shape ``[bsize]``
                if None, inputs_embeds must be not None
            pad_mask: BoolTensor, shape ``[bsize]``
                1s/Trues denote padding, 0s/Falses denote not padding
            hs: list of Tensors with shape (B, D)
                a list of h vectors for each lstm
            temperature: float
                a parameter to adjust the entropy of the
                token sampling. high temperature means high entropy
            prev_logits: torch Tensor [bsize, n_tokens] or None
                optionally argue the previous logits as a vector to
                contain the most recent predictions
            inputs_embeds: None or Tensor, shape (B,D)
                optionally argue the embeddings directly instead of
                token ids.
        Returns:
            dict:
                logits: Tensor of shape (B, N)
                pred_ids: Tensor of shape (B,)
                hs: list of Tensors with shape (B, D)
                    a list of updated h vectors for each lstm
        """
        if inpts is None:
            B = inputs_embeds.shape[0]
            pred_ids = torch.zeros(B,device=self.get_device()).long()
        else:
            B = inpts.shape[0]
            pred_ids = inpts.detach().data.clone()

        if pad_mask is None:
            idx = torch.ones(B, device=self.get_device()).bool()
        else:
            idx = ~pad_mask.bool()

        if prev_logits is None:
            # will be used to store the predicted logits
            logits = torch.zeros(B,self.out_tokens).to(self.get_device())
        else:
            logits = prev_logits.detach().data.clone()

        if inputs_embeds is None:
            inpt = self.embeddings(inpts)
        else: inpt = inputs_embeds
        
        new_hs = [] # h.clone() for h in hs ]
        if self.l_norm:
            inpt = self.pre_norm(inpt)
        if len(inpt)>0:
            # Loop through rnn layers of model
            for l in range(len(self.rnns)):
                h = self.step_core(
                    inpt=inpt, h=hs[l], layer=l, idx=idx)
                inpt = h
                new_hs.append(h)
            logits[idx]   = self.lm_head(self.decoder(inpt[idx]))
            pred_ids[idx] = self.sample_with_temperature(
                logits[idx].data, temperature
            )
        return {
            "logits":   logits,
            "pred_ids": pred_ids,
            "hs": new_hs, 
        }

    def step_core(self, inpt, h, layer, idx):
        hcopy = h.clone()
        h = self.rnns[layer](inpt[idx], h[idx])
        if self.h_norm or self.l_norm:
            h = self.layer_norms[layer](h)
        hcopy[idx] = h
        return hcopy

    def forward(self, inpts:torch.Tensor=None,
                      pad_mask:torch.Tensor=None,
                      n_steps:int=0,
                      temperature=None,
                      inputs_embeds=None,
                      input_ids=None,
                      attention_mask=None,
                      stop_ids=None,
                      *args, **kwargs):
        """
        Arguments:
            inpts: Tensor, shape ``[bsize, seq_len]``
            pad_mask: Tensor, shape ``[bsize, seq_len]``
                true means padding
            n_steps: int
                the number of prediction steps if not using teacher
                forcing
            temperature: float
                a parameter to adjust the entropy of the
                token sampling. high temperature means high entropy
            inputs_embeds: None or Tensor, shape (B,S,D)
                optionally argue the embeddings directly instead of
                token ids.
            stop_ids: set of ints
                the prediction loop will terminate if the model produces
                a token that is contained within stop_ids. The resulting
                return sequence will be the sequence including the stop
                id
            input_ids: long tensor (B,S)
            attention_mask: bool tensor (B,S)
                pad mask except that 0s denote padding tokens
        Returns:
            ret_dict: dict
                logits: Tensor of shape ``[bsize,seq_len+n_steps,n_tokens]``
                pred_ids: Tensor of shape ``[bsize,seq_len+n_steps,n_tokens]``
        """
        if input_ids is not None:
            inpts = input_ids
        input_ids = inpts
        if stop_ids is not None:
            if type(stop_ids)==int: stop_ids = [stop_ids]
            if len(stop_ids)>0:
                stop_ids = torch.LongTensor(list(stop_ids))
                stop_ids = stop_ids.to(self.get_device())
            else: stop_ids = None
        else: stop_ids = None
        if not inputs_embeds:
            embs = self.embeddings(inpts)
        else: embs = inputs_embeds

        # TODO this if block needs testing
        if attention_mask is not None:
            pad_mask = ~attention_mask.bool()
        if pad_mask is None:
            pad_mask = torch.zeros(
                embs.shape[:2],device=self.get_device()).bool()

        B,S,D = embs.shape
        logits = []
        pred_ids = []
        hs = self.get_fresh_recurrent_vectors(B)

        # Loop through sequence
        for step in range(S+n_steps):
            if step<embs.shape[1]:
                pmask = pad_mask[:,step]
                inpt = embs[:,step]
            else:
                inpt = self.embeddings(pred_ids[-1])
            ret_dict = self.step(
                inputs_embeds=inpt,
                pad_mask=pmask,
                hs=hs,
                temperature=temperature
            )
            hs = ret_dict["hs"]
            logits.append(ret_dict["logits"])
            pred_ids.append(ret_dict["pred_ids"])
            if stop_ids is not None and torch.isin(pred_ids[-1],stop_ids):
                break
        return {
            "logits": torch.stack(logits, dim=1),
            "pred_ids": torch.stack(pred_ids,dim=1),
            "hs": hs
        }

class LinearRNN(RNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d_hid = self.d_model*4
        modules = []
        modules.append(torch.nn.Linear( self.d_model, d_hid ))
        if self.l_norm:
            modules.append(torch.nn.LayerNorm(d_hid))
        modules.append(torch.nn.Dropout(self.drop_p))
        self.decoder = torch.nn.Sequential( *modules )

class GRU(RNN):
    def __init__(self, rnn_type="GRUCell", *args, **kwargs):
        super().__init__(*args, rnn_type=rnn_type, **kwargs)
        self.model_type = "GRU"

class LSTM(RNN):
    def __init__(self, rnn_type="LSTMCell", *args, **kwargs):
        super().__init__(*args, rnn_type=rnn_type, **kwargs)
        self.model_type = 'LSTM'

    def get_fresh_recurrent_vectors(self, batch_size=1):
        """
        Args:
            batch_size: int
        Returns:
            hs: list of tensors with shape (B, H)
                the length of the list is determined by the number of lstms
                in the model.
            cs: list of tensors with shape (B, H)
                the length of the list is determined by the number of lstms
                in the model.
        """
        hs = super().get_fresh_recurrent_vectors(batch_size=batch_size)
        cs = super().get_fresh_recurrent_vectors(batch_size=batch_size)
        return hs,cs
    
    def step(self,
             inpts=None,
             pad_mask=None,
             hs=None,
             cs=None,
             temperature=None,
             prev_logits=None,
             inputs_embeds=None,
             *args, **kwargs):
        """
        Arguments:
            inpts: Tensor, shape ``[bsize]``
                if None, inputs_embeds must be not None
            pad_mask: BoolTensor, shape ``[bsize]``
                1s/Trues denote padding, 0s/Falses denote not padding
            hs: list of Tensors with shape (B, D)
                a list of h vectors for each lstm
            cs: list of Tensors with shape (B, D)
                a list of c vectors for each lstm
            temperature: float
                a parameter to adjust the entropy of the
                token sampling. high temperature means high entropy
            prev_logits: torch Tensor [bsize, n_tokens] or None
                optionally argue the previous logits as a vector to
                contain the most recent predictions
            inputs_embeds: None or Tensor, shape (B,D)
                optionally argue the embeddings directly instead of
                token ids.
        Returns:
            dict:
                logits: Tensor of shape (B, N)
                pred_ids: Tensor of shape (B,)
                hs: list of Tensors with shape (B, D)
                    a list of updated h vectors for each lstm
                cs: list of Tensors with shape (B, D)
                    a list of updated c vectors for each lstm
        """
        if inpts is None:
            B = inputs_embeds.shape[0]
            pred_ids = torch.zeros(B,device=self.get_device()).long()
        else:
            B = inpts.shape[0]
            pred_ids = inpts.detach().data.clone()

        if pad_mask is None:
            idx = torch.zeros(B, device=self.get_device()).bool()
        else:
            idx = ~pad_mask.bool()

        if prev_logits is None:
            # will be used to store the predicted logits
            logits = torch.zeros(B,self.out_tokens).to(self.get_device())
        else:
            logits = prev_logits.detach().data.clone()

        if inputs_embeds is None:
            inpt = self.embeddings(inpts)
        else: inpt = inputs_embeds
        
        new_hs = [] #h.clone() for h in hs ]
        new_cs = [] #c.clone() for c in cs ]
        if self.l_norm:
            inpt = self.pre_norm(inpt)
        if len(inpt)>0:
            # Loop through lstm layers of model
            for l in range(len(self.rnns)):
                (h,c) = self.step_core(
                    inpt=inpt,
                    h=(hs[l],cs[l]),
                    layer=l,
                    idx=idx)
                inpt = h
                new_hs.append(h)
                new_cs.append(c)
            logits[idx] = self.lm_head(self.decoder(inpt[idx]))
            pred_ids[idx] = self.sample_with_temperature(
                logits[idx], temperature
            )
        return {
            "logits":   logits,
            "pred_ids": pred_ids,
            "hs": new_hs, 
            "cs": new_cs,
        }

    def step_core(self, inpt, h, layer, idx):
        """
        This function is meant to allow for easier DAS interventions
        """
        hcopy = [h[0].clone(), h[1].clone()]
        h = self.rnns[layer](inpt[idx], (h[0][idx], h[1][idx]))
        if self.h_norm or self.l_norm:
            h,c = h
            h = self.layer_norms[layer](h)
            h = (h,c)
        hcopy[0][idx] = h[0]
        hcopy[1][idx] = h[1]
        return hcopy

    def forward(self, inpts:torch.Tensor=None,
                      pad_mask:torch.Tensor=None,
                      n_steps:int=0,
                      temperature=None,
                      inputs_embeds=None,
                      input_ids=None,
                      attention_mask=None,
                      stop_ids=None,
                      *args, **kwargs):
        """
        Arguments:
            inpts: Tensor, shape ``[bsize, seq_len]``
            pad_mask: Tensor, shape ``[bsize, seq_len]``
                true means padding
            n_steps: int
                the number of prediction steps if not using teacher
                forcing
            temperature: float
                a parameter to adjust the entropy of the
                token sampling. high temperature means high entropy
            inputs_embeds: None or Tensor, shape (B,S,D)
                optionally argue the embeddings directly instead of
                token ids.
            stop_ids: set of ints
                the prediction loop will terminate if the model produces
                a token that is contained within stop_ids. The resulting
                return sequence will be the sequence including the stop
                id
            input_ids: long tensor (B,S)
            attention_mask: bool tensor (B,S)
                pad mask except that 0s denote padding tokens
        Returns:
            ret_dict: dict
                logits: Tensor of shape ``[bsize,seq_len+n_steps,n_tokens]``
                pred_ids: Tensor of shape ``[bsize,seq_len+n_steps,n_tokens]``
        """
        if input_ids is not None:
            inpts = input_ids
        input_ids = inpts
        if stop_ids is not None:
            if type(stop_ids)==int: stop_ids = [stop_ids]
            if len(stop_ids)>0:
                stop_ids = torch.LongTensor(list(stop_ids))
                stop_ids = stop_ids.to(self.get_device())
            else: stop_ids = None
        else: stop_ids = None
        if not inputs_embeds:
            embs = self.embeddings(inpts)
        else: embs = inputs_embeds

        if attention_mask is not None:
            pad_mask = ~attention_mask.bool()
        if pad_mask is None:
            pad_mask = torch.zeros(
                embs.shape[:2],device=self.get_device()).bool()


        B,S,D = embs.shape
        logits = []
        pred_ids = []
        hs,cs = self.get_fresh_recurrent_vectors(B)

        # Loop through sequence
        for step in range(S+n_steps):
            if step<embs.shape[1]:
                pmask = pad_mask[:,step]
                inpt = embs[:,step]
            else:
                inpt = self.embeddings(pred_ids[-1])
            ret_dict = self.step(
                inputs_embeds=inpt,
                pad_mask=pmask,
                hs=hs, cs=cs,
                temperature=temperature
            )
            hs,cs = ret_dict["hs"], ret_dict["cs"]
            logits.append(ret_dict["logits"])
            pred_ids.append(ret_dict["pred_ids"])
            if stop_ids is not None and torch.isin(pred_ids[-1], stop_ids):
                break
        return {
            "logits": torch.stack(logits, dim=1),
            "pred_ids": torch.stack(pred_ids,dim=1),
            "hs": hs, "cs": cs
        }


class Transformer(SequenceModule):
    """
    This is a vanilla transformer that uses a PyTorch backend with the
    ability to only compute the most recent token if desired. This
    model has a lower chance of becoming deprecated.
    """
    def __init__(self,
                encoder_layer_class="RotaryEncoderLayer", # SimpleEncoderLayer
                pos_enc_class="IdentityPositionalEncoding",
                *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = 'Transformer'
        encoder_layer_class = getattr(tmods, encoder_layer_class)
        pos_enc_class = getattr(tmods, pos_enc_class)
        self.embeddings = torch.nn.Embedding(self.n_tokens,self.d_model)
        if self.seq_len is None: self.seq_len = 2048
        if "rot_dim" in kwargs: rot_dim = kwargs["rot_dim"]
        else: rot_dim = int(self.d_model//self.n_heads)
        self.pos_encs = pos_enc_class(
            d_model=self.d_model,
            max_len=max(2048, self.seq_len),
            learnable=self.learn_posencs,
        )
        self.layers = torch.nn.ModuleList([])
        for el in range(self.n_layers):
            self.layers.append(encoder_layer_class(
                d_model=self.d_model,
                nhead=self.n_heads,
                rot_dim=rot_dim,
                dim_feedforward=self.d_model*self.h_mult,
                dropout=self.drop_p,
                activation=F.gelu,
                batch_first=True,
                norm_first=True,
                llama=kwargs.get("llama", False),
            ))
        self.decoder = nn.LayerNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, self.n_tokens)
        self.init_weights()

    def prep_encoder_mask(self,
            S=None,
            attention_mask=None,
            pad_mask=None,
            is_causal=True,
        ):
        """
        This function is the highest level of abstraction in this
        class for manually combining attention masks.

        Args:
            S: int
                size of the sequence length
            attention_mask: BoolTensor (B,S,S) or (S,S)
                true values mean do attend to
            pad_mask: BoolTensor (B,S)
                true values mean do attend to (false means padding ids)
            is_causal: bool
                if true, will apply a causal mask if `attention_mask`
                is None.
        Returns:
            attn_mask: bool tensor
                true values mean do attend to
        """
        attn_mask = None
        if attention_mask is not None:
            if attention_mask.dtype==torch.bool:
                attn_mask = attention_mask
            else:
                attn_mask = attention_mask==0
        if attn_mask is None and is_causal:
            attn_mask = generate_square_subsequent_mask(S)
            attn_mask = ~attn_mask.to(self.get_device())
        if pad_mask is not None:
            pad_mask = pad_mask
            if attn_mask is None:
                attn_mask = pad_mask
            else:
                attn_mask = self.prep_mask(
                    mask=~attn_mask,
                    pad_mask=~pad_mask,)
        if len(attn_mask.shape)>2:
            attn_mask = torch.repeat_interleave(
                input=attn_mask,
                repeats=self.n_heads,
                dim=0,
            )
        og_shape = attn_mask.shape
        attn_mask = attn_mask.reshape(-1,og_shape[-1])
        will_be_nan = attn_mask.long().sum(-1)==0
        if torch.any(will_be_nan):
            attn_mask[will_be_nan] = 1
        attn_mask = attn_mask.reshape(og_shape)
        return attn_mask

    def encoder(self,
                input_ids=None,
                attention_mask=None,
                pad_mask=None,
                use_cache=False,
                past_key_values=None,
                inputs_embeds=None,
                position_ids=None,
                output_attentions=False,
                is_causal=True,
                *args, **kwargs):
        """
        Arguments:
            input_ids: Long Tensor, shape ``[bsize, seq_len]``
                the input ids. one of this or inputs_embeds must be not
                None
            attention_mask: Tensor, shape (B,S,S) or (B,S)
                false values mean masked, not attended to indices.
            pad_mask: Tensor, shape (B,S)
                false values mean masked, not attended to indices.
            use_cache: bool
                if true, will return the updated past_key_values for
                future speedups
            past_key_values: list of lists of tensors
                the cached computations returned by the layer when
                `use_cache` is true.
            inputs_embeds: None or torch FloatTensor (B,S,E)
                the input embeddings. this must not be None if input_ids
                is None. input_ids overrides this argument if both are
                not None.
            position_ids: None or LongTensor (S,)
                optionally argue the position ids for the positional
                encodings.
            output_attentions: bool
                if true, will return the attention weights
        Returns:
            BaseModelOutputWithPast
        """
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        hidden_states = inputs_embeds

        attn_mask = self.prep_encoder_mask(
            S=inputs_embeds.shape[1],
            attention_mask=attention_mask,
            pad_mask=pad_mask,
            is_causal=is_causal,
        )

        if past_key_values is not None and position_ids is None:
            # we create the full position ids because this will be
            # used for the keys if using rotary encoder attention module
            position_ids = torch.arange(
                past_key_values[0][0].shape[-2]+inputs_embeds.shape[1],
            ).long().to(self.get_device())
        pids = None
        if position_ids is not None:
            pids = position_ids[-inputs_embeds.shape[1]:]
        hidden_states = self.pos_encs(
                inputs_embeds,
                pids=pids)

        past_key_value = None
        next_cache = [] if use_cache else None

        all_hidden_states = []
        attentions = []
        next_cache = []
        S = 0
        for i,layer in enumerate(self.layers):
            if past_key_values is not None:
                past_key_value = past_key_values[i]
            ret_dict = layer(
                src=hidden_states,
                src_mask=attn_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
                position_ids=position_ids,
            )
            hidden_states = ret_dict["hidden_states"]
            hidden_states = self.identities[i](hidden_states)

            if use_cache:
                next_cache.append(ret_dict["past_key_value"])
            if output_attentions:
                attentions.append(ret_dict.get("attentions", None))
            hidden_states[torch.isnan(hidden_states)] = 0
            all_hidden_states.append(hidden_states)
        if use_cache:
            if past_key_values is not None:
                pkv = past_key_values[-1]
                hidden_states = torch.cat([pkv,hidden_states],dim=1)
            next_cache.append(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=attentions,
        )

    def prep_mask(self, mask=None,pad_mask=None):
        """
        This function should be applied before the encoder function to
        allow reusability with Huggingface models. Assumes true means
        padding/unattended locations.

        Args:
            mask: None or bool tensor
                true means unattended locations
            pad_mask: None or bool tensor
                true means unattended locations
        Returns:
            attn_mask: bool tensor
                true denotes attended locations. note that this is
                flipped from the incoming arguments.
        """
        attn_mask = None
        if pad_mask is not None:
            if pad_mask.dtype==torch.bool: attn_mask = ~pad_mask
            else: attn_mask = pad_mask==0
        if mask is not None:
            if mask.dtype==torch.bool: mask = ~mask
            else: mask = mask==0
            if attn_mask is not None:
                attn_mask = padmask2attnmask(attn_mask)
                attn_mask = attn_mask&mask
            else: attn_mask = mask
        return attn_mask

    def tforce_fwd(self,
                   inpts:torch.Tensor=None,
                   mask:torch.Tensor=None,
                   pad_mask:torch.Tensor=None,
                   inputs_embeds:torch.Tensor=None,
                   attention_mask:torch.Tensor=None,
                   past_key_values=None,
                   use_cache=False,
                   temperature=None,
                   hidden_states_only=False,
                   output_attentions=False,
                   position_ids=None,
                   *args, **kwargs):
        """
        Arguments:
            inpts: Tensor, shape ``[bsize, seq_len]``
            mask: Tensor, shape ``[seq_len, seq_len]``
                true means unattended locations
            pad_mask: Tensor, shape ``[bsize, seq_len]``
                true means padding
            inputs_embeds: tensor (B,S,E)
                optionally argue embeddings instead of token ids
            past_key_values: tuple of tuple of tensors
                the output of a huggingface cache. used to speed up
                computations. See huggingface documentation for more
                details
            hidden_states_only: bool
                if true, will not bother computing the logits
            output_attentions: bool
                if true, will return the attention weights
            position_ids: None or LongTensor (S,)
                optionally argue the position ids for the positional
                encodings.
        Returns:
            ret_dict: dict
                "pred_ids": torch LongTensor (B,S)
                "logits": torch FloatTensor (B,S,N)
                "past_key_values": None or tuple of tuple of tensors
                "attentions": None or tuple of tensors [(B,N,S,S)]
        """
        # flipped so that true means attend to
        attn_mask = self.prep_mask(mask=mask, pad_mask=pad_mask)

        ## Left Padding
        #if position_ids is None and pad_mask is not None and pad_mask[0,0]:
        #    n_els = (~pad_mask).float().sum(-1)
        #    pids = torch.cat([torch.arange(n).long() for n in n_els])
        #    position_ids = torch.zeros(pad_mask.shape).long()
        #    position_ids[~pad_mask] = pids
        #    position_ids = position_ids.to(self.get_device())

        output = self.encoder(
            inpts,
            attention_mask=attn_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            position_ids=position_ids,
        )
        if hidden_states_only:
            return {
                "hidden_states": output.hidden_states,
                "last_hidden_state": output.last_hidden_state,
                "past_key_values": getattr(output,"past_key_values",None),
                "logits": getattr(output,"logits",None),
                "attentions": getattr(output, "attentions", None),
            }
        if not hasattr(output, "logits"):
            og_shape = output.last_hidden_state.shape
            state = output.last_hidden_state.reshape(-1,og_shape[-1])
            logits = self.lm_head(
                self.decoder(state)
            ).reshape(*og_shape[:-1], -1)
        else: logits = output.logits
        pred_ids = self.sample_with_temperature(
            logits, temperature
        )
        return {
            "hidden_states": output.hidden_states,
            "last_hidden_state": output.last_hidden_state,
            "logits": logits,
            "pred_ids": pred_ids,
            "past_key_values": getattr(output,"past_key_values",None),
            "attentions": getattr(output, "attentions", None),
        }

    def freedom_fwd(self,
                    inpts:torch.Tensor=None,
                    mask:torch.Tensor=None,
                    pad_mask:torch.Tensor=None,
                    n_steps:int=0,
                    incl_all_inpts:bool=False,
                    temperature=None,
                    inputs_embeds=None,
                    past_key_values=None,
                    stop_ids=None,
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
            n_steps: int
                the number of prediction steps if not using teacher
                forcing
            incl_all_inpts: bool
                if true, will include all input tokens in the output
                prediction tensor. otherwise only includes "predicted
                spaces". "predicted spaces" includes the shifted initial
                inputs. This is useful to save a concatenation during
                the data bootstrapping phase.
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
            output_attentions: bool
                if true, will return the attention weights
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
        n_loops = n_steps + 1

        if stop_ids is not None:
            if type(stop_ids)==int: stop_ids = [stop_ids]
            if len(stop_ids)>0:
                stop_ids = torch.LongTensor(list(stop_ids))
                stop_ids = stop_ids.to(self.get_device())
            else: stop_ids = None
        else: stop_ids = None

        if inpts is None:
            B,S = inputs_embeds.shape[:2]
        else:
            B,S = inpts.shape

        pred_ids = torch.zeros(
            (B,S+n_loops+int(incl_all_inpts)), device=self.get_device()
        ).long()
        if inpts is not None:
            pred_ids[:,:S] = inpts
        logits = torch.zeros(
            (B,S+n_steps+incl_all_inpts,self.n_tokens),
            device=self.get_device()
        )
        if inpts is not None:
            logits[:,:S-1+incl_all_inpts].scatter_(
                dim=-1,
                index=inpts[:, 1-incl_all_inpts:S, None],
                src=torch.ones_like(logits[:, :S-1+incl_all_inpts])
            )

        # Need to ensure we use the appropriate input type between
        # the inpts ids and the input embeddings
        if inpts is None:
            inpt_emb = inputs_embeds
            inpt = None
        else:
            inpt = pred_ids[:,:S]
            inpt_emb = None

        # Masks
        if pad_mask is not None:
            pad_mask = ~(pad_mask.bool())
            if pad_mask.shape[-1]<S+n_steps:
                p = S+n_steps - pad_mask.shape[-1]
                pad_mask = torch.nn.functional.pad(
                    pad_mask, (0, p), value=True
                )
        # Custom attention mask
        if mask is not None:
            mask = ~(mask.bool())
            if mask.shape[-1]<S+n_steps:
                p = S+n_steps - mask.shape[-1]
                mask = torch.nn.functional.pad(
                    mask, (0, p, 0, p), value=True
                )

        # Positional indices
        if position_ids is not None and position_ids.shape[-1]<S+n_steps:
            device = device_fxn(position_ids.get_device())
            p = S+n_steps - position_ids.shape[-1]
            new_positions = torch.arange(p).to(device)
            new_positions += position_ids[-1]
            position_ids = torch.cat([position_ids,new_positions],dim=-1)

        # Need to ensure the padding mask is the full length of the
        # past_key_values if past_key_values is not none
        p_end = S
        if past_key_values is not None and past_key_values[0] is not None:
            # pkv idxs: layer, k or v, values
            p_end = past_key_values[0][0].shape[1]

        h_states = []
        attentions = None
        pids = None

        for step in range(n_loops):
            ## The masks are inverted at this point, 1 means do attend
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
                position_ids=pids,
            )
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
            logits[:,S-1+step+incl_all_inpts] = pred
            argmaxs = self.sample_with_temperature(
                pred, temperature
            ).squeeze()
            pred_ids[:,S+step+int(incl_all_inpts)] = argmaxs
            if step < n_loops-1:
                inpt_emb = None
                s = S+step+int(incl_all_inpts)
                inpt = pred_ids[:,s:s+1]
                if stop_ids is not None and torch.isin(inpt, stop_ids):
                    logits = logits[:,:S+step+1]
                    pred_ids = pred_ids[:,:S+step+1]
                    break
        ret_dict = {
            "logits": logits,
            "pred_ids":  pred_ids[:,int(not incl_all_inpts):],
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


class HFTransformer(SequenceModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = 'Transformer'

        if self.pretrained:
            # We will load the weights and then transfer them to our
            # custom model type.
            temp_model = AutoModelForCausalLM.from_pretrained(
                self.hf_model_type
            )
            self.encoder = tmods.FlexibleLlamaModel(temp_model.config)
            self.encoder.load_state_dict(temp_model.state_dict())

            print("Properties:")
            for name in dir(self.encoder):
                print(name)
            print(self.encoder)
            d = self.encoder.get_input_embeddings().weight.shape[-1]
            self.d_model = d
        else:
            config = self.get_config()
            self.encoder = tmods.FlexibleLlamaModel( config )
            if hasattr(self.encoder, "transformer"):
                if hasattr(self.encoder.transformer, "wpe"):
                    wpe = self.encoder.transformer.wpe
                    for name, p in wpe.named_parameters():
                        p.requires_grad = self.learn_posencs

        if self.n_tokens:
            self.embeddings = self.encoder.get_input_embeddings()
        else:
            self.encoder.set_input_embeddings(None)
            self.embeddings = None

        self.decoder = nn.LayerNorm(self.d_model)
        if hasattr(self.encoder, "lm_head"):
            if self.n_tokens and self.n_tokens==self.out_tokens:
                self.lm_head = self.encoder.lm_head
            else:
                self.lm_head =  nn.Linear( self.d_model, self.out_tokens )
            self.encoder.lm_head = self.lm_head
        else:
            self.lm_head =  nn.Linear( self.d_model, self.out_tokens )
        self.init_weights()

    def tforce_fwd(self,
                   inpts:torch.Tensor=None,
                   mask:torch.Tensor=None,
                   pad_mask:torch.Tensor=None,
                   inputs_embeds:torch.Tensor=None,
                   past_key_values=None,
                   use_cache=False,
                   temperature=None,
                   *args, **kwargs):
        """
        Arguments:
            inpts: Tensor, shape ``[bsize, seq_len]``
            mask: Tensor, shape ``[seq_len, seq_len]``
                true means unattended locations
            pad_mask: Tensor, shape ``[bsize, seq_len]``
                true means padding
            inputs_embeds: tensor (B,S,E)
                optionally argue embeddings instead of token ids
            past_key_values: tuple of tuple of tensors
                the output of a huggingface cache. used to speed up
                computations. See huggingface documentation for more
                details
            temperature: float
                a parameter to adjust the entropy of the
                token sampling. high temperature means high entropy
        Returns:
            ret_dict: dict
                "pred_ids": torch LongTensor (B,S)
                "logits": torch FloatTensor (B,S,N)
                "past_key_values": None or tuple of tuple of tensors
        """
        # flipped so that true means attend to
        attn_mask = None
        if pad_mask is not None:
            attn_mask = ~(pad_mask.bool())
        if mask is not None:
            raise NotImplemented
            if attn_mask is not None:
                # be careful with padmask2attnmask function. make sure
                # mask values are false for padding.
                attn_mask = padmask2attnmask(attn_mask)
                attn_mask = attn_mask&~mask
            else: attn_mask = ~mask.bool()
        output = self.encoder(
            inpts,
            attention_mask=attn_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
        if not hasattr(output, "logits"):
            og_shape = output.last_hidden_state.shape
            state = output.last_hidden_state.reshape(-1,og_shape[-1])
            logits = self.lm_head(
                self.decoder(state)
            ).reshape(*og_shape[:-1], -1)
        else: logits = output.logits
        pred_ids = self.sample_with_temperature(
            logits, temperature
        )
        return {
            "last_hidden_state": output.last_hidden_state,
            "logits": logits,
            "pred_ids": pred_ids,
            "past_key_values": getattr(output,"past_key_values",None),
        }

    def freedom_fwd(self,
                    inpts:torch.Tensor=None,
                    mask:torch.Tensor=None,
                    pad_mask:torch.Tensor=None,
                    is_causal:bool=None,
                    n_steps:int=1,
                    incl_all_inpts:bool=False,
                    pad_pos_skip:bool=False,
                    temperature=None,
                    inputs_embeds=None,
                    past_key_values=None,
                    stop_ids=None,
                    *args, **kwargs):
        """
        Arguments:
            inpts: Tensor, shape ``[bsize, seq_len]``
            mask: Tensor, shape ``[seq_len, seq_len]``
                true means unattended locations
            pad_mask: Tensor, shape ``[bsize, seq_len]``
                true means padding
            is_causal: bool
                If specified, applies a causal mask as mask (optional)
                and ignores attn_mask for computing scaled dot product
                attention.
            n_steps: int
                the number of prediction steps if not using teacher
                forcing
            incl_all_inpts: bool
                if true, will include all input tokens in the output
                prediction tensor. otherwise only includes "predicted
                spaces". "predicted spaces" includes the shifted initial
                inputs. This is useful to save a concatenation during
                the data bootstrapping phase.
            pad_pos_skip: bool
                if true, will skip over masked tokens when applying
                positional encodings based on the pad mask. True values
                in the mask will be skipped.
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
                sequence will be the sequence including the stop id.
                All sequences in the batch are terminated from a single
                stop id in any sample
        Returns:
            ret_dict: dict
                "pred_ids": torch LongTensor (B,S+NSteps)
                "logits": torch FloatTensor (B,S+NSteps,NTokens)
                "past_key_values": None or tuple of tuple of tensors
        """
        if stop_ids is not None:
            if type(stop_ids)==int: stop_ids = [stop_ids]
            if len(stop_ids)>0:
                stop_ids = torch.LongTensor(list(stop_ids))
                stop_ids = stop_ids.to(self.get_device())
            else: stop_ids = None
        else: stop_ids = None

        n_loops = n_steps + 1
        if inpts is None:
            B,S = inputs_embeds.shape[:2]
        else:
            B,S = inpts.shape

        if pad_mask is None:
            pad_mask = torch.ones(B,S+n_loops).bool().to(self.get_device())
        else:
            pad_mask = torch.nn.functional.pad(
                ~(pad_mask.bool()), (0, n_loops), value=True
            )
        if mask is not None:
            raise NotImplemented
            mask = torch.nn.functional.pad(
                ~(mask.bool()), (0, n_loops, 0, n_loops), value=True
            )
        pred_ids = torch.zeros(
            (B,S+n_loops), device=self.get_device()
        ).long()
        if inpts is not None:
            pred_ids[:,:S] = inpts
        logits = torch.zeros(
            (B,S+n_steps+incl_all_inpts,self.n_tokens),
            device=self.get_device()
        )
        if inpts is not None:
            logits[:,:S-1+incl_all_inpts].scatter_(
                dim=-1,
                index=inpts[:, 1-incl_all_inpts:S, None],
                src=torch.ones_like(logits[:, :S-1+incl_all_inpts])
            )

        # Need to ensure we use the appropriate input type between
        # the inpts ids and the input embeddings
        if inpts is None:
            inpt_emb = inputs_embeds
            inpt = None
        else:
            inpt = pred_ids[:,:S]
            inpt_emb = None

        # Need to ensure the padding mask is the full length of the
        # past_key_values if past_key_values is not none
        p_end = S
        if past_key_values is not None and past_key_values[0] is not None:
            p_end = past_key_values[0][0].shape[1]

        h_states = []
        for step in range(n_loops):
            attn_mask = None
            e = p_end+step
            if pad_mask is not None:
                attn_mask = pad_mask[:,:e]
            if mask is not None:
                if attn_mask is None:
                    attn_mask = mask[:,:e,:e]
                else:
                    attn_mask = padmask2attnmask(attn_mask)
                    attn_mask = mask[:,:e,:e]&attn_mask
                    if past_key_values is not None:
                        attn_mask = attn_mask[:,-1:]
            output = self.encoder(
                input_ids=inpt,
                attention_mask=attn_mask,
                use_cache=True,
                past_key_values=past_key_values,
                inputs_embeds=inpt_emb,
            )
            past_key_values = output.past_key_values
            if len(h_states)==0:
                h_states.append(output.last_hidden_state)
            else:
                h_states.append(output.last_hidden_state[:, -1:])
            ## TODO: change FlexibleLlama model to output logits
            if not hasattr(output, "logits"):
                states = output.last_hidden_state[:,-1]
                pred = self.lm_head(self.decoder(states))
            else: pred = output.logits[:,-1]
            logits[:,S-1+step+incl_all_inpts] = pred
            argmaxs = self.sample_with_temperature(
                pred, temperature
            ).squeeze()
            pred_ids[:,S+step+incl_all_inpts] = argmaxs
            if step < n_steps:
                inpt_emb = None
                inpt = pred_ids[:,S+step:S+step+1]
                if stop_ids is not None and torch.isin(inpt, stop_ids):
                    logits = logits[:,:S+step+1]
                    pred_ids = pred_ids[:,:S+step+1]
                    break
        return {
            "logits": logits,
            "pred_ids":  pred_ids[:,int(not incl_all_inpts):],
            "past_key_values": past_key_values,
            "last_hidden_state": torch.cat(h_states, dim=1),
        }

class TorchTransformer(SequenceModule):
    """
    This is a vanilla transformer that uses a PyTorch backend. This
    model has a lower chance of becoming deprecated.
    """
    def __init__(self,
            encoder_layer_class=torch.nn.TransformerEncoderLayer,
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = 'Transformer'
        self.embeddings = torch.nn.Embedding(self.n_tokens,self.d_model)
        # TODO: enable rotary encodings
        self.pos_encs = tmods.SinPositionalEncoding(
            d_model=self.d_model,
            max_len=max(2048, self.seq_len),
            learnable=self.learn_posencs,
        )
        self.layers = torch.nn.ModuleList([])

        for el in range(self.n_layers):
            self.layers.append(encoder_layer_class(
                d_model=self.d_model,
                nhead=self.n_heads,
                dim_feedforward=self.d_model*self.h_mult,
                dropout=self.drop_p,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ))
        self.decoder = nn.LayerNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, self.n_tokens)
        self.init_weights()

    def combine_attn_masks(self,
                sz=0,
                attention_mask=None,
                pad_mask=None):
        """
        This function manually combines attention masks because pytorch
        does it stupidly.

        Args:
            sz: int
            attention_mask: BoolTensor (B,S,S) or (S,S)
                True values mean masked, not attended to indices
            pad_mask: BoolTensor (B,S)
        """
        if attention_mask is None:
            attn_mask = generate_square_subsequent_mask(
                sz,
                device=self.get_device(),
                dtype="float"
            )
        else:
            if attention_mask.dtype==torch.bool:
                attention_mask = attention_mask.float().masked_fill_(
                    attention_mask, float("-inf"))
            attn_mask = attention_mask
        if pad_mask is not None:
            pad_mask = pad_mask.float().masked_fill_(
                pad_mask,float("-inf"))
            S = pad_mask.shape[1]
            attn_mask = attn_mask + pad_mask[:,None].repeat((1,S,1))
            attn_mask[(attn_mask==0).sum(-1)==0] = 0
        if len(attn_mask.shape)>2:
            attn_mask = torch.repeat_interleave(
                input=attn_mask,
                repeats=self.n_heads,
                dim=0,
            )
        return attn_mask

    def encoder(self,
                input_ids=None,
                attention_mask=None,
                pad_mask=None,
                use_cache=False,
                past_key_values=None,
                inputs_embeds=None,
                position_ids=None,
                output_attentions=False,
                *args, **kwargs):
        """
        Arguments:
            input_ids: Long Tensor, shape ``[bsize, seq_len]``
                the input ids. one of this or inputs_embeds must be not
                None
            attention_mask: Tensor, shape (B,S,S) or (B,S)
                true values mean masked, not attended to indices.
            pad_mask: Tensor, shape (B,S)
                true values mean masked, not attended to indices.
            use_cache: bool
                if true, will return the updated past_key_values for
                future speedups
            past_key_values: list of lists of tensors
                the cached computations returned by the layer when
                `use_cache` is true.
            inputs_embeds: None or torch FloatTensor (B,S,E)
                the input embeddings. this must not be None if input_ids
                is None. input_ids overrides this argument if both are
                not None.
            position_ids: None or LongTensor (S,)
                optionally argue the position ids for the positional
                encodings.
            output_attentions: bool
                if true, will return the attention weights
        Returns:
            BaseModelOutputWithPast
        """
        past_key_values = None
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        hidden_states = inputs_embeds

        attn_mask = self.combine_attn_masks(
            sz=inputs_embeds.shape[1],
            attention_mask=attention_mask,
            pad_mask=pad_mask,
        )
        hidden_states = self.pos_encs(inputs_embeds, pids=position_ids)

        past_key_value = None
        all_hidden_states = []
        attn_weights = []
        next_cache = []
        S = 0
        for i,layer in enumerate(self.layers):
            hidden_states = layer(
                src=hidden_states,
                src_mask=attn_mask,
            )
            hidden_states[torch.isnan(hidden_states)] = 0
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            #past_key_values=next_cache,
            past_key_values=None,
            #hidden_states=all_hidden_states,
        )

    def tforce_fwd(self,
                   inpts:torch.Tensor=None,
                   mask:torch.Tensor=None,
                   pad_mask:torch.Tensor=None,
                   inputs_embeds:torch.Tensor=None,
                   past_key_values=None,
                   use_cache=False,
                   temperature=None,
                   hidden_states_only=False,
                   *args, **kwargs):
        """
        Arguments:
            inpts: Tensor, shape ``[bsize, seq_len]``
            mask: Tensor, shape ``[seq_len, seq_len]``
                true means unattended locations
            pad_mask: Tensor, shape ``[bsize, seq_len]``
                true means padding
            inputs_embeds: tensor (B,S,E)
                optionally argue embeddings instead of token ids
            past_key_values: tuple of tuple of tensors
                the output of a huggingface cache. used to speed up
                computations. See huggingface documentation for more
                details
            hidden_states_only: bool
                if true, will not bother computing the logits
        Returns:
            ret_dict: dict
                "pred_ids": torch LongTensor (B,S)
                "logits": torch FloatTensor (B,S,N)
                "past_key_values": None or tuple of tuple of tensors
        """
        if pad_mask is not None:
            pad_mask = pad_mask.bool()
        if mask is not None:
            mask = mask.bool()

        output = self.encoder(
            inpts,
            attention_mask=mask,
            pad_mask=pad_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
        if hidden_states_only:
            return {
                "last_hidden_state": output.last_hidden_state,
                "past_key_values": getattr(output,"past_key_values",None),
                "logits": getattr(output,"logits",None),
            }
        if not hasattr(output, "logits"):
            og_shape = output.last_hidden_state.shape
            state = output.last_hidden_state.reshape(-1,og_shape[-1])
            logits = self.lm_head(
                self.decoder(state)
            ).reshape(*og_shape[:-1], -1)
        else: logits = output.logits
        pred_ids = self.sample_with_temperature(
            logits, temperature
        )
        return {
            "last_hidden_state": output.last_hidden_state,
            "logits": logits,
            "pred_ids": pred_ids,
            "past_key_values": getattr(output,"past_key_values",None),
        }

    def freedom_fwd(self,
                    inpts:torch.Tensor=None,
                    mask:torch.Tensor=None,
                    pad_mask:torch.Tensor=None,
                    n_steps:int=1,
                    incl_all_inpts:bool=False,
                    temperature=None,
                    inputs_embeds=None,
                    past_key_values=None,
                    stop_ids=None,
                    *args, **kwargs):
        """
        Arguments:
            inpts: Tensor, shape ``[bsize, seq_len]``
            mask: Tensor, shape ``[seq_len, seq_len]``
                true means padding, or unattended locations
            pad_mask: Tensor, shape ``[bsize, seq_len]``
                true means padding
            n_steps: int
                the number of prediction steps if not using teacher
                forcing
            incl_all_inpts: bool
                if true, will include all input tokens in the output
                prediction tensor. otherwise only includes "predicted
                spaces". "predicted spaces" includes the shifted initial
                inputs. This is useful to save a concatenation during
                the data bootstrapping phase.
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
        Returns:
            ret_dict: dict
                "pred_ids": torch LongTensor (B,S+NSteps)
                "logits": torch FloatTensor (B,S+NSteps,NTokens)
                "past_key_values": None or tuple of tuple of tensors
                "last_hidden_state": torch FloatTensor (B,S+NSteps,E)
        """
        n_loops = n_steps + 1

        if stop_ids is not None:
            if type(stop_ids)==int: stop_ids = [stop_ids]
            if len(stop_ids)>0:
                stop_ids = torch.LongTensor(list(stop_ids))
                stop_ids = stop_ids.to(self.get_device())
            else: stop_ids = None
        else: stop_ids = None

        if inpts is None:
            B,S = inputs_embeds.shape[:2]
        else:
            B,S = inpts.shape

        pred_ids = torch.zeros(
            (B,S+n_loops+int(incl_all_inpts)), device=self.get_device()
        ).long()
        if inpts is not None:
            pred_ids[:,:S] = inpts
        logits = torch.zeros(
            (B,S+n_steps+incl_all_inpts,self.n_tokens),
            device=self.get_device()
        )
        if inpts is not None:
            logits[:,:S-1+incl_all_inpts].scatter_(
                dim=-1,
                index=inpts[:, 1-incl_all_inpts:S, None],
                src=torch.ones_like(logits[:, :S-1+incl_all_inpts])
            )

        inpt_embs = inputs_embeds

        # Masks
        attn_mask = None
        if pad_mask is not None:
            pad_mask = pad_mask.bool()
            if pad_mask.shape[-1]<S+n_loops:
                p = S+n_loops - pad_mask.shape[-1]
                pad_mask = torch.nn.functional.pad(
                    pad_mask, (0, p), value=False
                )
        # Custom attention mask
        if mask is not None:
            mask = mask.bool()
            if mask.shape[-1]<S+n_steps:
                p = S+n_steps - mask.shape[-1]
                mask = torch.nn.functional.pad(
                    mask, (0, p, 0, p), value=False
                )
        # Need to ensure the padding mask is the full length of the
        # past_key_values if past_key_values is not none
        p_end = S
        if past_key_values is not None and past_key_values[0] is not None:
            # pkv idxs: layer, k or v, values
            p_end = past_key_values[0][0].shape[1]

        h_states = []

        for step in range(n_loops):
            if pad_mask is not None:
                e = p_end+step
                pmask = pad_mask[:,:e]
            if mask is not None:
                e = p_end+step
                attn_mask = mask[:,:e,:e]

            output = self.encoder(
                input_ids=inpts,
                attention_mask=attn_mask,
                pad_mask=pmask,
                use_cache=True,
                past_key_values=past_key_values,
                inputs_embeds=inpt_embs,
            )
            past_key_values = output.past_key_values

            if len(h_states)==0:
                h_states.append(output.last_hidden_state)
            else:
                h_states.append(output.last_hidden_state[:,-1:])
            if not hasattr(output, "logits"):
                state = h_states[-1][:,-1]
                pred = self.lm_head(self.decoder(state))
            else: pred = output.logits[:,-1]
            logits[:,S-1+step+incl_all_inpts] = pred
            argmaxs = self.sample_with_temperature(
                pred, temperature
            ).squeeze()
            pred_ids[:,S+step+int(incl_all_inpts)] = argmaxs
            if step < n_loops-1:
                inpt = pred_ids[:,S+step:S+step+1]
                inpt_emb = self.embeddings(inpt)
                if stop_ids is not None and torch.isin(inpt, stop_ids):
                    logits = logits[:,:S+step+1]
                    pred_ids = pred_ids[:,:S+step+1]
                    break
                if inpts is None:
                    inpt_embs = torch.cat([inpt_embs, inpt_emb],dim=1)
                else:
                    inpts = torch.cat([inpts, inpt],dim=1)
        return {
            "logits": logits,
            "pred_ids":  pred_ids[:,int(not incl_all_inpts):],
            "past_key_values": past_key_values,
            "last_hidden_state": torch.cat(h_states, dim=1),
        }

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
                sprout_len=3,
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
            sprout_len: int
                the amount of seed text if using `tforce=False`
            output_attentions: bool
                if true, model will return scaled attention weights
        Returns:
            ret_dict: dict (keys: str, vals: torch tensor)
                "loss": torch tensor (1,) or (B,)
                "acc": torch tensor (1,) or (B,)
                    the raw accuracy for the non-rmb task
                "pred_ids": torch tensor (B,S,P)
                    the prediction logits. only returned if ret_preds is
                    true
        """
        ret_dict = dict()
        pad_id = self.pad_id
        bos_id = self.bos_id
        eos_id = self.eos_id
        if sprout_len is None or sprout_len<0: sprout_len = 0
        if "input_pad_mask" not in data:
            inpt_pad_mask = (data["input_ids"]==pad_id)
            inpt_pad_mask = inpt_pad_mask|(data["input_ids"]==eos_id)
        else: inpt_pad_mask = data["input_pad_mask"].clone()
        if "output_pad_mask" not in data:
            out_pad_mask = data["output_ids"]==pad_id
            out_pad_mask = out_pad_mask==bos_id
        else: out_pad_mask = data["output_pad_mask"].clone()


        if "input_ids" in data:
            inpts = data["input_ids"]
            if not tforce or type(self.model) in {LSTM,GRU,RNN,LinearRNN}:
                leading_pad = torch.max(torch.argmax(
                    (~inpt_pad_mask).long(), dim=1
                ))
                inpts = inpts[:,:leading_pad+int(sprout_len)]
            n_steps = inpt_pad_mask.shape[-1]-inpts.shape[-1]
        elif "inputs" in data:
            inpts = data["inputs"]
            n_steps = inpts.shape[1]
        outputs = data["output_ids"]

        device = self.model.get_device()
        if inpts.get_device()!=self.model.get_device():
            inpts = inpts.to(device)
            inpt_pad_mask = inpt_pad_mask.to(device)
            outputs = outputs.to(device)
            out_pad_mask = out_pad_mask.to(device)

        ret_dict = self.model(
            inpts,
            pad_mask=inpt_pad_mask,
            tforce=tforce,
            n_steps=n_steps,
            temperature=temperature,
            output_attentions=output_attentions,
        )

        ## Loss
        #################################
        logits = ret_dict["logits"]
        inpt_mask = ~inpt_pad_mask.reshape(-1)
        out_mask =  ~out_pad_mask.reshape(-1)
        #print("n_steps:", n_steps)
        #print("inpts:", inpts.shape)
        #print("logits:", logits.shape)
        #print("inpt-logit diff:", logits.shape[1]-inpts.shape[1])
        #print("outputs:", outputs.shape)
        #print("inpt_mask:", inpt_mask.shape)
        #print("out_mask:", out_mask.shape)
        ps = logits.reshape(
            -1, logits.shape[-1]
        )[inpt_mask]
        labels = outputs.reshape(-1)[out_mask]
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
            temp = torch.zeros_like(outputs).float()
            temp[out_mask.reshape(outputs.shape)] = loss
            loss = temp
        else:
            loss = loss.mean()
        ret_dict["loss"] = loss

        ## Acc
        #################################
        if "pred_ids" in ret_dict:
            pred_ids = ret_dict["pred_ids"]
            pred_ids = pred_ids.reshape(-1)[inpt_mask]
        else:
            pred_ids = torch.argmax(ps, dim=-1)
            ret_dict["pred_ids"] = torch.argmax(logits, dim=-1)
        acc = (pred_ids==labels).float()
        if reduce_metrics: acc = acc.mean()
        else: 
            temp = torch.zeros_like(outputs).float()
            temp[out_mask.reshape(outputs.shape)] = acc.long()
            acc = temp
        ret_dict["acc"] = acc

        ret_dict["top_k"] = top_k_acc(
            ps, labels, top_k, as_tensor=True
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
            print("Full inpt mask", inpt_pad_mask[i])
            print("Full outpt mask", out_pad_mask[i])
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

        idx = inpt_pad_mask.float().sum(-1)!=out_pad_mask.float().sum(-1)
        print()
        print()
        print()
        print()
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
    config["encoder_layer_class"] = "RotaryEncoderLayer"
    model = globals()[config.get("model_type","LSTM")](**config)
    #model = globals()[config.get("model_type","LinearRNN")](**config)
    return LossWrapper(model=model, config=config, **config)


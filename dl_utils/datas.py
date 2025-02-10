import pickle
import dl_utils.tokenizer
from dl_utils.utils import rolling_window
import torch
from tensordict import TensorDict
import numpy as np
from tqdm import tqdm
import os


class CausalDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 labels=None,
                 masks=None,
                 pad_id=0,
                 bos_id=1,
                 eos_id=2,
                 concat=False,
                 dynamic_concat=False,
                 cat_seq_len=None,
                 *args, **kwargs):
        """
        Args:
            data: str or list or ndarray or torch LongTensor
                if string, the path to the data. Otherwise, 
                should be a number of sequences that are already
                tokenized in id form. If concat is false, all of the
                sequences should be the same length.
            labels: None or str or dict or tensordict
                If labels is not none and data is a string, labels
                should also be a string. The value of the labels should
                be a dict or a tensor dict of labels. If a dict is
                argued, assumes values are ndarrays. Currently
                incompatible with concat. Any shape is legal as long
                as the batch dimension matches.
            masks: None or same as data or dict
                optionally boolean masks. If a dict is argued, will assume
                that all values are masks of the same shape and type as
                data. Do not argue padding masks.
            pad_id: int
                the padding id
            bos_id: int
                the beginning of sequence id
            eos_id: int
                the end of sequence id
            concat: bool
                if true, will concat all samples together into one long
                sequence that will then be sampled from uniformly.
            dynamic_concat: bool
                if true, will shuffle and reconcat the samples at the
                beginning of every new epoch.
            cat_seq_len: None or int
                if concatenate is true, the data will be sampled in
                sequence lengths of cat_seq_len.
        """
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.seq_len = cat_seq_len
        self.concat = concat
        self.dynamic_concat = dynamic_concat

        if type(data)==str:
            data_path = data
            if ".p" in data_path:
                with open(self.data_path, "rb") as f:
                    data = pickle.load(f)
            if labels:
                data_path = labels
                if ".p" in data_path:
                    with open(self.data_path, "rb") as f:
                        labels = pickle.load(f)
            if masks:
                if type(masks)==str:
                    data_path = masks
                    if ".p" in data_path:
                        with open(self.data_path, "rb") as f:
                            masks = pickle.load(f)
                elif type(masks)==dict:
                    for k in masks:
                        data_path = masks[k]
                        if ".p" in data_path:
                            with open(self.data_path, "rb") as f:
                                masks[k] = pickle.load(f)

        self.inpt_seqs = data
        self.labels = labels
        self.masks = masks
        if self.masks is not None and type(self.masks)!=dict:
            self.masks = { "mask": self.masks }

        if self.concat:
            assert type(self.labels)!=TensorDict
            if self.dynamic_concat:
                self.og_seqs = data
                self.og_labels = labels
                self.og_masks = {**self.masks}

        if self.seq_len is None:
            if self.concat:
                self.seq_len = 3*len(self.inpt_seqs[0])
            else:
                self.seq_len = len(self.inpt_seqs[0])
        if hasattr(self.inpt_seqs, "shape") and self.concat:
            B,S = self.inpt_seqs.shape
            self.inpt_seqs = self.inpt_seqs.reshape(B*S,-1).squeeze()
            if self.labels is not None:
                self.labels = self.labels.reshape(B*S,-1).squeeze()
            if self.masks is not None:
                for k in self.masks:
                    self.masks[k] = self.masks[k].reshape(B*S,-1).squeeze()
        ndarr = type(np.zeros((1,)))
        if type(self.inpt_seqs)==ndarr:
            self.inpt_seqs = torch.LongTensor(self.inpt_seqs)
            if self.labels is not None and type(self.labels)==dict:
                self.labels = TensorDict({
                    k: torch.from_numpy(v) for k,v in self.labels.items()
                }, batch_size=len(self.inpt_seqs))
            if self.masks is not None:
                for k in self.masks:
                    self.masks[k] = torch.BoolTensor(self.masks[k])
        elif type(self.inpt_seqs)==list and self.concat:
            if not self.dynamic_concat:
                seqs = []
                for seq in self.inpt_seqs: seqs += seq
                self.inpt_seqs = torch.LongTensor(seqs)
                if self.masks is not None:
                    for k in self.masks:
                        seqs = []
                        for seq in self.masks[k]: seqs += seq
                        self.masks[k] = torch.BoolTensor(seqs)
        else:
            self.inpt_seqs = torch.LongTensor(self.inpt_seqs)
            if self.labels is not None:
                if type(self.labels)==dict:
                    self.labels = TensorDict({
                      k: torch.LongTensor(v) for k,v in self.labels.items()
                    }, batch_size=len(self.inpt_seqs))
            if self.masks is not None:
                for k in self.masks:
                    self.masks[k] = torch.BoolTensor(self.masks[k])

    def __len__(self):
        if self.concat and not self.dynamic_concat:
            return len(self.inpt_seqs)-self.seq_len-1
        return len(self.inpt_seqs)

    def dynamic_concat_getitem(self, idx):
        """
        This function will concat a number of uniformly sampled sequences
        together in addition to the indexed sequence to dynamically
        vary the concatenated data.
        """
        ridx = np.random.randint(len(self.inpt_seqs))
        samps = [i for i in self.inpt_seqs[ridx]]
        samps += [i for i in self.inpt_seqs[idx]]
        out_samps = None
        masks = {k: [i for i in self.masks[k][ridx]] for k in self.masks}
        for k in masks:
            masks[k] += [i for i in self.masks[k][idx]]
        end_len = self.seq_len+len(samps)
        while len(samps) < end_len:
            ridx = np.random.randint(len(self.inpt_seqs))
            samps += [i for i in self.inpt_seqs[ridx]]
            for k in masks:
                masks[k] += [i for i in self.masks[k][ridx]]

        # Shift and cut at seq_len
        shift = np.random.randint(len(samps)-self.seq_len)
        end = self.seq_len+shift
        input_ids = torch.LongTensor(samps[shift:end])
        masks = {k: torch.BoolTensor(masks[k][shift:end]) for k in masks}
        if out_samps is not None:
            output_ids = torch.LongTensor(out_samps[shift:end])
        else:
            output_ids = input_ids[1:]
            input_ids = input_ids[:-1]
        if "input_pad_mask" not in masks:
            m = (input_ids==self.pad_id)|(input_ids==self.eos_id)
            masks["input_pad_mask"] = m
        if "output_pad_mask" not in masks:
            m = (output_ids==self.pad_id)|(output_ids==self.bos_id)
            masks["output_pad_mask"] = m

        return { "input_ids":input_ids, "output_ids":output_ids, **masks}


    def __getitem__(self, idx):
        """
        Returns:
            ret_dict: dict
                "input_ids": long tensor (S,)
                "output_ids": long tensor (S,)
                "input_pad_mask": bool tensor (S,)
                "output_pad_mask": bool tensor (S,)
        """
        if self.concat:
            if self.dynamic_concat:
                return self.dynamic_concat_getitem(idx)
            samp = self.inpt_seqs[idx:idx+self.seq_len]
        else:
            samp = self.inpt_seqs[idx]
        input_ids = samp[:-1]
        output_ids = samp[1:]

        ## Masks
        masks = dict() if self.masks is None else\
                {k:v for k,v in self.masks.items()}
        for k in masks:
            if self.concat:
                masks[k] = masks[k][idx:idx+self.seq_len]
            else:
                masks[k] = masks[k][idx]
        if "input_pad_mask" not in masks:
            m = (input_ids==self.pad_id)|(input_ids==self.eos_id)
            masks["input_pad_mask"] = m
        if "output_pad_mask" not in masks:
            m = (output_ids==self.pad_id)|(output_ids==self.bos_id)
            masks["output_pad_mask"] = m

        ret_dict = {
            "input_ids": input_ids,
            "output_ids": output_ids,
            **masks,
        }
        if self.labels is None: return ret_dict

        ## Labels
        for k in self.labels.keys():
            ret_dict[k] = self.labels[k][idx]
        return ret_dict


def get_datasets(config):
    """
    This function creates a toy dataset of sequences. The sequences
    consist of a beginning of sequence token, then a starting token k
    that can take K possible values, a string of N ordered digits
    ranging somewhere in the range of 1-100, and a final output of the
    starting token k.

    Args:
        config: dict
            a dict of configuration settings

    Returns:
        tokenizer: huggingface tokenizer or dl_utils tokenizer
        train_dataset: torch Dataset
        val_dataset: torch Dataset
    """
    tokenizer = dl_utils.tokenizer.Tokenizer()
    pad_id = tokenizer.pad_id
    bos_id = tokenizer.bos_id
    eos_id = tokenizer.eos_id
    special_ids = tokenizer.special_ids
    for k,v in special_ids.items(): config[k] = v
    K = config.get("K", 5)
    N = config.get("N", 10)
    seq_max = config.get("seq_max", 100)
    n_padding = config.get("n_padding", 4)
    k_offset = len(special_ids)
    seq_offset = k_offset + K
    n_samples = config.get("n_samples", 1000)
    samples = []
    for i in range(n_samples):
        leading_pad = np.random.randint(n_padding)
        samp = [pad_id for _ in range(leading_pad)]
        samp.append(bos_id)
        k = np.random.randint(0,K) + k_offset
        samp.append(k)
        seq_start = np.random.randint(0,seq_max-N)
        s = seq_start+seq_offset
        samp += list(range(s, s+N))
        samp.append(k)
        samp.append(eos_id)
        samp += [pad_id for _ in range(n_padding-leading_pad)]
        samples.append(samp)
    samples = np.asarray(samples)
    config["n_tokens"] = seq_offset+seq_max
    config["seq_len"] = samples.shape[-1]
    n_train = int(n_samples*0.8)
    train_samps =   samples[:n_train]
    val_samps =     samples[n_train:]
    train_dataset = CausalDataset(data=train_samps, **special_ids)
    val_dataset =   CausalDataset(data=val_samps, **special_ids)

    # Kinda hacky, but will help with printing example
    ids = np.unique(samples.reshape(-1))
    id2word = tokenizer.id2word
    id2word = {
        **id2word,
        **{i:str(i) for i in ids if i not in id2word}
    }
    tokenizer.id2word = id2word
    tokenizer.word2id = {v:k for k,v in tokenizer.id2word.items()}
    return tokenizer, train_dataset, val_dataset

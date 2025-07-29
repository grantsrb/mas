import torch
from dl_utils.utils import device_fxn

def get_tforced_stepwise_hook(comms_dict):
    def hook_fn(module, input, output):
        if "loop_count" not in comms_dict:
            comms_dict["loop_count"] = 0
        # output is assumed to be of shape (batch, seq_length, hidden_size)
        src_idx = comms_dict.get("src_idx",0)
        trg_idx = comms_dict.get("trg_idx",1)
        varb_idx = comms_dict.get("varb_idx",None)

        if hasattr(output,"hidden_states"):
            main_trg_actvs = output["hidden_states"]
        elif type(output)==tuple:
            main_trg_actvs = output[0]
        else:
            main_trg_actvs = output
        if comms_dict.get("intrv_vecs",None) is None:
            comms_dict["intrv_vecs"] = torch.zeros_like(main_trg_actvs)

        # Prep source vectors, shape (B,S2,D)
        main_src_actvs = comms_dict["src_activations"]

        src_swap_mask = comms_dict["src_swap_masks"]>=0
        trg_swap_mask = comms_dict["trg_swap_masks"]>=0

        placeholder = torch.empty_like(main_trg_actvs)
        trg_actvs = main_trg_actvs[trg_swap_mask]
        src_actvs = main_src_actvs[:,:-1][src_swap_mask]

        intrv_module = comms_dict["intrv_module"]
        outs = intrv_module(
            target=trg_actvs,
            source=src_actvs,
            target_idx=trg_idx,
            source_idx=src_idx,
            varb_idx=varb_idx,
        )

        comms_dict["intrv_vecs"][trg_swap_mask] = outs

        placeholder[trg_swap_mask] = outs
        placeholder[~trg_swap_mask] = main_trg_actvs[~trg_swap_mask]

        main_trg_actvs = placeholder

        # device = main_trg_actvs.get_device()
        # param = torch.nn.Parameter(torch.ones((1,))).to(device)
        # main_trg_actvs = main_trg_actvs*param

        if hasattr(output,"hidden_states"):
            output["hidden_states"] = main_trg_actvs
            return output
        elif type(output)==tuple:
            return (main_trg_actvs,) + output[1:]
        else:
            return main_trg_actvs

    return hook_fn

def get_stepwise_hook(comms_dict):
    def hook_fn(module, input, output):
        if "loop_count" not in comms_dict:
            comms_dict["loop_count"] = 0
        # output is assumed to be of shape (batch, seq_length, hidden_size)
        src_idx = comms_dict.get("src_idx",0)
        trg_idx = comms_dict.get("trg_idx",1)
        varb_idx = comms_dict.get("varb_idx",None)

        if hasattr(output,"hidden_states"):
            main_trg_actvs = output["hidden_states"]
        elif type(output)==tuple:
            main_trg_actvs = output[0]
        else:
            main_trg_actvs = output

        # ensure targ vectors shas shape (B,S1,D)
        seq_dim = True
        if len(main_trg_actvs.shape)==2:
            seq_dim = False
            main_trg_actvs = main_trg_actvs[:,None]
        n_loops = main_trg_actvs.shape[1]

        # Prep source vectors, shape (B,S2,D)
        main_src_actvs = comms_dict["src_activations"]

        outs_list = []
        ret_output = True
        for loopi in range(n_loops):
            trg_actvs = main_trg_actvs[:,loopi]
            lc = comms_dict["loop_count"]
            comms_dict["loop_count"] += 1
            tsmask = comms_dict["trg_swap_masks"][:,lc]
            batch_bools = tsmask>=0
            if comms_dict.get("intrv_vecs",None) is None:
                comms_dict["intrv_vecs"] = []
            comms_dict["intrv_vecs"].append(torch.empty_like(trg_actvs))
            if not torch.any(batch_bools):
                outs_list.append(trg_actvs)
                continue
            ret_output = False

            placeholder = torch.empty_like(trg_actvs)
            placeholder[~batch_bools] = trg_actvs[~batch_bools]
            device = device_fxn(batch_bools.get_device())
            src_rows = torch.arange(len(batch_bools)).long().to(device)[batch_bools]
            # Find the locations in the src swap mask that have the same swap
            # order as the trg swap mask for this loop
            src_cols = (comms_dict["src_swap_masks"]==tsmask[:,None]).long()
            src_cols = torch.argmax(src_cols, dim=-1)[batch_bools]
            src_actvs = main_src_actvs[src_rows,src_cols]
            trg_actvs = trg_actvs[batch_bools]

            intrv_module = comms_dict["intrv_module"]
            outs = intrv_module(
                target=trg_actvs,
                source=src_actvs,
                target_idx=trg_idx,
                source_idx=src_idx,
                varb_idx=varb_idx,
            )

            comms_dict["intrv_vecs"][-1][batch_bools] = outs

            placeholder[batch_bools] = outs
            outs = placeholder
            outs_list.append(outs)

        if ret_output: return output # only occurs if no interventions

        outs = torch.stack(outs_list, dim=1)
        if not seq_dim: # handle case where original shape was (B,D)
            outs = outs[:,0]
        if hasattr(output,"hidden_states"):
            output["hidden_states"] = outs
            return output
        elif type(output)==tuple:
            return (outs,) + output[1:]
        else:
            return outs

    return hook_fn

def get_indywise_hook(comms_dict):
    def hook(module, inp, out):
        """
        out: tensor (B,M,D) or dict
        """
        h = out
        if type(out)==dict:
            h = h["hidden_states"]
        device = device_fxn(h.get_device())
        og_h_shape = h.shape
        intrv_modu = comms_dict["intrv_module"]
        src_idx = comms_dict.get("src_idx",0)
        trg_idx = comms_dict.get("trg_idx",1)
        varb_idx = comms_dict.get("varb_idx",None)

        # General to multi-dimensional states or single vector states
        source_actvs = comms_dict["src_activations"]
        B,S = source_actvs.shape[:2]
        source_actvs = source_actvs.reshape(B,S,-1)
        source_actvs = source_actvs.to(device)

        # Get positional indices of the interchange for each sample in
        # the batch.
        source_seq_idxs = comms_dict["src_swap_idxs"].long()
        trg_seq_idxs = comms_dict["trg_swap_idxs"].long()
        batch_bools = trg_seq_idxs==comms_dict["loop_count"]
        h = h.reshape(B,-1) # assume no seq dim
        intr_out = h.clone()

        comms_dict["loop_count"] += 1
        if comms_dict.get("intrv_vecs", None) is None:
            comms_dict["intrv_vecs"] = []
        comms_dict["intrv_vecs"].append(torch.empty_like(intr_out))
        if not torch.any(batch_bools):
            h = h.reshape(og_h_shape)
            if type(out)==dict:
                out["hidden_states"] = h
                h = out
            return h

        # Get appropriate inputs for interchange
        idxs = torch.arange(len(batch_bools)).long().to(device)
        idxs = idxs[batch_bools]
        #trg_idxs = trg_seq_idxs[batch_bools]
        source_idxs = source_seq_idxs[batch_bools]

        trg_inpts = h[idxs]
        source_inpts = source_actvs[idxs, source_idxs]

        # Perform causal interchange
        outs = intrv_modu(
            target=trg_inpts,
            source=source_inpts.to(device),
            target_idx=trg_idx,
            source_idx=src_idx,
            varb_idx=varb_idx,)

        comms_dict["intrv_vecs"][-1][idxs] = outs

        # Place causally intervened outputs into appropriate locations
        # in original output tensor. We do it this way to avoid auto-grad
        # errors for in-place operations
        intr_out[idxs] = 0
        intr_out[idxs] += outs

        intr_out = intr_out.reshape(og_h_shape)
        if type(out)==dict:
            out["hidden_states"] = intr_out
            intr_out = out
        return intr_out
    return hook

# Helper: get the module corresponding to the chosen layer.
def get_hook_module(model, hook_layer):
    if type(hook_layer)==str: # optionally specify module name string
        for name,modu in model.named_modules():
            if name==hook_layer:
                return modu
    # For LLaMA-style models the transformer layers might be stored in model.model.layers or model.transformer.h.
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[hook_layer]
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[hook_layer]
    else:
        raise ValueError("Cannot locate hook layer in the model.")

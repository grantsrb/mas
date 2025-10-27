import torch
import torch.nn.functional as F
import math

def device_fxn(device):
    if type(device)==int and device<0:
        return "cpu"
    return device

def equalize_shapes(trg_actvs, src_actvs):
    """
    We need to handle cases where the src and trg activations are a sequence
    and a grid. In these cases, typically the first element of the sequence
    is the "CLS" token, which we want to remove.

    Args:
        trg_actvs: tensor (B,C,H,W) or (B,S,D)
        src_actvs: tensor (B,C,H,W) or (B,S,D)

    Returns:
        trg_actvs: tensor (B*H*W, C) or (B*S,D)
        src_actvs: tensor (B*H*W, C) or (B*S,D)
    """
    if trg_actvs.shape!=src_actvs.shape:
        if len(trg_actvs.shape)==4:
            B,Dt,H,W = trg_actvs.shape
            if len(src_actvs.shape)==3:
                B,S,Ds = src_actvs.shape
                S = S-1
                R = int(math.sqrt(S))
                src_actvs = src_actvs[:,1:].reshape(B,R,R,Ds).permute(0,3,1,2)
            src_actvs = F.interpolate(
                src_actvs, size=(H, W), mode="bilinear", align_corners=True,
            )
        elif len(trg_actvs.shape)==3: # src is a grid
            B,S,Dt = trg_actvs.shape
            S = S-1
            R = int(math.sqrt(S))
            trg_actvs = trg_actvs[:,1:].reshape(B,R,R,Dt).permute(0,3,1,2)
            if len(src_actvs.shape)==3:
                B,S,Ds = src_actvs.shape
                S = S-1
                Rs = int(math.sqrt(S))
                src_actvs = src_actvs[:,1:].reshape(B,Rs,Rs,Ds).permute(0,3,1,2)
            src_actvs = F.interpolate(
                src_actvs, size=(R, R), mode="bilinear", align_corners=True,
            )
    if len(trg_actvs.shape)==4:
        B,Dt,H,W = trg_actvs.shape
        _,Ds,_,_ = src_actvs.shape
        trg_actvs = trg_actvs.permute(0,2,3,1)
        src_actvs = src_actvs.permute(0,2,3,1)
    else:
        B,S,Dt = trg_actvs.shape
        _,S,Ds = src_actvs.shape
    return trg_actvs.reshape(-1,Dt), src_actvs.reshape(-1,Ds)
        

def get_vision_hook(comms_dict):
    def hook(module, inp, out):
        """
        out: tensor (B,C,H,W) or dict
        """
        trg_actvs = out
        if type(out)==dict:
            trg_actvs = trg_actvs["hidden_states"]
        elif hasattr(out, "hidden_states"):
            trg_actvs = trg_actvs.hidden_states
        elif type(out)==tuple:
            trg_actvs = out[0]
        device = device_fxn(trg_actvs.get_device())
        og_actvs = trg_actvs.clone()
        og_shape = og_actvs.shape

        intrv_modu = comms_dict["intrv_module"].to(device)
        src_idx = comms_dict.get("src_idx",0)
        trg_idx = comms_dict.get("trg_idx",1)
        varb_idx = comms_dict.get("varb_idx",None)

        grad_state = comms_dict.get("req_grad", None)
        if grad_state is not None:
            prev_grad_state = torch.is_grad_enabled()
            torch.set_grad_enabled(grad_state)

        src_actvs = comms_dict["src_activations"]
        trg_actvs, src_actvs = equalize_shapes(trg_actvs, src_actvs)
        src_actvs = src_actvs.to(device)

        # Get positional indices of the interchange for each sample in
        # the batch.
        default_bools = torch.ones(src_actvs.shape[0]).bool()
        src_swap_bools = comms_dict.get("src_swap_bools", default_bools)
        trg_swap_bools = comms_dict.get("trg_swap_bools", default_bools)
        if src_swap_bools is None:
            src_swap_bools = default_bools
        if trg_swap_bools is None:
            trg_swap_bools = default_bools

        trg_inputs = trg_actvs.to(device)[trg_swap_bools]
        src_inputs = src_actvs.to(device)[src_swap_bools]

        if comms_dict.get("low_rank_transformation", None) is not None:
            low_rank_transformation = comms_dict["low_rank_transformation"].to(device)
            trg_inputs = low_rank_transformation(trg_inputs)
            src_inputs = low_rank_transformation(src_inputs)

        # Perform causal interchange
        intrv_out = intrv_modu(
            target=trg_inputs,
            source=src_inputs,
            target_idx=trg_idx,
            source_idx=src_idx,
            varb_idx=varb_idx,
        )
        if comms_dict.get("low_rank_transformation", None) is not None:
            low_rank_transformation = comms_dict["low_rank_transformation"]
            intrv_out = low_rank_transformation(intrv_out, inverse=True)
        if len(og_shape)==4:
            B,C,H,W = og_shape
            intrv_out = intrv_out.reshape(B,H,W,C).permute(0,3,1,2)
        elif len(og_shape)==3:
            B,S,D = og_shape
            intrv_out = intrv_out.reshape(B,-1,D)
            B,S2,D = intrv_out.shape
            if S2!=S:
                intrv_out = torch.cat([
                    og_actvs[:,0:1], intrv_out.reshape(B,S,D)
                ], dim=1)
        comms_dict["intrv_vectors"] = intrv_out

        if grad_state is not None:
            torch.set_grad_enabled(prev_grad_state)

        if type(out)==dict:
            out["hidden_states"] = intrv_out
            intrv_out = out
        elif type(out)==tuple:
            intrv_out = (intrv_out,) + out[1:]
        elif hasattr(out, "hidden_states"):
            out.hidden_states = intrv_out
            intrv_out = out
        return intrv_out
    return hook

def hook_vision_model(model, layer_name, alignment):
    """
    Hooks the model at the desired layer to collect the activations
    """
    if not hasattr(alignment, "comms_dict"):
        alignment.comms_dict = dict()
        alignment.comms_dict["intrv_module"] = alignment
        alignment.comms_dict["src_idx"] = 0
        alignment.comms_dict["trg_idx"] = 1
        alignment.comms_dict["varb_idx"] = None
        alignment.comms_dict["src_activations"] = None
        alignment.comms_dict["src_swap_bools"] = None
        alignment.comms_dict["trg_swap_bools"] = None

    hook = get_vision_hook(alignment.comms_dict)
    for name, module in model.named_modules():
        if name == layer_name:
            return module.register_forward_hook(hook)
    raise ValueError(f"Layer {layer_name} not found in model")

def get_actvs_hook(comms_dict, key):
    def hook_fn(module, input, output):
        comms_dict[key].append(output.cpu())
    return hook_fn

def hook_model_layer(model, layer_name, key=None):
    # hook the model at the desired layer to collect the activations
    comms_dict = dict()
    if key is None:
        key = layer_name
    comms_dict[key] = []
    hook = get_actvs_hook(comms_dict, key)
    handle = None
    for name, module in model.named_modules():
        if name == layer_name:
            handle = module.register_forward_hook(hook)
            break
    if handle is None:
        valid_layers = sorted(list([name for name, _ in model.named_modules()]))
        raise ValueError(f"Layer {layer_name} not found in model. Valid layers: {valid_layers}")
    return handle, comms_dict

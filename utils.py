import torch
import numpy as np
import json
import yaml
import os
from tqdm import tqdm
import copy
import sys

def device_fxn(device):
    if device<0: return "cpu"
    return device

def get_len_hist(arr):
    """
    Creates a histogram of the lengths of the contents of the argued array.
    Defaults to none in cases of objects with no length
    """
    hist = dict()
    for el in arr:
        try:
            hist[len(el)] = hist.get(len(el), 0)+1
        except:
            hist[None] = hist.get(None, 0)+1
    return hist

def get_activations_hook(comms_dict, key="source", to_cpu=False):
    """
    Returns a hook that can be used to collect activations at the
    specified layer. The argued comms_dict is expected to have a key
    for the activations that stores a list of past activations. Each
    call to the hook will append the latest activations to the list
    stored in the comms dict.

    Args:
        comms_dict: dict
            This is the communications dict. It will store a list of the
            activations that come through the hooked layer. Each new entry
            in the list corresponds to a call to the model. The argued "key"
            to this function will be the key to access the activation list.
        key: str
            the key to use to store the activations list.
        to_cpu: bool
            if true, the activations will be stored on the CPU. Otherwise
            they will stay on their current device.
    Returns:
        hook: pytorch hook function
            this function will simply collect the activations that exit
            the hooked layer.
    """
    if key not in comms_dict or type(comms_dict[key])!=list:
        comms_dict[key] = []

    if to_cpu:
        def hook(module, inp, out):
            if type(out)==dict:
                if "hidden_states" in out:
                    comms_dict[key].append(out["hidden_states"].cpu())
                else:
                    comms_dict[key].append(out["attentions"].cpu())
            elif type(out)==tuple:
                comms_dict[key].append(out[0].cpu())
            elif hasattr(out, "hidden_states"):
                if type(out.hidden_states)==tuple:
                    hstates = out.hidden_states[-1].cpu()
                else:
                    hstates = out.hidden_states.cpu()
                comms_dict[key].append(hstates.cpu())
            else:
                comms_dict[key].append(out.cpu())
    else:
        def hook(module, inp, out):
            if type(out)==dict:
                comms_dict[key].append(out["hidden_states"])
            elif type(out)==tuple:
                comms_dict[key].append(out[0])
            elif hasattr(out, "hidden_states"):
                if type(out.hidden_states)==tuple:
                    hstates = out.hidden_states[-1]
                else:
                    hstates = out.hidden_states
                comms_dict[key].append(hstates)
            else:
                comms_dict[key].append(out)
    return hook

def register_activation_hooks(
        model,
        layers,
        comms_dict,
        to_cpu=True):
    """Helper function to register forward hooks at multiple layers"""
    handles = []
    hooked_layers = set()
    for name, mod in model.named_modules():
        if name in layers:
            hooked_layers.add(name)
            hook = get_activations_hook(
                comms_dict=comms_dict,
                key=name,
                to_cpu=to_cpu,)
            handle = mod.register_forward_hook(hook)
            handles.append(handle)
    try:
        missing_layers = set(layers)-hooked_layers
        if len(missing_layers) > 0:
            print("Layers", missing_layers, "not found")
            print(model)
    except:
        print("Failed to analyze missing layers")
        print("layers:", layers)
        print("hooked:", hooked_layers)
    layers = list(hooked_layers)
    return handles, layers


def collect_activations(
        model,
        input_ids,
        attention_mask=None,
        pad_mask=None,
        task_mask=None,
        layers=None,
        comms_dict=None,
        batch_size=500,
        to_cpu=True,
        ret_attns=False,
        ret_pred_ids=False,
        ret_logits=False,
        tforce=False,
        n_steps=0,
        ret_gtruth=False,
        verbose=False,):
    """
    Get the response from the argued layers in the model.

    Args:
        model: torch Module or torch gpu Module
        input_ids: long tensor (N,S)
        pad_mask: bool tensor (N,S)
            true means padding
        task_mask: bool tensor (N,S)
            false means teacher force the corresponding id
        layers: None or sequence of str
            Name of layers to collect activations from. If None will
            collect outputs from every layer.
        batch_size: int
            Optionally break the processing into batches.
        to_cpu: bool
            If true, torch tensors will be placed onto the cpu.
        ret_attn: bool
            if true, will return the attention values as well.
        ret_pred_ids: bool
            if true, will return the prediction ids under "pred_ids"
        ret_logits: bool
            if true, will return the prediction logits under "logits"
        tforce: bool
            will use teacher forcing on all inputs if true. if false,
            uses the task mask to determine when to teacher force.
            false indices of the task mask indicate do teacher force.
        n_steps: int
            the number of additional steps to generate on top of the
            initial input.
        ret_gtruth: bool
            if true, the model will return the ground truth pred_ids
            where tmask is false
    returns: 
        comms_dict: dict
            The keys will consist of the corresponding layer name. The
            values will be the activations at that layer.

            "layer_name": torch tensor (N, ...)
    """
    if comms_dict is None: comms_dict = dict()

    # Layers is modified to only the layers that were found in the model.
    # If you argue a layer not found in the model, it will be ignored.
    # There will, however, be a print statement that indicates that the
    # layer was not found.
    handles, layers = register_activation_hooks(
        model=model, layers=layers, comms_dict=comms_dict, to_cpu=to_cpu,)

    if batch_size is None: batch_size = len(input_ids)

    device = device_fxn(next(model.parameters()).get_device())
    outputs = {key:[] for key in layers}
    if ret_attns:
        assert len(input_ids)<=batch_size
        outputs["attentions"] = []
    if ret_pred_ids: outputs["pred_ids"] = []
    if ret_logits: outputs["logits"] = []
    rnge = range(0,len(input_ids), batch_size)
    if verbose: rnge = tqdm(rnge)
    for batch in rnge:
        x = input_ids[batch:batch+batch_size]
        amask = None
        pmask = None
        tmask = None
        if attention_mask is not None:
            amask = attention_mask[batch:batch+batch_size].to(device)
        if pad_mask is not None:
            pmask = pad_mask[batch:batch+batch_size].to(device)
        if task_mask is not None and not tforce:
            tmask = task_mask[batch:batch+batch_size].to(device)
        try:
            out_dict = model(
                inpts=x.to(device),
                task_mask=tmask,
                pad_mask=pmask,
                output_attentions=ret_attns,
                n_steps=n_steps,
                ret_gtruth=ret_gtruth,
                tforce=tforce,)
        except:
            out_dict = model(
                input_ids=x.to(device),
                attention_mask=amask if amask is not None else pmask,
                output_attentions=ret_attns,)
        if ret_attns:
            outputs["attentions"].append(out_dict["attentions"][0])
        if ret_logits:
            outputs["logits"].append(out_dict["logits"])
        if ret_pred_ids:
            if "pred_ids" in out_dict:
                outputs["pred_ids"].append(out_dict["pred_ids"])
            else:
                outputs["pred_ids"].append(torch.argmax(out_dict.logits, dim=-1))
        for k in layers:
            output = comms_dict[k]
            if type(output)==list: # There could be an internal for loop in model
                if len(output)==0:
                    print(k, "isn't producing")
                    assert False
                if len(output[0].shape)<=4:
                    try:
                        output = torch.stack(output, dim=1)
                    except:
                        print("Failed for", k)
                        output = torch.stack(output, dim=1)
                elif len(output)==1:
                    raise NotImplemented
                    #output = output[0]
                else:
                    print(output[0].shape)
                    raise NotImplemented
                    #output = torch.cat(output, dim=1)
            outputs[k].append(output)
            comms_dict[k] = []
    if len(outputs[layers[0]])>1:
        # Concat batches together
        outputs = {k:torch.cat(v,dim=0) for k,v in outputs.items()}
    else:
        outputs = {k:v[0] for k,v in outputs.items()}
    if to_cpu:
        outputs = {k:v.cpu() for k,v in outputs.items()}

    # Ensure we do not create a memory leak
    for i in range(len(handles)):
        handles[i].remove()
    del handles

    return outputs

def load_yaml(file_name):
    """
    Loads a yaml file as a python dict

    file_name: str
        the path of the yaml file
    """
    file_name = os.path.expanduser(file_name)
    with open(file_name,'r') as f:
        yam = yaml.safe_load(f)
    return yam

def load_json(file_name):
    """
    Loads a json file as a python dict

    file_name: str
        the path of the json file
    """
    file_name = os.path.expanduser(file_name)
    with open(file_name,'r') as f:
        s = f.read()
        j = json.loads(s)
    return j

def default_to_list(val, n_el):
    """
    Simplifies extracting the appropriate type for parameters that
    are lists.

    Args:
        val: list or anything else
        n_el: int
            desired number of elements in the final list
    Returns:
        list: length n_el
            returns a list of length n_el
    """
    if type(val) != list:
        return [val for _ in range(n_el)]
    elif len(val)<n_el:
        return val + [val[0] for _ in range(n_el-len(val))]
    return val[:n_el]

def load_text(file_name, strip=True):
    file_name = os.path.expanduser(file_name)
    with open(file_name, "r") as f:
        lines = f.readlines()
    if strip:
        lines = [line.strip() for line in lines]
    return lines

def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False

def save_json(data, file_name):
    """
    saves a dict to a json file

    data: dict
    file_name: str
        the path that you would like to save to
    """
    failure = True
    n_loops = 0
    while failure and n_loops<10*len(data):
        failure = False
        n_loops += 1
        try:
            with open(file_name, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        except (TypeError, OverflowError):
            data = {**data}
            keys = list(data.keys())
            for k in keys:
                if not is_jsonable(data[k]):
                    if type(data[k])==dict:
                        data = {**data, **data[k]}
                        del data[k]
                    elif type(data[k])==set:
                        data[k] = list(data[k])
                    elif hasattr(data[k],"__name__"):
                        data[k] = data[k].__name__
                    else:
                        try:
                            data[k] = str(data[k])
                        except:
                            del data[k]
                            print("Removing", k, "from json")
            try:
                with open(file_name, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
            except:
                print("trying again")
                failure = True



def load_json_or_yaml(file_name):
    """
    Loads a json or a yaml file (determined by its extension) as a python
    dict.

    Args:
        file_name: str
            the path of the json/yaml file
    Returns:
        d: dict
            a dict representation of the loaded file
    """
    if ".json" in file_name:
        return load_json(file_name)
    elif ".yaml" in file_name:
        return load_yaml(file_name)
    raise NotImplemented

def str_to_typed_value(val):
    if val.lower() in {"none", "null", "na"}:
        val = None
    elif "," in val:
        val = val.split(",")
        val = [str_to_typed_value(v) for v in val if v!=""]
    elif val.lower()=="true":
        val = True
    elif val.lower()=="false":
        val = False
    elif val.isnumeric():
        val = int(val)
    elif val.replace(".", "").isnumeric() and val.count(".")<=1:
        val = float(val)
    return val

def get_command_line_args(args=None):
    if args is None: args = sys.argv[1:]
    config = {}
    command_keys = []
    for arg in args:
        if ".yaml" in arg or ".json" in arg:
            if "=" in arg: arg = arg.split("=")[-1].strip()
            config = {**config, **load_json_or_yaml(arg)}
        elif "=" in arg:
            key,val = arg.split("=")
            config[key] = str_to_typed_value(val)
            command_keys.append(key)
    return config, command_keys


def extract_ids(string, tokenizer):
    """
    Returns the token_ids for the string, but does so without the
    bos_id and without the eos_id.

    Args:
        string: str
        tokenizer: Tokenizer object
    """
    ids = tokenizer.convert_tokens_to_ids(string)
    if ids is None or (type(ids)==int and ids == 0):
        ids = tokenizer(string, return_tensors="pt")["input_ids"]
        assert len(ids.shape)<=2
        if len(ids.shape)==2: ids = ids[0] # just removes wrapping
        if ids[0]==tokenizer.bos_token_id:
            ids = ids[1:]
        if ids[-1]==tokenizer.eos_token_id and len(ids)>1:
            ids = ids[:-1]
    elif not hasattr(ids, "__len__"):
        ids = torch.LongTensor([ids])
    return ids

def tensor2str(seq, n=2, delim=","):
    """
    seq: tensor
        must have numeric values
    """
    t = float if "float" in str(seq.dtype) else int
    return delim.join( ["{:2}".format(str(t(s))[:n]) for s in seq] )

def run_cmodel_to_completion(
        cmodel,
        inpt_token,
        varbs=None,
        post_varbs=False,
        info=None,
        end_tokens={},
        incl_all_varbs=False,
        *args, **kwargs,
    ):
    """
    Runs a causal model until it produces None or a token contained
    in the end_tokens dict.

    Args:
        cmodel: CausalModel
            accepts a token and a dict of variables
        inpt_token: tokenized string
        varbs: (optional) dict of variables or None
            optionally argue variables to argue to the causal model
            with the first input token.
        post_varbs: bool
            if true, will return the variables after each
            input token. Otherwise returns the variables
            before each token.
        info: (optional) dict
            some cmodels require additional info which can be specified
            in the info dict
        end_tokens: set
            optionally specify tokens that indicate the model is finished.
            None is assumed to be an end condition for all cmodels
        incl_all_varbs: bool
            if true, will include the output variables even if post_varbs
            is false and will include the initial input variables even
            if post_varbs is True
    Returns:
        outp_tokens: list of tokens
            the output produced by the causal model after each
            input token.
        varb_list: list of dicts
            the input variables used by the causal model with each
            input token.
        outp_task_mask: list of bools
            a task mask output by the causal model
    """
    return run_for_n_steps(
        cmodel=cmodel,
        inpt_token=inpt_token,
        varbs=varbs,
        post_varbs=post_varbs,
        info=info,
        end_tokens=end_tokens,
        n_steps=None,
        incl_all_varbs=incl_all_varbs,
        *args, **kwargs,
    )

def run_for_n_steps(
        cmodel,
        inpt_token,
        n_steps=None,
        varbs=None,
        post_varbs=False,
        info=None,
        end_tokens=set(),
        incl_all_varbs=False,
        *args, **kwargs,
    ):
    """
    Runs a causal model until it prduces None or a token contained in the end_tokens
    dict.

    Args:
        cmodel: CausalModel
            accepts a token and a dict of variables
        inpt_token: tokenized string
        n_steps: (optional) int or None
            if int, will run the causal model for only that many steps or until
            an end condition is met. If None, will run the causal model until an
            end condition.
        varbs: (optional) dict of variables or None
            optionally argue variables to argue to the causal model
            with the first input token.
        info: (optional) dict
            some cmodels require additional info which can be specified
            in the info dict
        end_tokens: set
            optionally specify tokens that indicate the model is finished.
            None is assumed to be an end condition for all cmodels
        post_varbs: bool
            if true, will return the variables after each
            input token. Otherwise returns the variables
            before each token.
        incl_all_varbs: bool
            if true, will include the output variables even if post_varbs
            is false and will include the initial input variables even
            if post_varbs is True
    Returns:
        outp_tokens: list of tokens
            the output produced by the causal model after each
            input token.
        varb_list: list of dicts
            the input variables used by the causal model with each
            input token.
        outp_task_mask: list of bools
            a task mask output by the causal model
    """
    if end_tokens is None: end_tokens = set()
    outp_token_ids = []
    if not varbs:
        varbs = cmodel.init_varbs
    varb_list = []
    if incl_all_varbs and post_varbs:
        varb_list.append(varbs)
    task_mask = []
    token = inpt_token
    step = 0
    while n_steps is None or step < n_steps:
        step += 1
        if not post_varbs:
            varb_list.append(copy.deepcopy(varbs))
        token, varbs, tmask = cmodel(
            token_id=token, varbs=varbs, info=info, *args, **kwargs
        )
        if post_varbs:
            varb_list.append(copy.deepcopy(varbs))
        task_mask.append(tmask)
        outp_token_ids.append(token)
        if token in end_tokens or token is None:
            break
        elif step > 10000:
            print("Infinite Loop in intrv_datas!!!!")
            print("inpt token:", inpt_token)
            print("outpts:", outp_token_ids[:100])
            print("varbs step1:", varb_list[0])
            print("varbs:", varbs)
            assert False
    if incl_all_varbs and not post_varbs:
        varb_list.append(varbs)
    return outp_token_ids, task_mask, varb_list

def run_til_idx(
        cmodel,
        seq,
        idx=None,
        varbs=None,
        post_varbs=False,
        info=None,
        *args, **kwargs,
    ):
    """
    Runs a causal model on the provided sequence until the
    argued idx. If the idx is None, will run on all tokens
    in the sequence.

    Args:
        cmodel: CausalModel
            accepts a token and a dict of variables
        seq: list of ids
        idx: (optional) int or None
            (inclusive) the index in the sequence to inclusively
            stop at
        varbs: (optional) dict of variables or None
            optionally argue variables to argue to the causal model
            with the first input token.
        info: (optional) dict
            some cmodels require additional info which can be specified
            in the info dict
        post_varbs: bool
            if true, will return the variables after each
            input token. Otherwise returns the variables
            before each token.
    Returns:
        outp_tokens: list of tokens
            the output produced by the causal model after each
            input token.
        varb_list: list of dicts
            the input variables used by the causal model with each
            input token.
        outp_task_mask: list of bools
            a task mask output by the causal model
    """
    if idx is None: idx = len(seq)
    outp_token_ids = []
    if not varbs:
        varbs = cmodel.init_varbs
    varb_list = []
    task_mask = []
    for si,token in enumerate(seq):
        if not post_varbs:
            varb_list.append(copy.deepcopy(varbs))
        token, varbs, tmask = cmodel(
            token_id=token, varbs=varbs, info=info, *args, **kwargs
        )
        if post_varbs:
            varb_list.append(copy.deepcopy(varbs))
        task_mask.append(tmask)
        outp_token_ids.append(token)
        if si==idx:
            break
    return outp_token_ids, task_mask, varb_list
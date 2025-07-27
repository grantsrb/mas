import torch
import pickle
import os
import json
import yaml
import copy
from datetime import datetime
from .utils import get_git_revision_hash, package_versions, get_datetime_str, remove_ending_slash
import numpy as np

BEST_CHECKPT_NAME = "best_checkpt_0.pt.best"

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

def save_checkpt(save_dict,
        save_folder,
        epoch,
        save_name="checkpt",
        ext=".pt",
        del_prev_sd=True,
        best=False):
    """
    Saves a dictionary that contains a statedict

    save_dict: dict
        a dictionary containing all the things you want to save
    save_folder: str
        the full path to save the checkpt file to
    save_name: str
        the name of the file that the save dict will be saved to. This
        function will automatically append the epoch to the end of the
        save name followed by the extention, `ext`.
    epoch: int
        an integer to be associated with this checkpoint
    ext: str
        the extension of the file
    del_prev_sd: bool
        if true, the state_dict of the previous checkpoint will be
        deleted
    best: bool
        if true, additionally saves this checkpoint as the best
        checkpoint under the filename set by BEST_CHECKPT_NAME
    """
    if del_prev_sd and epoch is not None:
        prev_paths = get_checkpoints(save_folder)
        if len(prev_paths) > 0:
            prev_path = prev_paths[-1]
            delete_sds(prev_path)
        elif epoch != 0:
            print("Failed to find previous checkpoint")
    if epoch is None: epoch = 0
    path = "{}_{}{}".format(save_name,epoch,ext)
    path = os.path.join(save_folder, path)
    path = os.path.abspath(os.path.expanduser(path))
    torch.save(save_dict, path)
    if best: save_best_checkpt(save_dict, save_folder)

def delete_sds(checkpt_path):
    """
    Deletes the state_dicts from the argued checkpt path.

    Args:
        checkpt_path: str
            the full path to the checkpoint
    """
    if not os.path.exists(checkpt_path): return
    checkpt = load_checkpoint(checkpt_path)
    keys = list(checkpt.keys())
    for key in keys:
        if "state_dict" in key or "optim_dict" in key:
            del checkpt[key]
    torch.save(checkpt, checkpt_path)

def save_best_checkpt(save_dict, folder):
    """
    Saves the checkpoint under the name set in BEST_CHECKPT_PATH to the
    argued folder

    save_dict: dict
        a dictionary containing all the things you want to save
    folder: str
        the path to the folder to save the dict to.
    """
    path = os.path.join(folder,BEST_CHECKPT_NAME)
    path = os.path.abspath(path)
    torch.save(save_dict,path)

def get_checkpoints(folder, checkpt_exts={'p', 'pt', 'pth'}):
    """
    Returns all .p, .pt, and .pth file names contained within the
    folder. They're sorted by their epoch.

    BEST_CHECKPT_PATH is not included in this list. It is excluded using
    the assumption that it has the extension ".best"

    folder: str
        path to the folder of interest
    checkpt_exts: set of str
        a set of checkpoint extensions to include in the checkpt search.

    Returns:
        checkpts: list of str
            the full paths to the checkpoints contained in the folder
    """
    folder = os.path.expanduser(folder)
    assert os.path.isdir(folder)
    checkpts = []
    for f in os.listdir(folder):
        splt = f.split(".")
        if len(splt) > 1 and splt[-1] in checkpt_exts:
            path = os.path.join(folder,f)
            checkpts.append(path)
    def sort_key(x): return int(x.split(".")[-2].split("_")[-1])
    filt_checkpts = []
    for c in checkpts:
        try:
            sort_key(c)
            filt_checkpts.append(c)
        except: pass
    checkpts = filt_checkpts
    checkpts = sorted(checkpts, key=sort_key)
    return checkpts

def foldersort(x):
    """
    A sorting key function to order folder names with the format:
    <path_to_folder>/<exp_name>_<exp_num>_<ending_folder_name>/

    Assumes that the experiment number will always be the rightmost
    occurance of an integer surrounded by underscores (i.e. _1_)

    x: str
    """
    if x[-1] == "/": x = x[:-1]
    splt = x.split("/")
    if len(splt) > 1: splt = splt[-1].split("_")
    else: splt = splt[0].split("_")
    for s in reversed(splt[1:]):
        try:
            return int(s)
        except:
            pass
    print("Failed to sort:", x)
    return np.inf

def prep_search_keys(s):
    """
    Removes unwanted characters from the search keys string. This
    allows you to easily append a string representing the search
    keys to the model folder name.
    """
    return s.replace(
            " ", ""
        ).replace(
            "[", ""
        ).replace(
            "]", ""
        ).replace(
            "\'", ""
        ).replace(
            ",", ""
        ).replace(
            "/", ""
        )

def get_exp_num(path):
    """
    Finds and returns the experiment number from the argued path.
    """
    return foldersort(path)

def get_exp_name(x):
    """
    Finds and returns the string before the experiment number from
    the argued path.
    
    A sorting key function to order folder names with the format:
    <path_to_folder>/<exp_name>_<exp_num>_<ending_folder_name>/

    Assumes that the experiment number will always be the rightmost
    occurance of an integer surrounded by underscores (i.e. _1_)

    x: str
    """
    if x[-1] == "/": x = x[:-1]
    splt = x.split("/")
    if len(splt) > 1: splt = splt[-1].split("_")
    else: splt = splt[0].split("_")
    exp_num = None
    for i,s in enumerate(reversed(splt)):
        try:
            exp_num = int(s)
            break
        except:
            pass 
    if exp_num is None: return None
    else: return "_".join(splt[:-(i+1)])
  
def is_model_folder(path, exp_name=None, incl_empty=True):
    """
    checks to see if the argued path is a model folder or otherwise.
    i.e. does the folder contain checkpt files and a hyperparams.json?

    path: str
        path to check
    exp_name: str or None
        optionally include exp_name to determine if a folder is a model
        folder based on the name instead of the contents.
    incl_empty: bool
        if true, will include folders without checkpoints as possible
        model folders.
    """
    check_folder = os.path.expanduser(path)
    if not os.path.isdir(check_folder): return False
    if incl_empty and exp_name is not None:
        # Remove ending slash if there is one
        if check_folder[-1]=="/": check_folder = check_folder[:-1]
        folder_splt = check_folder.split("/")[-1]
        # Need to split on underscores and check for entirety of
        # exp_name because exp_name is only the first part of any
        # model folder
        name_splt = exp_name.split("_")
        folder_splt = folder_splt.split("_")
        match = True
        for i in range(len(name_splt)):
            if i >= len(folder_splt) or name_splt[i] != folder_splt[i]:
                match = False
                break
        if match:
            return True
    contents = os.listdir(check_folder)
    is_empty = True
    has_hyps = False
    for content in contents:
        if ".pt" in content: is_empty = False
        if "hyperparams" in content: has_hyps = True
    if incl_empty: return has_hyps or not is_empty
    return not is_empty

def is_incomplete_folder(path):
    """
    checks to see if the argued path is an empty model folder. 
    i.e. does the folder contain a hyperparams.json without checkpt
    files? Generally it is okay to delete empty model folders.

    WARNING: ONLY RETURNS TRUE IF THE FOLDER CONTAINS A HYPERPARAMETERS
    JSON WITHOUT ANY CHECKPOINTS. WILL RETURN FALSE FOR COMPLETELY
    EMPTY FOLDERS!!!!

    path: str
        path to check
    exp_name: str or None
    """
    check_folder = os.path.expanduser(path)
    if not os.path.isdir(check_folder): return False
    contents = os.listdir(check_folder)
    is_empty = True
    has_hyps = False
    for content in contents:
        if ".pt" in content: is_empty = False
        if "hyperparams" in content: has_hyps = True
    return has_hyps and is_empty

def is_exp_folder(path):
    """
    Checks to see if the argued path is an exp folder. i.e. does it
    contain at least 1 model folder.

    Args:
        path: str
            full path to the folder in question.
    Returns:
        is_folder: bool
            if the argued path is to an experiment folder, will return
            true. Otherwise returns false.
    """
    if not os.path.isdir(path): return False
    mfs = get_model_folders(path)
    return len(mfs)>0

def get_model_folders(exp_folder, incl_full_path=False, incl_empty=True):
    """
    Returns a list of paths to the model folders contained within the
    argued exp_folder

    exp_folder - str
        full path to experiment folder
    incl_full_path: bool
        include extension flag. If true, the expanded paths are
        returned. otherwise only the end folder (i.e.  <folder_name>
        instead of exp_folder/<folder_name>)
    incl_empty: bool
        if true, will include folders without checkpoints as possible
        model folders.

    Returns:
        list of folder names (see incl_full_path for full path vs end
        point)
    """
    folders = []
    exp_folder = os.path.expanduser(exp_folder)
    if exp_folder[-1]=="/":
        exp_name = exp_folder[:-1].split("/")[-1]
    else:
        exp_name = exp_folder.split("/")[-1]
    if ".pt" in exp_folder[-4:]:
        # if model file, return the corresponding folder
        folders = [ "/".join(exp_folder.split("/")[:-1]) ]
    else:
        for d, sub_ds, files in os.walk(exp_folder):
            for sub_d in sub_ds:
                check_folder = os.path.join(d,sub_d)
                is_mf = is_model_folder(
                    check_folder,exp_name=exp_name,incl_empty=incl_empty
                )
                if is_mf:
                    if incl_full_path:
                        folders.append(check_folder)
                    else:
                        folders.append(sub_d)
        if is_model_folder(exp_folder,incl_empty=incl_empty):
            folders.append(exp_folder)
    folders = list(set(folders))
    if incl_full_path: folders = [os.path.expanduser(f) for f in folders]
    return sorted(folders, key=foldersort)

def load_checkpoint(path, use_best=True, ret_path=False):
    """
    Loads the save_dict into python. If the path is to a model_folder,
    the loaded checkpoint is the BEST checkpt if available, otherwise
    the checkpt of the last epoch

    Args:
        path: str
            path to checkpoint file or model_folder
        use_best: bool
            if true, will load the best checkpt based on validation metrics
        ret_path: bool
            if true, will return the path used for the checkpoint
    Returns:
        checkpt: dict
            a dict that contains all the valuable information for the
            training.
    """
    path = os.path.expanduser(path)
    hyps = None
    if os.path.isdir(path):
        best_path = os.path.join(path,BEST_CHECKPT_NAME)
        if use_best and os.path.exists(best_path):
            path = best_path 
        else:
            checkpts = get_checkpoints(path)
            if len(checkpts)==0: return None
            path = checkpts[-1]
    data = torch.load(
        path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    data["loaded_path"] = path
    if "config" in data:
        data["hyps"] = data["config"]
    if "hyps" not in data: 
        data["hyps"] = get_hyps(path)
    if "epoch" not in data:
        # TODO Untested!!
        ext = path.split(".")[-1]
        data["epoch"] = int(path.split("."+ext)[0].split("_")[-1])
        torch.save(data, path) 
    if ret_path:
        return data, path
    return data

def load_model(path, models, load_sd=True, use_best=False,
                                           hyps=None,
                                           verbose=True):
    """
    Loads the model architecture and state dict from a .pt or .pth
    file. Or from a training save folder. Defaults to the last check
    point file saved in the save folder.

    path: str or dict
        either .pt,.p, or .pth checkpoint file; or path to save folder
        that contains multiple checkpoints. if dict, must be a checkpt
        dict.
    models: dict
        A dict of the potential model classes. This function is
        easiest if you import each of the model classes in the calling
        script and simply pass `globals()` as the argument for this
        parameter. If None is argued, `globals()` is used instead.
        (can usually pass `globals()` as the arg assuming you have
        imported all of the possible model classes into the script
        that calls this function)

        keys: str
            the class names of the potential models
        vals: Class
            the potential model classes
    load_sd: bool
        if true, the saved state dict is loaded. Otherwise only the
        model architecture is loaded with a random initialization.
    use_best: bool
        if true, will load the best model based on validation metrics
    hyps: dict (optional)
        if you would like to argue your own hyps, you can do that here
    """
    if type(path) == type(str()):
        path = os.path.expanduser(path)
        hyps = None
        data = load_checkpoint(path,use_best=use_best)
    else: data = path
    if 'hyps' in data:
        kwargs = data['hyps']
    elif 'model_hyps' in data:
        kwargs = data['model_hyps']
    elif "config" in data:
        kwargs = data["config"]
    else:
        kwargs = get_hyps(path)
    if models is None: models = globals()
    model = models[kwargs['model_type']](**kwargs)
    if "state_dict" in data and load_sd:
        print("loading state dict")
        try:
            model.load_state_dict(data["state_dict"])
        except:
            try:
                sd = {k:v.clone() for k,v in data["state_dict"].items()}
                m_sd = model.state_dict()
                keys = list(sd.keys())
                for k in keys:
                    if "model." in key and key not in m_sd:
                        # Simply remove "model." from keys
                        new_key = ".".join(key.split(".")[1:])
                        sd[new_key] = sd[key]
                        del sd[key]
                model.load_state_dict(sd)
            except:
                print("failed to load state dict, attempting fix")
                sd = data["state_dict"]
                m_sd = model.state_dict()
                keys = {*sd.keys(), *m_sd.keys()}
                for k in keys:
                    if k not in sd:
                        print("Error for", k)
                        sd[k] = getattr(model, k)
                    if k not in m_sd:
                        print("Error for", k)
                        setattr(model, k, sd[k])
                model.load_state_dict(sd)
                print("succeeded!")
    else:
        print("state dict not loaded!")
    return model

def get_hyps(folder):
    """
    Returns a dict of the hyperparameters collected from the json or
    yaml save file in the model folder.

    folder: str
        path to the folder that contains checkpts and a hyps json file
    """
    folder = os.path.expanduser(folder)
    if not os.path.isdir(folder):
        folder = "/".join(folder.split("/")[:-1])
    for name in ["hyperparams", "config"]:
        for ext in ["json", "yaml"]:
            f = os.path.join(folder, f"{name}.{ext}")
            if os.path.exists(f):
                hyps = load_json_or_yaml(f)
                return hyps
    print("failed for", folder, f)
    return None

def load_hyps(folder):
    """
    Returns a dict of the hyperparameters collected from the json
    save file in the model folder.

    folder: str
        path to the folder that contains checkpts and a hyps json file
    """
    return get_hyps(folder)

def load_config(folder):
    """
    Returns a dict of the hyperparameters collected from the json
    save file in the model folder.

    folder: str
        path to the folder that contains checkpts and a hyps json file
    """
    return get_hyps(folder)

def get_config(folder):
    """
    Returns a dict of the hyperparameters collected from the json
    save file in the model folder.

    folder: str
        path to the folder that contains checkpts and a hyps json file
    """
    return get_hyps(folder)

def load_init_checkpt(model, config):
    """
    Easily load a checkpoint into the model at initialization.

    model: torch module
    config: dict
        "init_checkpt": str
    """
    init_checkpt = config.get("init_checkpt", None)
    if init_checkpt is not None and init_checkpt.strip()!="":
        if not os.path.exists(init_checkpt):
            init_checkpt = os.path.join(config["save_root"], init_checkpt)
        print("Initializing from checkpoint", init_checkpt)
        checkpt = load_checkpoint(init_checkpt)
        try:
            model.load_state_dict(checkpt["state_dict"])
        except:
            print("Failed to load checkpt, attempting fix...")
            sd = checkpt["state_dict"]
            mskeys = set(model.state_dict().keys())
            sym_diff = mskeys.symmetric_difference(set(sd.keys()))
            if len(sym_diff)>0:
                print("State Dict Symmetric Difference")
                for k in sym_diff: 
                    if k in mskeys:
                        print("MODEL:", k, model.state_dict()[k].shape)
                    else:
                        print("CHECKPT:", k, sd[k].shape)

            for key in sync_keys:
                if key in model.state_dict():
                    sd[key] = model.state_dict()[key]
            model.load_state_dict(sd)
    return model

def exp_num_exists(exp_num, exp_folder):
    """
    Determines if the argued experiment number already exists for the
    argued experiment name.

    exp_num: int
        the number to be determined if preexisting
    exp_folder: str
        path to the folder that contains the model folders
    """
    folders = get_model_folders(exp_folder)
    for folder in folders:
        num = foldersort(folder)
        if exp_num == num:
            return True
    return False

def make_save_folder(hyps, incl_full_path=False):
    """
    Creates the save name for the model. Will add exp_num to hyps if
    it does not exist when argued.

    hyps: dict
        keys:
            exp_save_path: str
                path to the experiment folder where all experiments
                sharing the same `exp_name` are saved.
                i.e. /home/user/all_saves/exp_name/
            exp_name: str
                the experiment name
            exp_num: int
                the experiment id number
            search_keys: str
                the identifying keys for this hyperparameter search
    incl_full_path: bool
        if true, prepends the exp_save_path to the save_folder.
    """
    return get_save_folder(hyps, incl_full_path=incl_full_path)

def get_save_folder(hyps, incl_full_path=False):
    """
    Creates the save name for the model. Will add exp_num to hyps if
    it does not exist when argued.

    hyps: dict
        keys:
            exp_folder: str or None
                path to the experiment folder where all experiments
                sharing the same `exp_name` are saved.
                i.e. /home/user/all_saves/<exp_name>/
                If None is argued, will use "./<exp_name>"
            exp_name: str
                the experiment name.
            exp_num: int
                the experiment id number
            search_keys: str
                the identifying keys for this hyperparameter search
    incl_full_path: bool
        if true, prepends the exp_folder to the save_folder.
    """
    if "exp_num" not in hyps:
        hyps["exp_folder"] = hyps.get(
          "exp_folder", os.path.join("./", hyps.get("exp_name", "myexp"))
        )
        hyps["exp_num"] = get_new_exp_num(
            hyps["exp_folder"], hyps["exp_name"]
        )
    model_folder = "{}_{}".format( hyps["exp_name"], hyps["exp_num"] )
    model_folder += prep_search_keys(hyps.get("search_keys","_"))
    if "exp_name" in model_folder:
        splt = model_folder.split("exp_name")
        right = splt[-1].split("_")
        if len(right)>1:
            model_folder = splt[0] + "_".join(right[1:])
        else:
            model_folder = splt[0]
    if incl_full_path: 
        return os.path.join(hyps["exp_folder"], model_folder)
    return model_folder

def get_new_exp_num(exp_folder, exp_name, offset=0):
    """
    Finds the next open experiment id number by searching through the
    existing experiment numbers in the folder.

    If an offset is argued, it is impossible to have an exp_num that is
    less than the value of the offset. The returned exp_num will be
    the next available experiment number starting with the value of the
    offset.

    Args:
        exp_folder: str
            path to the main experiment folder that contains the model
            folders. i.e. if the `exp_name` is "myexp" and there is
            a folder that contains a number of model folders, then
            exp_folder would be "/path/to/myexp/"
            If None is argued, assumes "./<exp_name>/
        exp_name: str
            the name of the experiment
        offset: int
            a number to offset the experiment numbers by.

    Returns:
        exp_num: int
    """
    if not exp_folder: exp_folder = os.path.join("./", exp_name)
    name_splt = exp_name.split("_")
    namedex = 1
    if len(name_splt) > 1:
        namedex = len(name_splt)
    exp_folder = os.path.expanduser(exp_folder)
    if not os.path.exists(exp_folder):
        os.mkdir(exp_folder)
    _, dirs, _ = next(os.walk(exp_folder))
    exp_nums = set()
    for d in dirs:
        splt = d.split("_")
        if len(splt) >= 2:
            num = None
            for i in reversed(range(len(splt))):
                try:
                    num = int(splt[i])
                    break
                except:
                    pass
            if namedex > 1 and i > 1:
                name = "_".join(splt[:namedex])
            else: name = splt[0]
            if name == exp_name and num is not None:
                exp_nums.add(num)
    for i in range(len(exp_nums)):
        if i+offset not in exp_nums:
            return i+offset
    return len(exp_nums) + offset

def save_yaml(data, file_name):
    with open(file_name, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

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

def is_jsonable(x):
    try:
        json.dumps(x, ensure_ascii=False, indent=4)
        return True
    except (TypeError, OverflowError):
        pass
    return False

def make_jsonable(x):
    if is_jsonable(x): return x
    if type(x)==dict:
        for k in list(x.keys()):
            newk = make_jsonable(k)
            x[newk] = make_jsonable(x[k])
            if newk!=k or type(newk)!=type(k):
                print("K:", k, x[k])
                del x[k]
    elif hasattr(x, "__len__"):
        x = [make_jsonable(xx) for xx in x]
    elif hasattr(x,"__name__"):
        x = x.__name__
    else:
        try:
            x = str(x)
        except:
            print("Removing", x, "from json")
            x = ""
    return x

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
        jdata = make_jsonable(copy.deepcopy(data))
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(jdata, f, ensure_ascii=False, indent=4)

def record_session(config, model, globals_dict=None, verbose=False):
    """
    Writes important parameters to file. If 'resume_folder' is an entry
    in the config dict, then the txt file is appended to instead of being
    overwritten.

    config: dict
        dict of relevant hyperparameters
    model: torch nn.Module
        the model to be trained
    globals_dict: dict
        just argue `globals()`
    """
    try:
        config["git_hash"] = config.get(
            "git_hash", get_git_revision_hash()
        )
    except:
        s="you aren't using git?! you should really version control..."
        config["git_hash"] = s
        print(s)
    git_hash = config["git_hash"]
    sf = config['save_folder']
    if not os.path.exists(sf):
        os.mkdir(sf)
    h = "config"
    mode = "a" if "resume_folder" in config else "w"
    packages = package_versions(globals_dict=globals_dict,verbose=verbose)
    with open(os.path.join(sf,h+".txt"),mode) as f:
        dt_string = get_datetime_str()
        f.write(dt_string)
        f.write("\nGit Hash: {}".format(git_hash))
        f.write("\nPackage Versions:")
        for module_name,v in packages.items():
            f.write("\t{}: {}\n".format(module_name, v))
        f.write("\n"+str(model)+'\n')
        for k in sorted(config.keys()):
            f.write(str(k) + ": " + str(config[k]) + "\n")
    temp_hyps = dict()
    keys = list(config.keys())
    temp_hyps = {k:v for k,v in config.items()}
    if verbose:
        print("\nConfig:")
    for k in keys:
        if verbose and k!="packages":
            print("\t{}:".format(k), temp_hyps[k])
        if type(config[k]) == type(np.array([])):
            del temp_hyps[k]
        elif type(config[k])==np.int64:
            temp_hyps[k] = int(temp_hyps[k])
        elif type(config[k])==type(set()):
            temp_hyps[k] = list(config[k])
    if "packages" not in temp_hyps:
        temp_hyps["packages"] = packages
    save_json(temp_hyps, os.path.join(sf,h+".json"))

def get_folder_from_path(path):
    if os.path.isdir(path): return path
    return "/".join(path.split("/")[:-1])

def get_num_duplicates(folder, fname, ext=".csv"):
    n_dupls = 0
    folder = get_folder_from_path(folder)
    for f in os.listdir(folder):
        n_dupls += int(fname == f[:len(fname)] and ext in f)
    return n_dupls


def get_save_name(
        save_folder,
        kwargs,
        config,
        abbrevs = {
            "model_names": "mdls",
            "dataset_name": "dset",
            "dataset_kwargs": "dkwgs",
            "filtered_dataset_path": "fltdset",
            "hook_layers": "lyrs",
            "mtx_types": "mtxtyps",
            "identity_init": "ideninit",
            "identity_rot": "idenrot",
            "mask_type":   "msktyp",
            "n_units": "nunits",
            "learnable_addition": "lrnadd",
            "num_training_steps": "ntrn",
            "batch_size": "bsz",
            "grad_accumulation_steps": "gradstps",
            "max_length": "mxln",
            "eval_batch_size": "vlbsz",
            "learning_rate": "lr",
            "stepwise": "stpwse",
            "swap_keys": "swpks",
        },
        ignores = {
            "print_every",
            "model_names",
            "mtx_kwargs",
            "save_keys",
            "dataset_names",
            "source_files",
        },):
    # Get intial save folder root
    exp_name = kwargs.get("exp_name", config.get("exp_name", "mas_"))
    save_name = f"{exp_name}_"

    # always add model names to save name
    if "model_names" not in config:
        if "source_files" in kwargs:
            kwargs["model_names"] = kwargs["source_files"]
            config["model_names"] = kwargs["source_files"]
        else:
            config["model_names"] = config["source_files"]
            kwargs["model_names"] = config["source_files"]
    kwargs["model_names"] = kwargs.get("model_names", config["model_names"])
    m2 = "".join([x[:3] for x in remove_ending_slash(kwargs["model_names"][-1]).split("/")[-1].split("_")])
    save_name = save_name + abbrevs["model_names"] + "-" + m2 + "_"

    # Get datetime
    dtime = datetime.now().strftime("%Y-%m-%d_t%H%M%S")

    # add key value pairs to folder name
    if "save_keys" in kwargs:
        s = set(kwargs["save_keys"])
    elif "save_keys" in config:
        s = set(config["save_keys"])
    else:
        s = set(kwargs.keys())
        d = set(kwargs.keys()).symmetric_difference(set(config.keys()))
        if len(s)==0 or len(d)==0: s = config.get("save_keys", set())
    if len(s)==0:
        n_dupls = get_num_duplicates(folder=save_folder, fname=save_name, ext=".csv")
        return save_name + f"_d{dtime}_v{n_dupls}"
    for k in sorted(list(s)):
        if k in ignores: continue
        if k not in kwargs: kwargs[k] = config[k]
        has_len = hasattr(kwargs[k],"__len__")
        if k=="swap_keys":
            val = config["swap_keys"]
            swap_keys = []
            for v in val:
                skey = "".join(v)
                swap_keys.append(skey)
            val = "-".join(swap_keys)
        elif k!="hook_layers" and type(kwargs[k])!=str and has_len:
            val = "".join([
              str(e)[:3]+str(e)[-2:] if len(str(e))>3 else str(e)[:3] for e in kwargs[k]
            ][:3])
        else:
            if k=="hook_layers" and type(kwargs[k])==str:
                val = "".join([e[:4] for e in kwargs[k].split(".")])
            elif k=="hook_layers" and type(kwargs[k])==list:
                if "self_attn." in kwargs[k][0]:
                    val = "".join([e.split("self_attn.")[-1][:3] for e in kwargs[k]])
                else:
                    val = "".join([e[:3]+e[-2:] for e in kwargs[k]])
            elif hasattr(kwargs[k], "__name__"):
                val = kwargs[k].__name__[:5]
            elif type(kwargs[k]) in {int, float}:
                sci = "{:e}".format(kwargs[k])
                val = str(kwargs[k])
                if len(val)>len(sci):
                    val = sci
            else:
                val = str(kwargs[k])[:5]
        save_name += abbrevs.get(k, k).replace("_","")[:7]+val+"_"
    save_name = save_name[:-1]
    save_name = save_name\
        .replace("source", "src")\
        .replace("arith","arth")\
        .replace("base","bse")\
        .replace("causal", "casl")\
        .replace("boundle","bndls")\
        .replace("nepochs","neps")\
        .replace("ntrains","ntrn")\
        .replace("relaxed","rlxd")\
        .replace("layers","lyrs")\
        .replace("True","T")\
        .replace("False","F")\
        .replace("auto","ato")\
        .replace("swap_keys","swpks")\
        .replace("encoder","enc")
    n_dupls = get_num_duplicates(save_folder, save_name, ext=".csv")
    return save_name + f"_d{dtime}_v{n_dupls}"

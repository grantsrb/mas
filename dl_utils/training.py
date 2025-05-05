import sys
import os
import numpy as np
import torch
import time
from collections import deque
import select
import shutil
import copy
import torch.multiprocessing as mp

import dl_utils.save_io as io

def parse_type(val):
    """
    Determines the appropriate data type for the argued string value.

    Args:
        val: str
    Returns:
        val: any
    """
    val = str(val)
    if val.lower() in {"none", "null", "na"}:
        val = None
    elif "," in val:
        val = [parse_type(v) for v in val.split(",") if v!=""]
    elif val.lower()=="true":
        val = True
    elif val.lower()=="false":
        val = False
    elif val.isnumeric():
        val = int(val)
    elif val.replace(".", "").isnumeric():
        val = float(val)
    return val

def read_command_line_args(args=None):
    if args is None: args = sys.argv[1:]
    model_folders = []
    command_args = []
    command_kwargs = dict()

    for arg in args:
        if io.is_model_folder(arg):
            model_folders.append(arg)
        elif io.is_exp_folder(arg):
            mfs = io.get_model_folders(
                arg, incl_full_path=True, incl_empty=False)
            for f in mfs:
                model_folders.append(f)
        elif "checkpt" in arg and ".pt" in arg:
            model_folders.append(arg)
        elif ".yaml" in arg or ".json" in arg:
            command_kwargs = {**command_kwargs,
                              **io.load_json_or_yaml(arg)}
        elif "=" in arg:
            key,val = arg.split("=")
            command_kwargs[key] = parse_type(val)
        else:
            command_args.append(arg)
    return model_folders, command_args, command_kwargs

def config_error_catching(config):
    """
    This function just makes sure that some obvious hyperparameter
    choices are set and some obviously wrong hyperparameter settings
    are changed to what the experimenter meant.
    """
    config["exp_name"] = config.get("exp_name", "myexp")
    config["use_accelerate"] = config.get(
        "use_accelerate", config.get("use_accelerator", True)
    )
    return config

def empirical_batch_size(config, model, dataset, reduction_factor=0.5):
    """
    Empirically finds a batch size based on cuda errors. Updates the
    config with a new batch size and a new n_grad_loops.

    Args:
        config: dict
        model: torch Module
        dataset: torch Dataset
        reduction_factor: float in interval (0,1)
            the factor to reduce the batch size by.
    """
    #torch.cuda.OutOfMemoryError
    config["batch_size"] = config.get("batch_size", 128)
    config["vbatch_size"] = config.get("vbatch_size", 1000)
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=config["batch_size"],
    )
    cuda_error = True
    while cuda_error:
        cuda_error = False
        try:
            itr = iter(data_loader)
            for _ in range(2):
                data = next(itr)
                package = model(
                    data,
                    ret_preds=True,
                    tforce=config.get("tforce_train", True),
                )
                loss = package["loss"]
                acc = package["acc"]

                loss.backward()
        except torch.cuda.OutOfMemoryError:
            cuda_error = True
            factor = reduction_factor
            config["batch_size"] =  int(factor*config["batch_size"])
            config["vbatch_size"] = int(factor*config["vbatch_size"])
            #config["n_grad_loops"] = max(config["n_grad_loops"]*div, 8)
            data_loader = torch.utils.data.DataLoader(
                dataset, shuffle=True, batch_size=config["batch_size"],
            )
        except StopIteration:
            pass
    return data_loader

def get_resume_checkpt(config, in_place=False, verbose=True):
    """
    This function cleans up the code to resume from a particular
    checkpoint or folder.

    Args:
        config: dict
            dictionary of hyperparameters
            keys: str
                "resume_folder": str
                    must be a key present in config for this function to
                    act.
                "ignore_keys": list of str
                    an optional key used to enumerate keys to be ignored
                    when loading the old hyperparameter set
            vals: varies
        in_place: bool
            if true, changes the argued hyperparameters in place.
            otherwise has no effect on the argued hyperparameters
    Returns:
        checkpt: dict or None
            a loaded checkpoint dict of the model that will be resumed
            from. None if the checkpoint does not exist
        config: dict
            the modified hyperparameters. This does not reference the
            argued config object. But will be a deep copy of the argued
            config if `in_place` is true.
    """
    # Ignore keys are used to override the hyperparams json associated
    # with the loaded training. In otherwords, use the ignore keys
    # to specify new hyperparameters in the current training instead
    # of the hyperparameters from the training that is being loaded.
    ignore_keys = [
        'n_epochs',
        'rank',
        "n_eval_steps",
        "n_eval_eps",
        "save_root",
        "resume_folder",
    ]
    ignore_keys = config.get('ignore_keys',ignore_keys)
    resume_folder = config.get('resume_folder', None)
    if not in_place: 
        config = {**config}
    if resume_folder is not None and resume_folder != "":
        checkpt = io.load_checkpoint(resume_folder)
        # Check if argued number of epochs exceeds the total epochs
        # of the to-be-resumed training
        if verbose and training_exceeds_epochs(config, checkpt):
            print("Could not resume due to epoch count")
            print("Performing fresh training")
        else:
            if verbose:
                print("Loading config from", resume_folder)
            temp_hyps = checkpt['config']
            for k,v in temp_hyps.items():
                if k not in ignore_keys:
                    config[k] = v
            config["seed"] = config.get( "seed", 0)
            if config["seed"] is None: config["seed"] = 0
            config['seed'] += int(time.time()) # For fresh data
            s = " Restarted training from epoch "+str(checkpt['epoch'])
            config['description'] = config.get("description", "")
            config['description'] += s
            config['ignore_keys'] = ignore_keys
            config["save_folder"] = resume_folder
            return checkpt, config
    return None, config

def training_exceeds_epochs(config, checkpt):
    """
    Helper function to deterimine if the training to be resumed has
    already exceeded the number of epochs argued in the resumed
    training.

    Args:
        config: dict
        checkpt: dict
    """
    n_epochs = -1
    if "n_epochs" in config: n_epochs = config["n_epochs"]
    return (checkpt['epoch']>=(n_epochs-1) or n_epochs == -1)

def fill_hyper_q(config, hyp_ranges, keys, hyper_q, idx=0):
    """
    Recursive function to load each of the hyperparameter combinations
    onto a queue.

    config - dict of hyperparameters created by a HyperParameters object
        type: dict
        keys: name of hyperparameter
        values: value of hyperparameter
    hyp_ranges - dict of lists
        these are the ranges that will change the hyperparameters for
        each search
        type: dict
        keys: name of hyperparameters to be searched over
        values: list or dict of lists of equal length
            if a list is given, it should be a list of values to search
            over for the hyperparameter specified by the corresponding
            key. If a dict of lists is given, the name of the key is
            ignored and the items in the lists are paired together for
            a hyperparameter combination corresponding to their
            respective keys. i.e.

            {
                "param0": [foo1],
                "combos": {
                    "param1":[p1,p2],
                    "param2":[v1,v2]
                }
            }

            will result in the folowing 2 hyperparameter search values:

            {"param0": foo1, "param1": p1, "param2": v1}
            and
            {"param0": foo1, "param1": p2, "param2": v2}
    keys - keys of the hyperparameters to be searched over. Used to
        specify order of keys to search
    hyper_q - deque to hold all parameter sets
    idx - the index of the current key to be searched over

    Returns:
        hyper_q: deque of dicts `config`
    """
    # Base call, saves the hyperparameter combination
    if idx >= len(keys):
        # Load q
        config['search_keys'] = ""
        for k in keys:
            if k=="exp_name":
                config["search_keys"] += "_expname"
            elif isinstance(hyp_ranges[k],dict):
                for rk in hyp_ranges[k].keys():
                    s = io.prep_search_keys(str(config[rk]))
                    config['search_keys'] += "_" + str(rk)+s
            else:
                s = io.prep_search_keys(str(config[k]))
                config['search_keys'] += "_" + str(k)+s
        hyper_q.append({**config})

    # Non-base call. Sets a hyperparameter to a new search value and
    # passes down the dict.
    else:
        key = keys[idx]
        # Allows us to specify combinations of hyperparameters
        if isinstance(hyp_ranges[key],dict):
            rkeys = list(hyp_ranges[key].keys())
            for i in range(len(hyp_ranges[key][rkeys[0]])):
                for rkey in rkeys:
                    config[rkey] = hyp_ranges[key][rkey][i]
                hyper_q = fill_hyper_q(config, hyp_ranges, keys, hyper_q,
                                                               idx+1)
        else:
            for param in hyp_ranges[key]:
                config[key] = param
                hyper_q = fill_hyper_q(config, hyp_ranges, keys, hyper_q,
                                                               idx+1)
    return hyper_q

def make_hyper_range(low, high, range_len, method="log"):
    """
    Creates a list of length range_len that is a range between two
    values. The method dictates the spacing between the values.

    low: float
        the lowest value in the range

    """
    if method.lower() == "random":
        param_vals = np.random.random(low, high+1e-5, size=range_len)
    elif method.lower() == "uniform":
        step = (high-low)/(range_len-1)
        param_vals = np.arange(low, high+1e-5, step=step)
    else:
        range_low = np.log(low)/np.log(10)
        range_high = np.log(high)/np.log(10)
        step = (range_high-range_low)/(range_len-1)
        arange = np.arange(range_low, range_high+1e-5, step=step)
        param_vals = 10**arange
    param_vals = [float(param_val) for param_val in param_vals]
    return param_vals

def hyper_search(config, hyp_ranges, train_fxn):
    """
    The top level function to create hyperparameter combinations and
    perform trainings.

    config: dict
        the initial hyperparameter dict
        keys: str
        vals: values for the hyperparameters specified by the keys
    hyp_ranges: dict
        these are the ranges that will change the hyperparameters for
        each search. A unique training is performed for every
        possible combination of the listed values for each key
        keys: str
        vals: lists of values for the hyperparameters specified by the
              keys
    train_fxn: function
        args:
            config: dict
            verbose: bool
        a function that performs the desired training given the argued
        hyperparams
    """
    starttime = time.time()

    config['multi_gpu'] = config.get('multi_gpu',False)
    if config['multi_gpu']:
        config["n_gpus"] = config.get("n_gpus",torch.cuda.device_count())
        os.environ['MASTER_ADDR'] = '127.0.0.1'     
        os.environ['MASTER_PORT'] = '8021'

    exp_folder = config["exp_folder"]
    results_file = os.path.join(exp_folder, "results.txt")
    with open(results_file,'a') as f:
        f.write("Hyperparameters:\n")
        for k in config.keys():
            if k not in hyp_ranges:
                f.write(str(k) + ": " + str(config[k]) + '\n')
        f.write("\nHyperranges:\n")
        for k in hyp_ranges.keys():
            if isinstance(hyp_ranges[k],dict):
                s = str(k)+":\n"
                for rk in hyp_ranges[k].keys():
                    rs = ",".join([str(v) for v in hyp_ranges[k][rk]])
                    s += "  "+str(rk) + ": [" + rs +']\n'
            else:
                rs = ",".join([str(v) for v in hyp_ranges[k]])
                s = str(k) + ": [" + rs +']\n'
            f.write(s)
        f.write('\n')

    hyper_q = deque()
    hyper_q = fill_hyper_q(config, hyp_ranges, list(hyp_ranges.keys()),
                                                      hyper_q, idx=0)
    total_searches = len(hyper_q)
    print("n_searches:", total_searches)

    result_count = 0
    print("Starting Hyperloop")
    while not len(hyper_q)==0:
        print("\n\nSearches left:", len(hyper_q),"-- Running Time:",
                                             time.time()-starttime)
        config = hyper_q.popleft()

        res = config.get("resume_folder", None)
        if res is None or res=="":
            config["model_folder"] = io.get_save_folder(config)
            config["save_folder"] = os.path.join(
                config["exp_folder"], config["model_folder"]
            )
            if not os.path.exists(config["save_folder"]):
                os.mkdir(config["save_folder"])
            print("Saving to", config["save_folder"])

        verbose = True
        if config['multi_gpu']:
            mp.spawn(train_fxn, nprocs=config['n_gpus'],
                                join=True,
                                args=(config,verbose))
        else:
            train_fxn(0, config=config, verbose=verbose)

def simple_parse(value):
    value = str(value)
    try:
        if value[-1].isnumeric():
            if "." in value:
                return float(value)
            return int(value)
        elif value.lower() in {"false", "true"}:
            value = value.lower()=="true"
        elif value[0]=="[":
            raise NotImplemented
    except:
        pass
    return value

def run_training(train_fxn):
    """
    This function extracts the hyperparams and hyperranges from the
    command line arguments and asks the user if they would like to
    proceed with the training and/or overwrite the previous save
    folder.

    train_fxn: function
        this the training function that will carry out the training
        args:
            config: dict
            verbose: bool
    """
    config = io.load_json_or_yaml(sys.argv[1])
    print()
    print("Using hyperparams file:", sys.argv[1])
    config["lr"] = config.get("lr", 1e-3)
    if len(sys.argv) < 3:
        ranges = {"lr": [config['lr']]}
    else:
        ranges = None
        for arg in sys.argv[2:]:
            if ".yaml" in arg or ".json" in arg:
                print("Using hyperranges file:", sys.argv[2])
                ranges = io.load_json_or_yaml(arg)
            elif "=" in arg:
                splt = arg.split("=")
                config[splt[0]] = parse_type(splt[1])
        if ranges is None:
            ranges = {"lr": [config['lr']]}
    print()

    keys = sorted(list(config.keys()))
    hyps_str = ""
    for k in keys:
        if k not in ranges:
            hyps_str += "{}: {}\n".format(k,config[k])
    print("Hyperparameters:")
    print(hyps_str)
    print("\nSearching over:")
    print("\n".join(["{}: {}".format(k,v) for k,v in ranges.items()]))

    exp_names = [config.get('exp_name', "myexp")]
    for k in ranges.keys():
        if type(ranges[k])==dict:
            if "exp_name" in ranges[k]:
                exp_names = list(set(ranges[k]["exp_name"]))
                if len(exp_names)>1: raise NotImplemented
    if "exp_name" in ranges:
        exp_names = ranges["exp_name"]
    og_config = copy.deepcopy(config)
    og_ranges = copy.deepcopy(ranges)
    for exp_name in exp_names:
        ranges = copy.deepcopy(og_ranges)
        config = copy.deepcopy(og_config)
        config["exp_name"] = exp_name
        exp_folder = exp_name
        if "save_root" in config:
            config['save_root'] = os.path.expanduser(config['save_root'])
            if not os.path.exists(config['save_root']):
                os.mkdir(config['save_root'])
            exp_folder = os.path.join(config['save_root'], exp_folder)
        print("Main Exp Folder:", exp_folder)
        sleep_time = 8
        if os.path.exists(exp_folder):
            dirs = io.get_model_folders(exp_folder)
            if len(dirs) > 0:
                s = "Overwrite last folder {}? (No/yes)".format(dirs[-1])
                print(s)
                i,_,_ = select.select([sys.stdin], [],[],sleep_time)
                if i and "y" in sys.stdin.readline().strip().lower():
                    print("Are you sure?? This will delete the data (Y/n)")
                    i,_,_ = select.select([sys.stdin], [],[],sleep_time)
                    if i and "n" not in sys.stdin.readline().strip().lower():
                        path = os.path.join(exp_folder, dirs[-1])
                        shutil.rmtree(path, ignore_errors=True)
            else:
                s = "You have {} seconds to cancel experiment name {}:"
                print(s.format(sleep_time, config['exp_name']))
                i,_,_ = select.select([sys.stdin], [],[],sleep_time)
        else:
            s = "You have {} seconds to cancel experiment name {}:"
            print(s.format(sleep_time, config['exp_name']))
            i,_,_ = select.select([sys.stdin], [],[],sleep_time)
        print()

        keys = list(ranges.keys())
        start_time = time.time()

        # Make results file
        exp_folder = config['exp_name']
        if "save_root" in config:
            config['save_root'] = os.path.expanduser(config['save_root'])
            if not os.path.exists(config['save_root']):
                os.mkdir(config['save_root'])
            exp_folder = os.path.join(config['save_root'], exp_folder)
        config["exp_folder"] = exp_folder
        if not os.path.exists(exp_folder):
            os.mkdir(exp_folder)

        hyper_search(config, ranges, train_fxn)
    print("Total Execution Time:", time.time() - start_time)

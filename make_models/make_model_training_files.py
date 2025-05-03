"""
Run this script to update the model creation files.
"""
import os
import sys
import json
import yaml
import copy


## Update these values
root = "make_models" # the name of the directory in which this script resides
main_save_directory = "/mnt/fs2/grantsrb/mas_neurips2025/"
main_d_model = 128
n_epochs = 2500
seeds =    [12345, 23456,]
devices =  [0,1,2,3,4,5,6,7,8,9]
unk_p = 0.2
tasks = ["MultiObject", "SameObject", "MultiObjectMod", "MultiObjectRound"]
tformer_tasks = ["MultiObject", "MultiObjectMod", "MultiObjectRound"]
rnns = ["GRU", "LSTM"]

if len(sys.argv)>=2:
    for arg in sys.argv[1:]:
        if "main_save_directory=" in arg:
            main_save_directory = arg.split("main_save_directory=")[-1]
        elif "devices=" in arg:
            devices = arg.split("devices=")[-1].split(",")
            devices = [int(d) for d in devices]
        elif "seeds=" in arg:
            seeds = arg.split("seeds=")[-1].split(",")
            seeds = [int(s) for s in seeds]

print("Making Config Files")
print("Devices:", devices)
print("Seeds:", seeds)
print("DModels:", seeds)

if os.path.exists(root):
    os.chdir(root)


for folder in ["ranges", "configs", "metas", "run_scripts"]:
    if not os.path.exists(folder):
        os.mkdir(folder)

if not os.path.exists(main_save_directory):
    os.mkdir(main_save_directory)

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

def save_yaml(data, file_name):
    with open(file_name, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


# Get Main Config
config = load_yaml("config.yaml")
config["save_root"] = main_save_directory
config["n_epochs"] = n_epochs
og_config = config

# Make RNN Configs
config = copy.deepcopy(og_config)
unk = ""
incr = len(devices)//len(tasks)
for rnn in rnns:
    print(rnn)
    run_script = f"#!/bin/bash\n\n"
    for ti,task in enumerate(tasks):
        task_low = task.lower()
        rnn_low = rnn.lower()
        exp_name = f"{task_low}_{rnn_low}{unk}"
        # Make Config
        config["exp_name"] = exp_name
        config["task_type"] = task
        config["model_type"] = rnn
        config["n_layers"] = 1
        config["d_model"] = main_d_model
        config["tforce_train"] = False
        config["unk_p"] = bool("unk" in unk)*unk_p
        cpath = os.path.join("configs/", exp_name + "_config.json")
        save_json(config, cpath)

        # Make Ranges
        ranges = {
            "seed": seeds,
        }
        rpath = os.path.join("ranges/", exp_name + "_ranges.json")
        save_json(ranges, rpath)

        # Make Meta Config
        start = ti*incr
        end = (ti+1)*incr
        ds = devices[start:end]
        print(ds)
        meta = {
            "devices": ds,
            "key_order": ["seed"],
            "hyperparams": os.path.join(root, cpath),
            "hyperranges": os.path.join(root, rpath),
        }
        mpath = os.path.join("metas/", exp_name + "_meta.json")
        save_json(meta, mpath)
        run_script += f"python3 distr.py train.py {root}/{mpath}\n"
    script_name = f"{rnn_low}.sh"
    script_path = os.path.join("run_scripts", script_name)
    with open(script_path, "w") as f:
        f.write(run_script)


# Make Transformer Configs
config = copy.deepcopy(og_config)
unks = ["_unk",]
incr = len(devices)//(len(tformer_tasks) * len(unks))
for enc_type in ["rope"]:
    print("Tformer", enc_type)
    run_script = f"#!/bin/bash\n\n"
    for ti,task in enumerate(tformer_tasks):
        for ui,unk in enumerate(unks):
            task_low = task.lower()
            exp_name = f"{task_low}_{enc_type}_tformer{unk}"
            # Make Config
            config["exp_name"] = exp_name
            config["task_type"] = task
            config["model_type"] = "Transformer"
            config["tforce_train"] = True
            config["lnorm"] = True
            config["llama"] = True
            config["d_model"] = main_d_model
            config["n_layers"] = 2
            config["unk_p"] = bool("unk" in unk)*unk_p
            config["encoder_layer_class"] = {
                "rope": "RotaryEncoderLayer",
                "nope": "SimpleEncoderLayer"
            }[enc_type]
            cpath = os.path.join("configs/", exp_name + "_config.json")
            save_json(config, cpath)

            # Make Ranges
            ranges = {
                "seed": seeds,
            }
            rpath = os.path.join("ranges/", exp_name + "_ranges.json")
            save_json(ranges, rpath)

            # Make Meta Config
            start = int((ti*len(unks)+ui)*incr)
            end = int((ti*len(unks)+ui+1)*incr)
            ds = devices[start: end]
            print("ds:", ds)
            meta = {
                "devices": ds,
                "key_order": ["seed"],
                "hyperparams": os.path.join(root, cpath),
                "hyperranges": os.path.join(root, rpath),
            }
            mpath = os.path.join("metas/", exp_name + "_meta.json")
            save_json(meta, mpath)
            run_script += f"python3 distr.py train.py {root}/{mpath}\n"
    script_name = f"{enc_type}_tformer.sh"
    script_path = os.path.join("run_scripts", script_name)
    with open(script_path, "w") as f:
        f.write(run_script)

print("Done")

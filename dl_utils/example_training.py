import torch
import transformers # Imported for versioning
import datasets # Imported for versioning

import numpy as np
import time
from tqdm import tqdm
import os
import sys
import accelerate
try:
    import dl_utils
except:
    sys.path.append("./")
    sys.path.append("../")

from dl_utils.save_io import (
    save_checkpt, load_json_or_yaml, record_session, get_save_folder,
)
from dl_utils.utils import package_versions
from dl_utils.schedulers import DecayScheduler
import dl_utils.training
from dl_utils.seq_models import make_model
from dl_utils.datas import get_datasets


def train(rank, config, verbose=True, *args, **kwargs):
    verbose = verbose and rank==0
    torch.cuda.empty_cache()

    # Hyperparameters
    config = config_error_catching(config) # Make sure we have valid config
    config["packages"] = package_versions(globals())
    config["seed"] = config.get("seed", int(time.time()))
    if config["seed"] is None: config["seed"] = int(time.time())
    torch.manual_seed(config["seed"]+rank)
    np.random.seed(   config["seed"]+rank)
    config["rank"] = rank

    # Dataset/Tokenizer
    #######################################
    if verbose: print("Collecting Data")
    # This function updates the config dict and returns DataSet objects
    tokenizer, train_dataset, val_dataset = get_datasets(config)
    config["tokenizer"] = tokenizer
    if verbose:
        print("Train Samples:", len(train_dataset))
        print("Val Samples:", len(val_dataset))
        print("Using Sequence Length:", config.get("seq_len",None))
    if "inputs" in train_dataset[0]:
        config["inpt_shape"] = train_dataset[0]["inputs"][0].shape

    # Model
    #######################################
    model = make_model(config)
    n_params = 0
    for p in model.parameters():
        if hasattr(p, "data"):
            n_params += p.data.numel()
    config["n_params"] = n_params
    if verbose:
        print("NParameters:", n_params)

    # Optimizer
    #######################################
    if verbose:
        print("Creating Optimizer")
    # important to have an lr in config for the scheduler
    config["lr"] = config.get("lr", 0.001)
    optimizer = getattr(torch.optim, config.get("optim_type","Adam"))(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("l2", 0),
    )

    # Scheduler
    #######################################
    if config.get("plateau_scheduler", False):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
          optimizer, threshold=0.001, patience=config.get("patience",10)
        )
    else:
        scheduler = DecayScheduler( optimizer, **config )

    # Logging
    #######################################
    if verbose:
        print("Model Type:", type(model))
        print("Recording Session")
    if rank==0:
        record_session(
            config, model, globals_dict=globals(), verbose=verbose
        )

    #######################################
    # Distributed Wrapper
    #######################################
    if verbose and torch.cuda.device_count()>1:
        print("Handling multiple GPUs")
    accelerator = None
    if config["use_accelerate"]:
        accelerator = accelerate.Accelerator()
        model, optimizer = accelerator.prepare( model, optimizer )
        device = accelerator.device
    elif "device" in config:
        device = config["device"]
    else:
        device = rank
    model.to(device)
    train_loader = dl_utils.training.empirical_batch_size(
        config, model, train_dataset
    )
    if rank==0 and verbose:
        print("New Batch Size:", config["batch_size"])
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=True,
        batch_size=config["vbatch_size"],
    )
    if config["use_accelerate"]:
        train_loader, val_loader = accelerator.prepare(
            train_loader, val_loader
        )
        print("Warning! Accelerate may turn off data shuffling!")

    if rank==0 and verbose:
        print(model)

    #############################################################
    # Training
    #############################################################
    n_epochs = config.get("n_epochs", 100)
    for epoch in range(n_epochs):
        epochtime = time.time()
        torch.cuda.empty_cache()
        if verbose:
            print()
            s = "Beginning Epoch {} - {}".format(
                epoch, config["save_folder"]
            )
            print(s)
            logstr = s + "\n"

        #############################################################
        # Train Loop
        #############################################################
        model.train()
        avg_loss = 0
        avg_acc = 0
        nloops = config.get("n_train_loops", len(train_loader))
        nloops = min(nloops,len(train_loader))
        checkpt_mod = config.get( "checkpt_mod", np.inf )
        val_mod = config.get( "val_mod", 1)
        optimizer.zero_grad()
        for i,data in enumerate(train_loader):
            starttime = time.time()
            package = model(
                data,
                ret_preds=True,
                tforce=config.get("tforce_train", True),
            )
            loss = package["loss"]
            acc = package["acc"]

            if config["use_accelerate"]:
                accelerator.backward(loss)
            else:
                loss.backward()

            avg_acc += acc.item()
            avg_loss += loss.item()

            if i%config.get("n_grad_loops",1)==0 or i==len(train_loader)-1:
                if config.get("grad_clip",0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config["grad_clip"]
                    )
                optimizer.step()
                optimizer.zero_grad()
                try:
                    scheduler.step()
                except:
                    pass

            if verbose and i%10==0:
                dec = 4
                l = round(loss.item(), dec)
                a = round(acc.item(), dec)
                c = round(100*i/nloops, 2)
                t = round(time.time()-starttime, 3)
                s = "Loss: {} -Acc: {}".format(l,a)
                s += " - {}% {}s   ".format(c,t)
                print(s, end=int(len(s)/2)*" " + "\r")


            if config.get("exp_name","myexp")=="test" and i>=30: break
            if i>=(nloops-1): break
            if i>0 and checkpt_mod and i%checkpt_mod==0 and rank==0:
                if config.get("save_folder", None) is not None:
                    dec = 5
                    train_loss = round(avg_loss/i, dec)
                    train_acc = round(avg_acc/i, dec)
                    save_dict = {
                        "mid_epoch": True,
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_acc":  train_acc,
                        "val_loss": None,
                        "val_acc":  None,
                        "state_dict": model.state_dict(),
                        "optim_dict": optimizer.state_dict(),
                        "lr": optimizer.param_groups[0]["lr"],
                        "config": config,
                    }
                    ep = round(epoch+i/len(train_loader), 3)
                    save_checkpt(
                        save_dict=save_dict,
                        save_folder=config["save_folder"],
                        save_name="checkpt",
                        epoch=ep,
                        ext=".pt"
                    )
        div = (i+1)
        dec = 5
        train_loss = round(avg_loss/div, dec)
        train_acc  = round(avg_acc/div, dec)

        if verbose:
            examps = "Example Train Preds:\n"
            preds = package["pred_ids"]
            targs = data["output_ids"]
            pmask = ~data["input_pad_mask"].cpu()
            tmask = ~data["output_pad_mask"].cpu()
            for i in range(min(3,len(preds))):
                s = "Targ: "+", ".join(
                    [str(t) for t in targs[i].cpu()[tmask[i]].tolist()]
                )
                examps += s+"\n"
                s = "Pred: "+", ".join(
                    [str(p) for p in preds[i].cpu()[pmask[i]].tolist()]
                )
                examps += s+"\n"
                if tokenizer:
                    s = "Targ Str: "+ tokenizer.decode(
                        targs[i].cpu()[tmask[i]]
                    )[0]
                    examps += s+"\n"
                    s = "Pred Str: "+ tokenizer.decode(
                        preds[i].cpu()[pmask[i]]
                    )[0]
                    examps += s+"\n"
                examps += "\n"
            logstr += examps
            print(examps)

        #############################################################
        # Validation Loop
        #############################################################
        val_loss =     0
        val_acc =      0
        if rank==0 and (epoch%val_mod==0 or epoch==n_epochs-1):
            model.eval()
            if verbose: print("Validating...")
            with torch.no_grad():
                nloops = config.get("max_val_loops",len(val_loader))
                nloops = min(nloops, len(val_loader))
                avg_loss = 0
                avg_acc = 0
                for i,data in enumerate(val_loader):
                    starttime = time.time()
                    package = model(
                        data,
                        ret_preds=True,
                        tforce=False,
                        temperature=config.get(
                            "sampling_temperature", None
                        ),
                        sprout_len=config.get("sprout_len", 3),
                    )
                    loss = package["loss"]
                    acc = package["acc"]
                    avg_loss += loss.item()
                    avg_acc += acc.item()
                    if verbose:
                        p = round(100*(i+1)/nloops, 2)
                        t = round(time.time()-starttime, 4)
                        print("{}% -- {}s".format(p,t), end="         \r")
                    if i>=nloops-l: break
            div = (i+1)
            dec = 5
            val_loss = round(avg_loss/div, 5)
            val_acc =  round(avg_acc/div, 5)
            scheduler.step(val_loss)
            if config["exp_name"]=="test": break

            if verbose:
                examps = "Example Val Preds:\n"
                preds = package["pred_ids"]
                targs = data["output_ids"]
                pmask = ~data["input_pad_mask"].cpu()
                tmask = ~data["output_pad_mask"].cpu()
                for i in range(min(3,len(preds))):
                    s = "Targ: "+", ".join(
                        [str(t) for t in targs[i].cpu()[tmask[i]].tolist()]
                    )
                    examps += s+"\n"
                    s = "Pred: "+", ".join(
                        [str(p) for p in preds[i].cpu()[pmask[i]].tolist()]
                    )
                    examps += s+"\n"
                    if tokenizer:
                        s = "Targ Str: "+ tokenizer.decode(
                            targs[i].cpu()[tmask[i]]
                        )[0]
                        examps += s+"\n"
                        s = "Pred Str: "+ tokenizer.decode(
                            preds[i].cpu()[pmask[i]]
                        )[0]
                        examps += s+"\n"
                    examps += "\n"
                logstr += examps
                print(examps)
                print()
                s = "Final Stats, Epoch: {}".format(epoch)
                print(s)
                logstr += "\n" + s + "\n"

                s = "Train Loss: {} - Train Acc: {}".format(
                    train_loss,train_acc
                )
                logstr += s + "\n"
                print(s)

                s = "Val Loss: {} Val Acc: {}".format( val_loss,val_acc)
                logstr += s + "\n"
                print(s)

                s = "Epoch Dur: {}s".format(round(time.time()-epochtime))
                logstr += s + "\n\n\n\n"
                print(s)
                print()
                print()

        ##############################################################
        #### SAVING
        ##############################################################
        if rank==0 and epoch%val_mod==0 and config.get( "save", True ):
            if config.get("save_folder", None) is not None:
                save_dict = {
                    "mid_epoch": False,
                    "epoch":       epoch,
                    "train_loss":  train_loss,
                    "train_acc":   train_acc,
                    "val_loss":    val_loss,
                    "val_acc":     val_acc,
                    "state_dict":  model.state_dict(),
                    "optim_dict":  optimizer.state_dict(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "config":        config,
                }
                mod = config.get("sd_save_mod", None)
                keep_prev_sd = mod and epoch%mod==0
                save_checkpt(
                    save_dict=save_dict,
                    save_folder=config["save_folder"],
                    save_name="checkpt",
                    epoch=epoch,
                    ext=".pt",
                    del_prev_sd=not keep_prev_sd
                )
                save_training_log(config, logstr)

        # Clean up
        keys = list(package.keys())
        for k in keys: del package[k]
        if config.get("exp_name","myexp")=="test" and epoch>2: break
    return model


def save_training_log(config, logstr, fname="training_log.txt", reset=False):
    """
    Saves the logstr to the save folder under the name training_log.txt

    config: dict
    logstr: str
        the string to save
    fname: str
        the name of the file to save to
    reset: bool
        if true, resets the training log and then writes. otherwise
        appends to training log
    """
    mode = "w" if reset else "a"
    with open(os.path.join(config["save_folder"], fname),mode) as f:
        f.write(logstr)

def config_error_catching(config):
    """
    This function just makes sure that some obvious hyperparameter
    choices are set and some obviously wrong hyperparameter settings
    are changed to what the experimenter meant.
    """
    config = dl_utils.training.config_error_catching(config)
    config["model_folder"] = get_save_folder(config)
    config["save_folder"] = os.path.join(
        config["exp_folder"], config["model_folder"]
    )
    return config

def parse(value):
    value = str(value)
    try:
        if value[-1].isnumeric():
            if "." in value:
                return float(value)
            return int(value)
        elif value.lower() in {"false", "true"}:
            value = value.lower()=="true"
    except:
        pass
    return value

if __name__=="__main__":
    config = { }
    if len(sys.argv)>1:
        config = {}
        for arg in  sys.argv[1:]:
            if ".yaml" in arg or ".json" in arg:
                config = {**config, **load_json_or_yaml(arg)}
            elif "=" in arg:
                splt = arg.split("=")
                config[splt[0]] = parse(splt[1])
            
    train(0, config)


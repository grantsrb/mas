import torch
import numpy as np
import time
from tqdm import tqdm
import os
import sys
import pandas as pd
try:
    import accelerate
except:
    print("install accelerate")

from dl_utils.save_io import (
    save_checkpt, load_json_or_yaml, record_session, load_init_checkpt,
    load_checkpoint,
)
from dl_utils.schedulers import DecayScheduler, PlateauTracker
from dl_utils.utils import package_versions, pad_to
from dl_utils.datas import CausalDataset
from dl_utils.tokenizer import Tokenizer

from seq_models import make_model
from utils import tensor2str
import tasks
from dl_utils.training import run_training
import torch.multiprocessing as mp

torch.autograd.set_detect_anomaly(True)


def make_tokenizer(info):
    words = set()
    for v in info.values():
        if type(v)==list:
            words = words.union(set(v))
        else: words.add(v)
    print("Words:", words)
    word2id = {
        info["pad_token"]: 0,
        info["bos_token"]: 1,
        info["eos_token"]: 2,
    }
    for w in sorted(list(words)):
        if w not in word2id: word2id[w] = len(word2id)
    print("word2id:", word2id)
    tokenizer = Tokenizer(
        word2id=word2id,
        **info,
    )
    return tokenizer

def tokenize_samples(
        samples,
        info,
        tokenizer=None,
        add_bos=True,
        *args, **kwargs):
    """
    This function takes a list of samples and tokenizes them using the
    tokenizer. It also pads the samples to the maximum length of the
    samples.

    Args:
        samples: list of lists of str
            The tokenized strings that need to be converted to token ids
        info: dict
            The information about the task, including the tokenizer
        tokenizer: Tokenizer
            The tokenizer to use for tokenizing the samples

    Returns:
        samples: list of list of int
            The tokenized and padded samples
        tokenizer: Tokenizer
            The tokenizer used for tokenizing the samples
        info: dict
            The information about the task, including the tokenizer
        max_len: int
            The maximum length of the samples after padding
    """
    if tokenizer is None:
        tokenizer = Tokenizer(**info)
        tokenizer.train(tok_X=samples)
    sample_ids = []
    max_len = 0
    unk_id = tokenizer.unk_token_id
    for sample in samples:
        ids = [
            tokenizer.word2id.get(word, unk_id) for word in sample
        ]
        if add_bos:
            ids = [tokenizer.bos_token_id] + ids
        sample_ids.append(ids)
        if len(ids)>max_len: max_len = len(ids)
    return sample_ids, tokenizer, info, max_len

def get_datasets(config):
    """
    Returns a tokenizer, a train CausalDataset, and a valid CausalDataset.

    Returns:
        tokenizer: Tokenizer
        train_dataset: CausalDataset
        valid_dataset: CausalDataset
    """
    n_train = config.get("n_train_samples", 1000)
    n_valid = config.get("n_train_samples", 500)
    task_config = config.get("task_config", dict())
    task = getattr(tasks, config["task_type"])(**task_config)
    info = task.info
    tokenizer = make_tokenizer( info=info )
    train_samps, train_tmasks, _ = task.generate_samples(n_train)
    valid_samps, valid_tmasks, _ = task.generate_samples(n_valid)
    train_samps, tokenizer, info, tmax_len = tokenize_samples(
        samples=train_samps, info=info, tokenizer=tokenizer, **config)
    valid_samps, _, _, vmax_len = tokenize_samples(
        samples=valid_samps, info=info, tokenizer=tokenizer,**config)

    # Pad to insure equal length
    pad = tokenizer.pad_token_id
    add_bos = config.get("add_bos", True)
    train_samps = [
        pad_to(s, tot_len=tmax_len, fill_val=pad) for s in train_samps
    ]
    train_tmasks = [
        pad_to(add_bos*[0]+t, tot_len=tmax_len, fill_val=0) for t in train_tmasks
    ]
    valid_samps = [
        pad_to(s, tot_len=vmax_len, fill_val=pad) for s in valid_samps
    ]
    valid_tmasks = [
        pad_to(add_bos*[0]+t, tot_len=vmax_len, fill_val=0) for t in valid_tmasks
    ]

    train_samps = torch.LongTensor(train_samps)
    train_tmasks = torch.BoolTensor(train_tmasks)
    valid_samps = torch.LongTensor(valid_samps)
    valid_tmasks = torch.BoolTensor(valid_tmasks)

    temp = {**config, **info, **tokenizer.special_ids}
    train_dataset = CausalDataset(
        data=train_samps,
        labels=None,
        masks={"task_mask": train_tmasks},
        **temp)
    valid_dataset = CausalDataset(
        data=valid_samps,
        labels=None,
        masks={"task_mask": valid_tmasks},
        **temp)
    config["info"] = info
    config["seq_len"] = tmax_len
    config["n_tokens"] = len(tokenizer.word2id)
    config["out_tokens"] = len(tokenizer.word2id)
    return tokenizer, train_dataset, valid_dataset


def train(rank, config, verbose=True, *args, **kwargs):
    torch.cuda.empty_cache()

    # Hyperparameters
    config = config_error_catching(config) # Make sure we have valid config
    config["save_folder"] = config.get("save_folder", "./mytraining")
    config["seed"] = config.get("seed", int(time.time()))
    if config["seed"] is None: config["seed"] = int(time.time())
    torch.manual_seed(config["seed"]+rank)
    np.random.seed(   config["seed"]+rank)
    config["rank"] = rank

    # Dataset/Tokenizer
    #######################################
    if verbose and rank==0: print("Making Data")
    # This function updates the config dict and returns DataSet objects
    tokenizer, train_dataset, val_dataset = get_datasets(config)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config.get("batch_size", 128)
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=True,
        batch_size=config.get("batch_size", 1000)
    )
    if verbose and rank==0:
        print("Train Samples:", len(train_dataset))
        print("Val Samples:", len(val_dataset))
        print("Using Sequence Length:", config["seq_len"])

    # Model
    #######################################
    model = make_model(config)
    model = load_init_checkpt(model, config)
    model = load_embeddings(model, config)
    n_params = 0
    for p in model.parameters():
        if hasattr(p, "data"):
            n_params += p.data.numel()
    config["n_params"] = n_params
    print("NParameters:", n_params)

    # Optimizer
    #######################################
    if verbose and rank==0:
        print("Creating Optimizer")
    config["lr"] = config.get("lr", 0.001)
    optimizer = getattr(torch.optim, config.get("optim_type","Adam"))(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("l2", 0),
    )

    # Scheduler
    #######################################
    scheduler = DecayScheduler( optimizer, **config )
    plateau_tracker = PlateauTracker(**config)

    # Distributed Wrapper
    #######################################
    if rank==0 and verbose and torch.cuda.device_count()>1:
        print("Handling multiple GPUs")
    try:
        accelerator = accelerate.Accelerator()
        model, optimizer, train_loader = accelerator.prepare(
            model, optimizer, train_loader
        )
        val_loader = accelerator.prepare(val_loader)
    except:
        print("error with accelerator")
        model.to(rank)

    #############################################################
    # Save Configuration
    #############################################################
    record_session(config, model, globals_dict=globals())

    #############################################################
    # Training
    #############################################################
    df_dict = {
        "epoch":         [],
        "train_loss":    [],
        "train_acc":     [],
        "train_correct": [],
        "val_loss":      [],
        "val_acc":       [],
        "val_correct":   [],
        "lr":            [],
    }
    n_epochs = config.get("n_epochs", 100)
    best_val_correct = 0
    best_train_correct = 0
    for epoch in range(n_epochs):
        epochtime = time.time()
        torch.cuda.empty_cache()
        if rank==0 and verbose:
            print()
            s = "Beginning Epoch {} - {}".format(
                epoch, config.get("save_folder", "No Save Folder")
            )
            print(s)
            logstr = s + "\n"

        #############################################################
        # Train Loop
        #############################################################
        model.train()
        avg_loss = 0
        avg_acc = 0
        avg_correct = 0
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
            corrects = package["corrects"]

            try:
                accelerator.backward(loss)
            except:
                loss.backward()

            avg_acc += acc.item()
            avg_loss += loss.item()
            avg_correct += corrects.float().mean().item()

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

            if verbose and i%10==0 and rank==0:
                dec = 4
                l = round(loss.item(), dec)
                a = round(acc.item(), dec)
                c = round(100*i/nloops, 2)
                t = round(time.time()-starttime, 3)
                s = "Loss: {} -Acc: {}".format(l,a)
                s += " - {}% {}s   ".format(c,t)
                print(s, end=int(len(s)/2)*" " + "\r")


            if config.get("exp_name","deleteme")=="test" and i>=30: break
            if i>=(nloops-1): break
        div = (i+1)
        dec = 5
        train_loss = round(avg_loss/div, dec)
        train_acc  = round(avg_acc/div, dec)
        train_correct = round(avg_correct/div, dec)
        if verbose:
            s = "Example Train Preds:"
            print()
            print(s)
            logstr += s+"\n"

            print("Token Map:", tokenizer.id2word)
            preds = package["pred_ids"]
            targs = data["output_ids"]
            tok = tokenizer
            for i in range(min(3,len(preds))):
                s = "\nIdx : " + tensor2str(torch.arange(len(targs[i])))
                s += "\nTarg: " + tensor2str(targs[i],)
                s += "\nPred: " + tensor2str(preds[i],)
                s += "\nMask: " + tensor2str(data["task_mask"][i],)
                s += "\nRawT: " + " ".join([tok.id2word[int(w)] for w in targs[i]])
                s += "\nRawP: " + " ".join([tok.id2word[int(w)] for w in preds[i]])
                print(s)
                print()
                logstr += s+"\n"

            incorrects = ~(corrects==1)
            if incorrects.float().sum()>0:
                s = "Wrong Train Examples:"
                print(s)
                logstr += s+"\n"
                arr = torch.arange(len(incorrects)).long()
                preds = package["pred_ids"]
                targs = data["output_ids"]
                tok = tokenizer
                for i in range(min(3,incorrects.long().sum().item())):
                    i = arr[incorrects.cpu()][i]
                    s = "\nIdx : " + tensor2str(torch.arange(len(targs[i])))
                    s += "\nTarg: " + tensor2str(targs[i],)
                    s += "\nPred: " + tensor2str(preds[i],)
                    s += "\nMask: " + tensor2str(data["task_mask"][i],)
                    s += "\nRawT: " + " ".join([tok.id2word[int(w)] for w in targs[i]])
                    s += "\nRawP: " + " ".join([tok.id2word[int(w)] for w in preds[i]])
                    print(s)
                    print()
                    logstr += s+"\n"


        #############################################################
        # Validation Loop
        #############################################################
        val_loss =     0
        val_acc =      0
        val_correct =  0
        end_training = False
        if rank==0 and (epoch%val_mod==0 or epoch==n_epochs-1):
            model.eval()
            if verbose: print("Validating...")
            with torch.no_grad():
                nloops = config.get("max_val_loops",len(val_loader))
                nloops = min(nloops, len(val_loader))
                avg_loss = 0
                avg_acc = 0
                avg_correct = 0
                for i,data in enumerate(val_loader):
                    starttime = time.time()
                    package = model(
                        data,
                        ret_preds=True,
                        tforce=False,
                        temperature=config.get(
                            "sampling_temperature", None
                        )
                    )
                    loss = package["loss"]
                    acc = package["acc"]
                    corrects = package["corrects"]

                    avg_loss += loss.item()
                    avg_acc += acc.item()
                    avg_correct += corrects.float().mean().item()

                    if verbose:
                        p = round(100*(i+1)/nloops, 2)
                        t = round(time.time()-starttime, 4)
                        print("{}% -- {}s".format(p,t), end="         \r")
                    if i>=nloops-1: break
            div = (i+1)
            dec = 5
            val_loss = round(avg_loss/div, 5)
            val_acc =  round(avg_acc/div, 5)
            val_correct =  round(avg_correct/div, 5)
            scheduler.step(val_loss)

            df_dict["epoch"].append(epoch)
            df_dict["train_loss"].append(train_loss)
            df_dict["train_acc"].append(train_acc)
            df_dict["train_correct"].append(train_correct)
            df_dict["val_loss"].append(val_loss)
            df_dict["val_acc"].append(val_acc)
            df_dict["val_correct"].append(val_correct)
            df_dict["lr"].append(optimizer.param_groups[0]["lr"])

            if config.get("exp_name", "deleteme")=="test": break
            if verbose:
                print()
                s = "Example Val Preds:"
                print(s)
                logstr += s+"\n"
                preds = package["pred_ids"]
                targs = data["output_ids"]
                tok = tokenizer
                for i in range(min(3,len(preds))):
                    s = "\nIdx : " + tensor2str(torch.arange(len(targs[i])))
                    s += "\nTarg: " + tensor2str(targs[i],)
                    s += "\nPred: " + tensor2str(preds[i],)
                    s += "\nMask: " + tensor2str(data["task_mask"][i],)
                    s += "\nRawT: " + " ".join([tok.id2word[int(w)] for w in targs[i]])
                    s += "\nRawP: " + " ".join([tok.id2word[int(w)] for w in preds[i]])
                    print(s)
                    print()
                    logstr += s+"\n"
                print()

                s = "Final Stats, Epoch: {}".format(epoch)
                print(s)
                logstr += "\n" + s + "\n"

                s = "Train Loss: {} - Train Acc: {} - Train Correct: {}".format(
                    train_loss,train_acc,train_correct
                )
                logstr += s + "\n"
                print(s)

                s = "Val Loss: {} Val Acc: {} - Val Correct: {}".format(
                    val_loss,val_acc,val_correct)
                logstr += s + "\n"
                print(s)

                s = "Epoch Dur: {}s".format(round(time.time()-epochtime))
                logstr += s + "\n\n\n\n"
                print(s)

                print()
                print()

            end_training = plateau_tracker.update(
                val_loss=val_loss, 
                val_acc=val_correct)

        ##############################################################
        #### SAVING
        ##############################################################
        save_mod = config.get("sd_save_mod", np.inf)
        if save_mod is None or save_mod<0: save_mod = np.inf
        if rank==0 and (epoch%val_mod==0 or epoch%save_mod==0 or end_training):
            if config.get( "save", False ):
                save_dict = {
                    "mid_epoch": False,
                    "epoch":       epoch,
                    "train_loss":  train_loss,
                    "train_acc":   train_acc,
                    "train_correct": train_correct,
                    "val_loss":    val_loss,
                    "val_acc":     val_acc,
                    "val_correct": val_correct,
                    "state_dict":  model.state_dict(),
                    "optim_dict":  optimizer.state_dict(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "config":        config,
                }
                # Determine whether to keep the previous save
                keep_prev_sd = save_mod and epoch%save_mod==0

                if keep_prev_sd:
                    # Double the saving increment
                    dmod = config.get("sd_save_double_every", None)
                    double = dmod and epoch%dmod == 0
                    if config.get("save_mod",None) is not None and double:
                        config["sd_save_mod"] *= 2

                best = False
                if train_correct>=best_train_correct-0.001:
                    best_train_correct = train_correct
                    if val_correct>=best_val_correct:
                        best = True
                        best_val_correct = val_correct
                save_checkpt(
                    save_dict=save_dict,
                    save_folder=config["save_folder"],
                    save_name="checkpt",
                    epoch=epoch,
                    ext=".pt",
                    del_prev_sd=not keep_prev_sd,
                    best=best,
                )
                save_training_log(config, logstr)
                save_training_csv(config, df_dict)

        # Clean up
        keys = list(package.keys())
        for k in keys: del package[k]
        if config.get("exp_name", "deleteme")=="test" and epoch>2: break
        if end_training:
            print("Early stopping due to performance plateau!!")
            break
    if verbose:
        print("Ending model", config["save_folder"])
    return model


def save_training_log(config,
                      logstr,
                      fname="training_log.txt",
                      reset=False):
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

def save_training_csv(config,
                      df_dict,
                      fname="training_data.csv"):
    """
    Saves the logstr to the save folder under the name training_log.txt

    config: dict
    logstr: str
        the string to save
    fname: str
        the name of the file to save to
    """
    path = os.path.join(config["save_folder"], fname)
    pd.DataFrame(df_dict).to_csv(path, index=False, header=True)

def load_embeddings(model, config):
    """
    This function assists in loading embedding structures from
    pretrained models at the beginning of training.

    Args:
        model: torch module
        config: dict
            A configuration dict that holds the state dict to be loaded.
            "state_dict": torch state dict
    Returns:
        model: torch module
            the model is updated in place, so returning the model is
            actually unnecessary.
    """
    init_checkpt = config.get("init_checkpt", None)
    if init_checkpt is not None and init_checkpt.strip()!="":
        if not os.path.exists(init_checkpt):
            init_checkpt = os.path.join(config["save_root"], init_checkpt)
        checkpt = load_checkpoint(init_checkpt)
        word2id = config["word2id"]
        init_word2id = checkpt["config"]["word2id"]
        print("checkpt:","\n".join([str(k) for k in checkpt["state_dict"].keys()]))
        print()
        print("model:","\n".join([str(k) for k in model.state_dict().keys()]))
        init_emb_weight = checkpt["state_dict"]["model.embeddings.weight"]
        embs = model.model.embeddings
        for word,id_ in init_word2id.items():
            if word in word2id:
                embs.weight.data[word2id[word]] = init_emb_weight[id_]
    return model

def extract_task_config(config):
    """
    This will make it easier to search over task settings using the
    hyperparameter search paradigm in this progrect. 
    """
    task_keys = {
        # Task Agnostic
        "sep_digits", "reverse_digits", "numeral_base",
        # Numeric Equivalence
        "n_demo_types", "chain_of_num", "strategy", "incl_trigger",
        "pre_trigger", "multi_trigger", "max_count", "hold_outs",
        "max_demo_tokens", "unk_p",
        # Arithmetic
        "min_ops", "max_ops", "max_val", "min_val", "max_new",
        "sep_every", "ops", "n_ops",

        # Induction Heads
        "min_resp_dist", "max_resp_dist", "n_trigs", "n_trig_types",
        "n_token_types", "max_first_idx", "trig_first", "incl_sep",
        "allow_dupls",
    }
    task_config = config.get("task_config", dict())
    for k in task_config: task_keys.add(k)
    for k in task_keys:
        if k in config:
            print("Making changes to", k,"!!!!",
                task_config.get(k,None), "->", config[k])
            task_config[k] = config[k]
    if "chain_of_count" in task_config:
        print("Chain of count is deprecated. Using chain_of_num instead!!")
        task_config["chain_of_num"] = task_config["chain_of_count"]
        del task_config["chain_of_count"]
    config["task_config"] = task_config
    if config.get("task", None) is not None:
        if config["task"]== "multiobj":
            config["task_config"]["copy_task"] = False
            config["task_config"]["n_demo_types"] = 3
        elif config["task"]== "singleobj":
            config["task_config"]["copy_task"] = False
            config["task_config"]["n_demo_types"] = 1
        elif config["task"]== "sameobj":
            config["task_config"]["copy_task"] = True
            config["task_config"]["n_demo_types"] = 1
    return config

def config_error_catching(config):
    """
    This function just makes sure that some obvious hyperparameter
    choices are set and some obviously wrong hyperparameter settings
    are changed to what the experimenter meant.
    """
    config = extract_task_config(config)
    return config

if __name__ == "__main__":
    mp.set_start_method('forkserver')
    run_training(train)


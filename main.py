import sys
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset, load_from_disk
import torch.nn.functional as F

from datas import (
    get_dataset, tokenize_dataset, ensure_equal_length,
    collate_fn, default_replacement_dict,
)
from utils import (
    collect_activations, device_fxn, get_command_line_args,
    default_to_list,
)
import seq_models as smods
from dl_utils.save_io import (
    get_save_name, load_checkpoint, get_folder_from_path, save_json, load_yaml,
)
from dl_utils.tokenizer import Tokenizer
from interchange import InterventionModule

import pandas as pd # import after transformers

def fill_in_prompts_and_replacements(config, yaml_path="./prompts.yaml"):
    prompts = load_yaml(yaml_path)
    config["prompts"] = []
    config["replacements"] = []
    for model_name in config["model_names"]:
        config["prompts"].append(prompts["prompts"].get(model_name, ""))
        config["replacements"].append(
            prompts["replacements"].get(
                model_name,
                [default_replacement_dict]
            )[0]
        )
    return config

def gsm8k_is_correct_batch(model, tokenizer, examples, device, max_new_tokens=50):
    """
    Given a list of examples (each with a "question" and "answer"), tokenizes the questions,
    runs batch generation, and returns a list of booleans indicating whether the answer appears
    in the generated text for each example.
    """
    questions = examples["question"]
    answers = examples["answer"]
    # Tokenize the entire batch; we use padding so that each example is the same length.
    inputs = tokenizer(
        questions, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    # For each example, check whether its answer appears in the generated text.
    return [ans.split("####")[-1] in gen for ans, gen in zip(answers, generated_texts)]


def get_hook(comms_dict):
    def hook_fn(module, input, output):
        if "loop_count" not in comms_dict:
            comms_dict["loop_count"] = 0
        # output is assumed to be of shape (batch, seq_length, hidden_size)
        src_idx = comms_dict.get("src_idx",0)
        trg_idx = comms_dict.get("trg_idx",1)

        if hasattr(output,"hidden_states"):
            trg_actvs = output["hidden_states"]
        else:
            trg_actvs = output

        # Prep source vectors
        src_actvs = comms_dict["src_activations"]

        # Handle case where we have a specific swap mask
        if comms_dict.get("trg_swap_idxs", None) is not None:
            trg_swap_idxs = comms_dict["trg_swap_idxs"]
            src_swap_idxs = comms_dict["src_swap_idxs"]
            i = comms_dict["loop_count"]
            if len(src_actvs.shape)==2: # in contrast to len 3
                trg_swap_idxs = trg_swap_idxs[:,i]
                trg_swap_mask = trg_swap_idxs>-1
                src_swap_mask = (src_swap_idxs==trg_swap_idxs[:,None])
                src_swap_mask = src_swap_mask&(src_swap_idxs>0)
            else:
                trg_swap_mask = trg_swap_idxs>-1
                src_swap_mask = src_swap_idxs>-1
            placeholder = torch.empty_like(trg_actvs)
            placeholder[~trg_swap_mask] = trg_actvs[~trg_swap_mask]
            src_actvs = src_actvs[src_swap_mask]
            trg_actvs = trg_actvs[trg_swap_mask]

        comms_dict["loop_count"] += 1

        ## DEBUGGING
        #device = src_actvs.get_device()
        #p = torch.nn.Parameter(torch.ones_like(trg_actvs))
        #return output * p.to(device)
        ## DEBUGGING

        # Perform causal interchange
        #p = torch.nn.Parameter(torch.ones_like(src_actvs))
        #outs = src_actvs.to(device)*p
        intrv_module = comms_dict["intrv_module"]
        outs = intrv_module(
            target=trg_actvs,
            source=src_actvs,
            target_idx=trg_idx,
            source_idx=src_idx,)

        if comms_dict.get("trg_swap_idxs", None) is not None:
            placeholder[trg_swap_mask] = outs
            outs = placeholder

        if hasattr(output,"hidden_states"):
            output["hidden_states"] = outs
            return output
        else:
            return outs

    return hook_fn

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

def get_model_and_tokenizer(model_name, device=0, padding_side="left"):
    print(f"Loading model and tokenizer for {model_name}...")
    if os.path.exists(model_name):
        checkpt = load_checkpoint(model_name)
        mconfig = checkpt["config"]
        temp = smods.make_model(mconfig)
        temp.load_state_dict(checkpt["state_dict"])
        model = temp.model
        tokenizer = Tokenizer(
            words=set(),
            unk_token=None,
            word2id=mconfig.get("word2id",{}),
            padding_side=padding_side)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side=padding_side)
        if not tokenizer.pad_token:
            tokenizer.pad_token = "<PAD>"
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("okay")
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
    model.to(device)
    model.eval()
    return model, tokenizer

def main():
    arg_config = get_command_line_args(sys.argv)
    ##########################
    #    Default configuration
    ##########################
    defaults = {
        "save_root": "/data2/grantsrb/icml_mas/",
        "exp_name": "myexp",
        "save_memory": True,
        # Use two identical models by default (replace with real LLaMA repo names as needed)
        "model_names": [
            #"meta-llama/Llama-3.2-1B",
            "gpt2",
            "gpt2",
        ], #[, "gpt2"], #
        "dataset_names": ["gsm8k", "gsm8k"],           # gsm8k dataset
        "dataset_kwargs": [
            {"name": "main", "split":"train", } for _ in range(2)
        ],
        "filter_by_correct": False,
        "padding_sides": ["left", "left"],
        "filtered_dataset_paths": [
            "./data/filtered_gsm8k",  # where to save/load the filtered dataset
            "./data/filtered_gsm8k",  # where to save/load the filtered dataset
        ],
        "layers": [ # layers at which to attach the hooks
            "model.embed_tokens",
            "transformer.wte"
        ],  
        "mtx_types": ["RotationMatrix", "RotationMatrix"],
        "identity_init": False,
        "identity_rot": False,
        "mask_type":   "FixedMask", # BoundlessMask
        "n_units": None,
        "learnable_addition": False,

        "num_training_steps": 50000,
        "print_every": 100,
        "batch_size": 32,
        "grad_accumulation_steps": 8,
        "lr": 1e-3,
        "max_length": 128,                 # max token length for our (toy) examples
        "eval_batch_size": 16,             # batch size for correctness evaluation

        "save_keys": ["mtx_types", "mask_type", "layers", "dataset_names"],
    }
    config = {**defaults}
    for k in arg_config: config[k] = arg_config[k]
    print("Config:")
    for k in sorted(list(config.keys())):
        print(k, config[k])

    config["mtx_kwargs"] = [
        {**config} for _ in range(len(config["model_names"]))
    ]
    config["mask_kwargs"] = {**config}
    config = fill_in_prompts_and_replacements(config)

    if torch.cuda.is_available():
        if torch.cuda.device_count()>1:
            devices = [0,1] 
        else:
            devices = [0,0] 
    else:
        devices = ["cpu","cpu"]
    config["padding_sides"] = default_to_list(
        config["padding_sides"],
        n_el=len(config["model_names"])
    )
    padding_sides = config["padding_sides"]

    save_folder = get_folder_from_path(config["model_names"][0])
    if not os.path.exists(save_folder):
        save_folder = os.path.join(
            config.get("save_root", "./"),
            config.get("exp_name", "myexperiment")
        )
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    save_name = get_save_name(
        save_folder=save_folder,
        kwargs=arg_config,
        config=config)
    print("Saving to:", save_folder)

    jpath = os.path.join(save_folder, save_name + ".json")
    save_json(config, jpath)

    ##########################
    #    Load two models and tokenizers
    ##########################
    models = []
    tokenizers = []
    m_sizes = []
    for mi,model_name in enumerate(config["model_names"]):
        model, tokenizer = get_model_and_tokenizer(
            model_name,
            padding_side=padding_sides[mi],
            device=devices[mi]
        )
        model.to(devices[mi])
        model.eval()

        # Freeze model parameters so that only our rotation matrix is trained.
        for param in model.parameters():
            param.requires_grad = False
        print("Model", mi, "-", model_name)
        print(model)
        models.append(model)
        tokenizers.append(tokenizer)
        
        # Just collect a single step to determine the dimensionality of
        # the hooked layer
        with torch.no_grad():
            actvs = collect_activations(
                model,
                input_ids=torch.LongTensor([[0]]),
                layers=[config["layers"][mi]],
                batch_size=500,
                to_cpu=True,)
        m_sizes.append(actvs[config["layers"][mi]].shape[-1])
    
    ##########################
    #    Load the dataset
    ##########################
    print("Loading datasets...")
    datasets = []
    for mi in range(len(config["dataset_names"])):
        dkwargs = {**config["dataset_kwargs"][mi]}
        dkwargs["split"] = "train"
        dkwargs["data_path"] = config.get(
            "train_data_paths",
            ["./data/multiobj.json", "./data/multiobj.json"]
        )[mi]
        dataset = get_dataset(config["dataset_names"][mi], **dkwargs)
        datasets.append(dataset)
    #dkwargs["split"] = "valid"
    #valid_dataset = get_dataset(config["dataset_name"], **dkwargs)

    ##########################
    #    Filter the dataset to keep examples that both models answer correctly.
    #    Save the filtered dataset to disk so that future runs can reload it.
    ##########################
    if not config.get("filter_by_correct", True):
        filtered_dataset = dataset
    else:
        filtered_dataset_path = config["filtered_dataset_path"]
        if filtered_dataset_path and os.path.exists(filtered_dataset_path):
            print(f"Loading filtered dataset from disk at {filtered_dataset_path} ...")
            filtered_dataset = load_from_disk(filtered_dataset_path)
        else:
            print("Filtering dataset by correctness (this may take a while)...")
            filtered_examples = []
            num_examples = len(dataset)
            eval_bs = config["eval_batch_size"]
            for start in range(0, num_examples, eval_bs):
                batch = dataset[start: start + eval_bs]
                correct_mask1 = gsm8k_is_correct_batch(
                    models[0], tokenizers[0], batch, devices[0], max_new_tokens=50)
                correct_mask2 = gsm8k_is_correct_batch(
                    models[1], tokenizers[1], batch, devices[1], max_new_tokens=50)
                # Only keep examples where both models are correct.
                answers = batch["answer"]
                questions = batch["question"]
                for ans, q, corr1, corr2 in zip(answers, questions, correct_mask1, correct_mask2):
                    if corr1 and corr2:
                        filtered_examples.append({"answer": ans, "question": q})
                if start % (eval_bs * 5) == 0:
                    print(f"Processed {start + len(batch)} examples; {len(filtered_examples)} kept so far.")
            if len(filtered_examples) == 0:
                print("WARNING: No examples passed the correctness filter. Using a small subset of the dataset instead.")
                filtered_examples = dataset.select(range(min(10, len(dataset))))
            filtered_dataset = Dataset.from_list(filtered_examples)
            print(f"Saving filtered dataset to disk at {filtered_dataset_path} ...")
            filtered_dataset.save_to_disk(filtered_dataset_path)
    
    ##########################
    #    Tokenize the filtered dataset for autoregressive training.
    #    Here we form an input by concatenating the question and
    #    answer (with a newline and “Answer:” marker).
    ##########################
    train_tokenized_datasets = []
    for mi,tokenizer in enumerate(tokenizers):
        temp = {**config}
        temp["dataset_name"] = temp["dataset_names"][mi]
        temp["replacements"] = temp["replacements"][mi]
        temp["prompt"] = temp["prompts"][mi]
        train_tokenized_datasets.append(
            tokenize_dataset(
                dataset=filtered_dataset,
                tokenizer=tokenizer,
                config=temp,
            )
        )
    #train_tokenized_datasets = ensure_equal_length(
    #    train_tokenized_datasets, pad_sides=padding_sides)

    # Create a DataLoader that iterates over indices of the filtered dataset.
    indices = list(range(len(filtered_dataset)))
    train_loader = DataLoader(
        indices,
        batch_size=config["batch_size"],
        shuffle=True
    )
    
    ##########################
    #    Collect Source Activations
    ##########################
    with torch.no_grad():
        src_activations = []
        src_swap_idxs = []
        src_task_masks = []
        src_pred_ids = []
        src_logits = []
        src_probs = []
        pad_masks = []
        print("Collecting Activations")
        for mi,model in enumerate(models):
            device = devices[mi]
            print("Model", mi, config["model_names"][mi])
            print("Device:", device)
            vbsize = config.get("eval_batch_size", 128)
            batch = collate_fn(
                torch.arange(len(train_tokenized_datasets[mi])).long(),
                train_tokenized_datasets[mi],
                device="cpu")

            actvs = collect_activations(
                model,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                layers=[config["layers"][mi], "lm_head"],
                ret_pred_ids=True,
                batch_size=vbsize,
                to_cpu=True,
                verbose=True,
            )

            src_activations.append(actvs[config["layers"][mi]])
            src_activations[-1] = src_activations[-1].squeeze()

            logits = actvs["lm_head"].squeeze()
            src_logits.append(logits)

            src_probs.append(torch.softmax(logits, dim=-1))
             
            pad_id = tokenizers[mi].pad_token_id
            eos_id = getattr(tokenizers[mi], "eos_id", -1)
            pad_masks.append(
              (batch["input_ids"]==pad_id)|(batch["input_ids"]==eos_id))

            src_pred_ids.append(actvs["pred_ids"].squeeze())
            src_pred_ids[-1][pad_masks[-1]] = pad_id

            if "swap_idxs" in batch:
                src_swap_idxs.append(batch["swap_idxs"].cpu())
            if "task_mask" in batch:
                src_task_masks.append(batch["task_mask"].cpu())

            if "task_mask" in batch:
                tmask = batch["task_mask"].to(device)
                flat_tmask = tmask.reshape(-1)
            else:
                tmask = pad_masks[-1].to(device)
                flat_tmask = tmask.reshape(-1)
            corrects = torch.ones_like(tmask)
            pids = src_pred_ids[-1].to(device)[tmask]
            tids = batch["labels"] .to(device)[tmask]
            idx = pids==tids
            corrects[tmask] = idx
            corrects = corrects.float().sum(-1)==corrects.shape[-1]
            tokacc = (idx).float().mean().item()
            fullacc = corrects.float().mean().item()
            print("TokAcc:", tokacc)
            print("FullAcc:", fullacc)
            ## TODO
            #idxs = torch.arange(len(corrects)).long().to(device)[~corrects]
            #for _ in idxs:
            #    amask = ~batch["attention_mask"][_]
            #    pmask = ~pad_masks[-1][_]
            #    t = tmask[_].cpu()
            #    print("Inpts:", batch["input_ids"][_][pmask])
            #    print("Preds:", src_pred_ids[-1][_][pmask])
            #    print("Targs:", batch["labels"][_][pmask])
            #    print("TInpt:", batch["input_ids"][_][t])
            #    print("TPred:", src_pred_ids[-1][_][t])
            #    print("TTarg:", batch["labels"][_][t])
            #break


    ##########################
    #    Define a single rotation matrix as a learnable parameter.
    #    (We then “force” it to be orthogonal after each optimizer step.)
    ##########################
    intrv_module = InterventionModule(
        sizes=m_sizes,
        **config,
    )
    intrv_module.eval()
    optimizer = torch.optim.Adam(
        intrv_module.parameters(),
        lr=config["lr"])
    
    ##########################
    #    Define and attach forward hooks to a specified layer in each model.
    #    The hook for model 1 applies the rotation (matrix multiplication);
    #    the hook for model 2 applies the transpose.
    ##########################
    comms_dict = {
        "intrv_module": intrv_module,
        "src_activations": None,
        "src_idx": 0,
        "trg_idx": 1,
    }
    hook_fn_model1 = get_hook(comms_dict)
    hook_fn_model2 = get_hook(comms_dict)
    
    hook_module_model1 = get_hook_module(models[0], config["layers"][0])
    hook_module_model2 = get_hook_module(models[1], config["layers"][1])
    hook_handle1 = hook_module_model1.register_forward_hook(hook_fn_model1)
    hook_handle2 = hook_module_model2.register_forward_hook(hook_fn_model2)
    
    ##########################
    #    Training loop: adjust the rotation matrix so that the models (with hooked activations)
    #    autoregressively predict the (filtered) training text.
    #    (Since the underlying models are frozen, only the rotation is updated.)
    ##########################
    global_step = 0
    print("Starting training of the rotation matrix ...")
    models = [model.eval() for model in models]
    optimizer.zero_grad()
    df_dict = {
        "loss": [],
        "tok_acc": [],
        "trial_acc": [],
        "src_idx": [],
        "trg_idx": [],
    }
    while global_step < config["num_training_steps"]:
        for batch_indices in train_loader:
            # Forward passes. The hook functions will transform activations at the chosen layer.
            losses = []
            trial_accs = []
            tok_accs = []
            tot_loss = 0
            tot_tok = 0
            tot_trial = 0
            for sidx in range(len(models)):
                losses.append([])
                trial_accs.append([])
                tok_accs.append([])
                for tidx in range(len(models)):
                    ## Get batch
                    batch = collate_fn(
                        batch_indices,
                        train_tokenized_datasets[tidx],
                        device=devices[tidx])

                    ## Model 1 -> Model 2
                    comms_dict["src_idx"] = sidx
                    comms_dict["trg_idx"] = tidx
                    comms_dict["loop_count"] = 0
                    comms_dict["intrv_module"].to(devices[tidx])
                    comms_dict["src_activations"] =\
                        src_activations[sidx][batch_indices].to(devices[tidx])
                    if "swap_idxs" in batch:
                        ssm = src_swap_idxs[sidx][batch_indices].to(devices[tidx])
                        comms_dict["src_swap_idxs"] = ssm
                        tsm = batch["swap_idxs"].to(devices[tidx])
                        comms_dict["trg_swap_idxs"] = tsm

                    ## Run model
                    outputs = models[tidx](
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],)

                    # Calc Loss
                    if "logits" in outputs:
                        logits = outputs["logits"]
                    else:
                        logits = outputs.logits

                    V = logits.shape[-1]
                    flat = logits.reshape(-1,V)
                    labels = batch["labels"].reshape(-1)
                    tmask = batch["attention_mask"]

                    loss = F.cross_entropy(
                        flat[tmask.reshape(-1)],
                        labels[tmask.reshape(-1)]
                    )

                    accum = config.get("grad_accumulation_steps", 1)
                    loss = loss/accum/4.0
                    if config["save_memory"]:
                        loss.backward()
                    losses[-1].append(loss.item())
                    tot_loss += loss.to(devices[0])

                    if "task_mask" in batch:
                        tmask = batch["task_mask"].to(devices[tidx])
                    else:
                        tmask = batch["attention_mask"]
                    trial = torch.ones_like(batch["labels"]).bool()
                    pids = torch.argmax(logits, dim=-1)
                    #print("Labels:", batch["labels"][0][tmask[0]])
                    #print("\t", tokenizers[tidx].decode(batch["labels"][0][tmask[0]]))
                    #print("Pids:", pids[0][tmask[0]])
                    #print("\t", tokenizers[tidx].decode(pids[0][tmask[0]]))
                    #print("Tids:", tids[0][smask[0]])
                    #print("pids", pids.shape)
                    #print("tmask", tmask.shape)
                    #print("tids", tids.shape)
                    #print("smask", smask.shape)
                    labels = batch["labels"]
                    eq = pids[tmask]==labels[tmask]
                    trial[tmask] = eq
                    trial_acc = trial.sum(-1)==trial.shape[-1]
                    trial_acc = trial_acc.float().mean()
                    tok_acc = eq.float().mean()

                    tot_trial += trial_acc.item()/4.0
                    tot_tok += tok_acc.item()/4.0
                    trial_accs[-1].append(trial_acc.item())
                    tok_accs[-1].append(tok_acc.item())

                    df_dict["loss"].append(loss.item())
                    df_dict["tok_acc"].append(tok_acc.item())
                    df_dict["trial_acc"].append(trial_acc.item())
                    df_dict["src_idx"].append(sidx)
                    df_dict["trg_idx"].append(tidx)

                    # Print a sample generation every print_every steps.
                    if global_step % config["print_every"] == 0:

                        print("\n\nSource Model", sidx, "- Target Model", tidx)
                        print(f"Step {global_step}, Loss: {loss.item()}")
                        perm = torch.randperm(len(logits)).long()
                        labels = batch["labels"]
                        inpts = batch["input_ids"]
                        outs = torch.argmax(logits, dim=-1)#[perm[:2]]
                        pmask = ~pad_masks[tidx][batch_indices]
                        input_mask = pmask
                        if len(src_task_masks)>0:
                            tmask = src_task_masks[tidx][batch_indices]
                            input_mask = pmask&(~tmask)
                        else:
                            tmask = pmask

                        trg_pad_id = tokenizers[tidx].pad_token_id
                        trg_pad_tok = tokenizers[tidx].pad_token

                        for i in range(min(2,len(outs))):
                            # Input Text
                            input_text = tokenizers[tidx].decode(inpts[i][input_mask[i]])
                            if type(input_text)!=str:
                                input_text = input_text[0]
                            input_text = input_text.replace(trg_pad_tok, "")

                            # Target Text
                            target_text = tokenizers[tidx].decode(labels[i][tmask[i]])
                            if type(target_text)!=str:
                                target_text = target_text[0]
                            target_text = target_text.replace(trg_pad_tok, "")

                            # Generated Text
                            generated_text = tokenizers[tidx].decode(outs[i][tmask[i]])
                            if type(generated_text)!=str:
                                generated_text = generated_text[0]
                            generated_text = generated_text.replace(trg_pad_tok, "")

                            print("Input    :", input_text.replace("\n", "\\n"))
                            print("Target   :", target_text.replace("\n", "\\n"))
                            print("Generated:", generated_text.replace("\n", "\\n"))
                            print()
                            print("GenIds:", outs[i][tmask[i]])
                            print("TrgIds:", labels[i][tmask[i]])
                            print()
                        print("Mtx  Type:", config["mtx_types"][0])
                        print("Mask Type:", config["mask_type"],
                                "- Learn:", config["learnable_addition"],
                                "- Units:", intrv_module.swap_mask.n_units)
                        print()


            if not config["save_memory"]:
                tot_loss.backward()
            if global_step % accum==0:
                optimizer.step()
                optimizer.zero_grad()

            if global_step % config["print_every"] == 0:
                print("Step:", global_step)
                print("Tot Loss:", tot_loss.item())
                print("Tok Acc:",  tot_tok)
                print("\tM1->M1:", tok_accs[0][0])
                print("\tM1->M2:", tok_accs[0][1])
                print("\tM2->M1:", tok_accs[1][0])
                print("\tM2->M2:", tok_accs[1][1])
                print("Trial Acc:",tot_trial)
                print("\tM1->M1:", trial_accs[0][0])
                print("\tM1->M2:", trial_accs[0][1])
                print("\tM2->M1:", trial_accs[1][0])
                print("\tM2->M2:", trial_accs[1][1])
            
            ### Save loss and state dict
            if global_step%config.get("save_every_steps", 100):
                csv = os.path.join(save_folder, save_name + ".csv")
                df = pd.DataFrame(df_dict)
                df.to_csv(csv, header=True, index=False)

                pt = os.path.join(save_folder, save_name + ".pt")
                sd = {
                    "config": config,
                    "state_dict": intrv_module.state_dict(),
                }
                torch.save(sd, pt)

            ### Stop training
            global_step += 1
            if global_step >= config["num_training_steps"]:
                break

    ##########################
    # 9. Clean up: remove hooks.
    ##########################
    hook_handle1.remove()
    hook_handle2.remove()
    print("Training complete.")

if __name__ == "__main__":
    main()

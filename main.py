import sys
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset, load_from_disk
import torch.nn.functional as F

from utils import collect_activations, device_fxn, get_command_line_args
from interchange import InterventionModule

def is_correct_batch(model, tokenizer, examples, device, max_new_tokens=50):
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
        # output is assumed to be of shape (batch, seq_length, hidden_size)
        src_idx = comms_dict.get("src_idx",0)
        trg_idx = comms_dict.get("trg_idx",1)
        S = output.shape[1]
        src_actvs = comms_dict["src_activations"][:,-S:]
        device = src_actvs.get_device()

        if hasattr(output,"hidden_states"):
            trg_actvs = output["hidden_states"]
        else:
            trg_actvs = output

        # DEBUGGING
        # p = torch.nn.Parameter(torch.ones_like(trg_actvs))
        # return output * p.to(device)

        # Perform causal interchange
        intrv_module = comms_dict["intrv_module"]
        outs = intrv_module(
            target=trg_actvs,
            source=src_actvs.to(device),
            target_idx=trg_idx,
            source_idx=src_idx,)

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

def main():
    arg_config = get_command_line_args(sys.argv)
    ##########################
    #    Default configuration
    ##########################
    defaults = {
        "exp_name": "myexp",
        # Use two identical models by default (replace with real LLaMA repo names as needed)
        "model_names": ["gpt2", "gpt2"], #["huggyllama/llama-7b", "huggyllama/llama-7b"],
        "dataset_name": "gsm8k",           # gsm8k dataset
        "dataset_kwargs": {
            "name": "main",
            "split":"train",
        },
        "filtered_dataset_path": "./filtered_gsm8k",  # where to save/load the filtered dataset
        "hook_layers": [ # layers at which to attach the hooks
            "transformer.wte",
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
        "max_length": 256,                 # max token length for our (toy) examples
        "eval_batch_size": 16,             # batch size for correctness evaluation
    }
    config = {**defaults}
    for k in arg_config: config[k] = arg_config[k]
    print("Config:")
    for k in sorted(list(config.keys())):
        print(k, config[k])

    config["mtx_kwargs"] = [{**config} for _ in range(len(config["model_names"]))]
    config["mask_kwargs"] = {**config}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_folder, save_name = get_save_paths(kwargs=arg_config, config=config)
    
    ##########################
    #    Load two models and tokenizers
    ##########################
    models = []
    tokenizers = []
    m_sizes = []
    for mi,model_name in enumerate(config["model_names"]):
        print(f"Loading model and tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        tokenizer.pad_token = "|<PAD>|"
        tokenizer.pad_token_id = tokenizer.eos_token_id
        # (If the model is very large, consider using load_in_8bit or device_map="auto".)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(device)
        model.eval()
        # Freeze model parameters so that only our rotation matrix is trained.
        for param in model.parameters():
            param.requires_grad = False
        print("Model", mi, "-", model_name)
        print(model)
        models.append(model)
        tokenizers.append(tokenizer)
        with torch.no_grad():
            actvs = collect_activations(
                model,
                torch.LongTensor([[[0]]]),
                layers=[config["hook_layers"][mi]],
                batch_size=500,
                to_cpu=True,)
        m_sizes.append(actvs[config["hook_layers"][mi]].shape[-1])
    
    ##########################
    #    Load the dataset (gsm8k)
    ##########################
    print("Loading dataset...")
    # features: ["question", "answer"]
    # num_rows: 7473
    dataset = load_dataset(config["dataset_name"], **config["dataset_kwargs"])
    
    ##########################
    #    Filter the dataset to keep examples that both models answer correctly.
    #    Save the filtered dataset to disk so that future runs can reload it.
    ##########################
    filtered_dataset_path = config["filtered_dataset_path"]
    if os.path.exists(filtered_dataset_path):
        print(f"Loading filtered dataset from disk at {filtered_dataset_path} ...")
        filtered_dataset = load_from_disk(filtered_dataset_path)
    else:
        print("Filtering dataset by correctness (this may take a while)...")
        filtered_examples = []
        num_examples = len(dataset)
        eval_bs = config["eval_batch_size"]
        for start in range(0, num_examples, eval_bs):
            batch = dataset[start: start + eval_bs]
            correct_mask1 = is_correct_batch(
                models[0], tokenizers[0], batch, device, max_new_tokens=50)
            correct_mask2 = is_correct_batch(
                models[1], tokenizers[1], batch, device, max_new_tokens=50)
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
    def tokenize_training(example, tokenizer):
        text = example["question"] + "\nAnswer:" + example["answer"]
        return tokenizer(text, truncation=True, padding="max_length", max_length=config["max_length"])

    train_tokenized_datasets = []
    for tokenizer in tokenizers:
        tokenized = filtered_dataset.map(lambda ex: tokenize_training(ex, tokenizer), batched=False)
        train_tokenized_datasets.append(tokenized)
    
    # Create a DataLoader that iterates over indices of the filtered dataset.
    indices = list(range(len(filtered_dataset)))
    train_loader = DataLoader(indices, batch_size=config["batch_size"], shuffle=True)
    
    # A simple collate function that “batches” the tokenized examples.
    def collate_fn(batch_indices, tokenized_dataset, device=0):
        batch = tokenized_dataset.select(batch_indices)
        input_ids = torch.tensor(batch["input_ids"]).to(device)
        attention_mask = torch.tensor(batch["attention_mask"]).to(device)
        # In a standard LM objective the labels are the input_ids (shifted internally by the model)
        return {"input_ids": input_ids, "attention_mask": attention_mask}
    
    ##########################
    #    Collect Source Activations
    ##########################
    with torch.no_grad():
        src_activations = []
        src_pred_ids = []
        src_logits = []
        src_probs = []
        pad_masks = []
        for mi,model in enumerate(models):
            vbsize = 10
            batch = collate_fn(
                torch.arange(len(train_tokenized_datasets[mi])).long(),
                train_tokenized_datasets[mi],
                device="cpu")

            actvs = collect_activations(
                model,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                layers=[config["hook_layers"][mi], "lm_head"],
                ret_pred_ids=True,
                batch_size=vbsize,
                to_cpu=True,)

            src_activations.append(actvs[config["hook_layers"][mi]])
            src_activations[-1] = src_activations[-1].squeeze()

            logits = actvs["lm_head"].squeeze()
            src_logits.append(logits)

            src_probs.append(torch.softmax(logits, dim=-1))
             
            pad_id = tokenizers[mi].pad_token_id
            pad_masks.append(batch["input_ids"]==pad_id)

            src_pred_ids.append(actvs["pred_ids"].squeeze())
            src_pred_ids[-1][pad_masks[-1]] = pad_id


    ##########################
    #    Define a single rotation matrix as a learnable parameter.
    #    (We then “force” it to be orthogonal after each optimizer step.)
    ##########################
    intrv_module = InterventionModule(
        sizes=m_sizes,
        **config,
    )
    intrv_module.eval()
    intrv_module.to(device)
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
    
    hook_module_model1 = get_hook_module(models[0], config["hook_layers"][0])
    hook_module_model2 = get_hook_module(models[1], config["hook_layers"][1])
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
        "direction": [], # 1->2 or 2->1
    }
    while global_step < config["num_training_steps"]:
        for batch_indices in train_loader:
            # Forward passes. The hook functions will transform activations at the chosen layer.
            
            ## Model 1 -> Model 2
            sidx = 0
            tidx = 1
            comms_dict["src_idx"] = sidx
            comms_dict["trg_idx"] = tidx
            comms_dict["src_activations"] = src_activations[sidx][batch_indices].to(device)
            batch = collate_fn(batch_indices, train_tokenized_datasets[tidx],device=device)

            outputs1 = models[tidx](input_ids=batch["input_ids"],
                                  attention_mask=batch["attention_mask"],)


            # Calc Loss 1
            V = outputs1.logits.shape[-1]
            flat = outputs1.logits.reshape(-1,V)
            if config.get("soft_labels", True):
                labels1 = src_probs[sidx][batch_indices].reshape(-1,V).to(device)
            else:
                labels1 = src_pred_ids[sidx][batch_indices].to(device).reshape(-1)
            pmask = ~pad_masks[sidx][batch_indices].reshape(-1).to(device)
            loss1 = F.cross_entropy(flat[pmask], labels1[pmask])

            pids = torch.argmax(flat, dim=-1)[pmask]
            tids = torch.argmax(src_probs[sidx][batch_indices], dim=-1)
            tok_acc1 = (pids==tids.to(device).reshape(-1)[pmask]).float().mean()

            df_dict["loss"].append(loss1.item())
            df_dict["tok_acc"].append(tok_acc1.item())
            df_dict["direction"].append("1->2")


            ## Model 2 -> Model 1
            sidx = 1
            tidx = 0
            comms_dict["src_idx"] = sidx
            comms_dict["trg_idx"] = tidx
            comms_dict["src_activations"] = src_activations[sidx][batch_indices].to(device)
            batch = collate_fn(batch_indices, train_tokenized_datasets[tidx],device=device)
            outputs2 = models[tidx](input_ids=batch["input_ids"],
                                  attention_mask=batch["attention_mask"])

            # Calc Loss 2
            V = outputs2.logits.shape[-1]
            flat = outputs2.logits.reshape(-1,V)
            if config.get("soft_labels", True):
                labels2 = src_probs[sidx][batch_indices].to(device).reshape(-1,V)
            else:
                labels2 = src_pred_ids[sidx][batch_indices].to(device).reshape(-1)
            pmask = ~pad_masks[sidx][batch_indices].reshape(-1).to(device)
            loss2 = F.cross_entropy(flat[pmask], labels2[pmask])

            pids = torch.argmax(flat, dim=-1)[pmask]
            tids = torch.argmax(src_probs[sidx][batch_indices], dim=-1)
            tok_acc2 = (pids==tids.to(device).reshape(-1)[pmask]).float().mean()

            df_dict["loss"].append(loss2.item())
            df_dict["tok_acc"].append(tok_acc2.item())
            df_dict["direction"].append("2->1")


            # Combine losses and backprop
            accum = config.get("grad_accumulation_steps", 1)
            loss = (loss1 + loss2) / 2.0 / accum

            loss.backward()
            if global_step % accum==0:
                optimizer.step()
                optimizer.zero_grad()

            
            # Print a sample generation every print_every steps.
            if global_step % config["print_every"] == 0:

                # sample_prompt = filtered_dataset[0]["question"] + "\nAnswer:"
                # inputs = tokenizers[0](sample_prompt, return_tensors="pt").to(device)
                # with torch.no_grad():
                #     generated_ids = models[0].generate(**inputs, max_new_tokens=50)
                # generated_text = tokenizers[0].decode(generated_ids[0], skip_special_tokens=True)
                print(f"\nStep {global_step}, Loss: {loss.item()}")
                outputs = [outputs1, outputs2]
                labels = [torch.argmax(ps[batch_indices], dim=-1) for ps in src_probs]
                pmasks = [pm[batch_indices] for pm in pad_masks]
                perm = torch.randperm(len(outputs[0].logits)).long()
                for mi in range(2):
                    print("Sample Generation - Model", mi)
                    pmask = ~pmasks[1-mi]
                    labs = labels[1-mi] #[perm[:2]]
                    outs = torch.argmax(outputs[mi].logits, dim=-1)#[perm[:2]]

                    trg_pad_id = tokenizers[mi].pad_token_id
                    src_pad_id = tokenizers[1-mi].pad_token_id
                    trg_pad_tok = tokenizers[mi].pad_token
                    src_pad_tok = tokenizers[1-mi].pad_token

                    for i in range(min(2,len(outs))):
                        # Generated Text
                        generated_text = tokenizers[mi].decode(outs[i][pmask[i]])
                        generated_text = generated_text.replace(trg_pad_tok, "")

                        # Target Text
                        target_text = tokenizers[1-mi].decode(labs[i][pmask[i]])
                        target_text = target_text.replace(src_pad_tok, "")

                        print("Generated:", generated_text.replace("\n", "\\n"))
                        print("Target   :", target_text.replace("\n", "\\n"))
                        print()
                        print("GenIds:", outs[i][pmask[i]])
                        print("TrgIds:", labs[i][pmask[i]])
                        print()
                        print("Mtx  Type:", config["mtx_types"][0])
                        print("Mask Type:", config["mask_type"],
                                "- Learn:", config["learnable_addition"],
                                "- Units:", intrv_module.swap_mask.n_units)
                    print()
            
                print("Step:", global_step)
                print("Loss:")
                print("\tM1->M2:", loss1)
                print("\tM2->M1:", loss2)
                print("Tok Acc:")
                print("\tM1->M2:", tok_acc1)
                print("\tM2->M1:", tok_acc2)
            
            ### Save loss and state dict
            if global_step%config.get("save_every_steps", 100):
                csv = os.path.join(save_folder, save_name+".csv")
                df = pd.DataFrame(df_dict)
                df.to_csv(csv, header=True, index=False)

                pt = os.path.join(save_folder, save_name+".pt")
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

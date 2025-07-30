from datasets import concatenate_datasets, load_dataset

# ====== TOXICHAT ======
def toxichat_filter(dataset, filter_mode="both"):
    if filter_mode=="both":
        return dataset
    elif filter_mode=="toxic":
        return dataset.filter(
            lambda item: (item["toxicity"]>0) or (item["jailbreaking"]>0)
        )
    elif filter_mode=="nontoxic":
        return dataset.filter(
            lambda item: item["toxicity"]==0
        )
    else:
        raise ValueError("Invalid FILTER_MODE", filter_mode)

def get_toxichat_formatter(prompt_template, prompt, tokenizer, max_length):
    def format(example):
        input_text = prompt_template.format(prompt=example["user_input"])
        response = input_text + example["model_output"]

        full_text = prompt + response
        tokenized = tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        tokenized["labels"] = tokenized["input_ids"].clone()
        return {
            **{ k: v.squeeze(0) for k, v in tokenized.items() },
            **{ "text": full_text, "input_text": input_text, }
        }
    return format

# ====== HH-RLHF ======

def hhrlhf_filter(dataset, filter_mode="both"):
    if filter_mode=="both":
        reject_dataset = dataset.rename_column("rejected","text")
        reject_dataset = reject_dataset.remove_columns("chosen")

        chosen_dataset = dataset.rename_column("chosen","text")
        chosen_dataset = chosen_dataset.remove_columns("rejected")
        return concatenate_datasets([chosen_dataset, reject_dataset])
    elif filter_mode=="toxic":
        dataset = dataset.rename_column("rejected","text")
        dataset = dataset.remove_columns("chosen")
        return dataset
    elif filter_mode=="nontoxic":
        dataset = dataset.rename_column("chosen","text")
        dataset = dataset.remove_columns("rejected")
        return dataset
    else:
        raise ValueError("Invalid FILTER_MODE", filter_mode)

def get_hhrlhf_formatter(prompt_template, prompt, tokenizer, max_length):
    def format(example):
        input_text = prompt_template.format(prompt=example["text"])

        full_text = prompt + input_text
        tokenized = tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        tokenized["labels"] = tokenized["input_ids"].clone()
        return {
            **{ k: v.squeeze(0) for k, v in tokenized.items() },
            **{ "text": full_text, "input_text": input_text, }
        }
    return format


# ====== anitamaxvim/jigsaw-toxic-comments ======

def jigsaw_filter(dataset, filter_mode="both"):
    label_keys=[
        "toxic", "severe_toxic", "obscene", "threat", "insult",
        "identity_hate"
    ]
    if filter_mode=="both":
        return dataset
    elif filter_mode=="toxic":
        toxic = dataset.filter(
            lambda item: sum(item[key] for key in label_keys) > 0
        )
        return toxic
    elif filter_mode=="nontoxic":
        nontoxic = dataset.filter(
            lambda item: sum(item[key] for key in label_keys) == 0
        )
        return nontoxic
    else:
        raise ValueError("Invalid FILTER_MODE", filter_mode)


def get_jigsaw_formatter(prompt_template, prompt, tokenizer, max_length):
    def format(example):
        input_text = prompt_template.format(prompt=example["comment_text"])

        full_text = prompt = input_text
        tokenized = tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        tokenized["labels"] = tokenized["input_ids"].clone()
        return {
            **{ k: v.squeeze(0) for k, v in tokenized.items() },
            **{ "text": full_text, "input_text": input_text, }
        }
    return format


# ====== General ======

def filter_for_len(
    dataset,
    tokenizer,
    min_len=None,
    max_len=None,
):
    assert min_len is not None or max_len is not None
    assert min_len is not max_len
    if min_len:
        newset = dataset.filter(
            lambda item: len(tokenizer(item["text"]).input_ids)>min_len
        )
    else:
        newset = dataset.filter(
            lambda item: len(tokenizer(item["text"]).input_ids)<max_len
        )
    return newset


def balance_toxic_dataset(
    dataset,
    seed=42,
    label_keys=["toxicity", "jailbreaking"]
):
    # Count the distribution of labels
    toxic_count = sum(
        sum(item[key] for key in label_keys) > 0 for item in dataset
    )
    nontoxic_count = len(dataset) - toxic_count
    if toxic_count > nontoxic_count:
        nontoxic = dataset.filter(
            lambda item: sum(item[key] for key in label_keys) == 0
        )
        toxic = dataset.filter(
            lambda item: sum(item[key] for key in label_keys) > 0
        ).shuffle(seed=seed).select(range(nontoxic_count))
        dataset = concatenate_datasets([ toxic, nontoxic ])
    elif nontoxic_count > toxic_count:
        nontoxic = dataset.filter(
            lambda item: sum(item[key] for key in label_keys) == 0
        ).shuffle(seed=seed).select(range(toxic_count))
        toxic = dataset.filter(
            lambda item: sum(item[key] for key in label_keys) > 0
        )
        dataset = concatenate_datasets([ toxic, nontoxic ])
    return dataset.shuffle(seed=seed)



def get_filters_and_formatters(
    dataset_name,
    prompt_template,
    tokenizer,
    max_length,
    seed=42,
    prompt="",
):
    if dataset_name=="lmsys/toxic-chat":
        format_fn = get_toxichat_formatter(
            prompt_template=prompt_template,
            prompt=prompt,
            tokenizer=tokenizer,
            max_length=max_length
        )
        balance_fn = lambda dset: balance_toxic_dataset(
            dset, label_keys=["toxicity", "jailbreaking"], seed=seed,
        )
        return toxichat_filter, format_fn, balance_fn
    elif dataset_name=="Anthropic/hh-rlhf":
        format_fn = get_hhrlhf_formatter(
            prompt_template=prompt_template,
            prompt=prompt,
            tokenizer=tokenizer,
            max_length=max_length
        )
        balance_fn = lambda dset: dset
        return hhrlhf_filter, format_fn, balance_fn
    elif dataset_name=="anitamaxvim/jigsaw-toxic-comments":
        format_fn = get_jigsaw_formatter(
            prompt_template=prompt_template,
            prompt=prompt,
            tokenizer=tokenizer,
            max_length=max_length
        )
        balance_fn = lambda dset: balance_toxic_dataset(
            dset,
            label_keys=[
                "toxic", "severe_toxic", "obscene", "threat", "insult",
                "identity_hate"
            ],
            seed=seed,
        )
        return jigsaw_filter, format_fn, balance_fn
    else:
        raise NotImplemented

def prep_dataset(
    dataset_name,
    prompt_template,
    tokenizer,
    max_length,
    split,
    filter_mode,
    seed=42,
    prompt="",
    keeper_cols = ['text', 'input_ids', 'attention_mask', 'labels',],
):
    filter_dataset, format_fn, balance_fn = get_filters_and_formatters(
        dataset_name=dataset_name,
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        max_length=max_length,
        seed=seed,
        prompt=prompt,
    )
    if dataset_name=="lmsys/toxic-chat":
        dset = load_dataset(dataset_name, 'toxicchat0124', split=split)
    else:
        dset = load_dataset(dataset_name, split=split)
    dset = balance_fn(dset)
    dset = filter_dataset(
        dset,
        filter_mode=filter_mode
    )
    dset = dset.map(format_fn)
    remove_cols = [
        col for col in dset.column_names if col not in keeper_cols
    ]
    dset = dset.remove_columns(remove_cols)
    return dset
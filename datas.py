from utils import load_text
from datasets import load_dataset

def get_dataset(dataset_name, **kwargs):
    if dataset_name=="gsm8k":
        return load_dataset(dataset_name, **kwargs)
    elif dataset_name=="num_equivalence":
        path = kwargs.get("data_path", "./multiobj.txt")
        return load_text(file_path=path)
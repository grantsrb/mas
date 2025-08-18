This folder contains scripts to finetune, generate lm text, and perform DAS/MAS on the models.

You can reproduce all LLM results [Model Alignment Search](https://arxiv.org/abs/2501.06164) by first finetuning your huggingface language model using `finetune_script.sh`, then collecting activations and responses from each of the models using `collect_actvs_script.sh`, and lastly running MAS using the `run_lang_mas.sh` script.

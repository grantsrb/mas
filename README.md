# Emergent Symbol-like Number Variables in Artificial Neural Networks

This repo contains the code used to generate the analyses in the paper [Emergent Symbol-like Number Variables in Artificial Neural Networks](https://arxiv.org/abs/2501.06141) (and [Model Alignment Search](https://arxiv.org/abs/2501.06164)).

## 🔍 Overview

Many neural analyses focus on static representations, correlational analyses, and sufficient behavioral representations to interpret neural networks (NNs). Many of these analyses, however, can disregard how the activity causally affects the NN's behavior, and they can disregard the necessity of the activity for behavior. Methods like Distributed Alignment Search are designed to causally intervene on the activations so as to causally relate NN activity to behavior through causal interventions while also isolating necessary activation subspaces for the behavior.

## 🚀 Installation / 📦 Dependencies

You can install the requirements for this repo via pip:

```bash
pip install -r requirements.txt
```

## 🧠 Key Features

- ⚙️ Drop-in analysis tools for trained PyTorch models
- 🔬 Methods to perform DAS using generalized Alignment Functions
- 🔌 Compatible with custom sequence-based pytorch models and Huggingface models  

## 🧪 Example Usage

You will first need a model to analyze. You can create new models trained on the numeric equivalence tasks by first changing the `make_models/make_model_training_file.py` script and then running the following:

```bash
$ python make_models/make_model_training_file.py
$ bash make_models/run_scripts/gru.py
```

Once you have a working model, you can run a DAS or MAS experiment on that model by arguing a configuration yaml file to the main script:

```bash
$ python main.py configs/general_das_config.yaml
```

Look in the `configs` directory for example configuration files.

You can also override configuration settings using comand line arguments:

```bash
$ python main.py configs/general_das_config.yaml model_names=models/multiobject_gru/multiobject_gru_0_seed12345
```

To recreate the experiments used in [Emergent Symbol-like Number Variables in Artificial Neural Networks](https://arxiv.org/abs/2501.06141), you can use the scripts located in `scripts/das_scripts/` after editing the appropriate path variables in the respective scripts:

```bash
$ bash scripts/das_scripts/dispatch_exps.sh
```

## 🧑‍🔬 Citation

If you use this repo in your research, please cite:

> Satchel Grant, Noah D. Goodman, James L. McClelland (2025). *Emergent Symbol-like Number Variables in Artificial Neural Networks*. [Transactions on Machine Learning Research](https://arxiv.org/abs/2501.06141)

*BibTex*
```bibtex
@article{grant2025alignmentfunctions,
    title={Emergent Symbol-like Number Variables in Artificial Neural Networks}, 
    author={Satchel Grant and Noah D. Goodman and James L. McClelland},
    journal={Transactions on Machine Learning Research},
    year={2025},
    url={https://arxiv.org/abs/2501.06141}, 
}
```

## 🙌 Contributing

Contributions, suggestions, and issues are welcome! Open a pull request or file an issue.

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

# Repository info

This repository contains a PyTorch implementation of [Generative models with kernel distance in data space](https://arxiv.org/abs/2009.07327), proposed by Szymon Knop, Marcin Mazur, Przemysław Spurek, Jacek Tabor, Igor Podolak (2020).

## Contents of the repository

```text
|-- src/ - contains an implementation of the models proposed in the paper allowing to reproduce experiments from the original paper
|---- architectures/ - files containing architectures proposed in the paper
|---- externals/ - code adapted from the [external repository](https://github.com/mseitzer/pytorch-fid) to compute FID Score of models
|---- evaluators/ - implementation of evaluators of metrics that will be reported in experiments
|---- factories/ - factories used to create objects proper objects base on command line arguments. Subfolders contain factories for specific models
|---- lighting_modules/ - implementation of experiments in pytorch lightning
|---- metrics/ - directory containing the implementation of all of the metrics used in paper
|---- modules/ - custom neural network layers used in models
|---- tests/ - a bunch of unit tests
|---- train_autoencoder.py - the main script to run all of the experiments
|---- precalc_fid.py - additional script that can be used to precalculate FID statistics for datasets
|-- results/ - directory that will be created to store the results of conducted experiments
|-- data/ - default directory that will be used as a source of data and place to download datasets
```

Experiments are written in `pytorch-lightning` to decouple the science code from the engineering. The `LightningModule` implementation is in `src/lightning_modules/{autoencoder|generator|latent_generator}_module.py` files. For more details refer to [PyTorch-Lightning documentation](https://github.com/PyTorchLightning/pytorch-lightning)

## Conducting the experiments

To execute experiments described in Table 1 in the paper run scripts located in `src/reproduce_table1.sh`
To execute experiments described in Table 3 in the paper run scripts located in `src/reproduce_table3.sh`. Because the training includes two-stage training it is necessary to provide path to checkpoint created in the first stage as an input parameter to second stage.

The repository supports running CWAE and reuses code provided in SWAE paper. All of the implementations are based on the respective papers and repositories.

- For Cramer-Wold AutoEncoders [arXiv](https://arxiv.org/abs/1805.09235) and [GitHub repository](https://github.com/gmum/cwae-pytorch)

- For Sliced-Wasserstein AutoEncoders [arXiv](https://arxiv.org/pdf/1804.01947.pdf) and [GitHub repository](https://github.com/skolouri/swae)

### Browsing the results

Results are stored in tensorboard format. To browse them run the following command:
`tensorboard --logdir results`

## Datasets

The repository uses default datasets provided by PyTorch for MNIST, FashionMNIST, and CELEBA. To convert CELEB-A to 64x64 images we first center crop images to 140x140 and then resize them to 64x64.

## Environment

- python3
- pytorch
- torchvision
- numpy
- pytorch-lightning

## Additional links

To compute FID Scores we have adapted the code from:

- [Pytorch FID](https://github.com/mseitzer/pytorch-fid)

Commit: 011829daeccc84341c1e8e6061d10a640a495573)\*

## License

This implementation is licensed under the MIT License

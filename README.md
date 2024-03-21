# Repository info

This repository contains a PyTorch implementation of [Generative models with kernel distance in data space](https://arxiv.org/abs/2009.07327), proposed by Szymon Knop, Marcin Mazur, Przemysław Spurek, Jacek Tabor, Igor Podolak (2020).

## Contents of the repository

```text
|-- src/ - contains an implementation of the models proposed in the paper allowing to reproduce experiments from the original paper
|---- architecture/ - files containing architectures proposed in the paper
|---- lightning_callbacks/ - implementation of evaluators of metrics reported in our experiments
|---- factories/ - factories used to create objects proper objects base on command line arguments. Subfolders contain factories for specific models
|---- lighting_modules/ - implementation of experiments in pytorch lightning
|---- metrics/ - directory containing the implementation of all of the metrics used in paper
|---- modules/ - custom neural network layers used in models
|---- train_autoencoder.py - the main script to run all of the experiments
|-- results/ - directory that will be created to store the results of conducted experiments
|-- data/ - default directory that will be used as a source of data and place to download datasets
```

Experiments are written in `pytorch-lightning` to decouple the science code from the engineering. The `LightningModule` implementation is in `src/lightning_modules/{autoencoder|generator|latent_generator}_module.py` files. For more details refer to [PyTorch-Lightning documentation](https://github.com/PyTorchLightning/pytorch-lightning)

## Conducting the experiments

To execute experiments described in Table 4 in the paper run scripts located in `src/reproduce_table4.sh`

The repository supports running CWAE and reuses code provided in SWAE paper. All of the implementations are based on the respective papers and repositories.

- For Cramer-Wold AutoEncoders [arXiv](https://arxiv.org/abs/1805.09235) and [GitHub repository](https://github.com/gmum/cwae-pytorch)

- For Sliced-Wasserstein AutoEncoders [arXiv](https://arxiv.org/pdf/1804.01947.pdf) and [GitHub repository](https://github.com/skolouri/swae)

### Browsing the results

Results are stored in tensorboard format. To browse them run the following command:
`tensorboard --logdir results`

## Datasets

The repository uses default datasets provided by PyTorch for MNIST, FashionMNIST, and CELEBA. To convert CELEB-A to 64x64 images we first center crop images to 140x140 and then resize them to 64x64.


### Silverman rule of thumb values

As mentioned in paper we are using precalculated values of Silverman rule of thumb. Below is a table that contains precalculated values for used datasets. To compute these values use `compute_cw_dataset_statistics.py` script.

<center>
    <table>
        <thead>
            <tr>
                <th></th>
                <th colspan=6><center>Dataset</center></th>
            </tr>
            <tr>
                <th>Batch size</th>
                <th><center>MNIST</center></th>
                <th><center>F-MNIST</center></th>
                <th><center>KMNIST</center></th>
                <th><center>SVHN</center></th>
                <th><center>CIFAR-10</center></th>
                <th><center>CELEBA</center></th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><center><b>64</b></center></td>
                <td>0.0202</td><td>0.0265</td><td>0.0258</td><td>0.0084</td><td>0.0134</td><td>0.0166</td>
            </tr>
            <tr>
                <td><center><b>128</b></center></td>
                <td>0.0153</td><td>0.0201</td><td>0.0196</td><td>0.0064</td><td>0.0102</td><td>0.0124</td>
            </tr>
            <tr>
                <td><center><b>256</b></center></td>
                <td>0.0116</td><td>0.0152</td><td>0.0148</td><td>0.0049</td><td>0.0077</td><td>0.0094</td>
            </tr>
        </tbody>
    </table>
</center>

## Stacked MNIST experiment

To perform Stacked MNIST experiment you can use `train_mnist_classifier.py` to train classifier first.

To train classifier run script:
`python -m train_mnist_classifier --model classifier --dataset stacked_mnist --dataroot <dataroot_path>`

stop it when it reaches expected level of accuracy and later reuse classifier:

### CWAE

`
python -m train_autoencoder --model cwae --dataset stacked_mnist --dataroot <dataroot_path> --latent_dim 24 --classifier_checkpoint <classifier_ckpt_path> --gpus 1
`

### CW2

`
python -m train_autoencoder --model cw2_dynamic --dataset stacked_mnist --dataroot <dataroot_path> --latent_dim 24 --classifier_checkpoint <classifier_ckpt_path> --gpus 1 --save_checkpoint --verbose
`

### LCW

`
python -m train_latent_generator --model cwg_dynamic --dataset stacked_mnist --dataroot <dataroot_path> --noise_dim 24 --classifier_checkpoint <classifier_ckpt_path> --gpus 1 --ae_ckpt <cw2_checkpoint_path>
`

## Environment

The code was built and executed on `Python 3.9.13` and dependencies specified in [requirements.txt](./requirements.txt).

To create virtual environment please verify with `python --version` if you are using `Python 3.9.13` and execute:

`python -m venv .venv`

To install please activate your environment using source on Linux or `. \.venv\Scripts\Activate.ps1` on Windows.

and install dependencies:

`pip install -r .\requirements.txt`

Please note that torch dependencies require specifying find links parameter:

`https://download.pytorch.org/whl/torch_stable.html`

## License

This implementation is licensed under the MIT License

from data_modules.dataset_factory import DatasetFactory
from data_modules.latent_generator_dataset_data_module import LatentGeneratorDataModule
from data_modules.stacked_mnist_data_module import StackedMnistDataModule
from lightning_callbacks.silverman_gamma_evaluator import SilvermanGammaEvaluator
from common.noise_creator import NoiseCreator
from lightning_callbacks.generate_sample_images_callback import GenerateSampleImagesCallback
from data_modules.image_dataset_data_module import ImageDatasetDataModule
from lightning_callbacks.stacked_mnist_mode_collapse_evaluator import StackedMnistModeCollapseEvaluator
from lightning_modules.base_generative_module import BaseGenerativeModule
from common.args_parser import BaseArguments
import os
from typing import Iterable, Union
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks.base import Callback
from lightning_callbacks.eval_generative_metrics import EvalGenerativeMetrics
from lightning_callbacks.cw_distance_evaluator import CwDistanceEvaluator
from lightning_callbacks.sw_distance_evaluator import SwDistanceEvaluator
from lightning_modules.classifier_mnist_module import ClassifierMNIST


def get_data_module(hparams: BaseArguments) -> Union[ImageDatasetDataModule, StackedMnistDataModule]:
    dataset_factory = DatasetFactory(hparams.dataset, hparams.dataroot)
    if hparams.dataset == 'stacked_mnist':
        print('Stacked MNIST mode')
        data_module = StackedMnistDataModule(dataset_factory, hparams.batch_size, hparams.batch_size, hparams.workers)
    else:
        data_module = ImageDatasetDataModule(dataset_factory, hparams.batch_size,
                                             hparams.batch_size, hparams.workers)

    return data_module


def train(hparams: BaseArguments, lightning_module: BaseGenerativeModule, callbacks: list[Callback],
          output_dir: str, experiment_name: str, tags: Union[str, Iterable[str]],
          data_module: Union[ImageDatasetDataModule, StackedMnistDataModule, LatentGeneratorDataModule, None] = None):
    print(f'Using random seed: {pl.seed_everything(hparams.random_seed)}')

    os.makedirs(output_dir, exist_ok=True)
    print('Created output dir: ', output_dir)

    noise_creator = NoiseCreator(lightning_module.get_noise_dim())

    if data_module is None:
        data_module = get_data_module(hparams)

    if hparams.dataset != 'stacked_mnist':
        assert isinstance(data_module, ImageDatasetDataModule)

        callbacks = [
            SilvermanGammaEvaluator(),
            CwDistanceEvaluator(),
            SwDistanceEvaluator(),
            *callbacks
        ]

        if not hparams.disable_generative_metrics:
            callbacks = [
                EvalGenerativeMetrics(data_module),
                *callbacks
            ]
    else:
        print('Loading stacked MNIST classifier')
        classifier = ClassifierMNIST.load_from_checkpoint(hparams.classifier_checkpoint).cuda()
        mode_collapse_evaluator = StackedMnistModeCollapseEvaluator(noise_creator, 500, classifier)
        callbacks.append(mode_collapse_evaluator)


    all_callbacks = [GenerateSampleImagesCallback(noise_creator, 64),
                     *callbacks]

    if not hparams.disable_early_stop:
        early_stop = EarlyStopping(
            monitor=hparams.monitor,
            patience=hparams.patience,
            verbose=hparams.verbose,
            mode='min'
        )
        all_callbacks = [early_stop,
                        *all_callbacks]

    checkpoint_callback: Union[ModelCheckpoint, None] = None
    if hparams.save_checkpoint:
        checkpoint_callback = ModelCheckpoint(
            verbose=hparams.verbose,
            monitor=hparams.monitor,
            mode='min'
        )
        all_callbacks = [checkpoint_callback, *all_callbacks]


    trainer = pl.Trainer.from_argparse_args(hparams,
                                            default_root_dir=output_dir,
                                            callbacks=all_callbacks)

    trainer.fit(lightning_module, datamodule=data_module)

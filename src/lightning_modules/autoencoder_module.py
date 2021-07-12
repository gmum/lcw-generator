import torch
import pytorch_lightning as pl
import itertools
import argparse

from numpy.random import randint
from noise_creator import NoiseCreator
from torch.utils.data.dataloader import DataLoader
from evaluators.autoencoder_evaluator import AutoEncoderEvaluator
from evaluators.fid_evaluator import FidEvaluator
from factories.autoencoder.cost_function_factory import get_cost_function
from factories.autoencoder.architecture_factory import get_architecture
from factories.optimizer_factory import get_optimizer_factory
from lightning_modules.common import get_train_dataloader, get_val_dataloader, end_validation_epoch
from factories.fid_evaluator_factory import create_fid_evaluator
from factories.autoencoder_evaluator_factory import create_evaluator
from argparse import Namespace


class AutoEncoderModule(pl.LightningModule):

    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        self.__encoder, self.__decoder = get_architecture(hparams.dataset, hparams.latent_dim)
        self.__noise_creator = NoiseCreator(hparams.latent_dim)
        self.__fid_evaluator = create_fid_evaluator(hparams.eval_fid, self.__noise_creator) if hparams.eval_fid is not None else None
        self.__cost_function = get_cost_function(hparams.model, hparams.lambda_val, hparams.latent_dim, self.__noise_creator)
        self.__autoencoder_evaluator = create_evaluator(self.__noise_creator)
        self.__validation_dataloader: DataLoader = None
        self.__test_reconstruction_images: torch.Tensor = None
        self.__random_sampled_latent: torch.Tensor = None
        self.hparams = hparams

    def get_encoder(self) -> torch.nn.Module:
        return self.__encoder

    def get_decoder(self) -> torch.nn.Module:
        return self.__decoder

    def get_latent_dim(self) -> int:
        return self.hparams.latent_dim

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return get_optimizer_factory(self.parameters(), self.hparams.lr)

    def train_dataloader(self) -> DataLoader:
        return get_train_dataloader(self.hparams)

    def val_dataloader(self) -> DataLoader:
        dataset, dataloader = get_val_dataloader(self.hparams)
        self.__random_sampled_latent = self.__noise_creator.create(64)
        self.__test_reconstruction_images = torch.stack([dataset[i][0] for i in randint(0, len(dataset), size=32)])
        assert len(self.__test_reconstruction_images.size()) == 4 and self.__test_reconstruction_images.size(0) == 32
        return dataloader

    def forward(self, batch: torch.Tensor) -> tuple:
        latent = self.__encoder(batch)
        output_images = self.__decoder(latent)
        return latent, output_images

    def get_generator(self) -> torch.nn.Module:
        return self.__decoder

    def training_step(self, batch: torch.Tensor, _) -> dict:
        batch_images = batch[0]
        latent, output_images = self(batch_images)
        loss = self.__cost_function(batch_images, latent, output_images)

        return {
            'loss': loss,
            'log': {'loss': loss}
        }

    def validation_step(self, batch: torch.Tensor, _) -> dict:
        batch_images = batch[0]
        latent, output_images = self(batch_images)
        loss = self.__cost_function(batch_images, latent, output_images)

        evaluation_metrics = self.__autoencoder_evaluator.evaluate(batch_images, latent, output_images)
        evaluation_metrics['val_loss'] = loss

        return evaluation_metrics

    def validation_epoch_end(self, outputs: dict) -> dict:

        _, decoded_images = self(self.__test_reconstruction_images.to(self.device))

        reconstructions = torch.stack(list(itertools.chain(*zip(self.__test_reconstruction_images.to(self.device), decoded_images))))
        sampled_images = self.__decoder(self.__random_sampled_latent.to(self.device))

        self.logger.experiment.add_images('sampled_images', sampled_images, self.current_epoch)
        self.logger.experiment.add_images('reconstructions', reconstructions, self.current_epoch)

        return end_validation_epoch(outputs, self.__fid_evaluator, self.get_generator(), self.device)

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.Namespace:
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--latent_dim', required=True, type=int, help='latent dimension')
        parser.add_argument('--lambda_val', required=False, type=float, default=1.0, help='value of lambda parameter of a cost function')

        return parser

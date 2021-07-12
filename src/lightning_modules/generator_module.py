import torch
from torch.utils.data import DataLoader
from factories.generator.cost_function_factory import get_cost_function
from factories.generator.architecture_factory import get_architecture
from factories.optimizer_factory import get_optimizer_factory
from evaluators.fid_evaluator import FidEvaluator
from noise_creator import NoiseCreator
from lightning_modules.common import get_train_dataloader, get_val_dataloader, end_validation_epoch
import pytorch_lightning as pl
from factories.fid_evaluator_factory import create_fid_evaluator
from tqdm import tqdm
import argparse


class GeneratorModule(pl.LightningModule):

    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        self.__generator = get_architecture(hparams.dataset, hparams.noise_dim)
        self.__noise_creator = NoiseCreator(hparams.noise_dim)
        self.__cost_function = get_cost_function(hparams.model)
        self.__fid_evaluator = create_fid_evaluator(hparams.eval_fid, self.__noise_creator) if hparams.eval_fid is not None else None

        self.__validation_dataloader: DataLoader = None
        self.__random_sampled_noise = self.__noise_creator.create(64)
        self.hparams = hparams

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return get_optimizer_factory(self.__generator.parameters(), self.hparams.lr)

    def train_dataloader(self) -> DataLoader:
        return get_train_dataloader(self.hparams)

    def val_dataloader(self) -> DataLoader:
        return get_val_dataloader(self.hparams)[1]

    def forward(self, batch: torch.Tensor) -> tuple:
        return self.__generator(batch)

    def get_latent_generator(self) -> torch.nn.Module:
        return self.__latent_generator

    def get_generator(self) -> torch.nn.Module:
        return self.__generator

    def training_step(self, batch: torch.Tensor, _) -> dict:
        batch_images = batch[0]
        noise = self.__noise_creator.create(batch_images.size(0)).to(self.device)
        generated_images = self(noise)
        loss = self.__cost_function(generated_images, batch_images)

        return {
            'loss': loss,
            'log': {'loss': loss}
        }

    def validation_step(self, batch: torch.Tensor, _) -> dict:
        batch_images = batch[0]
        noise = self.__noise_creator.create(batch_images.size(0)).to(self.device)
        generated_latent = self(noise)
        loss = self.__cost_function(generated_latent, batch_images)

        evaluation_metrics = {}
        evaluation_metrics['val_loss'] = loss

        return evaluation_metrics

    def validation_epoch_end(self, outputs: dict) -> dict:
        sampled_images = self.__generator(self.__random_sampled_noise.to(self.device))
        self.logger.experiment.add_images('sampled_images', sampled_images, self.current_epoch)
        return end_validation_epoch(outputs, self.__fid_evaluator, self.get_generator(), self.device)

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--noise_dim', required=True, type=int, help='noise dimension')
        return parser

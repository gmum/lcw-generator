import torch
import argparse
from lightning_modules.base_generative_module import BaseGenerativeModule
from common.args_parser import BaseArguments
from factories.generator.cost_function_factory import get_cost_function
from factories.generator.architecture_factory import get_architecture
from factories.optimizer_factory import get_optimizer_factory
from common.noise_creator import NoiseCreator


class GeneratorParams(BaseArguments):
    noise_dim: int
    gamma: float


class GeneratorModule(BaseGenerativeModule[GeneratorParams]):

    def __init__(self, hparams: GeneratorParams):
        super().__init__(hparams)
        self.__generator = get_architecture(hparams.dataset, hparams.noise_dim)
        self.__noise_creator = NoiseCreator(hparams.noise_dim)
        self.__cost_function = get_cost_function(hparams.model, hparams.gamma)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return get_optimizer_factory(self.__generator.parameters(), self.get_hparams())

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.__generator(batch)

    def get_noise_dim(self) -> int:
        return self.get_hparams().noise_dim

    def get_generator(self) -> torch.nn.Module:
        return self.__generator

    def training_step(self, batch: torch.Tensor, _) -> torch.Tensor:
        noise = self.__noise_creator.create(batch.size(0), self.device)
        generated_images = self(noise)
        loss = self.__cost_function(generated_images, batch)
        self.log('loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch: torch.Tensor, _) -> torch.Tensor:
        noise = self.__noise_creator.create(batch.size(0), self.device)
        generated_images = self(noise)
        val_loss = self.__cost_function(generated_images, batch)
        self.log('val_loss', val_loss, prog_bar=True, on_epoch=True)
        return generated_images

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--noise_dim', required=True, type=int, help='noise dimension')
        parser.add_argument('--gamma', required=False, type=float, help='gamma')
        return parser

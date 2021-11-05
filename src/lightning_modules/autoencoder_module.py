from factories.optimizer_factory import get_optimizer_factory
import torch
import argparse
from lightning_modules.base_generative_module import BaseGenerativeModule
from common.args_parser import BaseArguments
from common.noise_creator import NoiseCreator
from factories.autoencoder.cost_function_factory import get_cost_function
from factories.autoencoder.architecture_factory import get_architecture


class AutoEncoderParams(BaseArguments):
    latent_dim: int
    lambda_val: float
    gamma_val: float


class AutoEncoderModule(BaseGenerativeModule[AutoEncoderParams]):

    def __init__(self, hparams: AutoEncoderParams):
        super().__init__(hparams)
        self.__encoder, self.__decoder = get_architecture(hparams.dataset, hparams.latent_dim)
        self.__noise_creator = NoiseCreator(hparams.latent_dim)
        self.__cost_function = get_cost_function(hparams.model, hparams.lambda_val, hparams.latent_dim, self.__noise_creator, hparams.gamma_val)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return get_optimizer_factory(self.parameters(), self.get_hparams())

    def get_generator(self) -> torch.nn.Module:
        return self.__decoder

    def get_encoder(self) -> torch.nn.Module:
        return self.__encoder

    def get_decoder(self) -> torch.nn.Module:
        return self.__decoder

    def get_noise_dim(self) -> int:
        return self.get_hparams().latent_dim

    def forward(self, batch: torch.Tensor) -> tuple:
        latent = self.__encoder(batch)
        output_images = self.__decoder(latent)
        return latent, output_images

    def training_step(self, batch: torch.Tensor, _) -> torch.Tensor:
        _, _, loss = self.calculate_loss(batch)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch: torch.Tensor, _) -> tuple[torch.Tensor, torch.Tensor]:
        latent, output_images, val_loss = self.calculate_loss(batch)
        self.log('val_loss', val_loss, prog_bar=True, on_epoch=True)
        return latent, output_images

    def calculate_loss(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent, output_images = self(batch)
        loss = self.__cost_function(output_images, latent, batch)
        return latent, output_images, loss

    @ staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--latent_dim', required=True, type=int, help='latent dimension')
        parser.add_argument('--lambda_val', required=False, type=float, default=1.0, help='value of lambda parameter of a cost function')
        parser.add_argument('--gamma_val', required=False, type=float, default=None, help='value of gamma parameter of a cost function')
        return parser

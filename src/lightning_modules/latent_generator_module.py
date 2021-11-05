from lightning_modules.base_generative_module import BaseGenerativeModule
from common.args_parser import BaseArguments
import torch
from factories.latent_generator.cost_function_factory import get_cost_function
from factories.latent_generator.architecture_factory import get_architecture
from factories.optimizer_factory import get_optimizer_factory
from common.noise_creator import NoiseCreator
import argparse
from lightning_modules.autoencoder_module import AutoEncoderModule


class LatentGeneratorParams(BaseArguments):
    ae_ckpt: str
    noise_dim: int


class LatentGeneratorModule(BaseGenerativeModule[LatentGeneratorParams]):

    def __init__(self, hparams: LatentGeneratorParams):
        super().__init__(hparams)

        self.__autoencoder_module: AutoEncoderModule = AutoEncoderModule.load_from_checkpoint(hparams.ae_ckpt)
        print('Autoencoder loading completed')

        self.__latent_generator = get_architecture(hparams.noise_dim, self.__autoencoder_module.get_noise_dim())
        
        self.__noise_creator = NoiseCreator(hparams.noise_dim)

        self.__generator = torch.nn.Sequential(*[
            self.__latent_generator,
            self.__autoencoder_module.get_generator()
        ])

        self.__cost_function = None
        self.__model = hparams.model

    def set_gamma_value(self, gamma_value: float):
        self.__cost_function = get_cost_function(self.__model, gamma_value)

    def get_generator(self) -> torch.nn.Module:
        return self.__generator

    def get_noise_dim(self) -> int:
        return self.get_hparams().noise_dim

    def get_latent_generator(self) -> torch.nn.Module:
        return self.__latent_generator

    def get_autoencoder(self) -> AutoEncoderModule:
        return self.__autoencoder_module

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return get_optimizer_factory(self.__latent_generator.parameters(), self.get_hparams())

    def forward(self, batch: torch.Tensor) -> tuple:
        return self.__latent_generator(batch)

    def training_step(self, encoded_latent: torch.Tensor, _) -> torch.Tensor:
        noise = self.__noise_creator.create(encoded_latent.size(0), self.device)
        generated_latent = self(noise)
        loss = self.__cost_function(generated_latent, encoded_latent)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, encoded_latent: torch.Tensor, _) -> torch.Tensor:
        noise = self.__noise_creator.create(encoded_latent.size(0), self.device)
        generated_latent = self(noise)
        val_loss = self.__cost_function(generated_latent, encoded_latent)
        self.log('val_loss', val_loss, prog_bar=True, on_epoch=True)
        return generated_latent

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--ae_ckpt', required=True, help='path to trained autoencoder checkpoint')
        parser.add_argument('--noise_dim', required=True, type=int, help='noise dimension')

        return parser

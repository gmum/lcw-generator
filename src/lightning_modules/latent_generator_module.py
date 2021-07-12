import torch
from torch.utils.data import DataLoader
from factories.latent_generator.cost_function_factory import get_cost_function
from factories.latent_generator.architecture_factory import get_architecture
from factories.optimizer_factory import get_optimizer_factory
from evaluators.fid_evaluator import FidEvaluator
from noise_creator import NoiseCreator
from lightning_modules.common import get_train_dataloader, get_val_dataloader, end_validation_epoch, convert_to_latent
import pytorch_lightning as pl
from tqdm import tqdm
import argparse
from factories.fid_evaluator_factory import create_fid_evaluator
from lightning_modules.autoencoder_module import AutoEncoderModule


class LatentGeneratorModule(pl.LightningModule):

    def __init__(self, hparams: argparse.Namespace):
        super().__init__()

        autoencoder_module = AutoEncoderModule.load_from_checkpoint(hparams.ae_ckpt)
        print('Autoencoder loading completed')
        self.__encoder = autoencoder_module.get_encoder()
        self.__decoder = autoencoder_module.get_decoder()

        self.__latent_generator = get_architecture(hparams.noise_dim, autoencoder_module.hparams.latent_dim)
        self.__cost_function = get_cost_function(hparams.model)
        self.__noise_creator = NoiseCreator(hparams.noise_dim)
        self.__fid_evaluator = create_fid_evaluator(hparams.eval_fid, self.__noise_creator) if hparams.eval_fid is not None else None

        self.__args = hparams
        self.__validation_dataloader: DataLoader = None
        self.__random_sampled_noise: torch.Tensor = None

        self.__generator = torch.nn.Sequential(*[
            self.__latent_generator,
            self.__decoder
        ])
        self.__validation_encoded_latent = None
        self.hparams = hparams

    def get_encoder(self) -> torch.nn.Module:
        return self.__encoder

    def get_decoder(self) -> torch.nn.Module:
        return self.__decoder

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return get_optimizer_factory(self.__latent_generator.parameters(), self.__args.lr)

    def train_dataloader(self) -> DataLoader:
        dataloader = get_train_dataloader(self.__args)

        print(f'Encoding train dataset')
        encoded_latent = convert_to_latent(dataloader, self.__encoder, self.device)
        return DataLoader(encoded_latent,
                          batch_size=self.__args.batch_size,
                          shuffle=True,
                          num_workers=int(self.__args.workers))

    def val_dataloader(self) -> DataLoader:
        if self.__validation_encoded_latent is None:
            self.__random_sampled_noise = self.__noise_creator.create(64)
            val_dataloader = get_val_dataloader(self.__args)[1]
            print(f'Encoding validation dataset')
            self.__validation_encoded_latent = convert_to_latent(val_dataloader, self.__encoder, self.device)

        return DataLoader(self.__validation_encoded_latent,
                          batch_size=self.__args.batch_size,
                          shuffle=False,
                          num_workers=1,
                          drop_last=True)

    def forward(self, batch: torch.Tensor) -> tuple:
        return self.__latent_generator(batch)

    def get_latent_generator(self) -> torch.nn.Module:
        return self.__latent_generator

    def get_generator(self) -> torch.nn.Module:
        return self.__generator

    def training_step(self, batch: torch.Tensor, _) -> dict:
        encoded_latent = batch
        noise = self.__noise_creator.create(encoded_latent.size(0)).to(self.device)
        generated_latent = self(noise)
        loss = self.__cost_function(generated_latent, encoded_latent)

        return {
            'loss': loss,
            'log': {'loss': loss}
        }

    def validation_step(self, batch: torch.Tensor, _) -> dict:
        encoded_latent = batch
        noise = self.__noise_creator.create(encoded_latent.size(0)).to(self.device)
        generated_latent = self(noise)
        loss = self.__cost_function(generated_latent, encoded_latent)

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

        parser.add_argument('--ae_ckpt', required=True, help='path to trained autoencoder checkpoint')
        parser.add_argument('--noise_dim', required=True, type=int, help='noise dimension')

        return parser

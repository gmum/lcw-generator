import itertools

from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.trainer.trainer import Trainer
import torch
from pytorch_lightning.callbacks import Callback
from torchvision.utils import make_grid
from pytorch_lightning import LightningModule
from lightning_modules.base_generative_module import BaseGenerativeModule


class DecodeImagesCallback(Callback):

    def __init__(self, samples_count: int):
        self.__test_reconstruction_images = None
        self.__samples_count = samples_count

    def on_validation_batch_start(self, _: Trainer, pl_module: LightningModule,  batch: torch.Tensor, ___: int, ____: int) -> None:
        if self.__test_reconstruction_images is None:
            assert pl_module.logger
            self.__test_reconstruction_images = batch[0:self.__samples_count].cpu()
            reconstructions = make_grid(self.__test_reconstruction_images).permute(1, 2, 0).numpy()
            logger: TensorBoardLogger = pl_module.logger
            logger.log_image('target_distribution_visualization', reconstructions, pl_module.current_epoch)

    def on_validation_epoch_end(self, _: Trainer, pl_module: BaseGenerativeModule) -> None:
        assert self.__test_reconstruction_images is not None
        assert pl_module.logger
        logger: TensorBoardLogger = pl_module.logger

        _, decoded_images = pl_module(self.__test_reconstruction_images.to(pl_module.device))
        reconstructions = torch.stack(list(itertools.chain(*zip(self.__test_reconstruction_images.cpu(), decoded_images.cpu()))))
        reconstructions = make_grid(reconstructions).permute(1, 2, 0).numpy()
        logger.log_image('reconstructions', reconstructions, pl_module.current_epoch)

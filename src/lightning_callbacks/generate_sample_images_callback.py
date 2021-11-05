from lightning_modules.base_generative_module import BaseGenerativeModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.utils import make_grid
from common.noise_creator import NoiseCreator


class GenerateSampleImagesCallback(Callback):

    def __init__(self, noise_creator: NoiseCreator, sample_count: int):
        self.__noise_creator = noise_creator
        self.__sample_count = sample_count
        self.__random_sampled_latent = None

    def on_validation_epoch_end(self, _, pl_module: BaseGenerativeModule):
        if self.__random_sampled_latent is None:
            print('Creating noise for sampling reporting...')
            self.__random_sampled_latent = self.__noise_creator.create(self.__sample_count, pl_module.device)
        generator = pl_module.get_generator()
        sampled_images = generator(self.__random_sampled_latent.to(pl_module.device)).cpu()
        sampled_images = sampled_images[0:self.__sample_count]
        tensorboard_logger: TensorBoardLogger = pl_module.logger  # type: ignore
        tensorboard_logger.experiment.add_images('sampled_images', sampled_images, pl_module.current_epoch)

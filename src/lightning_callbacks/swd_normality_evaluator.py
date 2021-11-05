import torch
from metrics.swd import sliced_wasserstein_distance
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer
from lightning_modules.base_generative_module import BaseGenerativeModule
from common.noise_creator import NoiseCreator


class SwdNormalityEvaluator(Callback):

    def __init__(self, noise_creator: NoiseCreator):
        self.__noise_creator = noise_creator

    def on_validation_batch_end(
            self,
            _trainer: Trainer,
            pl_module: BaseGenerativeModule,
            outputs: tuple[torch.Tensor, torch.Tensor],
            _batch: torch.Tensor,
            _batch_idx: int,
            _dataloader_idx: int) -> None:

        latent = outputs[0]
        comparision_sample = self.__noise_creator.create(latent.size(0), pl_module.device)
        swd_penalty_value = sliced_wasserstein_distance(latent, comparision_sample, 50)
        pl_module.log('swd_normality', swd_penalty_value, prog_bar=False, on_epoch=True)

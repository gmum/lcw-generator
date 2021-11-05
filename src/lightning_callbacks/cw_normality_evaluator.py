import torch
from metrics.cw import cw_normality
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer
from lightning_modules.base_generative_module import BaseGenerativeModule


class CwNormalityEvaluator(Callback):

    def on_validation_batch_end(
            self,
            _trainer: Trainer,
            pl_module: BaseGenerativeModule,
            outputs: tuple[torch.Tensor, torch.Tensor],
            _batch: torch.Tensor,
            _batch_idx: int,
            _dataloader_idx: int) -> None:

        latent = outputs[0]
        cw_distance = cw_normality(latent)
        pl_module.log('cw_normality', cw_distance, prog_bar=False, on_epoch=True)

from typing import Union
import torch
from metrics.cw import cw_sampling_silverman
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer
from lightning_modules.base_generative_module import BaseGenerativeModule


class CwDistanceEvaluator(Callback):

    def on_validation_batch_end(
            self,
            _trainer: Trainer,
            pl_module: BaseGenerativeModule,
            outputs: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
            batch: torch.Tensor,
            _batch_idx: int,
            _dataloader_idx: int) -> None:

        if type(outputs) == tuple:
            output_images = outputs[1]
        else:
            output_images: torch.Tensor = outputs  # type: ignore

        assert batch.size() == output_images.size()

        batch = batch.to(pl_module.device)
        flattened_x = torch.flatten(batch, start_dim=1)
        flattened_y = torch.flatten(output_images, start_dim=1)
        cw_distance = cw_sampling_silverman(flattened_x, flattened_y)

        pl_module.log('cw_distance', cw_distance, prog_bar=False, on_epoch=True)

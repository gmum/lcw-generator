from typing import Union
import torch
from metrics.swd import sliced_wasserstein_distance
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer
from lightning_modules.base_generative_module import BaseGenerativeModule


class SwDistanceEvaluator(Callback):

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

        sw_distance = sliced_wasserstein_distance(flattened_x, flattened_y, 5000)

        pl_module.log('sw_distance', sw_distance, prog_bar=False, on_epoch=True)

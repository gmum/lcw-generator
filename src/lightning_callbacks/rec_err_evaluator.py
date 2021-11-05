import torch
from metrics.rec_err import mean_per_image_se
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer
from lightning_modules.base_generative_module import BaseGenerativeModule


class RecErrEvaluator(Callback):

    def on_validation_batch_end(
            self,
            _trainer: Trainer,
            pl_module: BaseGenerativeModule,
            outputs: tuple[torch.Tensor, torch.Tensor],
            batch: torch.Tensor,
            _batch_idx: int,
            _dataloader_idx: int) -> None:
        output_images = outputs[1]
        assert batch.size() == output_images.size()

        batch = batch.to(pl_module.device)
        flattened_x = torch.flatten(batch, start_dim=1)
        flattened_y = torch.flatten(output_images, start_dim=1)

        rec_err = mean_per_image_se(flattened_x, flattened_y)

        pl_module.log('rec_err', rec_err, prog_bar=False, on_epoch=True)

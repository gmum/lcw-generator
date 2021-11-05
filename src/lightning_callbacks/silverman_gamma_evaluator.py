import torch
from metrics.cw import silverman_rule_of_thumb
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer
from lightning_modules.base_generative_module import BaseGenerativeModule


class SilvermanGammaEvaluator(Callback):

    def on_validation_batch_end(
            self,
            _trainer: Trainer,
            pl_module: BaseGenerativeModule,
            outputs: tuple[torch.Tensor, torch.Tensor],
            batch: torch.Tensor,
            _batch_idx: int,
            _dataloader_idx: int) -> None:

        if type(outputs) == tuple:
            output_images = outputs[1]
        else:
            output_images: torch.Tensor = outputs  # type: ignore

        assert batch.size() == output_images.size()

        batch = batch.to(pl_module.device)
        flattened_x = torch.flatten(output_images, start_dim=1)
        flattened_y = torch.flatten(batch, start_dim=1)

        sample_combined = torch.cat((flattened_x, flattened_y), 0)

        pl_module.log('gamma_val_x', silverman_rule_of_thumb(flattened_x.std(), flattened_x.size(0)), prog_bar=False, on_epoch=True)
        pl_module.log('gamma_val_y', silverman_rule_of_thumb(flattened_y.std(), flattened_y.size(0)), prog_bar=False, on_epoch=True)
        pl_module.log('gamma_val_combined', silverman_rule_of_thumb(sample_combined.std(), sample_combined.size(0) // 2), prog_bar=False, on_epoch=True)

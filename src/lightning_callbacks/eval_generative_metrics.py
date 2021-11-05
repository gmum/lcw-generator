import torch
import torch_fidelity
from pytorch_lightning import Trainer
from data_modules.image_dataset_data_module import ImageDatasetDataModule
from lightning_modules.base_generative_module import BaseGenerativeModule
from pytorch_lightning.callbacks import Callback


class GeneratorWrapper(torch.nn.Module):

    def __init__(self, wrapped_generator: torch.nn.Module):
        super().__init__()
        self.__wrapped_generator = wrapped_generator

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        result = self.__wrapped_generator(noise)
        channels_count = result.size(1)
        result = (result * 255).byte()
        if (channels_count == 1):
            result = result.repeat(1, 3, 1, 1)
        return result


class EvalGenerativeMetrics(Callback):

    def __init__(self, data_module: ImageDatasetDataModule):
        super().__init__()
        self.__data_module = data_module

    def on_validation_epoch_end(self, _: Trainer, pl_module: BaseGenerativeModule) -> dict[str, float]:
        generator = pl_module.get_generator()
        noise_dim = pl_module.get_noise_dim()
        internally_wrapped_generator = GeneratorWrapper(generator)
        assert pl_module.logger

        wrapped_generator = torch_fidelity.GenerativeModelModuleWrapper(internally_wrapped_generator, noise_dim, 'normal', 0)
        dataset = self.__data_module.generative_eval_dataset()

        result = torch_fidelity.calculate_metrics(
            input1=wrapped_generator,
            input1_model_num_samples=10000,
            input2=dataset,
            input2_cache_name=f'{self.__data_module.dataset_name()}-geneval',
            cuda=True,
            isc=False,
            fid=True,
            kid=True,
            verbose=True,
        )

        for key, value in result.items():
            pl_module.log(key, value, prog_bar=False, on_epoch=True)

        return result

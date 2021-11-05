from typing import Union
import torch
import torch.utils.data
from torch.utils.data.dataset import Dataset, TensorDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from tqdm import tqdm
from data_modules.drop_target_dataset_wrapper import DropTargetDatasetWrapper
from data_modules.image_dataset_data_module import ImageDatasetDataModule
from data_modules.stacked_mnist_data_module import StackedMnistDataModule


class LatentGeneratorDataModule(pl.LightningDataModule):

    def __init__(self, inner_data_module: Union[ImageDatasetDataModule, StackedMnistDataModule], encoder: torch.nn.Module):
        super().__init__()
        self.__inner_data_module = inner_data_module
        self.__encoder = encoder
        self.__validation_dataset = None
        self.__train_dataset = None
        self.__std = None

    def dataset_name(self) -> str:
        return self.__inner_data_module.dataset_name()

    def setup(self, stage=None):
        self.__inner_data_module.setup()

        if self.__train_dataset is None:
            base_train_dataloader = self.__inner_data_module.train_dataloader()
            self.__train_dataset = self.convert_to_latent(base_train_dataloader, True)

        if self.__validation_dataset is None:
            base_val_dataloader = self.__inner_data_module.val_dataloader()
            self.__validation_dataset = self.convert_to_latent(base_val_dataloader)

    def train_dataset_elements_count(self) -> int:
        return self.__inner_data_module.train_dataset_elements_count()
    
    def get_train_dataloader_std(self) -> float:
        return self.__std

    def train_dataloader(self, drop_last=True, shuffle=True) -> DataLoader:
        assert self.__train_dataset is not None
        return DataLoader(self.__train_dataset, batch_size=self.__inner_data_module.train_batch_size, shuffle=shuffle,
                          num_workers=2, drop_last=drop_last, pin_memory=False)

    def val_dataloader(self) -> DataLoader:
        assert self.__validation_dataset is not None
        return DataLoader(self.__validation_dataset, batch_size=self.__inner_data_module.validation_batch_size,
                          num_workers=4, drop_last=True, pin_memory=False)

    def generative_eval_dataset(self) -> Dataset:
        return self.__inner_data_module.generative_eval_dataset()

    def convert_to_latent(self, dataloader: torch.utils.data.DataLoader, save_std = False) -> DropTargetDatasetWrapper:
        encoded_latent = list()

        with torch.no_grad():
            self.__encoder.eval()
            self.__encoder = self.__encoder.cuda()
            for _, batch_images in tqdm(enumerate(dataloader, 0)):
                latent_vector = self.__encoder(batch_images.cuda())
                encoded_latent.append(latent_vector)

        encoded_latent = torch.cat(encoded_latent).cpu()
        if save_std:
            self.__std = encoded_latent.std()
            print('Storing std value: ', self.__std)
        return DropTargetDatasetWrapper(TensorDataset(encoded_latent))

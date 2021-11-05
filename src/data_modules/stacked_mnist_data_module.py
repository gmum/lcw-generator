from typing import Union
import torch
from torch.utils.data.dataset import Dataset
from data_modules.dataset_factory import DatasetFactory
from torch.utils.data import DataLoader
import pytorch_lightning as pl


def stack_mnist(batch):
    batch_length = len(batch)
    numbers_count = batch_length // 3
    first_number = batch[0:numbers_count]
    second_number = batch[numbers_count:2*numbers_count]
    third_number = batch[2*numbers_count:]
    result = torch.cat([torch.stack(first_number, dim=0), torch.stack(second_number, dim=0), torch.stack(third_number, dim=0)], dim=1)
    return result


class StackedMnistDataModule(pl.LightningDataModule):

    def __init__(self, dataset_factory: DatasetFactory, train_batch_size: int, validation_batch_size: int, workers: int):
        super().__init__()
        self.__dataset_factory = dataset_factory
        self.train_batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size
        self.workers = workers
        self.__validation_dataset = None
        self.__train_dataset = None
        self.__geneval_dataset = None

    def dataset_name(self) -> str:
        return self.__dataset_factory.get_dataset_name()

    def setup(self, stage=None):
        if self.__validation_dataset is None:
            self.__validation_dataset = self.__dataset_factory.get_dataset(False)
            print(f'Size of validation dataset: {len(self.__validation_dataset)}')

        if self.__train_dataset is None:
            self.__train_dataset = self.__dataset_factory.get_dataset(True)
            print(f'Size of train dataset: {len(self.__train_dataset)}')

        if self.__geneval_dataset is None:
            self.__geneval_dataset = self.__dataset_factory.get_eval_dataset()
            print(f'Size of geneval dataset: {len(self.__geneval_dataset)}')

    def train_dataset_elements_count(self) -> int:
        assert self.__train_dataset is not None
        return len(self.__train_dataset)

    def train_dataloader(self, drop_last=True, shuffle=True) -> DataLoader:
        assert self.__train_dataset is not None
        return DataLoader(self.__train_dataset, batch_size=self.train_batch_size * 3, shuffle=shuffle,
                          num_workers=self.workers, drop_last=drop_last, pin_memory=False, collate_fn=stack_mnist)

    def val_dataloader(self) -> DataLoader:
        assert self.__validation_dataset is not None
        return DataLoader(self.__validation_dataset, batch_size=self.validation_batch_size * 3,
                          num_workers=0, drop_last=True, pin_memory=False, collate_fn=stack_mnist)

    def generative_eval_dataset(self) -> Dataset:
        assert self.__geneval_dataset is not None
        return self.__geneval_dataset

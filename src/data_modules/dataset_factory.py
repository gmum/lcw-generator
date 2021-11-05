from data_modules.dataset_resolvers import get_fmnist_dataset, get_mnist_dataset, get_celeba_dataset, get_svhn_dataset, get_cifar10_dataset, get_kmnist_dataset
from typing import Callable
from torchvision.datasets import VisionDataset
from data_modules.drop_target_dataset_wrapper import DropTargetDatasetWrapper


class DatasetFactory:

    def __init__(self, identifier: str, dataroot: str):
        self.__identifier = identifier
        self.__dataroot = dataroot

    def get_dataset_name(self) -> str:
        return self.__identifier

    def get_dataset(self, train: bool) -> DropTargetDatasetWrapper:
        resolvers: dict[str, Callable[[str, bool, bool], VisionDataset]] = {
            'fmnist': get_fmnist_dataset,
            'mnist': get_mnist_dataset,
            'stacked_mnist': get_mnist_dataset,
            'celeba': get_celeba_dataset,
            'svhn': get_svhn_dataset,
            'cifar10': get_cifar10_dataset,
            'kmnist': get_kmnist_dataset
        }
        dataset_resolver = resolvers[self.__identifier]
        dataset = dataset_resolver(self.__dataroot, train, False)

        return DropTargetDatasetWrapper(dataset)

    def get_eval_dataset(self) -> VisionDataset:
        resolvers: dict[str, Callable[[str, bool, bool], VisionDataset]] = {
            'fmnist': get_fmnist_dataset,
            'mnist': get_mnist_dataset,
            'stacked_mnist': get_mnist_dataset,
            'celeba': get_celeba_dataset,
            'svhn': get_svhn_dataset,
            'cifar10': get_cifar10_dataset,
            'kmnist': get_kmnist_dataset
        }
        dataset_resolver = resolvers[self.__identifier]
        dataset = dataset_resolver(self.__dataroot, True, True)

        return DropTargetDatasetWrapper(dataset)

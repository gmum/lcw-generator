from data_modules.to_rgb_tensor import ToLongTensor, ToRgbTensor
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST, CelebA, SVHN, KMNIST, CIFAR10
from torchvision.transforms.transforms import Compose


def get_mnist_dataset(dataroot: str, train: bool, eval: bool) -> MNIST:
    dataset_transforms = get_transforms_for_grayscale_eval_dataset([
        transforms.ToTensor()
    ], eval)

    return MNIST(root=dataroot,
                 train=train,
                 download=True,
                 transform=dataset_transforms)


def get_fmnist_dataset(dataroot: str, train: bool, eval: bool) -> FashionMNIST:
    dataset_transforms = get_transforms_for_grayscale_eval_dataset([
        transforms.ToTensor()
    ], eval)

    return FashionMNIST(root=dataroot,
                        train=train,
                        download=True,
                        transform=dataset_transforms)


def get_kmnist_dataset(dataroot: str, train: bool, eval: bool) -> KMNIST:
    dataset_transforms = get_transforms_for_grayscale_eval_dataset([
        transforms.ToTensor()
    ], eval)

    return KMNIST(root=dataroot,
                  train=train,
                  download=True,
                  transform=dataset_transforms)


def get_celeba_dataset(dataroot: str, train: bool, eval: bool) -> CelebA:
    celeba_transforms = get_transforms_for_eval_dataset([
        transforms.CenterCrop(140),
        transforms.Resize([64, 64]),
        transforms.ToTensor()
    ], eval)
    split = 'train' if train else 'valid'
    return CelebA(root=dataroot,
                  split=split,
                  download=True,
                  transform=celeba_transforms)


def get_svhn_dataset(dataroot: str, train: bool, eval: bool) -> SVHN:
    celeba_transforms = get_transforms_for_eval_dataset([
        transforms.ToTensor()
    ], eval)
    split = 'train' if train else 'test'
    return SVHN(root=dataroot,
                split=split,
                download=True,
                transform=celeba_transforms)


def get_cifar10_dataset(dataroot: str, train: bool, eval: bool) -> CIFAR10:
    celeba_transforms = get_transforms_for_eval_dataset([
        transforms.ToTensor()
    ], eval)
    return CIFAR10(root=dataroot,
                   train=train,
                   download=True,
                   transform=celeba_transforms)


def get_transforms_for_grayscale_eval_dataset(transforms_list: list, eval: bool) -> Compose:
    if eval:
        transforms_list.append(ToRgbTensor())
    return get_transforms_for_eval_dataset(transforms_list, eval)


def get_transforms_for_eval_dataset(transforms_list: list, eval: bool) -> Compose:
    if eval:
        transforms_list.append(ToLongTensor())
    return transforms.Compose(transforms_list)

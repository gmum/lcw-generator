from torch.nn import Module
from architecture.generator.celeba_generator import Generator as CelebaGenerator
from architecture.generator.mnist_generator import Generator as MnistGenerator
from architecture.generator.svhn_generator import Generator as SvhnGenerator
from architecture.generator.stacked_mnist_generator import Generator as StackedMnistGenerator


def get_architecture(identifier: str, noise_dim: int) -> Module:
    if identifier == 'mnist' or identifier == 'fmnist' or identifier == 'kmnist':
        return MnistGenerator(noise_dim)

    if identifier == 'celeba':
        return CelebaGenerator(noise_dim)

    if identifier == 'svhn':
        return SvhnGenerator(noise_dim)

    if identifier == 'stacked_mnist':
        return StackedMnistGenerator(noise_dim)

    raise ValueError("Unknown architecture")

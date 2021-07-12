from torch.nn import Module
from architectures.celeba import Generator as CelebaGenerator
from architectures.mnist import Generator as MnistGenerator


def get_architecture(identifier: str, noise_dim: int) -> Module:
    if identifier == 'mnist' or identifier == 'fmnist':
        return MnistGenerator(noise_dim)

    if identifier == 'celeba':
        return CelebaGenerator(noise_dim)

    raise ValueError("Unknown architecture")

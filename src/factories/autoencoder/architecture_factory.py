from torch.nn import Module
from architectures.mnist import Encoder as MnistEncoder, Decoder as MnistDecoder
from architectures.celeba import Encoder as CelebaEncoder, Decoder as CelebaDecoder
from architectures.latent_generator import LatentGenerator


def get_architecture(identifier: str, z_dim: int) -> Module:
    if identifier == 'mnist' or identifier == 'fmnist':
        return MnistEncoder(z_dim), MnistDecoder(z_dim)

    if identifier == 'celeba':
        return CelebaEncoder(z_dim), CelebaDecoder(z_dim)

    raise ValueError("Unknown architecture")

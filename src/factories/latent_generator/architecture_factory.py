from torch.nn import Module
from architectures.latent_generator import LatentGenerator


def get_architecture(noise_dim: int, latent_size: int) -> Module:
    return LatentGenerator(noise_dim, latent_size)

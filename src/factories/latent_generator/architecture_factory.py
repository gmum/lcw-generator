from torch.nn import Module
from architecture.latent_generator.latent_generator import LatentGenerator


def get_architecture(noise_dim: int, latent_size: int) -> Module:
    print(noise_dim, latent_size)
    return LatentGenerator(noise_dim, latent_size)

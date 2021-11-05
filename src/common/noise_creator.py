
import torch
from typing import Union
from torch.distributions.multivariate_normal import MultivariateNormal


class NoiseCreator:

    def __init__(self, latent_size: int):
        self.__distribution = MultivariateNormal(torch.zeros(latent_size), torch.eye(latent_size))

    def create(self, batch_size: int, device: Union[torch.device, str]) -> torch.Tensor:
        return self.__distribution.sample([batch_size]).to(device)

import torch
import torch.nn as nn


class LatentGenerator(nn.Module):
    def __init__(self, noise_dim: int, latent_size: int, neurons_count: int = 512):
        super().__init__()
        self.__sequential_blocks = [
            nn.Linear(noise_dim, neurons_count),
            nn.BatchNorm1d(neurons_count),
            nn.ReLU(True),
            nn.Linear(neurons_count, neurons_count),
            nn.BatchNorm1d(neurons_count),
            nn.ReLU(True),
            nn.Linear(neurons_count, neurons_count),
            nn.BatchNorm1d(neurons_count),
            nn.ReLU(True),
            nn.Linear(neurons_count, neurons_count),
            nn.BatchNorm1d(neurons_count),
            nn.ReLU(True),
            nn.Linear(neurons_count, neurons_count),
            nn.BatchNorm1d(neurons_count),
            nn.ReLU(True),
            nn.Linear(neurons_count, latent_size)
        ]
        self.__latent_size = latent_size
        self.main = nn.Sequential(*self.__sequential_blocks)

    def forward(self, input_noise: torch.Tensor):
        assert len(input_noise.size()) == 2
        generated_latent = self.main(input_noise)
        assert generated_latent.size(1) == self.__latent_size
        return generated_latent

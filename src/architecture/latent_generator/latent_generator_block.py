import torch
import torch.nn as nn


class LatentGeneratorBlock(nn.Module):

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.__sequential_blocks = [
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(True)
        ]
        self.main = nn.Sequential(*self.__sequential_blocks)

    def forward(self, input_noise: torch.Tensor):
        assert len(input_noise.size()) == 2
        generated_latent = self.main(input_noise)
        return generated_latent

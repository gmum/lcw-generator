from architecture.generator.linear_generator_block import LinearGeneratorBlock
import torch
import torch.nn as nn
from modules.view import View


class Generator(nn.Module):
    def __init__(self, noise_dim: int):
        super().__init__()

        self.__sequential_blocks = [
            LinearGeneratorBlock(noise_dim, 512),
            LinearGeneratorBlock(512, 512),
            LinearGeneratorBlock(512, 512),
            LinearGeneratorBlock(512, 512),
            LinearGeneratorBlock(512, 512),
            LinearGeneratorBlock(512, 512),
            nn.Linear(512, 28 * 28),
            nn.Sigmoid(),
            View(-1, 1, 28, 28)
        ]
        self.main = nn.Sequential(*self.__sequential_blocks)

    def forward(self, noise: torch.Tensor):
        decoded_images = self.main(noise)
        assert decoded_images.size(1) == 1 and decoded_images.size(2) == 28 and decoded_images.size(3) == 28
        return decoded_images

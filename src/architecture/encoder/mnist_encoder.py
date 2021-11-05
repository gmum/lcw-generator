
import torch
import torch.nn as nn
from architecture.generator.linear_generator_block import LinearGeneratorBlock


class Encoder(nn.Module):
    def __init__(self, latent_size: int):
        super().__init__()
        self.__sequential_blocks = [
            nn.Flatten(start_dim=1),
            LinearGeneratorBlock(28*28, 256),
            LinearGeneratorBlock(256, 256),
            LinearGeneratorBlock(256, 256),
            nn.Linear(256, latent_size)
        ]
        self.main = nn.Sequential(*self.__sequential_blocks)

    def forward(self, input_images: torch.Tensor) -> torch.Tensor:
        assert input_images.size(1) == 1 and input_images.size(2) == 28 and input_images.size(3) == 28
        encoded_latent = self.main(input_images)
        return encoded_latent

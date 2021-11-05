from architecture.generator.linear_generator_block import LinearGeneratorBlock
import torch
import torch.nn as nn
from architecture.decoder.celeba_decoder import Decoder


class Generator(nn.Module):
    def __init__(self, noise_dim: int):
        super().__init__()
        self.__sequential_blocks = [
            LinearGeneratorBlock(noise_dim, 512),
            LinearGeneratorBlock(512, 512),
            LinearGeneratorBlock(512, 512),
            LinearGeneratorBlock(512, 512),
            Decoder(512)
        ]
        self.main = nn.Sequential(*self.__sequential_blocks)

    def forward(self, input_noise: torch.Tensor) -> torch.Tensor:
        return self.main(input_noise)

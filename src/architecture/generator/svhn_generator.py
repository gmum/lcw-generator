from architecture.decoder.cnn_decoder_block import CnnDecoderBlock
from modules.map_tanh_zero_one import MapTanhZeroOne
from modules.view import View
from architecture.generator.linear_generator_block import LinearGeneratorBlock
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, noise_dim: int):
        super().__init__()
        self.__sequential_blocks = [
            LinearGeneratorBlock(noise_dim, 128),
            LinearGeneratorBlock(128, 128),
            LinearGeneratorBlock(128, 128),
            LinearGeneratorBlock(128, 256*8*8),
            View(-1, 256, 8, 8),
            CnnDecoderBlock(256, 128),
            CnnDecoderBlock(128, 64),
            nn.ConvTranspose2d(64, 3, 3, 1, 1),
            nn.Tanh(),
            MapTanhZeroOne()
        ]
        self.main = nn.Sequential(*self.__sequential_blocks)

    def forward(self, input_noise: torch.Tensor) -> torch.Tensor:
        return self.main(input_noise)

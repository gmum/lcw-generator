import torch
import torch.nn as nn
from architecture.generator.linear_generator_block import LinearGeneratorBlock
from modules.view import View


class Decoder(nn.Module):
    def __init__(self, latent_size: int):
        super().__init__()
        self.__sequential_blocks = [
            LinearGeneratorBlock(latent_size, 256),
            LinearGeneratorBlock(256, 256),
            LinearGeneratorBlock(256, 256),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid(),
            View(-1, 1, 28, 28)
        ]
        self.main = nn.Sequential(*self.__sequential_blocks)

    def forward(self, input_latent: torch.Tensor):
        decoded_images = self.main(input_latent)
        assert decoded_images.size(1) == 1 and decoded_images.size(2) == 28 and decoded_images.size(3) == 28
        return decoded_images

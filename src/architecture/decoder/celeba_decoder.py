from architecture.decoder.cnn_decoder_block import CnnDecoderBlock
import torch
import torch.nn as nn
from modules.view import View
from modules.map_tanh_zero_one import MapTanhZeroOne


class Decoder(nn.Module):
    def __init__(self, latent_size: int):
        super().__init__()
        self.__sequential_blocks = [
            nn.Linear(latent_size, 1024*8*8),
            View(-1, 1024, 8, 8),
            CnnDecoderBlock(1024, 512),
            CnnDecoderBlock(512, 256),
            CnnDecoderBlock(256, 128),
            nn.ConvTranspose2d(128, 3, 3, 1, 1),
            nn.Tanh(),
            MapTanhZeroOne()
        ]
        self.main = nn.Sequential(*self.__sequential_blocks)

    def forward(self, input_latent: torch.Tensor) -> torch.Tensor:
        decoded_images = self.main(input_latent)
        return decoded_images

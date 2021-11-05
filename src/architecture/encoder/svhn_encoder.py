

from architecture.encoder.cnn_encoder_block import CnnEncoderBlock
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_size: int):
        super().__init__()
        self.__sequential_blocks = [
            CnnEncoderBlock(3, 64),
            CnnEncoderBlock(64, 128),
            CnnEncoderBlock(128, 256),
            nn.Flatten(start_dim=1),
            nn.Linear(256*4*4, latent_size)
        ]
        self.main = nn.Sequential(*self.__sequential_blocks)

    def forward(self, input_images: torch.Tensor):
        assert input_images.size(1) == 3 and input_images.size(2) == 32 and input_images.size(3) == 32
        encoded_latent = self.main(input_images)
        return encoded_latent

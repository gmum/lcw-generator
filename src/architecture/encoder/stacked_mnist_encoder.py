

from architecture.encoder.cnn_encoder_block import CnnEncoderBlock
import torch
import torch.nn as nn

from architecture.generator.linear_generator_block import LinearGeneratorBlock


class Encoder(nn.Module):
    def __init__(self, latent_size: int):
        super().__init__()
        self.__sequential_blocks = [
            CnnEncoderBlock(3, 128),
            CnnEncoderBlock(128, 256),
            nn.Flatten(start_dim=1),
            LinearGeneratorBlock(7 * 7 * 256, 256),
            nn.Linear(256, latent_size)
        ]
        self.main = nn.Sequential(*self.__sequential_blocks)

    def forward(self, input_images: torch.Tensor):
        assert input_images.size(1) == 3 and input_images.size(2) == 28 and input_images.size(3) == 28
        encoded_latent = self.main(input_images)
        return encoded_latent

import torch
import torch.nn as nn


class LinearGeneratorBlock(nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.__sequential_blocks = [
            nn.Linear(in_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(True)
        ]

        self.main = nn.Sequential(*self.__sequential_blocks)

    def forward(self, input_data: torch.Tensor):
        return self.main(input_data)

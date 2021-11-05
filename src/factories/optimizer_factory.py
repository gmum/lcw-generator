from common.args_parser import BaseArguments
from typing import Iterator
from torch.nn.parameter import Parameter
from torch.optim import Adam


def get_optimizer_factory(parameters: Iterator[Parameter], params: BaseArguments):
    return Adam(parameters, lr=params.lr)

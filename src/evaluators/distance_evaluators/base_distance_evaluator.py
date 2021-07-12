import abc
import torch
import numpy as np


class BaseDistanceEvaluator(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

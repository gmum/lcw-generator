import torch
from metrics.cw import cw_sampling_silverman
from evaluators.distance_evaluators.base_distance_evaluator import BaseDistanceEvaluator


class CWDistanceEvaluator(BaseDistanceEvaluator):

    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.size() == y.size()

        flattened_x = torch.flatten(x, start_dim=1)
        flattened_y = torch.flatten(y, start_dim=1)

        return cw_sampling_silverman(flattened_x, flattened_y)

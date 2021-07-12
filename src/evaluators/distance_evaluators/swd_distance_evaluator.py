import torch
from metrics.swd import sliced_wasserstein_distance
from evaluators.distance_evaluators.base_distance_evaluator import BaseDistanceEvaluator


class SWDDistanceEvaluator(BaseDistanceEvaluator):

    def __init__(self, projections_count: int = 1000):
        self.__projections_count = projections_count

    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.size() == y.size()

        flattened_x = torch.flatten(x, start_dim=1)
        flattened_y = torch.flatten(y, start_dim=1)

        return sliced_wasserstein_distance(flattened_x, flattened_y, self.__projections_count)

import torch
from metrics.rec_err import mean_per_image_se
from evaluators.distance_evaluators.base_distance_evaluator import BaseDistanceEvaluator


class RecErrEvaluator(BaseDistanceEvaluator):

    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.size() == y.size()

        flattened_x = torch.flatten(x, start_dim=1)
        flattened_y = torch.flatten(y, start_dim=1)

        return mean_per_image_se(flattened_x, flattened_y)

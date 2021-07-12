import torch
from metrics.cw import cw_normality
from evaluators.sample_evaluators.base_sample_evaluator import BaseSampleEvaluator


class CWNormalitySampleEvaluator(BaseSampleEvaluator):

    def evaluate(self, sample: torch.Tensor) -> torch.Tensor:
        return cw_normality(sample)

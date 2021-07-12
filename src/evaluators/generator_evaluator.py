import torch


class GeneratorEvaluator:

    def __init__(self, output_evaluators: list):
        self.__output_evaluators = output_evaluators

    def evaluate(self, target_points: torch.Tensor, output_points: torch.Tensor) -> dict:
        epoch_results = {}

        for metric_name, output_evaluator in self.__output_evaluators:
            metric_value = output_evaluator.evaluate(target_points, output_points)
            epoch_results[metric_name] = metric_value

        return epoch_results

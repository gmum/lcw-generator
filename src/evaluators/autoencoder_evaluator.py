import torch


class AutoEncoderEvaluator:

    def __init__(self, output_evaluators: list, latent_evaluators: list):
        self.__output_evaluators = output_evaluators
        self.__latent_evaluators = latent_evaluators

    def evaluate(self, input_images: torch.Tensor, latent: torch.Tensor, output_images: torch.Tensor) -> dict:
        epoch_results = {}

        for metric_name, latent_evaluator in self.__latent_evaluators:
            metric_value = latent_evaluator.evaluate(latent)
            epoch_results[metric_name] = metric_value

        for metric_name, output_evaluator in self.__output_evaluators:
            metric_value = output_evaluator.evaluate(input_images, output_images)
            epoch_results[metric_name] = metric_value

        return epoch_results

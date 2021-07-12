from evaluators.distance_evaluators.cw_distance_evaluator import CWDistanceEvaluator
from evaluators.distance_evaluators.swd_distance_evaluator import SWDDistanceEvaluator

from evaluators.generator_evaluator import GeneratorEvaluator


def create_evaluator() -> GeneratorEvaluator:

    output_evaluators = list()
    output_evaluators.append(('cw_output', CWDistanceEvaluator()))
    output_evaluators.append(('sw_output', SWDDistanceEvaluator()))

    return GeneratorEvaluator(output_evaluators)

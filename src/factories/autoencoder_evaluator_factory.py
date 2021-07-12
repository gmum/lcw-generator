from noise_creator import NoiseCreator

from evaluators.sample_evaluators.cw_normality_sample_evaluator import CWNormalitySampleEvaluator
from evaluators.sample_evaluators.swd_sample_evaluator import SWDSampleEvaluator
from evaluators.distance_evaluators.cw_distance_evaluator import CWDistanceEvaluator
from evaluators.distance_evaluators.rec_err_evaluator import RecErrEvaluator
from evaluators.distance_evaluators.swd_distance_evaluator import SWDDistanceEvaluator

from evaluators.autoencoder_evaluator import AutoEncoderEvaluator


def create_evaluator(noise_creator: NoiseCreator) -> AutoEncoderEvaluator:

    rec_err_evaluator = RecErrEvaluator()

    latent_evaluators = list()
    latent_evaluators.append(('cw_normal', CWNormalitySampleEvaluator()))
    latent_evaluators.append(('sw_normal', CWNormalitySampleEvaluator()))

    output_evaluators = list()
    output_evaluators.append(('cw_output', CWDistanceEvaluator()))
    output_evaluators.append(('sw_output', SWDDistanceEvaluator()))
    output_evaluators.append(('rec_err', RecErrEvaluator()))

    autoencoder_evaluator = AutoEncoderEvaluator(output_evaluators, latent_evaluators)

    return autoencoder_evaluator

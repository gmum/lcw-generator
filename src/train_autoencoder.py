from lightning_callbacks.decode_images_callback import DecodeImagesCallback
from common.noise_creator import NoiseCreator
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.base import Callback
from common.args_parser import parse_program_args
from lightning_modules.autoencoder_module import AutoEncoderModule, AutoEncoderParams
from train_generic import train
from lightning_callbacks.rec_err_evaluator import RecErrEvaluator
from lightning_callbacks.cw_normality_evaluator import CwNormalityEvaluator
from lightning_callbacks.swd_normality_evaluator import SwdNormalityEvaluator


def run():
    parser = parse_program_args()
    parser = AutoEncoderModule.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    hparams: AutoEncoderParams = parser.parse_args()  # type: ignore
    output_dir = f'../results/ae/{hparams.dataset}/{hparams.latent_dim}/{hparams.model}'
    autoencoder_model = AutoEncoderModule(hparams)
    noise_creator = NoiseCreator(hparams.latent_dim)

    callbacks: list[Callback] = [
        RecErrEvaluator(),
        CwNormalityEvaluator(),
        DecodeImagesCallback(64),
        SwdNormalityEvaluator(noise_creator)
    ]

    experiment_name = f'ae-{hparams.model}-{hparams.lambda_val}-{hparams.batch_size}-{hparams.optimizer}-{hparams.lr}'
    train(hparams, autoencoder_model, callbacks, output_dir, experiment_name, 'autoencoder')


if __name__ == '__main__':
    run()

from pytorch_lightning import Trainer
from common.args_parser import parse_program_args
from pytorch_lightning.callbacks.base import Callback
from lightning_modules.generator_module import GeneratorModule, GeneratorParams
from train_generic import train


def run():
    parser = GeneratorModule.add_model_specific_args(parse_program_args())
    parser = Trainer.add_argparse_args(parser)
    hparams: GeneratorParams = parser.parse_args()  # type: ignore
    output_dir = f'../results/gen/{hparams.dataset}/{hparams.noise_dim}/{hparams.model}'
    generator_model = GeneratorModule(hparams)

    callbacks: list[Callback] = []

    experiment_name = f'gen-{hparams.model}-{hparams.batch_size}-{hparams.noise_dim}-{hparams.lr}'
    train(hparams, generator_model, callbacks, output_dir, experiment_name, 'generator')


if __name__ == '__main__':
    run()

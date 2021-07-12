import torch
from args_parser import parse_program_args
from lightning_modules.latent_generator_module import LatentGeneratorModule
from train_generic import train


def run():
    parser = LatentGeneratorModule.add_model_specific_args(parse_program_args())
    hparams = parser.parse_args()
    output_dir = f'../results/lg/{hparams.dataset}/{hparams.noise_dim}/{hparams.model}'
    generator_model = LatentGeneratorModule(hparams)
    train(hparams, generator_model, output_dir)


if __name__ == '__main__':
    run()

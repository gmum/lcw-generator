import torch
from args_parser import parse_program_args
from lightning_modules.autoencoder_module import AutoEncoderModule
from train_generic import train


def run():
    parser = AutoEncoderModule.add_model_specific_args(parse_program_args())
    hparams = parser.parse_args()
    output_dir = f'../results/ae/{hparams.dataset}/{hparams.latent_dim}/{hparams.model}'
    autoencoder_model = AutoEncoderModule(hparams)
    train(hparams, autoencoder_model, output_dir)


if __name__ == '__main__':
    run()

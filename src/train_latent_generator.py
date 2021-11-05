from pytorch_lightning.callbacks.base import Callback
from common.args_parser import parse_program_args
from pytorch_lightning import Trainer
from data_modules.latent_generator_dataset_data_module import LatentGeneratorDataModule
from lightning_modules.latent_generator_module import LatentGeneratorModule, LatentGeneratorParams
from train_generic import train, get_data_module
from metrics.cw import silverman_rule_of_thumb

def run():
    parser = LatentGeneratorModule.add_model_specific_args(parse_program_args())
    parser = Trainer.add_argparse_args(parser)
    hparams: LatentGeneratorParams = parser.parse_args()  # type: ignore
    output_dir = f'../results/lg/{hparams.dataset}/{hparams.noise_dim}/{hparams.model}'
    generator_model = LatentGeneratorModule(hparams)

    data_module = get_data_module(hparams)
    wrapped_data_module = LatentGeneratorDataModule(data_module, generator_model.get_autoencoder().get_encoder())
    wrapped_data_module.setup()
    std_value = wrapped_data_module.get_train_dataloader_std()
    gamma_val = silverman_rule_of_thumb(std_value, hparams.batch_size)
    generator_model.set_gamma_value(gamma_val)
    callbacks: list[Callback] = []
    experiment_name = f'lg-{hparams.model}-{hparams.batch_size}-{hparams.noise_dim}-{hparams.optimizer}-{hparams.lr}'
    train(hparams, generator_model, callbacks, output_dir, experiment_name, 'latent-generator', wrapped_data_module)


if __name__ == '__main__':
    run()

import os
import torch
from args_parser import parse_program_args
import pytorch_lightning as pl
from lightning_modules.autoencoder_module import AutoEncoderModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def train(hparams, autoencoder_model, output_dir: str):
    print(f'Using random seed: {pl.seed_everything(hparams.random_seed)}')

    os.makedirs(output_dir, exist_ok=True)
    print('Created output dir: ', output_dir)

    early_stop = EarlyStopping(
        monitor=hparams.monitor,
        min_delta=0.10,
        patience=10,
        verbose=True,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        filepath=None,
        verbose=True,
        monitor=hparams.monitor,
        mode='min'
    )

    trainer = pl.Trainer(gpus=hparams.gpus,
                         max_epochs=hparams.max_epochs,
                         progress_bar_refresh_rate=20,
                         check_val_every_n_epoch=hparams.check_val_every_n_epoch,
                         default_root_dir=output_dir,
                         early_stop_callback=early_stop,
                         checkpoint_callback=checkpoint_callback)
    trainer.fit(autoencoder_model)

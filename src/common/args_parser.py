import argparse


class BaseArguments(argparse.Namespace):
    model: str
    dataset: str
    dataroot: str
    workers: int
    batch_size: int
    optimizer: str
    lr: float
    monitor: str
    random_seed: int
    offline_mode: bool
    patience: int
    verbose: bool
    save_checkpoint: bool
    extra_tag: str
    classifier_checkpoint: str
    disable_generative_metrics: bool
    disable_early_stop: bool


def parse_program_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset', required=True, help='mnist | fmnist | celeba')
    parser.add_argument('--dataroot', default='../data', help='path to dataset')
    parser.add_argument('--workers', type=int, default=16, help='number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='selected optimizer, only adam is supported')
    parser.add_argument('--lr', type=float, default=0.001, help='optimizers learning rate value')
    parser.add_argument('--monitor', type=str, default='val_loss', help='value to monitor')
    parser.add_argument('--random_seed', type=int, default=2137997, help='random seed to parameterize')
    parser.add_argument('--offline_mode', action='store_true', help='disables logging to neptune cloud')
    parser.add_argument('--patience', type=int, default=3, help='early stoping parameter')
    parser.add_argument('--verbose', action='store_true', help='whether to add extra logging')
    parser.add_argument('--save_checkpoint', action='store_true', help='whether to save checkpoints')
    parser.add_argument('--disable_generative_metrics', action='store_true', help='whether evaluate FID, KID, IS and Sample SWD')
    parser.add_argument('--disable_early_stop', action='store_true', help='whether disable early stopping behavior')    
    parser.add_argument('--extra_tag', type=str, default=None, help='extra tag to pass to neptune')
    parser.add_argument('--classifier_checkpoint', type=str, default=None, help="classifier path for stacked mnist experiment")
    return parser

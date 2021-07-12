import argparse


def parse_program_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset', required=True, help='mnist | fmnist | celeba')
    parser.add_argument('--dataroot', default='../data', help='path to dataset')
    parser.add_argument('--workers', type=int, default=16, help='number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='selected optimizer, only adam is supported')
    parser.add_argument('--lr', type=float, default=0.001, help='optimizers learning rate value')
    parser.add_argument('--max_epochs', type=int, default=5000, help='max number of epochs to train for')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--monitor', type=str, default='val_loss', help='value to monitor')
    parser.add_argument('--gpus', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--eval_fid', type=str, default=None, help='Reports FID Score if provided path to precalculated stats (npz file)')
    parser.add_argument('--random_seed', type=int, default=None, help='random seed to parameterize')
    return parser

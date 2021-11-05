import argparse
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from data_modules.dataset_factory import DatasetFactory
from data_modules.image_dataset_data_module import ImageDatasetDataModule
from metrics.cw import silverman_rule_of_thumb


def parse_program_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', required=True, help='mnist | fmnist | celeba')
    parser.add_argument('--batch_size', type=int, required=True, help='count of samples to be compared')
    parser.add_argument('--dataroot', default='../data', help='path to dataset')
    parser.add_argument('--workers', type=int, default=16, help='number of data loading workers')

    return parser


def compute_mean_and_stddev_for_dataset(dataloader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    sum_x = torch.FloatTensor([0.0])
    sum_x2 = torch.FloatTensor([0.0])
    elements_count = 0

    for batch in tqdm(dataloader):
        flattened_batch = torch.flatten(batch)
        sum_x += flattened_batch.sum()
        sum_x2 += flattened_batch.square().sum()
        elements_count += flattened_batch.size(0)

    mean = sum_x / elements_count
    stddev = torch.sqrt(sum_x2/elements_count - mean**2)

    return mean, stddev


def run():
    parser = parse_program_args()
    hparams = parser.parse_args()

    dataset_factory = DatasetFactory(hparams.dataset, hparams.dataroot)
    data_module = ImageDatasetDataModule(dataset_factory, hparams.batch_size, hparams.batch_size, hparams.workers)
    data_module.setup()

    dataloader = data_module.train_dataloader(shuffle=True, drop_last=True)

    mean, stddev = compute_mean_and_stddev_for_dataset(dataloader)
    gamma = silverman_rule_of_thumb(stddev, hparams.batch_size)

    print('Mean: ', mean.item())
    print('StdDev: ', stddev.item())
    print('Silverman rule of thumb: ', gamma.item())


if __name__ == '__main__':
    run()

# MNIST  mean: 0.1307, stddev: 0.3081
# FMNIST mean: 0.2860, stddev: 0.3530
# CELEBA mean: 0.4328, stddev: 0.2774
# (1.06*STDDEV*N^(-1/5))**2

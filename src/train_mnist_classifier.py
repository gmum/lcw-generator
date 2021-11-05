from pytorch_lightning.trainer.trainer import Trainer
from torchvision.datasets.vision import VisionDataset
from common.args_parser import BaseArguments, parse_program_args
from data_modules.dataset_factory import DatasetFactory
from data_modules.image_dataset_data_module import ImageDatasetDataModule
from data_modules.dataset_resolvers import get_mnist_dataset
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning_modules.classifier_mnist_module import ClassifierMNIST


class ClassificationDatasetFactory:

    def __init__(self, identifier: str, dataroot: str):
        self.__identifier = identifier
        self.__dataroot = dataroot

    def get_dataset_name(self) -> str:
        return self.__identifier

    def get_dataset(self, train: bool) -> VisionDataset:
        return get_mnist_dataset(self.__dataroot, train, False)

    def get_eval_dataset(self) -> VisionDataset:
        return get_mnist_dataset(self.__dataroot, False, False)


parser = parse_program_args()
hparams: BaseArguments = parser.parse_args()  # type: ignore


checkpoint_callback = ModelCheckpoint(
    verbose=True,
    monitor='avg_acc',
    mode='max'
)

dataset_factory: DatasetFactory = ClassificationDatasetFactory(hparams.dataset, hparams.dataroot)  # type: ignore
trainer = Trainer(gpus=1, callbacks=[checkpoint_callback])
image_dataset_data_module = ImageDatasetDataModule(dataset_factory, hparams.batch_size,
                                                   hparams.batch_size, hparams.workers)

model = ClassifierMNIST()
print('Starting training!')
trainer.fit(model, image_dataset_data_module)

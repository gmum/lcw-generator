from typing import Any, Sized, Union
from torch.utils.data.dataset import TensorDataset
from torchvision.datasets.vision import VisionDataset


class DropTargetDatasetWrapper(VisionDataset, Sized):
    def __init__(self, wrapped_dataset: Union[VisionDataset, TensorDataset]):
        self.wrapped_dataset = wrapped_dataset

    def __getitem__(self, index: int) -> Any:
        return self.wrapped_dataset.__getitem__(index)[0]

    def __len__(self) -> int:
        return self.wrapped_dataset.__len__()

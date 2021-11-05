from lightning_modules.base_generative_module import BaseGenerativeModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from common.noise_creator import NoiseCreator
from lightning_modules.classifier_mnist_module import ClassifierMNIST
import torch
import torch.nn.functional as F


class StackedMnistModeCollapseEvaluator(Callback):

    def __init__(self, noise_creator: NoiseCreator, sample_count: int, classifier: ClassifierMNIST):
        self.__noise_creator = noise_creator
        self.__sample_count = sample_count
        self.__random_sampled_latent = None
        self.__num_of_samples = 26000
        self.__classifier = classifier

    def on_validation_epoch_end(self, _, pl_module: BaseGenerativeModule):
        modes = [0.0 for i in range(1000)]

        for i in range(self.__num_of_samples // self.__sample_count):
            self.__random_sampled_latent = self.__noise_creator.create(self.__sample_count, pl_module.device)
            generator = pl_module.get_generator()
            sampled_images = generator(self.__random_sampled_latent)
            first_number_labels = self.__classifier.predict_labels(sampled_images[:, 0:1, :, :])
            second_number_labels = self.__classifier.predict_labels(sampled_images[:, 1:2, :, :])
            third_number_labels = self.__classifier.predict_labels(sampled_images[:, 2:3, :, :])
            found_modes_now = (first_number_labels * 100 + second_number_labels * 10 + third_number_labels).cpu().numpy()
            for i in range(len(found_modes_now)):
                modes[found_modes_now[i]] += 1.0

        q = torch.Tensor([0.001 for i in range(1000)])
        modes_tensor = torch.Tensor(modes)
        p = modes_tensor / modes_tensor.sum() + 1e-6
        kl_div = F.kl_div(p.unsqueeze_(0).log(), q.unsqueeze_(0), reduction='sum')
        print(kl_div)
        tensorboard_logger: TensorBoardLogger = pl_module.logger  # type: ignore
        tensorboard_logger.log_metric('modes_count', torch.count_nonzero(modes_tensor), pl_module.current_epoch)
        tensorboard_logger.log_metric('kl_div', kl_div, pl_module.current_epoch)

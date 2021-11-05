
import torch
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from torch.nn.modules.activation import ReLU
from torch.nn.modules.dropout import Dropout, Dropout2d
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import MaxPool2d


class ClassifierMNIST(LightningModule):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.model = nn.Sequential(*[
            nn.Conv2d(1, 32, 5),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5),
            MaxPool2d(2),
            nn.ReLU(),
            Dropout2d(0.5),
            nn.Conv2d(32, 64, 5),
            MaxPool2d(2),
            nn.ReLU(),
            Dropout2d(0.5),
            nn.Flatten(start_dim=1),
            Linear(3*3*64, 256),
            ReLU(),
            Dropout(),
            Linear(256, 10)
        ])

    def forward(self, x):
        return self.model(x)

    def predict_labels(self, x: torch.Tensor) -> torch.Tensor:
        logits = self(x)
        labels = logits.max(1).indices
        return labels

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def training_step(self, batch, _):
        x, y = batch
        logits = self(x)

        loss = self.criterion(logits, y)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        labels = logits.max(1).indices
        acc = torch.FloatTensor([(labels == y).sum()]).cuda() / labels.size(0)
        return {'val_loss': loss, 'acc': acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()
        print(avg_loss, avg_acc)
        self.log('avg_acc', avg_acc, prog_bar=True)
        tensorboard_logs = {'val_loss': avg_loss, 'avg_acc': avg_acc}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

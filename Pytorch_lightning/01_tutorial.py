from typing import Any
import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self) -> Any:
        return optim.Adam(self.parameters(), lr=2e-2)


train_loader = DataLoader(dataset=datasets.MNIST(
    os.getcwd(), download=True, transform=transforms.ToTensor()))

trainer = pl.Trainer()
model = LitModel()
trainer.fit(model, train_loader)

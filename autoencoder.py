import torch
import torch.nn as nn
import torchvision.models as models


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, 512)

        self.decoder = torch.nn.Sequential(
            nn.Linear(256*8, 128*14*14),
            nn.Unflatten(1, (128, 14, 14)),

            nn.Upsample(scale_factor=2.0, mode='nearest'),

            nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2.0, mode='nearest'),

            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2.0, mode='nearest'),

            nn.Conv2d(32, 16, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Upsample(scale_factor=2.0, mode='nearest'),

            nn.ConvTranspose2d(16, 8, 3, stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.ConvTranspose2d(8, 3, 3, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x) * 255
        return x

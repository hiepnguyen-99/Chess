import torch 
import torch.nn as nn
import copy

class DQN(nn.Module):
    def __init__(self, action_size): # int action_size
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(13, 32, kernel_size=3, padding=1), # (13, 8, 8) -> (32, 8, 8)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*8*8, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)
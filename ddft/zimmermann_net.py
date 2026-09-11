import torch
import torch.nn as nn
import torch.nn.functional as F

class ZimmermannNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv layers
        self.conv1 = nn.Conv1d(4, 16, kernel_size=20, bias=True)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=20, bias=True)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=20, bias=True)
        
        # Dense layers
        self.fc1 = nn.Linear(4544, 100, bias=True)
        self.fc2 = nn.Linear(100, 3, bias=True)

    def _pad_same_conv(self, x):
        # Keras 'same' with kernel 20 pads (left=9, right=10)
        return F.pad(x, (9, 10), mode='constant', value=0.0)

    def _pool_same(self, x):
        # If odd length, Keras pads 1 on the right before stride-2 avg pooling
        if x.shape[-1] % 2 != 0:
            x = F.pad(x, (0, 1), mode='constant', value=0.0)
        return F.avg_pool1d(x, kernel_size=2, stride=2)

    def forward(self, x):
        # Block 1: Conv -> Softplus -> AvgPool
        x = self.conv1(self._pad_same_conv(x))
        x = F.softplus(x)
        x = self._pool_same(x)
        
        # Block 2: Conv -> Softplus -> AvgPool
        x = self.conv2(self._pad_same_conv(x))
        x = F.softplus(x)
        x = self._pool_same(x)
        
        # Block 3: Conv -> Softplus -> AvgPool
        x = self.conv3(self._pad_same_conv(x))
        x = F.softplus(x)
        x = self._pool_same(x)

        # Permute from [Batch, Channels, Length] to [Batch, Length, Channels] 
        # so the memory order matches Keras exactly before flattening!
        x = x.permute(0, 2, 1)
        
        # Dense Head: Flatten -> Linear -> Softplus -> Linear
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.softplus(x)
        x = self.fc2(x)
        return x

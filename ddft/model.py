import torch
import torch.nn as nn

class SuperadiabaticNet(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        
        # Input has 2 channels: [Density, Velocity]
        self.net = nn.Sequential(
            # Using padding_mode='circular' mathematically guarantees periodic boundary conditions!
            nn.Conv1d(2, channels, kernel_size=5, padding=2, padding_mode='circular'),
            nn.Mish(), # Mish or Swish are infinitely smooth, perfect for PDEs
            nn.Conv1d(channels, channels, kernel_size=5, padding=2, padding_mode='circular'),
            nn.Mish(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2, padding_mode='circular'),
            nn.Mish(),
            # Output is 1 channel: The Superadiabatic Force (F_sup)
            nn.Conv1d(channels, 1, kernel_size=5, padding=2, padding_mode='circular')
        )
        
        # Initialize output weights to zero! 
        # This ensures that at Epoch 0, F_sup is exactly 0.0, 
        # so the simulation starts as pure, uncorrupted equilibrium DDFT.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, rho, v):
        # rho and v come in as shape (Batch, X)
        # We stack them into (Batch, 2, X)
        x = torch.stack([rho, v], dim=1)
        f_sup = self.net(x)
        return f_sup.squeeze(1) # Return shape (Batch, X)
import pytorch_lightning as pl
import torch
from .fmt_functional import FMT_Functional

class FMTFreeEnergy(pl.LightningModule):

    def __init__(self, ljmethod = "MFA", dz = 1/32):
        
        super().__init__()
        self.sigma = 1
        self.dz = dz
        self.Lz = 10*self.sigma
        self.kernel_range = torch.arange(-(4/5)*self.Lz/2,(4/5)*self.Lz/2+self.dz,self.dz)
        self.kernel_size = len(self.kernel_range)
        self.model = FMT_Functional(sigma=self.sigma, dz=self.dz, Lz=self.Lz, ljmethod = ljmethod)

    def forward(self, rho, mu):
        """
        The forward function takes in an image and returns the reconstructed image
        """
        x = self.model(rho, mu)
        return x
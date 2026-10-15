import pytorch_lightning as pl
from neural_ddft.models.cnn import CNN, CNN_100

class NeuralFreeEnergy(pl.LightningModule):

    def __init__(self,
                    model_hparams):
        super().__init__()
        self.save_hyperparameters()
        self.model = CNN(**model_hparams)

    def forward(self, x):
        z = self.model(x)
        return z

class NeuralFreeEnergyFine(pl.LightningModule):

    def __init__(self,
                    model_hparams):
        super().__init__()
        self.save_hyperparameters()
        self.model = CNN_100(**model_hparams)

    def forward(self, x):
        z = self.model(x)
        return z
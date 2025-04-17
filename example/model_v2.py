import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from model import SlotAttentionAutoEncoder

class LitSlotModel(LightningModule):
    def __init__(self, resolution=(128, 128), num_slots=7, num_iterations=3, hid_dim=64,
                 learning_rate=4e-4, warmup_steps=10000, decay_rate=0.5, decay_steps=100000):
        super().__init__()
        self.save_hyperparameters()

        self.model = SlotAttentionAutoEncoder(
            resolution=resolution,
            num_slots=num_slots,
            num_iterations=num_iterations,
            hid_dim=hid_dim
        )
        self.criterion = nn.MSELoss()
        self.example_input_array = torch.randn(1, 3, *resolution)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images = batch['image']
        recon_combined, recons, masks, slots = self.model(images)
        loss = self.criterion(recon_combined, images)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min((step + 1) / self.hparams.warmup_steps,
                                       self.hparams.decay_rate ** (step / self.hparams.decay_steps))
        )

        return [optimizer], [scheduler]

import os
from pytorch_lightning.cli import LightningCLI
from model_v2 import LitSlotModel
from dataset import PartnetDataModule


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))  # where train.py is located

    LightningCLI(
        LitSlotModel,
        PartnetDataModule,
        save_config_callback=None,
        trainer_defaults={
            "default_root_dir": base_dir,
            "max_epochs": 3,
            "callbacks": []
        }
    )

if __name__ == '__main__':
    main()
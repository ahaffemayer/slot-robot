from pathlib import Path
from natsort import natsorted
from model_v2 import LitSlotModel as Model  # Adjust if your class is named differently
from dataset import PartnetDataModule
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt

def main():
    # Path to lightning_logs
    path = Path(__file__).parent / "checkpoints" / "lightning_logs"

    # Get latest version
    version_path = natsorted(path.glob("version_*"))[-1]
    checkpoint_path = natsorted((version_path / "checkpoints").glob("*.ckpt"))[-1]
    print(f"Loading model from {checkpoint_path}")

    # Load model from checkpoint
    model: Model = Model.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.freeze()

    # Load test dataset
    data_module = PartnetDataModule()
    data_module.setup("test")
    test_loader = DataLoader(
        data_module.test_dataloader().dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for i, batch in enumerate(test_loader):
        image = batch["image"].to(device)
        with torch.no_grad():
            recon_combined, recons, masks, _ = model(image)

        # Visualization
        num_slots = masks.shape[1]
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, num_slots + 2, figsize=(15, 2))

        image = image.squeeze(0)
        recon_combined = recon_combined.squeeze(0)
        recons = recons.squeeze(0)
        masks = masks.squeeze(0)

        image = image.permute(1, 2, 0).cpu().numpy()
        recon_combined = recon_combined.permute(1, 2, 0).cpu().detach().numpy()
        recons = recons.cpu().detach().numpy()
        masks = masks.cpu().detach().numpy()

        ax[0].imshow(image)
        ax[0].set_title('Image')
        ax[1].imshow(recon_combined)
        ax[1].set_title('Recon.')

        for j in range(num_slots):
            picture = recons[j] * masks[j] + (1 - masks[j])
            ax[j + 2].imshow(picture)
            ax[j + 2].set_title(f'Slot {j + 1}')
        
        for a in ax:
            a.grid(False)
            a.axis('off')

        plt.tight_layout()
        plt.show()

        if i == 4:
            break

if __name__ == "__main__":
    main()

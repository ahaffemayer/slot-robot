import matplotlib.pyplot as plt
import random
from dataset import PARTNET


# Create the dataset
dataset = PARTNET(split='train')

# Pick a random index
idx = random.randint(0, len(dataset) - 1)

# Get the sample
sample = dataset[idx]
image_tensor = sample['image']

# Convert to numpy for plotting
image_np = image_tensor.permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]

# Display
plt.imshow(image_np)
plt.title(f"Sample #{idx}")
plt.axis('off')
plt.show()

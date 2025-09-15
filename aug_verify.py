import os
import random
import matplotlib.pyplot as plt
from PIL import Image

# Configuration
BASE_DIRS = [
    "train",
    "valid",
    "test"
]
NUM_SAMPLES = 3  # Number of augmented images to show per class
AUG_KEY = "_aug"  # Keyword to filter augmented images

def preview_augmented_images(base_path):
    for folder in BASE_DIRS:
        folder_path = os.path.join(base_path, folder)
        print(f"\n Scanning {folder_path}...")

        for class_name in sorted(os.listdir(folder_path)):
            class_dir = os.path.join(folder_path, class_name)
            if not os.path.isdir(class_dir):
                continue

            aug_images = [img for img in os.listdir(class_dir) if AUG_KEY in img.lower()]
            if len(aug_images) == 0:
                print(f" No augmented images found in {class_name}")
                continue

            sample_imgs = random.sample(aug_images, min(NUM_SAMPLES, len(aug_images)))

            print(f"Previewing {len(sample_imgs)} augmented images from {class_name}...")
            fig, axes = plt.subplots(1, len(sample_imgs), figsize=(4 * len(sample_imgs), 4))
            if len(sample_imgs) == 1:
                axes = [axes]

            for ax, img_file in zip(axes, sample_imgs):
                img_path = os.path.join(class_dir, img_file)
                image = Image.open(img_path)
                ax.imshow(image)
                ax.set_title(img_file, fontsize=8)
                ax.axis('off')

            plt.suptitle(f"{folder.upper()} / {class_name}", fontsize=12)
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    # Replace this with your dataset root path
    dataset_root = "/Users/hannansandhu/Desktop/Bliss/Food Calorie Estimation"
    preview_augmented_images(dataset_root)

import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

# Correct folder names exactly as they appear in `train`, `valid`, `test`
weak_classes = [
    "Kung Pao Chicken",
    "String Bean Chicken Breast",
    "black pepper rice bowl",
    "Fried Rice",
    "water_spinach",
    "fried_chicken"
]

base_dirs = ["train", "valid", "test"]

augmentor = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode="nearest"
)

def augment_images(folder, target_count=25):
    images = [img for img in os.listdir(folder) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    existing_count = len(images)

    if existing_count >= target_count:
        return

    to_generate = target_count - existing_count
    print(f"{folder}: augmenting {to_generate} images...")

    for img_name in tqdm(images):
        if to_generate <= 0:
            break
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img, axis=0)

        gen = augmentor.flow(img, batch_size=1)
        for i in range(3):
            if to_generate <= 0:
                break
            new_img = next(gen)[0].astype(np.uint8)

            filename = f"aug_{i}_{img_name}"
            save_path = os.path.join(folder, filename)
            cv2.imwrite(save_path, cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))
            to_generate -= 1

# Loop through all base dirs and augment weak classes
# Final version with path normalization
for base in base_dirs:
    base_path = os.path.join(os.getcwd(), base)
    print(f"\nScanning '{base}' directory at path: {base_path}")

    actual_folders = [folder for folder in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, folder))]

    for weak_class in weak_classes:
        matched_folder = None
        for real_folder in actual_folders:
            if real_folder.strip().lower() == weak_class.strip().lower():
                matched_folder = real_folder
                break

        if matched_folder:
            class_dir = os.path.join(base_path, matched_folder)
            augment_images(class_dir, target_count=25 if base == "train" else 10)
        else:
            print(f"Folder not found for class: '{weak_class}' in {base}")

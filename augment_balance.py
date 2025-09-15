import os
import cv2
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# List of all your class names (as exactly written in folder names)
all_classes = [
    "Beijing Beef",
    "Chow Mein",
    "Fried Rice",
    "Hashbrown",
    "Kung Pao Chicken",
    "String Bean Chicken Breast",
    "black pepper rice bowl",
    "burger",
    "carrot_eggs",
    "chicken waffle",
    "chicken_nuggets",
    "chinese_cabbage",
    "chinese_sausage",
    "curry",
    "french fries",
    "fried_chicken",
    "fried_dumplings",
    "fried_eggs",
    "mango chicken pocket",
    "mung_bean_sprouts",
    "rice",
    "tostitos cheese dip sauce",
    "water_spinach"
]

base_dirs = ["train", "valid", "test"]

# Augmentation pipeline
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

def count_images(folder):
    return len([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

# Step 1: Get max class size in 'train'
train_path = os.path.join(os.getcwd(), "train")
max_images = 0
folder_map = {}

for cls in all_classes:
    for folder in os.listdir(train_path):
        if folder.strip().lower() == cls.strip().lower():
            folder_map[cls] = folder
            count = count_images(os.path.join(train_path, folder))
            if count > max_images:
                max_images = count
            break

print(f"Max images in any class (train): {max_images}")

# Step 2: Augmentation logic
def augment_to_target(folder, target):
    images = [img for img in os.listdir(folder) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    existing = len(images)
    to_generate = target - existing
    if to_generate <= 0:
        return

    print(f"➕ Augmenting {to_generate} images in {folder}...")
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
            aug_img = next(gen)[0].astype(np.uint8)
            save_name = f"aug_{i}_{img_name}"
            save_path = os.path.join(folder, save_name)
            cv2.imwrite(save_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
            to_generate -= 1

# Step 3: Loop through dirs
for base in base_dirs:
    print(f"\nProcessing '{base}' directory...")
    base_path = os.path.join(os.getcwd(), base)

    for class_name in all_classes:
        if class_name not in folder_map:
            for folder in os.listdir(base_path):
                if folder.strip().lower() == class_name.strip().lower():
                    folder_map[class_name] = folder
                    break

        if class_name in folder_map:
            full_path = os.path.join(base_path, folder_map[class_name])
            target = max_images if base == "train" else int(0.2 * max_images)
            augment_to_target(full_path, target)
        else:
            print(f"Class folder not found for: {class_name} in {base}")

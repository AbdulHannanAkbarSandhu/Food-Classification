import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

TRAIN_DIR = "./train"
ALLOWED_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')

resolutions = []
brightnesses = []
sharpnesses = []

print("Scanning training images...")

for class_folder in tqdm(os.listdir(TRAIN_DIR)):
    class_path = os.path.join(TRAIN_DIR, class_folder)
    if not os.path.isdir(class_path) or class_folder.startswith("."):
        continue

    for img_file in os.listdir(class_path):
        if not img_file.lower().endswith(ALLOWED_EXTS) or img_file.startswith("."):
            continue

        img_path = os.path.join(class_path, img_file)
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Skipped unreadable: {img_path}")
                continue

            h, w, _ = img.shape
            resolutions.append((w, h))

            gray = cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2GRAY)

            brightness = np.mean(gray)
            brightnesses.append(brightness)

            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpnesses.append(sharpness)

        except Exception as e:
            print(f"Error in {img_file}: {e}")

# Check if any images processed
if len(resolutions) == 0:
    print("No valid images found. Check TRAIN_DIR path and image formats.")
    exit()

# Summary
avg_resolution = np.mean(resolutions, axis=0)
avg_brightness = np.mean(brightnesses)
avg_sharpness = np.mean(sharpnesses)

print(f"\n Summary:")
print(f"• Average Resolution: {avg_resolution[0]:.1f} x {avg_resolution[1]:.1f}")
print(f"• Average Brightness: {avg_brightness:.2f}")
print(f"• Average Sharpness: {avg_sharpness:.2f}")

# Plotting
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.hist(brightnesses, bins=30, color='goldenrod')
plt.title("Brightness Distribution")
plt.xlabel("Mean Intensity")
plt.ylabel("Image Count")

plt.subplot(1, 3, 2)
plt.hist(sharpnesses, bins=30, color='darkcyan')
plt.title("Sharpness Distribution")
plt.xlabel("Laplacian Variance")
plt.ylabel("Image Count")

plt.subplot(1, 3, 3)
res_w, res_h = zip(*resolutions)
plt.hist2d(res_w, res_h, bins=20, cmap="Blues")
plt.title("Resolution Distribution")
plt.xlabel("Width")
plt.ylabel("Height")
plt.colorbar(label='Image Count')

plt.tight_layout()
plt.show()

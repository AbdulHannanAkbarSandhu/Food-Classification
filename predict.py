import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import matplotlib.pyplot as plt
import json

# --- CONFIG ---
MODEL_PATH = "fine_tuned_model.keras"  # Updated model path
IMAGE_PATH = os.environ.get("IMAGE_PATH", "c7.jpg")  # default fallback if not set
# Change this as needed
IMAGE_SIZE = (224, 224)             # EfficientNetB0 default

# --- Load the trained model ---
model = load_model(MODEL_PATH)

# --- Load class names ---
with open("class_names.json", "r") as f:
    class_indices = json.load(f)
class_names = list(class_indices.keys())

# --- Load and preprocess the image ---
img = image.load_img(IMAGE_PATH, target_size=IMAGE_SIZE)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)  # EfficientNet-specific preprocessing

# --- Make prediction ---
predictions = model.predict(img_array)
predicted_class = class_names[np.argmax(predictions)]
confidence = np.max(predictions)

# --- Display result ---
print(f" Predicted: {predicted_class} ({confidence * 100:.2f}%)")

plt.imshow(img)
plt.title(f"{predicted_class} ({confidence * 100:.2f}%)")
plt.axis('off')
plt.show()

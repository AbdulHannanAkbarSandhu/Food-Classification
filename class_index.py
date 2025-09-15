import os
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

# Paths
base_dir = "."
train_dir = os.path.join(base_dir, "train")

# Data generator with EfficientNet preprocessing
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_gen = datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=1, class_mode='categorical'
)

# Save the class name to index mapping
with open("class_names.json", "w") as f:
    json.dump(train_gen.class_indices, f)

print(" Saved class_names.json with", len(train_gen.class_indices), "classes.")

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.efficientnet import preprocess_input

# Set paths
base_dir = "."
train_dir = os.path.join(base_dir, "train")
valid_dir = os.path.join(base_dir, "valid")
test_dir = os.path.join(base_dir, "test")

# Hyperparameters
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 10
LR = 1e-5  # Lower learning rate for fine-tuning
LABEL_SMOOTHING = 0.1

# Data generators with EfficientNet preprocessing
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True
)
valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)
valid_gen = valid_datagen.flow_from_directory(
    valid_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)
test_gen = test_datagen.flow_from_directory(
    test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)

# Load best pre-trained model
model = load_model("best_model.h5")

# --- UNFREEZE PART OF THE BASE MODEL ---
# We'll identify the EfficientNet base (usually at model.layers[0])
base_model = model.layers[0]
print(f"Base model: {base_model.name}")

# Unfreeze last 20 layers of EfficientNet base
for layer in model.layers[-20:]:
    layer.trainable = True

# Recompile with a smaller learning rate
from tensorflow.keras.losses import CategoricalCrossentropy
model.compile(
    optimizer=Adam(learning_rate=LR),
    loss=CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
    metrics=["accuracy"]
)

# Callbacks
checkpoint = ModelCheckpoint("fine_tuned_model.keras", save_best_only=True, monitor="val_accuracy", mode="max")

earlystop = EarlyStopping(monitor="val_accuracy", patience=3)

# Fine-tune
model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=EPOCHS,
    callbacks=[checkpoint, earlystop]
)

# Evaluate final performance
print("Evaluating fine-tuned model on test data:")
model.evaluate(test_gen)

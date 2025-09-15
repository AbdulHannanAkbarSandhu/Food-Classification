import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

model = load_model("best_model.h5")
test_dir = "./test"
IMG_SIZE = (160, 160)
BATCH_SIZE = 32

test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)

y_true = test_gen.classes
class_labels = list(test_gen.class_indices.keys())
y_pred_probs = model.predict(test_gen)
y_pred = np.argmax(y_pred_probs, axis=1)

misclassified_indices = np.where(y_pred != y_true)[0]

# Show some of them
for i in misclassified_indices[:10]:  # show top 10
    img_path = test_gen.filepaths[i]
    img = plt.imread(img_path)
    plt.imshow(img)
    true_label = class_labels[y_true[i]]
    pred_label = class_labels[y_pred[i]]
    confidence = np.max(y_pred_probs[i])
    plt.title(f"True: {true_label} | Pred: {pred_label} ({confidence*100:.2f}%)")
    plt.axis('off')
    plt.show()

# Food Calorie Estimation with EfficientNet-B0

This repository provides an end-to-end pipeline for classifying food images using EfficientNet-B0. The project demonstrates how image classification can serve as the foundation for calorie estimation by integrating with future portion estimation and nutrition APIs.

---

## Project Overview

- **Objective**: Build a food classification model that can recognize a variety of dishes and act as the backbone for calorie estimation.  
- **Dataset**: Images were curated from [Roboflow](https://universe.roboflow.com/eldoradooo/food-calorie-estimation) :contentReference[oaicite:0]{index=0} and supplemented with real-world images to improve class balance.  
- **Model**: EfficientNet-B0 with transfer learning and fine-tuning.  
- **Process**:
  1. Organize raw images into `train/`, `valid/`, and `test/`.
  2. Apply augmentations to handle class imbalance.
  3. Train the base EfficientNet-B0 with frozen layers, then fine-tune with all layers unfrozen.
  4. Save checkpoints (`best_model.h5`, `fine_tuned_model.keras`).
  5. Evaluate model performance and generate Grad-CAM visualizations.

---

## Repository Structure

Food-Calorie-Estimation/
│
├── train_model.py # initial training script
├── finetune_model.py # fine-tuning script
├── predict.py # run single-image prediction
├── predict_matched.py # prediction with class matching
├── analyze_test_performance.py # evaluation script for test set
├── visualize_attention.py # Grad-CAM visualizations
├── augment_balance.py # dataset balancing through augmentation
├── augmentation_for_weak.py # additional augmentations for weak classes
├── aug_verify.py # verify augmented images
├── delete_augmented.py # cleanup augmented data
├── show_miss.py # view misclassified samples
├── organize_images.py # dataset organization helper
├── image_data.py # dataset loading and preprocessing
├── class_names.json # class name to index mapping
├── class_index.py # alternate class index utility
│
├── best_model.h5 # best saved model (H5 format)
├── fine_tuned_model.keras(not available here) # fine-tuned model (Keras format)
│
├── gradcam_outputs/ # Grad-CAM examples
├── README.dataset.txt # dataset preparation notes
├── README.roboflow.txt # Roboflow export details
│
├── requirements.txt # dependencies
├── constraints.txt # pinned versions
└── README.md # project documentation

---

## Requirements

### It is recommended to use Python 3.10. On Apple Silicon, TensorFlow-MacOS with Metal acceleration is supported.

Create a new environment:

```
conda create -n food-cal python=3.10 -y
conda activate food-cal
Install the dependencies:
pip install -r requirements.txt -c constraints.txt
requirements.txt
tensorflow-macos==2.20.*
tensorflow-metal==1.2.*
keras>=3.3
numpy==2.0.*
ml-dtypes>=0.3.0
h5py>=3.10
pillow
opencv-python>=4.9
scikit-learn
matplotlib
constraints.txt
numpy==2.0.*
ml-dtypes>=0.3.0
```
### On Linux/Windows, replace tensorflow-macos and tensorflow-metal with tensorflow==2.20.*.
### Usage
##### Download the data:
https://universe.roboflow.com/myworkspace-awwul/ingredient-detection-d6nwz
#### Augment using teh augmenting scripts and Remove AW Cola and Crispy Corn class as they are higly problematic!!!
### Training
 ```python train_model.py```
### Fine-tuning
```python finetune_model.py```
### Single Image Prediction
``` python predict.py --image path/to/image.jpg```
### Evaluate on Test Data
```python analyze_test_performance.py```
### Grad-CAM Visualization
```python visualize_attention.py```

## Results
The model achieved strong accuracy across of around 86 % on validation and test sets.
Grad-CAM visualizations confirmed the model was focusing on the correct regions of images (for example, highlighting fried chicken pieces or rice grains).
Augmentation improved class balance, with previously weak classes such as Beijing Beef and Mango Chicken Pocket showing higher precision after retraining.

## References
Roboflow Dataset: Food Calorie Estimation 
README.dataset
Tan, M. & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
Chollet, F. (2017). Keras.

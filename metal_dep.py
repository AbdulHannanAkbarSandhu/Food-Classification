import platform
import subprocess
import tensorflow as tf

print("Apple Metal GPU Compatibility Check")
print("="*40)

# System info
print(f"Python Version: {platform.python_version()}")
print(f"System: {platform.system()} {platform.machine()}")
print(f"TensorFlow Version: {tf.__version__}\n")

# Check if tensorflow-metal is installed
try:
    import tensorflow_macos
    import tensorflow_metal
    print("`tensorflow-metal` is installed.")
except ImportError:
    print("`tensorflow-metal` NOT installed.")

# Check TensorFlow GPU devices
print("\n TensorFlow GPU Devices:")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        print(f"Found GPU: {gpu}")
else:
    print("No GPU devices found by TensorFlow.")

# Check PyTorch Metal support (only basic check — PyTorch Metal is still experimental)
try:
    import torch
    if torch.backends.mps.is_available():
        print("\n PyTorch MPS (Metal Performance Shaders) is available.")
    else:
        print("\n PyTorch MPS not available.")
except Exception as e:
    print(f"\n PyTorch error: {e}")

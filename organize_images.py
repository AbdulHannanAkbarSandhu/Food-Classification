import os
import shutil
import pandas as pd

# Set this for the current split you want to organize
SPLIT = 'test'  # or 'valid' or 'test'

# Path to the CSV file for the split
csv_path = f"./{SPLIT}/_classes.csv"

# Path to the image folder
image_dir = f"./{SPLIT}"

# Load CSV
df = pd.read_csv(csv_path)

# Loop over rows
for _, row in df.iterrows():
    filename = row['filename']
    # Get the label by finding the column with value 1
    label = row.drop('filename').idxmax()

    # Create class folder if not exists
    class_dir = os.path.join(image_dir, label)
    os.makedirs(class_dir, exist_ok=True)

    # Move the file to the class folder
    src_path = os.path.join(image_dir, filename)
    dst_path = os.path.join(class_dir, filename)

    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
    else:
        print(f"Missing image: {src_path}")

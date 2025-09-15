import os

# Class list as per your original dataset
all_classes = [
    "Beijing Beef", "Chow Mein", "Fried Rice", "Hashbrown", "Kung Pao Chicken", "String Bean Chicken Breast",
    "black pepper rice bowl", "burger", "carrot_eggs", "chicken waffle", "chicken_nuggets", "chinese_cabbage",
    "chinese_sausage", "curry", "french fries", "fried_chicken", "fried_dumplings", "fried_eggs",
    "mango chicken pocket", "mung_bean_sprouts", "rice", "tostitos cheese dip sauce", "water_spinach"
]

base_dirs = ["train", "valid", "test"]

# Normalize names (spaces and underscores removed, lowercase)
def normalize(name):
    return name.strip().lower().replace(" ", "").replace("_", "")

deleted_count = 0

for base in base_dirs:
    print(f"\n Cleaning '{base}' directory...")
    base_path = os.path.join(os.getcwd(), base)

    # Get all actual folder names
    all_folders = os.listdir(base_path)

    for class_name in all_classes:
        norm_class = normalize(class_name)
        matched_folder = None

        # Find matching folder
        for folder in all_folders:
            if normalize(folder) == norm_class:
                matched_folder = folder
                break

        if matched_folder:
            full_path = os.path.join(base_path, matched_folder)
            for fname in os.listdir(full_path):
                if fname.lower().startswith("aug_") and fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    try:
                        os.remove(os.path.join(full_path, fname))
                        deleted_count += 1
                    except Exception as e:
                        print(f"Error deleting {fname}: {e}")
        else:
            print(f"Folder not found for class: {class_name}")

print(f"\n Done! Deleted {deleted_count} augmented images.")

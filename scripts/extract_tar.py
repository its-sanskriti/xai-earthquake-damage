import tarfile
import os

tar_path = r"E:\train_images_labels_targets.tar"
extract_to = r"E:\earthquake_dataset"  # this folder will be created

# Create folder if not exists
os.makedirs(extract_to, exist_ok=True)

print("⏳ Extracting... This may take a while depending on file size...")

with tarfile.open(tar_path, 'r') as tar_ref:
    tar_ref.extractall(extract_to)

print("✅ Extraction complete!")

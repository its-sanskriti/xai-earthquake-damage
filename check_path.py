import os

path = r"E:\train_images_labels_targets.tar"

if os.path.exists(path):
    print("✅ File found!")
else:
    print("❌ File not found, check the path again.")

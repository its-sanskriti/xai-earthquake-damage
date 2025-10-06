import numpy as np
import glob
import os

# folder where chunks were saved
folder = r"E:\earthquake_npz_chunks"

# list all npz files
files = sorted(glob.glob(os.path.join(folder, "*.npz")))
print(f"Found {len(files)} chunks")

# load the first one
data = np.load(files[0])
images = data["images"]
names = data["filenames"]

print("✅ Loaded:", os.path.basename(files[0]))
print("Images shape:", images.shape)
print("Filenames shape:", names.shape)
print("First 5 filenames:", names[:5])

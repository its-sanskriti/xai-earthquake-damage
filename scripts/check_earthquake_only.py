import numpy as np
import glob
import os

path = r"E:\earthquake_only_npz\*.npz"
files = sorted(glob.glob(path))
print(f"Found {len(files)} earthquake chunks")

data = np.load(files[0])
images = data["images"]
names = data["filenames"]

print("✅ Loaded:", os.path.basename(files[0]))
print("Images shape:", images.shape)
print("First 5 filenames:", names[:5])

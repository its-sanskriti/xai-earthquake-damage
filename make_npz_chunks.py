import os
from PIL import Image
import numpy as np
from tqdm import tqdm

# ==== paths ====
src_dir = r"E:\earthquake_dataset"      # folder created after extraction
out_dir = r"E:\earthquake_npz_chunks"   # output folder for chunks

# ==== parameters ====
image_size = 224        # resize target (224×224)
chunk_size = 1000       # how many images per .npz file

# ==== make sure output dir exists ====
os.makedirs(out_dir, exist_ok=True)

def find_images(root):
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    paths = []
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(valid_exts):
                paths.append(os.path.join(r, f))
    return sorted(paths)

paths = find_images(src_dir)
print(f"Found {len(paths)} images")

chunk_imgs, chunk_names = [], []
part = 0

for i, p in enumerate(tqdm(paths, desc="Processing")):
    try:
        img = Image.open(p).convert("RGB").resize((image_size, image_size))
        arr = np.asarray(img, dtype=np.uint8)
        chunk_imgs.append(arr)
        chunk_names.append(os.path.basename(p))
    except Exception as e:
        print(f"❌ {p}: {e}")
        continue

    if len(chunk_imgs) >= chunk_size:
        out_path = os.path.join(out_dir, f"images_part_{part:04d}.npz")
        np.savez_compressed(out_path,
                            images=np.stack(chunk_imgs),
                            filenames=np.array(chunk_names))
        print(f"✅ Saved {out_path}")
        chunk_imgs, chunk_names = [], []
        part += 1

# save leftover images
if chunk_imgs:
    out_path = os.path.join(out_dir, f"images_part_{part:04d}.npz")
    np.savez_compressed(out_path,
                        images=np.stack(chunk_imgs),
                        filenames=np.array(chunk_names))
    print(f"✅ Saved final chunk {out_path}")

print("\n🎉 All chunks done!")

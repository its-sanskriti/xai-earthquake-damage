import numpy as np
import os, glob
from tqdm import tqdm

in_dir = r"E:\earthquake_npz_chunks"
out_dir = r"E:\earthquake_only_npz"
os.makedirs(out_dir, exist_ok=True)

files = sorted(glob.glob(os.path.join(in_dir, "*.npz")))
print(f"Found {len(files)} source chunks")

earthquake_imgs, earthquake_names = [], []
part = 0
chunk_limit = 1000  # save every 1000 earthquake images

for f in tqdm(files, desc="Filtering earthquake images"):
    data = np.load(f)
    imgs = data["images"]
    names = data["filenames"]
    mask = np.array(["earthquake" in n.lower() for n in names])

    selected_imgs = imgs[mask]
    selected_names = names[mask]

    earthquake_imgs.extend(selected_imgs)
    earthquake_names.extend(selected_names)

    # save in chunks
    while len(earthquake_imgs) >= chunk_limit:
        out_path = os.path.join(out_dir, f"earthquake_part_{part:04d}.npz")
        np.savez_compressed(out_path,
                            images=np.stack(earthquake_imgs[:chunk_limit]),
                            filenames=np.array(earthquake_names[:chunk_limit]))
        print(f"✅ Saved {out_path}")
        earthquake_imgs = earthquake_imgs[chunk_limit:]
        earthquake_names = earthquake_names[chunk_limit:]
        part += 1

# save remaining images
if earthquake_imgs:
    out_path = os.path.join(out_dir, f"earthquake_part_{part:04d}.npz")
    np.savez_compressed(out_path,
                        images=np.stack(earthquake_imgs),
                        filenames=np.array(earthquake_names))
    print(f"✅ Saved final chunk {out_path}")

print("\n🎉 Done! Filtered earthquake-only images saved to:", out_dir)

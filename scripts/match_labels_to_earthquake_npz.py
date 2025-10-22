import os, glob, json
import numpy as np
from tqdm import tqdm

# === Input paths ===
npz_dir = r"E:\earthquake_only_npz"
labels_dir = r"E:\Earthquake\data\earthquake_dataset\train\labels"   # change if your labels folder has a different name
out_dir = r"E:\earthquake_only_npz_labeled"
os.makedirs(out_dir, exist_ok=True)

# === Step 1: Collect labels from POST-disaster JSONs ===
label_map = {}
json_files = glob.glob(os.path.join(labels_dir, "*post_disaster.json"))

print(f"Reading POST label JSONs: {len(json_files)} files")

for jf in tqdm(json_files, desc="Extracting labels"):
    try:
        base = os.path.splitext(os.path.basename(jf))[0].replace("_post_disaster", "")
        with open(jf, "r") as f:
            data = json.load(f)
        
        features = data.get("features", {}).get("lng_lat", [])
        subtypes = [
            feat["properties"].get("subtype", None)
            for feat in features
            if feat["properties"].get("feature_type") == "building"
        ]
        
        # take the majority label if available
        if subtypes:
            valid_subtypes = [s for s in subtypes if s and s != "un-classified"]
            if valid_subtypes:
                label_map[base] = max(set(valid_subtypes), key=valid_subtypes.count)
            else:
                label_map[base] = "un-classified"
        else:
            label_map[base] = None
    except Exception as e:
        print(f"⚠️ Error in {jf}: {e}")

print(f"\n✅ Total post-disaster files processed: {len(label_map)}")

# === Step 2: Attach labels to each NPZ file ===
npz_files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))

for f in tqdm(npz_files, desc="Matching labels"):
    data = np.load(f)
    images = data["images"]
    filenames = data["filenames"]

    labels = []
    for name in filenames:
        key = os.path.splitext(name)[0].replace("_post_disaster", "").replace("_pre_disaster", "")
        labels.append(label_map.get(key, None))

    out_path = os.path.join(out_dir, os.path.basename(f))
    np.savez_compressed(out_path, images=images, filenames=filenames, labels=np.array(labels, dtype=object))
    print(f"✅ Saved labeled npz: {out_path}")

print("\n🎉 Done — labeled earthquake-only data saved to:", out_dir)

import numpy as np

data = np.load(r"E:\earthquake_only_npz_labeled\earthquake_part_0000.npz", allow_pickle=True)
print(data.files)  # should show ['images', 'filenames', 'labels']

print("Images shape:", data["images"].shape)
print("Labels count:", len(data["labels"]))
print("Example label:", data["labels"][0])

# Correct way to get unique labels
print("Unique labels:", set(data["labels"]))


labels = data["labels"]

from collections import Counter
print(Counter(labels))


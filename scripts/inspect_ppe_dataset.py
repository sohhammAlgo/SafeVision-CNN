from pathlib import Path
import yaml
from collections import Counter


DATASET_DIR = Path("data/raw/ppe_detection")
DATA_YAML = DATASET_DIR / "data.yaml"


# -----------------------------
# Load data.yaml
# -----------------------------

with open(DATA_YAML, "r") as f:
    config = yaml.safe_load(f)


names = config["names"]

print("\n========== DATASET INFO ==========")

print("Classes:")
for index, name in enumerate(names):
    print(f"  {index}: {name}")


# -----------------------------
# Count labels
# -----------------------------

splits = {
    "train": DATASET_DIR / "train" / "labels",
    "valid": DATASET_DIR / "valid" / "labels",
    "test": DATASET_DIR / "test" / "labels",
}


for split_name, label_dir in splits.items():

    counter = Counter()
    total_objects = 0
    total_files = 0

    if not label_dir.exists():
        print(f"\n[WARNING] {label_dir} does not exist")
        continue

    for label_file in label_dir.glob("*.txt"):

        total_files += 1

        with open(label_file, "r") as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                if len(parts) != 5:
                    print(
                        f"[WARNING] Invalid annotation: "
                        f"{label_file}"
                    )
                    continue

                class_id = int(parts[0])

                counter[class_id] += 1
                total_objects += 1

    print(f"\n========== {split_name.upper()} ==========")

    print(f"Label files: {total_files}")
    print(f"Objects:     {total_objects}")

    for class_id, count in sorted(counter.items()):

        class_name = names[class_id]

        print(
            f"{class_id} ({class_name}): {count}"
        )
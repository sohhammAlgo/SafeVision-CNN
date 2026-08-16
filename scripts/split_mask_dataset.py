from pathlib import Path
from collections import defaultdict, Counter
import random
import shutil


# ============================================================
# Configuration
# ============================================================

SOURCE_DIR = Path("data/raw/mask_classification")
OUTPUT_DIR = Path("data/processed/mask_classification")

CLASSES = [
    "mask",
    "no_mask",
    "improper_mask"
]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42


# ============================================================
# Validate ratios
# ============================================================

assert abs(
    TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0
) < 1e-6


# ============================================================
# Random seed
# ============================================================

random.seed(SEED)


# ============================================================
# Create output directories
# ============================================================

for split in ["train", "validation", "test"]:

    for class_name in CLASSES:

        directory = (
            OUTPUT_DIR /
            split /
            class_name
        )

        directory.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# Collect images and group by original image
# ============================================================

groups = defaultdict(list)

for class_name in CLASSES:

    class_dir = SOURCE_DIR / class_name

    for image_path in class_dir.iterdir():

        if image_path.suffix.lower() not in [
            ".jpg",
            ".jpeg",
            ".png"
        ]:
            continue

        # Our preparation script creates names like:
        #
        # original_image_0.jpg
        # original_image_1.jpg
        #
        # Remove the final "_index" to recover
        # the original image identifier.

        parts = image_path.stem.rsplit("_", 1)

        if len(parts) == 2 and parts[1].isdigit():
            group_id = parts[0]
        else:
            group_id = image_path.stem

        groups[group_id].append(
            (image_path, class_name)
        )


print("================================")
print("MASK DATASET SPLIT")
print("================================")

print(
    "Original image groups:",
    len(groups)
)

print(
    "Total crops:",
    sum(len(items) for items in groups.values())
)


# ============================================================
# Shuffle groups
# ============================================================

group_ids = list(groups.keys())

random.shuffle(group_ids)


# ============================================================
# Calculate split sizes
# ============================================================

total_groups = len(group_ids)

train_end = int(
    total_groups * TRAIN_RATIO
)

val_end = train_end + int(
    total_groups * VAL_RATIO
)

train_groups = group_ids[:train_end]

val_groups = group_ids[
    train_end:val_end
]

test_groups = group_ids[
    val_end:
]


# ============================================================
# Split function
# ============================================================

def copy_groups(group_ids, split_name):

    counts = Counter()

    for group_id in group_ids:

        for image_path, class_name in groups[group_id]:

            destination = (
                OUTPUT_DIR /
                split_name /
                class_name /
                image_path.name
            )

            shutil.copy2(
                image_path,
                destination
            )

            counts[class_name] += 1

    return counts


# ============================================================
# Copy datasets
# ============================================================

train_counts = copy_groups(
    train_groups,
    "train"
)

val_counts = copy_groups(
    val_groups,
    "validation"
)

test_counts = copy_groups(
    test_groups,
    "test"
)


# ============================================================
# Print results
# ============================================================

print("\nTRAIN")
print("Groups:", len(train_groups))

for class_name in CLASSES:
    print(
        f"{class_name}: "
        f"{train_counts[class_name]}"
    )


print("\nVALIDATION")
print("Groups:", len(val_groups))

for class_name in CLASSES:
    print(
        f"{class_name}: "
        f"{val_counts[class_name]}"
    )


print("\nTEST")
print("Groups:", len(test_groups))

for class_name in CLASSES:
    print(
        f"{class_name}: "
        f"{test_counts[class_name]}"
    )


print("\nTOTAL")

for class_name in CLASSES:

    total = (
        train_counts[class_name]
        + val_counts[class_name]
        + test_counts[class_name]
    )

    print(
        f"{class_name}: {total}"
    )

print(
    "\nTotal images:",
    sum(
        train_counts.values()
    )
    + sum(
        val_counts.values()
    )
    + sum(
        test_counts.values()
    )
)

print("\nDataset split complete.")
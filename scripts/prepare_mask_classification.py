from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image

# Paths
SOURCE_DIR = Path("data/raw/mask_source")

IMAGE_DIR = SOURCE_DIR / "images"
ANNOTATION_DIR = SOURCE_DIR / "annotations"

OUTPUT_DIR = Path("data/raw/mask_classification")

# Class mapping
CLASS_MAP = {
    "with_mask": "mask",
    "without_mask": "no_mask",
    "mask_weared_incorrect": "improper_mask",
}

# Create output directories
for class_name in CLASS_MAP.values():
    (OUTPUT_DIR / class_name).mkdir(
        parents=True,
        exist_ok=True
    )

# Counters
counts = {
    "mask": 0,
    "no_mask": 0,
    "improper_mask": 0,
}

skipped = 0

# Process annotations
xml_files = list(ANNOTATION_DIR.glob("*.xml"))

print(f"Found {len(xml_files)} annotation files")


for xml_file in xml_files:

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        filename = root.findtext("filename")

        if filename is None:
            print(f"[WARNING] No filename in {xml_file}")
            skipped += 1
            continue

        image_path = IMAGE_DIR / filename

        if not image_path.exists():

            # Sometimes extension/path can differ
            possible_files = list(
                IMAGE_DIR.glob(Path(filename).stem + ".*")
            )

            if not possible_files:
                print(
                    f"[WARNING] Image not found: {filename}"
                )
                skipped += 1
                continue

            image_path = possible_files[0]

        image = Image.open(image_path).convert("RGB")

        # --------------------------------------------------
        # Process every annotated object
        # --------------------------------------------------

        objects = root.findall("object")

        for index, obj in enumerate(objects):

            class_name = obj.findtext("name")

            if class_name not in CLASS_MAP:
                print(
                    f"[WARNING] Unknown class "
                    f"{class_name} in {xml_file.name}"
                )
                continue

            target_class = CLASS_MAP[class_name]

            bbox = obj.find("bndbox")

            if bbox is None:
                continue

            xmin = int(float(bbox.findtext("xmin")))
            ymin = int(float(bbox.findtext("ymin")))
            xmax = int(float(bbox.findtext("xmax")))
            ymax = int(float(bbox.findtext("ymax")))

            # Clamp coordinates
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(image.width, xmax)
            ymax = min(image.height, ymax)

            if xmax <= xmin or ymax <= ymin:
                print(
                    f"[WARNING] Invalid bbox "
                    f"in {xml_file.name}"
                )
                continue

            # Crop
            crop = image.crop(
                (xmin, ymin, xmax, ymax)
            )

            output_name = (
                f"{image_path.stem}_"
                f"{index}.jpg"
            )

            output_path = (
                OUTPUT_DIR /
                target_class /
                output_name
            )

            crop.save(
                output_path,
                quality=95
            )

            counts[target_class] += 1

    except Exception as e:

        print(
            f"[ERROR] {xml_file.name}: {e}"
        )

        skipped += 1

print("\n==============================")
print("Dataset preparation complete")
print("==============================")

print(f"Mask:             {counts['mask']}")
print(f"No Mask:          {counts['no_mask']}")
print(f"Improper Mask:    {counts['improper_mask']}")

print(
    f"Total crops:      "
    f"{sum(counts.values())}"
)

print(f"Skipped:          {skipped}")
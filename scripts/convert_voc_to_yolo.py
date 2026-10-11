from pathlib import Path
import xml.etree.ElementTree as ET
import shutil

# Project folders
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "raw"
DATASET_DIR = PROJECT_ROOT / "dataset"

# We only have one class (UAV)
CLASS_ID = 0

def convert_split(split):
    xml_dir = RAW_DIR / split / "xml"
    img_dir = RAW_DIR / split / "img"

    output_img_dir = DATASET_DIR / "images" / split
    output_label_dir = DATASET_DIR / "labels" / split

    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    xml_files = list(xml_dir.glob("*.xml"))

    print(f"\nConverting {split}: {len(xml_files)} annotations")

    for xml_file in xml_files:

        # Read XML
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Image information
        filename = root.findtext("filename")

        width = int(root.find("size/width").text)
        height = int(root.find("size/height").text)

        yolo_lines = []

        # Find every object in the image
        for obj in root.findall("object"):  # root.findall("object") -> script will work even if an image contains multiple UAVs

            class_name = obj.findtext("name")

            # We only want UAV objects
            if class_name != "UAV":
                continue

            bbox = obj.find("bndbox")

            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            # Convert Pascal VOC -> YOLO
            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height

            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height

            yolo_line = (
                f"{CLASS_ID} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{box_width:.6f} "
                f"{box_height:.6f}"
            )

            yolo_lines.append(yolo_line)

        # Creat YOLO label file
        label_path = output_label_dir / f"{xml_file.stem}.txt"

        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))

        # Copy corresponding image
        source_image = img_dir / filename

        if source_image.exists():
             shutil.copy2(source_image, output_img_dir / filename)
        else:
            print(f"WARNING: image not found for {xml_file.name}")

for split_name in ["train", "val", "test"]:
    convert_split(split_name)

print("\nConversion finished!")
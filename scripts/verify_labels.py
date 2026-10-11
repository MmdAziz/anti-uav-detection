from pathlib import Path
import cv2
import random


PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_ROOT / "dataset" / "images" / "train"
LABEL_DIR = PROJECT_ROOT / "dataset" / "labels" / "train"

OUTPUT_DIR = PROJECT_ROOT / "verification"
OUTPUT_DIR.mkdir(exist_ok=True)


image_files = list(IMAGE_DIR.glob("*.jpg"))

selected_images = random.sample(image_files, 5)


# Verify each selected image
for image_path in selected_images:

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    # Load image
    image = cv2.imread(str(image_path))

    height, width = image.shape[:2]

    # Read YOLO label
    with open(label_path, "r") as file:
        lines = file.readlines()

    for line in lines:

        class_id, x_center, y_center, box_width, box_height = map(
            float, line.split()
        )

        # Convert normalized YOLO coordinates back to pixels
        x_center *= width
        y_center *= height
        box_width *= width
        box_height *= height

        xmin = int(x_center - box_width / 2)
        ymin = int(y_center - box_height / 2)

        xmax = int(x_center + box_width / 2)
        ymax = int(y_center + box_height / 2)

        # Draw bounding box
        cv2.rectangle(
            image,
            (xmin, ymin),
            (xmax, ymax),
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            "UAV",
            (xmin, max(ymin - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    output_path = OUTPUT_DIR / f"{image_path.stem}_verified.jpg"

    cv2.imwrite(str(output_path), image)

    print(f"Saved: {output_path}")


print("\nVerification finished!")
from ultralytics import YOLO

# Load the pretrained YOLOv8 nano model
model = YOLO("yolov8n.pt")

print("Model loaded successfully!")

# Run object detection on a sample image
results = model.predict(
    source = "https://ultralytics.com/images/bus.jpg",
    conf = 0.25,
    save = True
)

print("Detection completed!")
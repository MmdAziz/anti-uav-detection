from ultralytics import YOLO

# Loading Pretrained YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Small sanity-test training run
model.train(
    data="data.yaml",
    epochs=1,
    imgsz=320,  # As we are testing the pipeline not the performance, the img size is deliberately smaller than eventual 640
    batch=2,    # To keep memory/CPU load reasonable
    device="cpu",
    workers=0,  # To keep the first test simpler and avoid multiprocessing related headaches
    fraction=0.05,  # 5% of the training images
    val=False   # Skip validation for now
)
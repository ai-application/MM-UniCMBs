from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.yaml')  # build a new model from YAML

# Train the model
model.train(data="cmb.yaml", epochs=1000, imgsz=512, batch=24, project = "CMB_detect-yolon-MRLA", device=2)
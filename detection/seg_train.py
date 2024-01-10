from ultralytics import YOLO

# Load a model
model = YOLO('yolov8x-seg.yaml')  # build a new model from YAML
model = YOLO('yolov8x-seg.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data='backQJ.yaml', epochs=300, imgsz=1024)
from ultralytics import YOLO
import time

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.engine")

# Run inference on 'bus.jpg' with arguments
start = time.time()
model.predict("datasets/coco/images/val2017/", half=True, save_txt=True, imgsz=640, conf=0.5)
print(time.time() - start)

from ultralytics import YOLO
import cv2
from PIL import Image

model = YOLO('C:/Users/PC/Desktop/DATASET_THERMAL/runs/detect/yolov8n_50e/weights/best.pt')
metrics = model.val()  # evaluate model performance on the validation set

results = model('1.jpg', show = True, conf=0.25, save=True)  

path = model.export(format="onnx")  # export the model to ONNX format
cv2.waitKey(0)
cv2.destroyAllWindows()  # close window
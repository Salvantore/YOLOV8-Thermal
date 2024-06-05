from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

  # train the model

results = model.train(data="C:/Users/PC/Desktop/Dataset_Thermal/dataset-person/mydata.yaml", epochs=50, imgsz=416, batch=16)  

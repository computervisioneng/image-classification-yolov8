from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

model.train(data='/home/phillip/Desktop/todays_tutorial/44_yolov8_image_classification_custom_data/code/data/weather_dataset',
            epochs=20, imgsz=64)

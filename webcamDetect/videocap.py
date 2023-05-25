from ultralytics import YOLO

model = YOLO('last.pt')
model.predict(source="0", show=True)

# yolo detect predict model=last.pt source=0 save=True
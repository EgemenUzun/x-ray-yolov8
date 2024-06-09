from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('best.pt')
    results = model.train(data='data.yaml', epochs=50, imgsz=640, device=0, batch=-1)

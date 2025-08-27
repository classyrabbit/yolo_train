from ultralytics import YOLO
import torch

if __name__ == "__main__":
    print("GPUs detected:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:", torch.cuda.get_device_name(i))

    # load YOLOv8 model
    model = YOLO("yolov5s.pt")  # nano model for speed

    # train
    model.train(
        data="data.yaml",
        epochs=10,
        imgsz=320,
        batch=16,
        workers=0,   # important for Windows!
        device=0     # must be 0 → your RTX 3060
    )

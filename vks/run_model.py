from ultralytics import YOLO

model = YOLO("runs/detect/train5/weights/best.pt")
results = model("46.jpg")  # predict a new image
results[0].show()  # visualize predictions
results[0].save()  # save image with boxes

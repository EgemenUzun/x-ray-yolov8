from flask import Flask, render_template, request, Response
import cv2
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
import numpy as np

app = Flask(__name__)
camera = cv2.VideoCapture(0)

model = YOLO("./best.pt", verbose=False)

model.conf = 0.35
model.iou = 0.45

greyscale = False
sharpening = False
edge = False
bilateral = False
negative = False

brightness = 0.0
contrast = 1.0

filters_list = []

def set_filter_status(filters_list):
    global greyscale, sharpening, bilateral, negative
    greyscale = "greyscale" in filters_list
    sharpening = "super" in filters_list
    bilateral = "bilateral" in filters_list
    negative = "negative" in filters_list

def generate_frames():
    global greyscale, sharpening, bilateral, negative, brightness, contrast

    while camera.isOpened():
        # read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)
            result = frame.copy()

            if greyscale:
                result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            else:
                if bilateral:
                    result = cv2.bilateralFilter(result, 9, 75, 75)
                if negative:
                    result = cv2.bitwise_not(result)
                if sharpening:
                    result = cv2.filter2D(
                        result, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))

            # Apply brightness and contrast adjustments
            result = cv2.convertScaleAbs(result, alpha=contrast, beta=brightness)

            # Convert result to 3-channel BGR if greyscale is active (required for YOLO)
            if greyscale:
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

            # Detect using the YOLOv8 model
            yolo_result = model(result)

            boxes = yolo_result[0].boxes
            names = yolo_result[0].names
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # get Coordinates 
                label = names[int(box.cls)]  # get class name
                score = box.conf[0]  # get score

                # draw rectangle and label text
                cv2.rectangle(result, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(result, f'{label} {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            ret, buffer = cv2.imencode('.jpg', result)
            result = buffer.tobytes()

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + result + b"\r\n")
        
def generate_frames_original():
    while True:
        success, frame = camera.read()
        frame = cv2.flip(frame, 1)
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route("/", methods=["GET", "POST"])
def index():
    global filters_list, brightness, contrast

    if request.method == "POST":
        filters_list = request.form.getlist("filters")
        set_filter_status(filters_list)

        brightness = float(request.form.get("brightness")) if request.form.get(
            "brightness") is not None else 0.0
        contrast = float(request.form.get("contrast")) if request.form.get(
            "contrast") is not None else 1.0

    return render_template("main.html", edge=edge)

@app.route("/edited_video")
def edited_video():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/video")
def video():
    return Response(generate_frames_original(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True, threaded=True)

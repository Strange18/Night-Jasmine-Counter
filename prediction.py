# for real time prediction
import cv2
from ultralytics import YOLO

def load_model(model_path="./trained_files/weights/best.pt"):

    model = YOLO(model_path)
    return model

# stream url for the webcam of camera using ipcam
def load_camera(camera_source=0, stream_url=None):
    if stream_url is not None:
        cap = cv2.VideoCapture(stream_url)
    else:
        cap = cv2.VideoCapture(camera_source)
    return cap

def perform_real_time_perdiction(cap=None):
    # height and width of the frame of the window!
    desired_width = 640
    desired_height = 480

    model = load_model()
    cap = load_camera()
    while True:
        ret, frame = (
            cap.read()
        )  # returns bool to denote if the data is returned or not and matrix of frame
        if not ret:
            print("Failed to grab frame.")
            break

        frame = cv2.resize(frame, (desired_width, desired_height))

        results = model.predict(frame)

        # variable to store the count the instances of Night Jasmine detected
        class_counts = {}

        for result in results:
            boxes = result.boxes

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])

                class_name = model.names[cls]
                if class_name in class_counts:
                    class_counts[class_name] += 1
                else:
                    class_counts[class_name] = 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"{class_name} {conf:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        y_offset = 30
        for class_name, count in class_counts.items():
            cv2.putText(
                frame,
                f"{class_name}: {count}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                3,
            )
            y_offset += 20

        cv2.imshow("Might Jasmine Detection!", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        cap.release()
        cv2.destroyAllWindows()

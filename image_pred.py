import cv2
from ultralytics import YOLO


# loading the weight from the trained files
def load_model(path="./trained_files/weights/best.pt"):
    model = YOLO(path)
    return model


def detction_from_image(image_path):
    # loading test image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image is None:
        print("Error: Could not read image.")
        exit()

    # prediction of the model
    model = load_model()
    results = model.predict(image)
    # print(f'The Results are {results}')

    detections = []
    class_counts = {}

    # overlaying the bounding boxes in the image where the model predicts for the classes
    for result in results:
        boxes = result.boxes
        # print(f'The boxes data is {boxes}')

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])

            detections.append(
                {
                    "class": model.names[cls],
                    "confidence": float(conf),
                    "coordinates": (x1, y1, x2, y2),
                }
            )
            class_name = model.names[cls]
            if class_name in class_counts:
                class_counts[class_name] += 1
            else:
                class_counts[class_name] = 1

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 5)

            label = f"{model.names[cls]} {conf:.2f}"
            cv2.putText(
                image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                (0, 255, 0),
                5,
            )

    # print("Detections:", detections)
    width = 600
    height = int(image.shape[0] * (width / image.shape[1]))
    resized_image = cv2.resize(image, (width, height))
    
    y_offset = 30
    for class_name, count in class_counts.items():
        cv2.putText(
            resized_image,
            f"{count} {class_name} Detected",
            (30, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
        y_offset += 20

    return resized_image


def view_detction_in_image(image_path):
    resized_image = detction_from_image(image_path)
    cv2.imshow("YOLOv11n Detection", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# view_detction_in_image("./test.jpg")

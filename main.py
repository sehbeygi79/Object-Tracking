from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO


def draw_bboxes(frame, boxes_object, object_ids_to_show={0}):
    bboxes = boxes_object.xyxy.cpu().numpy()  # shape (N,4)
    confs = boxes_object.conf.cpu().numpy()  # shape (N,)
    classes = boxes_object.cls.cpu().numpy().astype(int)
    track_ids = result.boxes.id.int().cpu().tolist()

    for bbox, conf, cls, track_id in zip(bboxes, confs, classes, track_ids):
        if cls not in object_ids_to_show:
            continue

        x1, y1, x2, y2 = map(int, bbox)
        label = f"ID {track_id} - {float(conf):.2f}"

        # draw
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # background for text
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(
            frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
        )

    return frame


# ---- Config ----
VIDEO_SOURCE = "./data/input.mp4"  # 0 = webcam, or "path/to/video.mp4"
MODEL_NAME = "yolo11n.pt"
DEVICE = "cpu"
IMG_SIZE = 320
CONFIDENCE = 0.25
TRACKER = "bytetrack.yaml"  # tracker choice (ByteTrack recommended)
OBJECTS_TO_SHOW = {"person"}
# ----------------


model = YOLO(MODEL_NAME)
id2label = model.model.names
object_ids_to_show = {k for k, v in id2label.items() if v in OBJECTS_TO_SHOW}


# ---- Video writer setup ----
cap = cv2.VideoCapture(VIDEO_SOURCE)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output = cv2.VideoWriter(
    VIDEO_SOURCE.replace(".mp4", "_processed.mp4"), fourcc, fps, (width, height)
)

track_history = defaultdict(lambda: [])
while cap.isOpened():
    success, frame = cap.read()

    if not success:  # The end of the video is reached
        break

    result = model.track(
        frame,
        persist=True,
        imgsz=IMG_SIZE,
        conf=CONFIDENCE,
        device=DEVICE,
        tracker=TRACKER,
    )[0]

    if result.boxes and result.boxes.is_track:
        boxes = result.boxes.xywh.cpu()
        track_ids = result.boxes.id.int().cpu().tolist()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        print(f"track_ids: {track_ids}")

        # frame = result.plot()
        frame = draw_bboxes(frame, result.boxes, object_ids_to_show)
        # draw_tracks()

    # Display the annotated frame
    cv2.imshow("People Tracking", frame)
    output.write(frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Release the video capture object and close the display window
cap.release()
output.release()
cv2.destroyAllWindows()

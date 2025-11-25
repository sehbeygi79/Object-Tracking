from collections import defaultdict
import time
import json

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from utils import display_fps, draw_policy_lines, update_crossing_status


def draw_bboxes(frame, detection_results, object_ids_to_show={0}):
    for bbox, conf, cls, track_id in zip(
        detection_results["bboxes"],
        detection_results["confs"],
        detection_results["classes"],
        detection_results["track_ids"],
    ):
        if cls not in object_ids_to_show:
            continue

        x1, y1, x2, y2 = map(int, bbox)
        label = f"ID {track_id} - {float(conf):.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(
            frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
        )

    return frame


def draw_bboxes_ult(frame, detection_results, object_ids_to_show={0}):
    annotator = Annotator(frame, line_width=2)

    for bbox, conf, cls, track_id in zip(
        detection_results["bboxes"],
        detection_results["confs"],
        detection_results["classes"],
        detection_results["track_ids"],
    ):
        if cls not in object_ids_to_show:
            continue

        x1, y1, x2, y2 = map(int, bbox)
        label = f"ID: {track_id} - {float(conf):.2f}"
        annotator.box_label([x1, y1, x2, y2], label, color=colors(int(track_id), True))

    return frame


# ---- Config ----
VIDEO_SOURCE = "./data/input.mp4"
POLICY_JSON_PATH = "./data/policy.json"
MODEL_NAME = "yolo11n.pt"
DEVICE = "cpu"
IMG_SIZE = 320
CONFIDENCE = 0.25
TRACKER = "bytetrack.yaml"
OBJECTS_TO_SHOW = {"person"}
LINE_ALERT_DURATION = 3 # Number of seconds to keep the alert when a line is crossed
SKIP_FRAMES = 5  # Set to 1 to disable skipping
SHOW_STREAM = True  # Set to True to display the processed video
SAVE_VIDEO = True  # Set to True to store the processed video
# ----------------


model = YOLO(MODEL_NAME)
id2label = model.model.names
object_ids_to_show = {k for k, v in id2label.items() if v in OBJECTS_TO_SHOW}


# ---- Video writer setup ----
cap = cv2.VideoCapture(VIDEO_SOURCE)
if SAVE_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = cv2.VideoWriter(
        VIDEO_SOURCE.replace(".mp4", "_processed.mp4"),
        fourcc,
        video_fps,
        (width, height),
    )


# --- Policy File Reading ---
try:
    with open(POLICY_JSON_PATH, "r") as f:
        policy_data = json.load(f)
    policy_lines = policy_data.get("lines", [])
    policy_fences = policy_data.get("fences", [])
    print(f"Loaded {len(policy_lines)} lines from policy file.")
except (FileNotFoundError, json.JSONDecodeError):
    print(f"Problem loading the policy file!")
    exit()


print(f"Starting video processing with detection every {SKIP_FRAMES} frames...")


tracker = None
tracker_initialized = False
track_history = defaultdict(lambda: [])
line_crossing_storage = defaultdict(lambda: [])
lines_triggered = [False] * len(policy_lines)
line_activity_due_times = [0] * len(policy_lines)

frame_count = 0
start_time = time.time()
while cap.isOpened():
    success, frame = cap.read()
    detection_results = {}

    if not success:  # The end of the video is reached
        break

    if frame_count % SKIP_FRAMES == 0:
        result = model.track(
            frame,
            persist=True,
            imgsz=IMG_SIZE,
            conf=CONFIDENCE,
            device=DEVICE,
            tracker=TRACKER,
            verbose=False,
        )[0]

        # Access the internal tracker instance
        tracker = model.predictor.trackers[0]
        tracker_initialized = True

        # Save results
        if result.boxes and result.boxes.is_track:
            detection_results["bboxes"] = result.boxes.xyxy.cpu().numpy()
            detection_results["track_ids"] = result.boxes.id.int().cpu().numpy()
            detection_results["classes"] = result.boxes.cls.int().cpu().numpy()
            detection_results["confs"] = (last_confs := result.boxes.conf.cpu().numpy())
        else:
            last_confs = np.array([])

        # --- Line Crossing Check ---
        if len(policy_lines):
            line_crossing_storage, lines_triggered = update_crossing_status(
                detection_results, policy_lines, track_history, line_crossing_storage
            )
    # For skipped detection frames, use the tracker's prediction
    else:
        if tracker_initialized:
            # Manually predict next locations using the tracker's motion model (Kalman filter)
            tracks = [t for t in tracker.tracked_stracks if t.is_activated]
            tracker.multi_predict(tracks)
            tracker.frame_id += 1

            estimated_boxes = []
            for t in tracks:
                # Format: [x1, y1, x2, y2, track_id, score, class]
                estimated_boxes.append(np.hstack([t.xyxy, t.track_id, t.score, t.cls]))

            # Save estimated results
            if len(estimated_boxes):
                detection_results["bboxes"] = np.array(
                    [box[:4] for box in estimated_boxes]
                )
                detection_results["track_ids"] = np.array(
                    [int(box[4]) for box in estimated_boxes]
                )
                detection_results["classes"] = np.array(
                    [int(box[6]) for box in estimated_boxes]
                )
                detection_results["confs"] = last_confs

        else:
            # Skip frames until the first detection has occurred
            frame_count += 1
            continue

    # --- Draw Bounding Boxes ---
    if len(detection_results):
        frame = draw_bboxes_ult(frame, detection_results, object_ids_to_show)

    line_activity_due_times = [
        time.time() + LINE_ALERT_DURATION if a else d
        for a, d in zip(lines_triggered, line_activity_due_times)
    ]
    # Draw policy lines (red if being crossed)
    if len(policy_lines):
        frame = draw_policy_lines(
            frame,
            policy_lines,
            line_crossing_storage,
            line_activity_due_times,
        )

    # --- Draw FPS Display ---
    frame = display_fps(frame, start_time, frame_count + 1)

    # Display the annotated frame
    if SHOW_STREAM:
        cv2.imshow("People Tracking", frame)
    if SAVE_VIDEO:
        output.write(frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_count += 1


cap.release()
if SAVE_VIDEO:
    output.release()
if SHOW_STREAM:
    cv2.destroyAllWindows()

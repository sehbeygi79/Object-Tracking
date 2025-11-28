import time
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from tqdm import tqdm

from line_crossing import LineCrossing


class ObjectTracking:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = YOLO(cfg.MODEL_NAME)

        self.id2label = self.model.model.names
        self.object_ids_to_show = {
            k for k, v in self.id2label.items() if v in cfg.OBJECTS_TO_SHOW
        }

        # State variables
        self.tracker = None
        self.track_history = defaultdict(lambda: [])

        # Video I/O setup
        self.cap = None
        self.video_fps = None
        self.total_frames = None
        self.output = None
        self._setup_video_io()

        # Line crossing manager
        self.line_crossing_manager = LineCrossing(
            cfg.POLICY_JSON_PATH, cfg.LINE_ALERT_DURATION, self.video_fps
        )
        self.line_crossing_wait_time = 5

    def _setup_video_io(self):
        self.cap = cv2.VideoCapture(self.cfg.INPUT_VIDEO_PATH)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video source: {self.cfg.INPUT_VIDEO_PATH}")

        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.cfg.SAVE_VIDEO:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.output = cv2.VideoWriter(
                self.cfg.OUTPUT_VIDEO_PATH, fourcc, self.video_fps, (width, height)
            )

    def _draw_bboxes(self, frame, detection_results, object_ids_to_show={0}):
        if not len(detection_results):
            return frame

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
            annotator.box_label(
                [x1, y1, x2, y2], label, color=colors(int(track_id), True)
            )

        return frame

    # Runs the YOLO model on the frame
    def _process_frame_via_detection(self, frame):
        result = self.model.track(
            frame,
            persist=True,
            imgsz=self.cfg.IMG_SIZE,
            conf=self.cfg.OD_CONF_THRESHOLD,
            device=self.cfg.DEVICE,
            tracker=self.cfg.TRACKER_CONFIG_PATH,
            verbose=False,
        )[0]

        # Access the internal tracker instance
        if self.tracker is None:
            self.tracker = self.model.predictor.trackers[0]

        detection_results = {}
        if result.boxes and result.boxes.is_track:
            detection_results["bboxes"] = result.boxes.xyxy.cpu().numpy()
            detection_results["track_ids"] = result.boxes.id.int().cpu().numpy()
            detection_results["classes"] = result.boxes.cls.int().cpu().numpy()
            detection_results["confs"] = (last_confs := result.boxes.conf.cpu().numpy())
        else:
            last_confs = np.array([])

        return detection_results, last_confs

    # Predicts next locations using the tracker's motion model
    def _process_frame_via_prediction(self, last_confs):
        detection_results = {}

        tracks = [t for t in self.tracker.tracked_stracks if t.is_activated]
        self.tracker.multi_predict(tracks)
        self.tracker.frame_id += 1  # Manually update frame ID

        estimated_boxes = []
        for t in tracks:
            # Format: [x1, y1, x2, y2, track_id, score, class]
            estimated_boxes.append(np.hstack([t.xyxy, t.track_id, t.score, t.cls]))

        # Save estimated results
        if len(estimated_boxes):
            detection_results["bboxes"] = np.array([box[:4] for box in estimated_boxes])
            detection_results["track_ids"] = np.array(
                [int(box[4]) for box in estimated_boxes]
            )
            detection_results["classes"] = np.array(
                [int(box[6]) for box in estimated_boxes]
            )
            detection_results["confs"] = last_confs

        return detection_results

    # FPS: total processed frames divided by the total time elapsed
    def _display_fps(self, frame, start_time, frame_count):
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            fps = frame_count / elapsed_time
        else:
            fps = 0

        # Display FPS on the top-left corner
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(
            frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

        return frame

    def _draw_all_annotations(self, frame, detection_results, frame_count, start_time):
        frame = self._draw_bboxes(frame, detection_results)
        frame = self.line_crossing_manager.draw_policy_lines(frame, frame_count)
        frame = self._display_fps(frame, start_time, frame_count + 1)
        return frame

    def _release_video_resources(self):
        if self.cap:
            self.cap.release()
        if self.cfg.SAVE_VIDEO and self.output:
            self.output.release()
        if self.cfg.SHOW_STREAM:
            cv2.destroyAllWindows()

    def run(self):
        print(
            f"Starting video processing with detection every {self.cfg.SKIP_FRAMES} frames..."
        )

        start_time = time.time()
        last_confs = np.array([])

        with tqdm(
            total=self.total_frames, desc="Processing Video", unit="frame"
        ) as pbar:
            frame_count = 0
            while self.cap.isOpened():
                success, frame = self.cap.read()
                detection_results = {}

                if not success:  # Reached the end of the video
                    break

                if frame_count % self.cfg.SKIP_FRAMES == 0:
                    detection_results, last_confs = self._process_frame_via_detection(
                        frame
                    )

                # Use tracker's predictions for skipped frames
                else:
                    if self.tracker is not None:
                        detection_results = self._process_frame_via_prediction(
                            last_confs
                        )
                    else:
                        frame_count += 1
                        continue

                self.line_crossing_manager.reset_line_triggers()
                if frame_count % self.line_crossing_wait_time == 0:
                    self.line_crossing_manager.update_crossing_status(
                        detection_results,
                        self.track_history,
                        curr_time=frame_count / self.video_fps,
                    )

                # Final rendering
                frame = self._draw_all_annotations(
                    frame, detection_results, frame_count, start_time
                )
                if self.cfg.SHOW_STREAM:
                    cv2.imshow("People Tracking", frame)
                if self.cfg.SAVE_VIDEO:
                    self.output.write(frame)

                # Break loop on user input: "q"
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                frame_count += 1
                pbar.update(1)

        self.line_crossing_manager.print_logs()
        self._release_video_resources()

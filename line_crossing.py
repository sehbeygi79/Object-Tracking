from collections import defaultdict
import json
import cv2


class LineCrossing:
    def __init__(self, policy_json_path, line_alert_duration, video_fps):
        self.policy_lines = self._load_policy_lines(policy_json_path)
        if self.policy_lines is None:
            raise ValueError(f"Failed to load policy lines from {policy_json_path}")

        self.video_fps = video_fps
        self.line_alert_duration = line_alert_duration

        # State variables
        self.line_crossing_storage = defaultdict(lambda: [])
        self.lines_triggered = [False] * len(self.policy_lines)
        self.line_activity_due_times = [0] * len(self.policy_lines)

    def _load_policy_lines(self, policy_file_path):
        """Loads policy lines and fences from a JSON file."""
        try:
            with open(policy_file_path, "r") as f:
                policy_data = json.load(f)
            policy_lines = policy_data.get("lines", [])
            return policy_lines
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"Error: Problem loading the policy file {policy_file_path}")
            return None

    def _is_crossing(self, p1, p2, q1, q2):
        """
        Checks if two line segments intersect using the general line intersection formula.

        Args:
            p1 (tuple): Start point of the first segment.
            p2 (tuple): End point of the first segment.
            q1 (tuple): Start point of the second segment.
            q2 (tuple): End point of the second segment.
        """
        tc1 = (p1[0] - p2[0]) * (q1[1] - p1[1]) + (p1[1] - p2[1]) * (p1[0] - q1[0])
        tc2 = (p1[0] - p2[0]) * (q2[1] - p1[1]) + (p1[1] - p2[1]) * (p1[0] - q2[0])
        td1 = (q1[0] - q2[0]) * (p1[1] - q1[1]) + (q1[1] - q2[1]) * (q1[0] - p1[0])
        td2 = (q1[0] - q2[0]) * (p2[1] - q1[1]) + (q1[1] - q2[1]) * (q1[0] - p2[0])
        return tc1 * tc2 < 0 and td1 * td2 < 0

    def draw_policy_lines(self, frame, frame_count):
        if not len(self.policy_lines):
            return frame

        self.line_activity_due_times = [
            # Duration is multiplied by 2 because both the frame_count and the due_times are moving towards each other
            (
                frame_count + (self.video_fps * self.line_alert_duration * 2)
                if a
                else d - 1
            )
            for a, d in zip(self.lines_triggered, self.line_activity_due_times)
        ]

        for line_id, line in enumerate(self.policy_lines):
            # Line format: [[x1, y1], [x2, y2]]
            p1 = tuple(line[0])
            p2 = tuple(line[1])

            is_active = self.line_activity_due_times[line_id] >= frame_count
            line_color = (
                (0, 0, 255) if is_active else (128, 255, 0)
            )  # Red if crossed, Green otherwise

            cv2.line(frame, p1, p2, line_color, 3)

            # Label the line
            mid_x = (p1[0] + p2[0]) // 2
            mid_y = (p1[1] + p2[1]) // 2
            cv2.putText(
                frame,
                f"Line {line_id}",
                (mid_x, mid_y - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                line_color,
                2,
            )

            cv2.putText(
                frame,
                f"Counter: {len(self.line_crossing_storage[line_id])}",
                (mid_x, mid_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                line_color,
                2,
            )

        return frame

    def update_crossing_status(self, detection_results, track_history):
        if not len(self.policy_lines):
            return

        self.lines_triggered = [False] * len(self.policy_lines)
        # Check for line crossing events
        for bbox, track_id in zip(
            detection_results.get("bboxes", []),
            detection_results.get("track_ids", []),
        ):
            x1, y1, x2, y2 = map(int, bbox)
            # Use the bottom-center of the bounding box as the tracking point
            current_center = (int((x1 + x2) / 2), y2)
            history = track_history[track_id]
            if len(history) > 0:
                prev_center = history[-1]

                for line_id, line in enumerate(self.policy_lines):
                    line_start = tuple(line[0])
                    line_end = tuple(line[1])

                    if self._is_crossing(
                        prev_center, current_center, line_start, line_end
                    ):
                        self.line_crossing_storage[line_id].append(track_id)
                        self.lines_triggered[line_id] = True
                        print(
                            f"--- Line Crossing Detected! --- Track ID {track_id} crossed Line {line_id}"
                        )

            # Update track history but keep only the last two points (necessary for crossing check)
            track_history[track_id].append(current_center)
            if len(track_history[track_id]) > 2:
                track_history[track_id].pop(0)

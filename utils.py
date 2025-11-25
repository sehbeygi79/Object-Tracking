import time
import cv2


# args: prev_center, current_center, line_start, line_end
def is_crossing(p1, p2, p3, p4):
    tc1 = (p1[0] - p2[0]) * (p3[1] - p1[1]) + (p1[1] - p2[1]) * (p1[0] - p3[0])
    tc2 = (p1[0] - p2[0]) * (p4[1] - p1[1]) + (p1[1] - p2[1]) * (p1[0] - p4[0])
    td1 = (p3[0] - p4[0]) * (p1[1] - p3[1]) + (p3[1] - p4[1]) * (p3[0] - p1[0])
    td2 = (p3[0] - p4[0]) * (p2[1] - p3[1]) + (p3[1] - p4[1]) * (p3[0] - p2[0])
    return tc1 * tc2 < 0 and td1 * td2 < 0


def is_crossing2(prev_center, current_center, line_start, line_end):
    """
    Checks if a line segment defined by (prev_center, current_center) crosses
    another line segment defined by (line_start, line_end).
    The line crossing is detected based on the orientation of the segments.
    """

    # Helper function to find the orientation of the ordered triplet (p, q, r).
    # 0 --> p, q and r are collinear
    # 1 --> Clockwise
    # 2 --> Counterclockwise
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0  # Collinear
        return 1 if val > 0 else 2  # Clockwise or Counterclockwise

    # Helper function to check if point q lies on segment pr
    def on_segment(p, q, r):
        return (
            q[0] <= max(p[0], r[0])
            and q[0] >= min(p[0], r[0])
            and q[1] <= max(p[1], r[1])
            and q[1] >= min(p[1], r[1])
        )

    p1, q1 = prev_center, current_center
    p2, q2 = line_start, line_end

    # Find the four orientations needed for general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if o1 != 0 and o2 != 0 and o3 != 0 and o4 != 0 and o1 != o2 and o3 != o4:
        return True

    # Special Cases (Collinear) - not strictly necessary for simple line-crossing
    # checks in typical video surveillance, but included for completeness.
    # if o1 == 0 and on_segment(p1, p2, q1): return True
    # if o2 == 0 and on_segment(p1, q2, q1): return True
    # if o3 == 0 and on_segment(p2, p1, q2): return True
    # if o4 == 0 and on_segment(p2, q1, q2): return True

    return False


def draw_policy_lines(
    frame,
    policy_lines,
    line_crossing_storage,
    line_activity_due_times,
):
    for line_id, line in enumerate(policy_lines):
        # Line = [[x1, y1], [x2, y2]]
        p1 = tuple(line[0])
        p2 = tuple(line[1])

        is_active = line_activity_due_times[line_id] >= time.time()
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
            f"Counter: {len(line_crossing_storage[line_id])}",
            (mid_x, mid_y + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            line_color,
            2,
        )

    return frame


def update_crossing_status(
    detection_results, policy_lines, track_history, line_crossing_storage
):
    policy_lines_activity = [False] * len(policy_lines)
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

            for line_id, line in enumerate(policy_lines):
                line_start = tuple(line[0])
                line_end = tuple(line[1])

                if is_crossing(prev_center, current_center, line_start, line_end):
                    line_crossing_storage[line_id].append(track_id)
                    policy_lines_activity[line_id] = True
                    print(
                        f"--- Line Crossing Detected! --- Track ID {track_id} crossed Line {line_id}"
                    )

        # Update track history but keep only the last two points necessary for crossing check
        track_history[track_id].append(current_center)
        if len(track_history[track_id]) > 2:
            track_history[track_id].pop(0)

    return line_crossing_storage, policy_lines_activity


# FPS is the total frames processed divided by the total time elapsed
def display_fps(frame, start_time, frame_count):
    elapsed_time = time.time() - start_time
    if elapsed_time > 0:
        fps = frame_count / elapsed_time
    else:
        fps = 0

    # Display FPS on the top-left corner
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

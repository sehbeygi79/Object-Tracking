import time
import cv2


def is_crossing(prev_center, current_center, line_start, line_end):
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


def draw_policy_lines(frame, policy_lines, detection_results, crossing_track_ids):
    for line_id, line in enumerate(policy_lines):
        # Line = [[x1, y1], [x2, y2]]
        p1 = tuple(line[0])
        p2 = tuple(line[1])

        # Color the line based on if any tracked object is currently crossing it
        is_being_crossed = any(
            line_id in crossing_track_ids[track_id]
            for track_id in detection_results.get("track_ids", [])
        )
        line_color = (
            (0, 0, 255) if is_being_crossed else (255, 255, 0)
        )  # Red if crossed, Cyan otherwise

        # Draw the line on the frame
        cv2.line(frame, p1, p2, line_color, 3)

        # Label the line
        mid_x = (p1[0] + p2[0]) // 2
        mid_y = (p1[1] + p2[1]) // 2
        cv2.putText(
            frame,
            f"Line {line_id}",
            (mid_x, mid_y - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            line_color,
            2,
        )

    return frame


def update_crossing_status(
    frame, detection_results, policy_lines, track_history, crossing_track_ids
):
    # Check for line crossing events
    for bbox, track_id in zip(
        detection_results.get("bboxes", []),
        detection_results.get("track_ids", []),
    ):
        x1, y1, x2, y2 = map(int, bbox)
        # Use the bottom-center of the bounding box as the tracking point
        # TODO: we should select the appropriate point based on the direction of the movement
        # bbox_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        # bbox_bottom = (int((x1 + x2) / 2), y2)
        # corner_points = ((x1, y1), (x2, y1), (x1, y2), (x2, y2))
        current_center = (int((x1 + x2) / 2), y2)
        

        # Get history for this track ID
        history = track_history[track_id]

        # Only check for crossing if there is a previous center point
        if len(history) > 0:
            prev_center = history[-1]

            # Clear crossings for the current object on this frame
            crossing_track_ids[track_id] = set()

            for line_id, line in enumerate(policy_lines):
                line_start = tuple(line[0])
                line_end = tuple(line[1])

                if is_crossing(prev_center, current_center, line_start, line_end):
                    crossing_track_ids[track_id].add(line_id)
                    print(
                        f"--- Line Crossing Detected! --- Track ID {track_id} crossed Line {line_id}"
                    )
                    # You can add further actions here (e.g., logging, alerting)

        # Update track history for the next frame
        track_history[track_id].append(current_center)
        # Keep only the last two points (current and previous) for crossing check
        if len(track_history[track_id]) > 2:
            track_history[track_id].pop(0)

    return frame, crossing_track_ids


def display_fps(frame, start_time, frame_count):
    # Calculate FPS
    elapsed_time = time.time() - start_time
    if elapsed_time > 0:
        # FPS is the total frames processed divided by the total time elapsed
        fps = frame_count / elapsed_time
    else:
        fps = 0

    # Format and display FPS on the top-left corner
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

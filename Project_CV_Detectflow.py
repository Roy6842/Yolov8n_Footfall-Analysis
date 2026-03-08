from collections import defaultdict
import cv2
from ultralytics import YOLO

# =========================
# Booth traffic counter config
# =========================
MODEL_PATH = "yolov8n.pt"       #nano model is usually sufficient for people counting, but you can try larger models for better accuracy at the cost of speed
CAMERA_INDEX = 1
TRACKER = "botsort.yaml"        # or "bytetrack.yaml"
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.5
SAVE_VIDEO = True
OUTPUT_VIDEO_PATH = "flow_output.mp4"
WINDOW_NAME = "Booth Traffic Counter"
COUNT_COOLDOWN_FRAMES = 15       # avoid repeated counts due to jitter
NEG_TO_POS_IS_IN = True          # flip to False if IN/OUT direction is reversed
SHOW_BOX_LABELS = True
SHOW_TRACK_ID = True

# Global state for line drawing
line_points = []
line_ready = False

def mouse_callback(event, x, y, flags, param):
    """Click 2 points to define the counting line."""
    global line_points, line_ready

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(line_points) < 2:
            line_points.append((x, y))
        else:
            line_points = [(x, y)]
            line_ready = False

        if len(line_points) == 2:
            line_ready = True


def point_side(point, a, b):
    """Return which side of the line AB the point is on.

    > 0 : one side
    < 0 : the other side
    = 0 : approximately on the line
    """
    cross = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
    if abs(cross) < 1e-6:
        return 0
    return 1 if cross > 0 else -1


def draw_text_block(frame, current_people, in_count, out_count, total_flow):
    cv2.rectangle(frame, (8, 8), (340, 135), (0, 0, 0), -1)
    cv2.rectangle(frame, (8, 8), (340, 135), (255, 255, 255), 2)

    cv2.putText(frame, f"Current People: {current_people}", (18, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(frame, f"IN: {in_count}", (18, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"OUT: {out_count}", (18, 104),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    cv2.putText(frame, f"TOTAL FLOW: {total_flow}", (18, 132),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


def setup_counting_line(cap):
    """Let the user click 2 points on the camera preview to set the counting line."""
    global line_points, line_ready

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Cannot read frame from camera while setting the counting line.")

        preview = frame.copy()

        instruction_1 = "Click 2 points to draw counting line"
        instruction_2 = "Press ENTER to start | R to reset | Q to quit"
        cv2.putText(preview, instruction_1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)
        cv2.putText(preview, instruction_2, (20, 62), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)

        for pt in line_points:
            cv2.circle(preview, pt, 6, (255, 0, 0), -1)

        if len(line_points) == 2:
            cv2.line(preview, line_points[0], line_points[1], (0, 255, 255), 3)

        cv2.imshow(WINDOW_NAME, preview)
        key = cv2.waitKey(1) & 0xFF

        if key == 13 and len(line_points) == 2:  # Enter
            return line_points[0], line_points[1]
        if key == ord('r'):
            line_points = []
            line_ready = False
        if key == ord('q'):
            raise SystemExit("User exited before counting started.")


def main():
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Cannot open camera.")
        return

    print("Set the counting line on the preview window.")
    line_p1, line_p2 = setup_counting_line(cap)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps == 0:
        fps = 20.0

    writer = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

    last_side_by_id = {}
    last_center_by_id = {}
    last_count_frame_by_id = defaultdict(lambda: -10_000)
    in_count = 0
    out_count = 0
    frame_index = 0

    print("Counting started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame_index += 1
        current_people = 0

        results = model.track(
            frame,
            persist=True,
            classes=[0],
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            tracker=TRACKER,
            verbose=False,
        )

        annotated = frame.copy()
        cv2.line(annotated, line_p1, line_p2, (0, 255, 255), 3)

        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            current_people = len(boxes)

            track_ids = None
            if boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
            else:
                track_ids = [None] * len(boxes)

            xyxy_list = boxes.xyxy.cpu().tolist()
            conf_list = boxes.conf.cpu().tolist() if boxes.conf is not None else [0.0] * len(boxes)

            for (xyxy, track_id, conf) in zip(xyxy_list, track_ids, conf_list):
                x1, y1, x2, y2 = map(int, xyxy)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                center = (cx, cy)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(annotated, center, 4, (255, 0, 255), -1)

                label_parts = ["person"]
                if SHOW_TRACK_ID and track_id is not None:
                    label_parts.append(f"ID:{track_id}")
                if SHOW_BOX_LABELS:
                    label_parts.append(f"{conf:.2f}")
                label = " ".join(label_parts)
                cv2.putText(annotated, label, (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if track_id is None:
                    continue

                current_side = point_side(center, line_p1, line_p2)
                prev_side = last_side_by_id.get(track_id)
                prev_center = last_center_by_id.get(track_id)

                if prev_center is not None:
                    cv2.line(annotated, prev_center, center, (255, 0, 0), 2)

                if prev_side is not None and current_side != 0 and prev_side != 0 and current_side != prev_side:
                    if frame_index - last_count_frame_by_id[track_id] > COUNT_COOLDOWN_FRAMES:
                        if prev_side < current_side:
                            if NEG_TO_POS_IS_IN:
                                in_count += 1
                            else:
                                out_count += 1
                        else:
                            if NEG_TO_POS_IS_IN:
                                out_count += 1
                            else:
                                in_count += 1
                        last_count_frame_by_id[track_id] = frame_index

                last_side_by_id[track_id] = current_side
                last_center_by_id[track_id] = center

        total_flow = in_count + out_count
        draw_text_block(annotated, current_people, in_count, out_count, total_flow)

        cv2.imshow(WINDOW_NAME, annotated)

        if writer is not None:
            writer.write(annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

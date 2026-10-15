#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import zmq


def import_yolo(repo_path: Path | None = None):
    if repo_path is not None:
        import sys
        sys.path.insert(0, str(repo_path.resolve()))
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise ImportError("Cannot import ultralytics YOLO. Install ultralytics or pass --yolo-repo.") from exc
    return YOLO


def select_best_detection(det, class_id: int, min_conf: float):
    if not hasattr(det, "boxes") or len(det.boxes) == 0:
        return None

    boxes = det.boxes.xyxy.cpu().numpy()
    classes = det.boxes.cls.cpu().numpy().astype(int)
    confidences = det.boxes.conf.cpu().numpy()

    best = None
    for box, cls, conf in zip(boxes, classes, confidences):
        if class_id >= 0 and cls != class_id:
            continue
        if conf < min_conf:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        area = max(0, x2 - x1) * max(0, y2 - y1)
        score = float(conf) * area
        if best is None or score > best["score"]:
            best = {
                "box": [x1, y1, x2, y2],
                "confidence": float(conf),
                "class_id": int(cls),
                "score": score,
            }
    return best


def draw_detection(frame, detection):
    annotated = frame.copy()
    if detection is None:
        cv2.putText(
            annotated,
            "UAV not found",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated, None

    x1, y1, x2, y2 = detection["box"]
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.circle(annotated, (center_x, center_y), 5, (0, 255, 0), -1)
    cv2.putText(
        annotated,
        f"UAV {detection['confidence']:.2f}",
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    return annotated, (center_x, center_y)


def main():
    parser = argparse.ArgumentParser(description="Receive Pi camera frames, run YOLO on Windows, publish target coordinates")
    parser.add_argument("--frame-connect", type=str, required=True, help="Pi frame publisher address, e.g. tcp://10.212.187.158:5556")
    parser.add_argument("--frame-topic", type=str, default="camera_frame", help="Frame topic")
    parser.add_argument("--coord-bind", type=str, default="tcp://*:5555", help="Coordinate publisher bind address")
    parser.add_argument("--coord-topic", type=str, default="green_target", help="Coordinate topic")
    parser.add_argument("--model", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--yolo-repo", type=str, default=None, help="Local ultralytics repo path")
    parser.add_argument("--img-size", type=int, default=640, help="YOLO image size")
    parser.add_argument("--device", type=str, default="cpu", help="YOLO device, for example cpu or 0")
    parser.add_argument("--conf", type=float, default=0.25, help="Minimum confidence")
    parser.add_argument("--class-id", type=int, default=0, help="UAV class id; use -1 for all classes")
    parser.add_argument("--debug", action="store_true", help="Print sent center coordinates")
    parser.add_argument("--raw-window", type=str, default="Pi Camera Feed")
    parser.add_argument("--detect-window", type=str, default="Windows YOLO Detection")
    args = parser.parse_args()

    YOLO = import_yolo(Path(args.yolo_repo) if args.yolo_repo else None)
    model = YOLO(args.model)

    context = zmq.Context()

    frame_subscriber = context.socket(zmq.SUB)
    frame_subscriber.setsockopt(zmq.RCVHWM, 1)
    frame_subscriber.setsockopt(zmq.LINGER, 0)
    frame_subscriber.setsockopt(zmq.CONFLATE, 1)
    frame_subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
    frame_subscriber.connect(args.frame_connect)

    coord_publisher = context.socket(zmq.PUB)
    coord_publisher.setsockopt(zmq.SNDHWM, 1)
    coord_publisher.setsockopt(zmq.LINGER, 0)
    coord_publisher.bind(args.coord_bind)

    prev_time = time.time()
    fps = 0.0

    print(f"Receiving frames from {args.frame_connect}, topic={args.frame_topic}")
    print(f"Publishing coordinates on {args.coord_bind}, topic={args.coord_topic}")
    print("Press q or ESC to exit.")

    try:
        while True:
            jpeg_bytes = frame_subscriber.recv()
            frame_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            if frame is None:
                coord_publisher.send_string(f"{args.coord_topic} -1 -1")
                continue

            results = model(frame, imgsz=args.img_size, device=args.device, verbose=False)
            detection = select_best_detection(results[0], args.class_id, args.conf)
            annotated, center = draw_detection(frame, detection)

            current_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, current_time - prev_time))
            prev_time = current_time
            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (10, annotated.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

            if center is None:
                coord_publisher.send_string(f"{args.coord_topic} -1 -1")
                if args.debug:
                    print("send target: -1 -1")
            else:
                coord_publisher.send_string(f"{args.coord_topic} {center[0]} {center[1]}")
                if args.debug:
                    print(f"send target: {center[0]} {center[1]}")

            cv2.imshow(args.raw_window, frame)
            cv2.imshow(args.detect_window, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in {27, ord("q"), ord("Q")}:
                break
    finally:
        frame_subscriber.close()
        coord_publisher.close()
        context.term()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

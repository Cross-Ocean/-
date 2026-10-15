#!/usr/bin/env python3
import argparse
import time

import cv2
import zmq


def main():
    parser = argparse.ArgumentParser(description="Publish Raspberry Pi camera frames to a remote YOLO detector")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--width", type=int, default=320, help="Camera width")
    parser.add_argument("--height", type=int, default=240, help="Camera height")
    parser.add_argument("--bind", type=str, default="tcp://*:5556", help="Frame publisher bind address")
    parser.add_argument("--topic", type=str, default="camera_frame", help="Frame topic")
    parser.add_argument("--jpeg-quality", type=int, default=70, help="JPEG quality from 1 to 100")
    parser.add_argument("--flip-code", type=int, default=0, help="OpenCV flip code, camera.py used 0")
    parser.add_argument("--no-flip", action="store_true", help="Disable image flipping")
    args = parser.parse_args()

    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.setsockopt(zmq.SNDHWM, 1)
    publisher.setsockopt(zmq.LINGER, 0)
    publisher.setsockopt(zmq.CONFLATE, 1)
    publisher.bind(args.bind)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]

    print(f"Publishing camera frames on {args.bind}, topic={args.topic}")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            if not args.no_flip:
                frame = cv2.flip(frame, args.flip_code)

            ok, encoded = cv2.imencode(".jpg", frame, encode_params)
            if not ok:
                continue

            publisher.send(encoded.tobytes())
    except KeyboardInterrupt:
        print("Stopping camera frame publisher.")
    finally:
        cap.release()
        publisher.close()
        context.term()


if __name__ == "__main__":
    main()

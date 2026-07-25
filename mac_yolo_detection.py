from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "yolov8n.pt"
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.35
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

PROJECT_FOLDER = Path(__file__).resolve().parent
OUTPUT_FOLDER = PROJECT_FOLDER / "outputs"
SCREENSHOT_FOLDER = OUTPUT_FOLDER / "screenshots"
METRICS_FILE = OUTPUT_FOLDER / "runtime_metrics.csv"
VIDEO_FILE = OUTPUT_FOLDER / "detection_recording.mp4"


def create_output_folders() -> None:
    OUTPUT_FOLDER.mkdir(exist_ok=True)
    SCREENSHOT_FOLDER.mkdir(exist_ok=True)


def initialize_metrics_file() -> None:
    if METRICS_FILE.exists():
        return

    with METRICS_FILE.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(
            [
                "timestamp",
                "frame_number",
                "fps",
                "inference_ms",
                "detection_count",
                "average_confidence",
                "brightness",
            ]
        )


def calculate_brightness(frame) -> float:
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(grayscale.mean())


def save_metrics(
    frame_number: int,
    fps: float,
    inference_ms: float,
    detection_count: int,
    average_confidence: float,
    brightness: float,
) -> None:
    with METRICS_FILE.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                frame_number,
                round(fps, 3),
                round(inference_ms, 3),
                detection_count,
                round(average_confidence, 4),
                round(brightness, 3),
            ]
        )


def main() -> None:
    create_output_folders()
    initialize_metrics_file()

    print("Loading YOLO model...")
    model = YOLO(MODEL_NAME)

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        raise RuntimeError(
            "The Mac camera could not be opened. "
            "Check macOS camera permissions and close other camera applications."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_writer = None
    recording = False
    frame_number = 0
    previous_time = time.perf_counter()

    print("\nYOLO Mac Webcam Detection")
    print("--------------------------------")
    print("Q = Quit")
    print("S = Save screenshot")
    print("R = Start/stop video recording")
    print("--------------------------------\n")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Unable to read a camera frame.")
                break

            frame_number += 1

            inference_start = time.perf_counter()

            results = model.predict(
                source=frame,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False,
            )

            inference_ms = (
                time.perf_counter() - inference_start
            ) * 1000

            annotated_frame = results[0].plot()

            current_time = time.perf_counter()
            elapsed = current_time - previous_time
            previous_time = current_time

            fps = 1.0 / elapsed if elapsed > 0 else 0.0

            boxes = results[0].boxes

            if boxes is not None and len(boxes) > 0:
                detection_count = len(boxes)
                confidence_values = boxes.conf.cpu().numpy()
                average_confidence = float(confidence_values.mean())
            else:
                detection_count = 0
                average_confidence = 0.0

            brightness = calculate_brightness(frame)

            cv2.putText(
                annotated_frame,
                f"FPS: {fps:.1f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                annotated_frame,
                f"Inference: {inference_ms:.1f} ms",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                annotated_frame,
                f"Objects: {detection_count}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                annotated_frame,
                f"Brightness: {brightness:.1f}",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            if recording:
                cv2.putText(
                    annotated_frame,
                    "RECORDING",
                    (actual_width - 190, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

                if video_writer is not None:
                    video_writer.write(annotated_frame)

            save_metrics(
                frame_number=frame_number,
                fps=fps,
                inference_ms=inference_ms,
                detection_count=detection_count,
                average_confidence=average_confidence,
                brightness=brightness,
            )

            cv2.imshow(
                "YOLO Real-Time Object Detection - Dominique McClaney",
                annotated_frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("s"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                screenshot_path = (
                    SCREENSHOT_FOLDER
                    / f"yolo_detection_{timestamp}.png"
                )

                cv2.imwrite(
                    str(screenshot_path),
                    annotated_frame,
                )

                print(f"Screenshot saved: {screenshot_path}")

            if key == ord("r"):
                recording = not recording

                if recording:
                    codec = cv2.VideoWriter_fourcc(*"mp4v")

                    video_writer = cv2.VideoWriter(
                        str(VIDEO_FILE),
                        codec,
                        20.0,
                        (actual_width, actual_height),
                    )

                    print(f"Recording started: {VIDEO_FILE}")

                else:
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None

                    print("Recording stopped.")

    finally:
        camera.release()

        if video_writer is not None:
            video_writer.release()

        cv2.destroyAllWindows()

        print("\nYOLO detection stopped.")
        print(f"Metrics saved to: {METRICS_FILE}")
        print(f"Screenshots saved to: {SCREENSHOT_FOLDER}")


if __name__ == "__main__":
    main()
import cv2
import numpy as np
import os
import time
import csv
from datetime import datetime
from pathlib import Path

# Globals
crop_coords = None
csv_path = None
output_dir = None
captures_dir = None
camera_index = 0


def initialize_camera(camera_index=0):
    """Initialize and configure the camera"""
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera with index {camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    return cap


def capture_image_from_camera(cap, filename=None, save_image=True):
    """Capture image from an already opened camera"""
    ret, frame = cap.read()

    if not ret or frame is None:
        raise RuntimeError("Failed to capture image from camera")

    if save_image:
        if filename is None:
            filename = 'captured_image.jpg'

        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        success = cv2.imwrite(filename, frame)
        if not success:
            raise RuntimeError(f"Failed to save image to {filename}")

        print(f"Image saved as {filename}")

    return frame


def select_crop_window(frame, window_name="Select ROI"):
    cv2.imshow(window_name, frame)
    print("Select a region of interest and press SPACE or ENTER to confirm")
    r = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_name)
    return tuple(map(int, r))


def binarize(img, threshold=127):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary


def analyse(binary_img):
    h, w = binary_img.shape
    white_area = int((binary_img == 255).sum())
    black_area = int((binary_img == 0).sum())

    white_fraction = (binary_img == 255).sum(axis=1) / w
    transition_rows = np.where(white_fraction >= 0.5)[0]
    transition_y = int(transition_rows[0]) if transition_rows.size else h

    white_height = h - transition_y
    black_height = transition_y

    return white_height, black_height, black_area, white_area, transition_y


def output(binary_img, transition_y, output_filename):
    vis = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    cv2.line(vis, (0, transition_y), (vis.shape[1], transition_y), (0, 0, 255), 2)
    cv2.imwrite(str(output_filename), vis)


def experimentloop(cap, interval=60, count=120):
    """Main experiment loop with persistent camera connection"""
    global crop_coords, csv_path, captures_dir, output_dir

    start_time = time.time()

    for i in range(1, count + 1):
        elapsed_min = (time.time() - start_time) / 60
        filename = captures_dir / f"capture_{i:03d}.jpg"
        
        # Capture image from the persistent camera connection
        frame = capture_image_from_camera(cap, filename=str(filename), save_image=True)

        x, y, w, h = crop_coords
        crop_frame = frame[y:y+h, x:x+w]
        binary_img = binarize(crop_frame)
        white_h, black_h, black_a, white_a, trans_y = analyse(binary_img)

        output_file = output_dir / f"output_{i:03d}.jpg"
        output(binary_img, trans_y, output_file)

        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                datetime.now().isoformat(), f"{elapsed_min:.2f}", black_a, white_a,
                black_h, white_h, trans_y, h, f"{white_a / (w*h):.4f}"
            ])

        print(f"[Img {i}] Elapsed: {elapsed_min:.2f} min | Black Area: {black_a}, White Area: {white_a}, Threshold Y: {trans_y}")

        next_capture_time = start_time + (i * interval)
        time_to_wait = next_capture_time - time.time()
        if time_to_wait > 0:
            time.sleep(time_to_wait)


def main():
    global crop_coords, csv_path, captures_dir, output_dir, camera_index

    base_dir = Path("experiments")  # Main experiments folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    experiment_dir = base_dir / f"experiment_{timestamp}"

    # Create subdirectories
    captures_dir = experiment_dir / "captures"
    output_dir = experiment_dir / "output"
    csv_path = experiment_dir / "results.csv"

    # Create all directories
    captures_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'timestamp', 'elapsed_min', 'black_area', 'white_area',
            'black_height', 'white_height', 'transition_y', 'total_height', 'white_percentage'
        ])

    # Initialize camera once
    print("Initializing camera...")
    cap = initialize_camera(camera_index)
    
    try:
        # Capture test frame for ROI selection
        test_frame = capture_image_from_camera(cap, save_image=False)
        crop_coords = select_crop_window(test_frame)
        
        print("Starting experiment loop")
        experimentloop(cap, interval=60, count=120)
        
    finally:
        # Always release the camera, even if an error occurs
        print("Releasing camera...")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
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
    return binary, gray  # Now returns both binary and grayscale


def analyse(binary_img, gray_img):
    h, w = binary_img.shape
    white_area = int((binary_img == 255).sum())
    black_area = int((binary_img == 0).sum())

    white_fraction = (binary_img == 255).sum(axis=1) / w
    transition_rows = np.where(white_fraction >= 0.5)[0]
    
    # Handle edge cases properly
    if transition_rows.size == 0:
        # All black image - no rows with >= 50% white
        transition_y = h - 1  # Bottom row (instead of out-of-bounds h)
    elif transition_rows[0] == 0 and white_fraction[0] >= 0.9:
        # Likely all or mostly white image
        transition_y = 0
    else:
        # Normal case - first row where white pixels dominate
        transition_y = int(transition_rows[0])
    
    # Ensure transition_y is within valid bounds
    transition_y = max(0, min(transition_y, h - 1))

    white_height = h - transition_y
    black_height = transition_y

    # NEW: Divide image into 10 vertical sections and compute average grayscale
    section_height = h // 10
    section_averages = []
    
    for i in range(10):
        start_y = i * section_height
        if i == 9:  # Last section includes any remaining pixels
            end_y = h
        else:
            end_y = (i + 1) * section_height
        
        section = gray_img[start_y:end_y, :]
        avg_gray = np.mean(section)
        section_averages.append(avg_gray)

    return white_height, black_height, black_area, white_area, transition_y, section_averages


def output(binary_img, transition_y, output_filename):
    vis = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    h, w = vis.shape[:2]
    
    # Ensure line is drawn within image bounds
    line_y = max(0, min(transition_y, h - 1))
    cv2.line(vis, (0, line_y), (w, line_y), (0, 0, 255), 2)
    
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
        binary_img, gray_img = binarize(crop_frame)  # Now gets both images
        white_h, black_h, black_a, white_a, trans_y, section_avgs = analyse(binary_img, gray_img)
        
        total_pixels = w * h

        output_file = output_dir / f"output_{i:03d}.jpg"
        output(binary_img, trans_y, output_file)

        # Prepare row with section averages
        row_data = [
            datetime.now().isoformat(), f"{elapsed_min:.2f}", black_a, white_a,
            black_h, white_h, trans_y, h, f"{white_a / (w*h):.4f}"
        ]
        # Add the 10 section averages
        row_data.extend([f"{avg:.2f}" for avg in section_avgs])

        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row_data)

        print(f"[Img {i}] Elapsed: {elapsed_min:.2f} min | Black Area: {black_a}, White Area: {white_a}, Threshold Y: {trans_y}")
        print(f"   Section averages (0-255): {[f'{avg:.1f}' for avg in section_avgs]}")
        
        # Log potential edge cases
        if white_a == 0:
            print(f"   ⚠️  All black image detected")
        elif white_a == total_pixels:
            print(f"   ⚠️  All white image detected")
        elif trans_y == 0:
            print(f"   ℹ️  Transition at top of image")
        elif trans_y == h - 1:
            print(f"   ℹ️  Transition at bottom of image")

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
        headers = [
            'timestamp', 'elapsed_min', 'black_area', 'white_area',
            'black_height', 'white_height', 'transition_y', 'total_height', 'white_percentage'
        ]
        # Add headers for 10 sections
        headers.extend([f'section_{i}_avg' for i in range(10)])
        writer.writerow(headers)

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
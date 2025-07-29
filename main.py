import cv2
import numpy as np
import time
import os
import csv

def capture_image(camera_index=0, filename=None, save_image=True):
    """
    Capture an image from the specified camera.
    
    Args:
        camera_index (int): Index of the camera to use (default: 0)
        filename (str): Output filename. If None, uses 'captured_image.jpg'
        save_image (bool): Whether to save the image to file (default: True)
    
    Returns:
        numpy.ndarray: The captured frame
        
    Raises:
        RuntimeError: If camera cannot be opened or image capture fails
    """
    # Initialize camera
    cap = cv2.VideoCapture(camera_index)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera with index {camera_index}")
    
    # Set camera properties for better quality (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    try:
        # Capture frame
        ret, frame = cap.read()
        
        if not ret or frame is None:
            raise RuntimeError("Failed to capture image from camera")
        
        # Save image if requested
        if save_image:
            if filename is None:
                filename = 'captured_image.jpg'
            
            # Ensure directory exists
            directory = os.path.dirname(filename)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            
            # Save the image
            success = cv2.imwrite(filename, frame)
            if not success:
                raise RuntimeError(f"Failed to save image to {filename}")
            
            print(f"Image saved as {filename}")
        
        return frame
        
    finally:
        # Always release the camera
        cap.release()

def selectcropwindow(frame, window_name="Select Crop Window"):
    """
    Allow user to select a crop window on the given frame.
   
    Args:
        frame (numpy.ndarray): The image frame to select from
        window_name (str): Name of the window for display
   
    Returns:
        tuple: Coordinates of the selected crop window (x, y, width, height)
    """
    # Display the image
    cv2.imshow(window_name, frame)
   
    # Wait for user to select a region
    r = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True)
   
    # Wait for any key press and close the window
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)
   
    return r  # Returns (x, y, width, height)


def binarize_image(image, thresh=127):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return binary

def crop_image(image, x, y, w, h):
    return image[y:y+h, x:x+w]

def calc_white_black_area(binary_img):
    white = np.count_nonzero(binary_img == 255)
    black = np.count_nonzero(binary_img == 0)
    return white, black

def analyze_black_shapes(binary_img, min_size=60):
    """
    Analyze black shapes in the image and return detailed information
    including area, perimeter, and bounding boxes.
    """
    # Invert so that black shapes become the foreground
    inv = cv2.bitwise_not(binary_img)
    _, inv_bin = cv2.threshold(inv, 127, 255, cv2.THRESH_BINARY)
    
    # Get connected components with stats
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inv_bin, connectivity=8)
    
    shapes_info = []
    
    # Skip label 0 (background)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        if area >= min_size:
            # Get bounding box coordinates
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Create mask for this specific component
            component_mask = (labels == i).astype(np.uint8) * 255
            
            # Find contours to calculate perimeter
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Calculate perimeter (should be only one contour per component)
            perimeter = cv2.arcLength(contours[0], True) if contours else 0
            
            shapes_info.append({
                'area': area,
                'perimeter': perimeter,
                'bbox': (x, y, w, h),
                'centroid': (int(centroids[i][0]), int(centroids[i][1]))
            })
    
    return shapes_info

def draw_bounding_boxes(image, shapes_info, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes and information on the image.
    """
    output_img = image.copy()
    
    # Convert to BGR if grayscale
    if len(output_img.shape) == 2:
        output_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2BGR)
    
    for i, shape in enumerate(shapes_info):
        x, y, w, h = shape['bbox']
        cv2.rectangle(output_img, (x, y), (x + w, y + h), color, thickness)
        
        # Add shape number and area as text
        text = f"#{i+1} A:{shape['area']}"
        cv2.putText(output_img, text, (x, y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return output_img

def calculate_averages(shapes_info):
    """
    Calculate average area and perimeter of shapes.
    """
    if not shapes_info:
        return 0, 0
    
    total_area = sum(shape['area'] for shape in shapes_info)
    total_perimeter = sum(shape['perimeter'] for shape in shapes_info)
    count = len(shapes_info)
    
    avg_area = total_area / count
    avg_perimeter = total_perimeter / count
    
    return avg_area, avg_perimeter

def main():
    timestamp = time.strftime("%m-%d_%H_%M")
    base_folder = timestamp
    
    # Create the main folder and subfolders
    captured_folder = os.path.join(base_folder, "captured")
    output_folder = os.path.join(base_folder, "output")
    
    # Create directories if they don't exist
    os.makedirs(captured_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Created folders: {base_folder}/captured and {base_folder}/output")
    
    # === USER SETTINGS ===
    min_shape_size = 60                # minimum pixel count for a connected shape
    thresh_value = 127                 # binarization threshold (0–255)
    interval_seconds = 60              # capture interval
    # ======================
    
    start_time = time.time()
    counter = 1
    
    # Wait for the first interval
    wait = interval_seconds - ((time.time() - start_time) % interval_seconds)
    time.sleep(wait)
    
    # CSV for general data
    csv_path = os.path.join(output_folder, "data.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["elapsed_time_s", "white_area", "black_area", "num_shapes", 
                     "avg_shape_area", "avg_shape_perimeter"])
    
    # CSV for detailed shape data
    shapes_csv_path = os.path.join(output_folder, "shapes_detail.csv")
    shapes_csv_file = open(shapes_csv_path, "w", newline="")
    shapes_writer = csv.writer(shapes_csv_file)
    shapes_writer.writerow(["image_num", "shape_num", "area", "perimeter", 
                            "bbox_x", "bbox_y", "bbox_w", "bbox_h"])
    
    print(f"Starting capture every {interval_seconds} seconds...")

    counter_max = 120

    
    
    try:
        while counter <= counter_max:

            try:
                # 1) Capture
                frame = capture_image()
                now = time.time()
                elapsed = now - start_time
                counter
                # 2) Save original
                filename = os.path.join(captured_folder, f"img_{counter}.jpeg")
                cv2.imwrite(filename, frame)
            except:
                print(f"Error capturing image {counter}. Retrying...")
                continue
        
            # 3) Binarize
            try:                
                # 4) Crop
                crop = crop_image(binary, x, y, w, h)
                binary = binarize_image(frame, thresh_value)
                # 5) Compute areas
                white_area, black_area = calc_white_black_area(crop)
                
                # 6) Analyze shapes (get detailed info)
                shapes_info = analyze_black_shapes(crop, min_shape_size)
                num_shapes = len(shapes_info)
                
                # 7) Calculate averages
                avg_area, avg_perimeter = calculate_averages(shapes_info)
                
                # 8) Create output image with bounding boxes
                # Draw on the cropped binary image
                output_crop = draw_bounding_boxes(crop, shapes_info)
                
                # Also create full frame with crop region and bounding boxes
                full_output = frame.copy()
                # Draw crop region on full frame
                cv2.rectangle(full_output, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(full_output, "Crop Region", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Save output images
                crop_output_path = os.path.join(output_folder, f"crop_bbox_{counter}.jpeg")
                full_output_path = os.path.join(output_folder, f"full_bbox_{counter}.jpeg")
                cv2.imwrite(crop_output_path, output_crop)
                cv2.imwrite(full_output_path, full_output)
                
                # Report
                print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}]")
                print(f"Img {counter}: white={white_area}, black={black_area}, shapes={num_shapes}")
                print(f"Average shape area: {avg_area:.2f}, Average perimeter: {avg_perimeter:.2f}")
                
                # Write to main CSV
                writer.writerow([f"{elapsed:.2f}", white_area, black_area, num_shapes, 
                                f"{avg_area:.2f}", f"{avg_perimeter:.2f}"])
                csv_file.flush()
                
                # Write shape data csv
                for i, shape in enumerate(shapes_info):
                    bbox_x, bbox_y, bbox_w, bbox_h = shape['bbox']
                    shapes_writer.writerow([counter, i+1, shape['area'], f"{shape['perimeter']:.2f}",
                                        bbox_x, bbox_y, bbox_w, bbox_h])
                shapes_csv_file.flush()
                print(f"Captured and processed image {counter} successfully.")
            except Exception as e:
                print(f"Error processing image {counter}: {e}")

            # Wait until next exact interval
            next_capture = start_time + counter * interval_seconds
            time_to_wait = next_capture - time.time()
            if time_to_wait > 0:
                time.sleep(time_to_wait)
                
    except KeyboardInterrupt:
        print("\nStopping capture...")
    finally:
        csv_file.close()
        shapes_csv_file.close()
        print("CSV files closed.")

if __name__ == "__main__":
    main()
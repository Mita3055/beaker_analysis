from types import NoneType
import cv2
import os
import glob
import numpy as np
from pathlib import Path

def select_crop_region(image_path):
    """
    Display an image and let user select a crop region
    Returns: (x, y, width, height) of the selected region
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Create a copy for display
    img_copy = img.copy()
    
    # Variables to store crop coordinates
    crop_coords = []
    drawing = False
    start_x, start_y = -1, -1
    
    def draw_rectangle(event, x, y, flags, param):
        nonlocal start_x, start_y, drawing, img_copy, crop_coords
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_x, start_y = x, y
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                img_copy = img.copy()
                cv2.rectangle(img_copy, (start_x, start_y), (x, y), (0, 255, 0), 2)
                
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            # Calculate crop coordinates
            x1, y1 = min(start_x, x), min(start_y, y)
            x2, y2 = max(start_x, x), max(start_y, y)
            crop_coords = [x1, y1, x2 - x1, y2 - y1]  # x, y, width, height
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Create window and set mouse callback
    window_name = 'Select Crop Region (Press ENTER to confirm, ESC to cancel)'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, draw_rectangle)
    
    print(f"Opening image: {image_path}")
    print("Draw a rectangle to select crop region. Press ENTER to confirm, ESC to cancel.")
    
    while True:
        cv2.imshow(window_name, img_copy)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13:  # Enter key
            if crop_coords:
                cv2.destroyAllWindows()
                return tuple(crop_coords)
            else:
                print("No region selected. Please draw a rectangle.")
                
        elif key == 27:  # ESC key
            cv2.destroyAllWindows()
            return None

def select_threshold_point(image_path, scale_factor=2.0):
    """
    Display a grayscale image with a slider to select threshold value
    Returns: threshold value
    """
    # Read the image in grayscale
    img_color = cv2.imread(image_path)
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Variables to store threshold
    threshold_value = [128]  # Default value in a list so it can be modified in callback
    
    def on_threshold_change(value):
        # Apply threshold
        _, img_binary = cv2.threshold(img, value, 255, cv2.THRESH_BINARY)
        
        # Scale up the images
        h, w = img.shape
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        # Resize both images
        img_scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        img_binary_scaled = cv2.resize(img_binary, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Create side-by-side display with larger images
        divider_width = 20  # Wider divider for larger display
        combined = np.zeros((new_h, new_w * 2 + divider_width), dtype=np.uint8)
        combined[:, :new_w] = img_scaled  # Original on left
        combined[:, new_w+divider_width:] = img_binary_scaled  # Binary on right
        
        # Add divider line (darker gray for better visibility)
        combined[:, new_w:new_w+divider_width] = 64
        
        cv2.imshow(window_name, combined)
        threshold_value[0] = value
    
    # Create window
    window_name = f'Threshold Adjustment - Value: [Use trackbar] (ENTER=confirm, ESC=cancel)'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    # Create trackbar
    cv2.createTrackbar('Threshold', window_name, 128, 255, on_threshold_change)
    
    print(f"\nOpening threshold selection interface (scaled {scale_factor}x)")
    print("Use the slider to adjust the threshold value (0-255).")
    print("Left: Original grayscale | Right: Binary result")
    print("Press ENTER to confirm, ESC to cancel.")
    
    # Initial display
    on_threshold_change(128)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13:  # Enter key
            cv2.destroyAllWindows()
            print(f"Threshold value selected: {threshold_value[0]}")
            return threshold_value[0]
                
        elif key == 27:  # ESC key
            cv2.destroyAllWindows()
            return NoneType

def crop_image(img, crop_region):
    """Crop image using the given region (x, y, width, height)"""
    x, y, w, h = crop_region
    return img[y:y+h, x:x+w]

def process_images(input_folder, crop_region):
    """Process all images in the input folder"""
    # Get the folder name for the processed folder
    folder_name = os.path.basename(os.path.abspath(input_folder))
    if not folder_name:
        folder_name = "images"
    
    # Create main processed folder
    base_dir = os.path.dirname(input_folder) if os.path.dirname(input_folder) else '.'
    processed_folder = os.path.join(base_dir, f'processed_{folder_name}')
    
    # Check if processed folder already exists
    if os.path.exists(processed_folder):
        response = input(f"\nWarning: '{processed_folder}' already exists. Continue and potentially overwrite files? (y/n): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            return None, None, 0
    
    # Create subfolders
    saved_folder = os.path.join(processed_folder, 'saved')
    cropped_folder = os.path.join(processed_folder, 'cropped')
    grayscale_folder = os.path.join(processed_folder, 'grayscale')
    
    # Create folders if they don't exist
    os.makedirs(saved_folder, exist_ok=True)
    os.makedirs(cropped_folder, exist_ok=True)
    os.makedirs(grayscale_folder, exist_ok=True)
    
    # Get all image files
    image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    for pattern in image_patterns:
        image_files.extend(glob.glob(os.path.join(input_folder, pattern)))
        image_files.extend(glob.glob(os.path.join(input_folder, pattern.upper())))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for file in image_files:
        # Normalize path for comparison
        normalized = os.path.normpath(file).lower()
        if normalized not in seen:
            seen.add(normalized)
            unique_files.append(file)
    
    image_files = unique_files
    
    # Sort files to maintain order
    image_files.sort()
    
    if not image_files:
        print(f"No image files found in {input_folder}")
        return None, None, 0
    
    print(f"Found {len(image_files)} unique images to process")
    
    # Debug: Show list of files if there aren't too many
    if len(image_files) <= 20:
        print("Files to process:")
        for f in image_files:
            print(f"  - {os.path.basename(f)}")
    
    # Counter for processed images
    processed_count = 0
    last_cropped_path = None
    
    for img_path in image_files:
        original_filename = os.path.basename(img_path)
        print(f"Processing: {original_filename}")
        
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {original_filename}, skipping...")
            continue
        
        # Crop image
        try:
            cropped = crop_image(img, crop_region)
            
            # Increment counter for successful processing
            processed_count += 1
            
            # Create new filename with sequential numbering
            # Get the file extension from the original file
            _, ext = os.path.splitext(original_filename)
            new_filename = f"capture_{processed_count:03d}{ext}"
            
            # Save original image with new name
            saved_path = os.path.join(saved_folder, new_filename)
            cv2.imwrite(saved_path, img)
            
            # Save cropped image
            cropped_path = os.path.join(cropped_folder, new_filename)
            cv2.imwrite(cropped_path, cropped)
            last_cropped_path = cropped_path
            
            # Convert to grayscale
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            gray_path = os.path.join(grayscale_folder, new_filename)
            cv2.imwrite(gray_path, gray)
            
            print(f"  -> Saved as: {new_filename}")
            
        except Exception as e:
            print(f"Error processing {original_filename}: {str(e)}")
            # Don't increment counter for failed images
            continue
    
    # Save crop coordinates to a file for future reference
    crop_info_path = os.path.join(saved_folder, 'crop_coordinates.txt')
    with open(crop_info_path, 'w') as f:
        f.write(f"Crop Region Coordinates:\n")
        f.write(f"X: {crop_region[0]}\n")
        f.write(f"Y: {crop_region[1]}\n")
        f.write(f"Width: {crop_region[2]}\n")
        f.write(f"Height: {crop_region[3]}\n")
        f.write(f"\nProcessed {processed_count} images\n")
    
    return processed_folder, last_cropped_path, processed_count

def apply_threshold_to_all(processed_folder, threshold_value):
    """Apply the selected threshold to all grayscale images"""
    grayscale_folder = os.path.join(processed_folder, 'grayscale')
    binarized_folder = os.path.join(processed_folder, 'binarized')
    
    # Create binarized folder
    os.makedirs(binarized_folder, exist_ok=True)
    
    # Get all grayscale images
    gray_images = glob.glob(os.path.join(grayscale_folder, '*'))
    gray_images.sort()
    
    print(f"\nApplying threshold value {threshold_value} to all images...")
    
    for gray_path in gray_images:
        filename = os.path.basename(gray_path)
        
        # Read grayscale image
        gray = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"Warning: Could not read {filename}, skipping...")
            continue
        
        # Apply threshold
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        
        # Save binarized image
        binary_path = os.path.join(binarized_folder, filename)
        cv2.imwrite(binary_path, binary)
    
    # Save threshold value to file
    threshold_info_path = os.path.join(processed_folder, 'saved', 'threshold_value.txt')
    with open(threshold_info_path, 'w') as f:
        f.write(f"Threshold Value: {threshold_value}\n")
    
    print(f"Binarized images saved to: {binarized_folder}")
    print(f"Threshold value saved to: {threshold_info_path}")

def main():
    # Get input folder
    input_folder = r"C:\Users\Daniel Meles\Box\experiments\experiment_20250729_1423\captures"
    if not input_folder:
        input_folder = '.'
    
    if not os.path.exists(input_folder):
        print(f"Error: Folder '{input_folder}' does not exist!")
        return
    
    # Get all image files to find one for crop selection
    image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    for pattern in image_patterns:
        image_files.extend(glob.glob(os.path.join(input_folder, pattern)))
        image_files.extend(glob.glob(os.path.join(input_folder, pattern.upper())))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for file in image_files:
        # Normalize path for comparison
        normalized = os.path.normpath(file).lower()
        if normalized not in seen:
            seen.add(normalized)
            unique_files.append(file)
    
    image_files = unique_files
    image_files.sort()
    
    if not image_files:
        print(f"No image files found in {input_folder}")
        return
    
    # Use the first image for crop selection
    print(f"\nUsing '{os.path.basename(image_files[0])}' for crop selection")
    crop_region = select_crop_region(image_files[0])
    
    if crop_region is None:
        print("Crop selection cancelled.")
        return
    
    print(f"Selected crop region: x={crop_region[0]}, y={crop_region[1]}, width={crop_region[2]}, height={crop_region[3]}")
    
    # Process all images
    processed_folder, last_cropped_path, processed_count = process_images(input_folder, crop_region)
    
    if processed_folder is None or last_cropped_path is None:
        print("Processing was cancelled or no images were processed.")
        return
    
    print(f"\nInitial processing complete! Successfully processed {processed_count} images.")
    print(f"All files saved to: {processed_folder}")
    
    # Select threshold on the last cropped image
    threshold_value = select_threshold_point(last_cropped_path)
    
    if threshold_value is None:
        print("Threshold selection cancelled. Using default Otsu's method for binarization...")
        # Apply Otsu's method to all images
        grayscale_folder = os.path.join(processed_folder, 'grayscale')
        binarized_folder = os.path.join(processed_folder, 'binarized')
        os.makedirs(binarized_folder, exist_ok=True)
        
        gray_images = glob.glob(os.path.join(grayscale_folder, '*'))
        for gray_path in gray_images:
            gray = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
            if gray is not None:
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                binary_path = os.path.join(binarized_folder, os.path.basename(gray_path))
                cv2.imwrite(binary_path, binary)
    else:
        # Apply selected threshold to all images
        apply_threshold_to_all(processed_folder, threshold_value)
    
    print(f"\nProcessing complete!")
    print(f"All processed files are in: {processed_folder}")
    print(f"  - Original images (renamed): {os.path.join(processed_folder, 'saved')}")
    print(f"  - Cropped images: {os.path.join(processed_folder, 'cropped')}")
    print(f"  - Grayscale images: {os.path.join(processed_folder, 'grayscale')}")
    print(f"  - Binarized images: {os.path.join(processed_folder, 'binarized')}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
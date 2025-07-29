import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

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

def list_available_cameras(max_index=10):
    """
    List all available cameras up to max_index.
    
    Args:
        max_index (int): Maximum camera index to check
        
    Returns:
        list: List of available camera indices
    """
    print("🔍 Scanning for available cameras...\n")
    available = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # Use CAP_DSHOW to avoid warning spam on Windows
        if cap.isOpened():
            print(f"✅ Camera found at index: {index}")
            available.append(index)
            cap.release()
        else:
            print(f"❌ No camera at index: {index}")
    if not available:
        print("\n⚠️ No cameras detected.")
    else:
        print(f"\n🎥 Total cameras detected: {len(available)}")
    return available

def select_crop_window(frame, window_name="Select ROI"):
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
    print("Select a region of interest and press SPACE or ENTER to confirm")
    r = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True)
   
    # Close the window
    cv2.destroyWindow(window_name)
   
    return r  # Returns (x, y, width, height)

def select_threshold(image, window_name="Threshold Selection"):
    """
    Interactive threshold selection for binarization.
    
    Args:
        image (numpy.ndarray): Grayscale image
        window_name (str): Name of the window
        
    Returns:
        int: Selected threshold value
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Create window and trackbar
    cv2.namedWindow(window_name)
    cv2.createTrackbar('Threshold', window_name, 127, 255, lambda x: None)
    
    print("Adjust the threshold using the trackbar. Press 'q' when done.")
    
    while True:
        # Get current trackbar position
        threshold = cv2.getTrackbarPos('Threshold', window_name)
        
        # Apply threshold
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Display the result
        cv2.imshow(window_name, binary)
        
        # Check for 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyWindow(window_name)
    return threshold

def process_image(image, crop_coords=None, threshold=None):
    """
    Process the image with cropping and binarization.
    
    Args:
        image (numpy.ndarray): Input image
        crop_coords (tuple): Crop coordinates (x, y, width, height)
        threshold (int): Threshold value for binarization
        
    Returns:
        dict: Dictionary containing processed images
    """
    result = {'original': image}
    
    # Apply crop if coordinates provided
    if crop_coords and crop_coords[2] > 0 and crop_coords[3] > 0:
        x, y, w, h = crop_coords
        cropped = image[y:y+h, x:x+w]
        result['cropped'] = cropped
        working_image = cropped
    else:
        working_image = image
    
    # Convert to grayscale
    if len(working_image.shape) == 3:
        gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = working_image
    result['grayscale'] = gray
    
    # Apply threshold if provided
    if threshold is not None:
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        result['binary'] = binary
    
    return result

def display_results(results):
    """
    Display all processed images using matplotlib.
    
    Args:
        results (dict): Dictionary of processed images
    """
    n_images = len(results)
    fig, axes = plt.subplots(1, n_images, figsize=(5*n_images, 5))
    
    if n_images == 1:
        axes = [axes]
    
    for idx, (name, img) in enumerate(results.items()):
        ax = axes[idx]
        
        # Handle color vs grayscale images
        if len(img.shape) == 3:
            # Convert BGR to RGB for matplotlib
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
        else:
            ax.imshow(img, cmap='gray')
        
        ax.set_title(name.capitalize())
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        # 1. List available cameras
        print("=== Step 1: Camera Detection ===")
        available_cameras = list_available_cameras()
        
        if not available_cameras:
            print("No cameras found. Exiting.")
            exit(1)
        
        # Select camera
        if len(available_cameras) == 1:
            camera_idx = available_cameras[0]
            print(f"\nUsing camera at index {camera_idx}")
        else:
            camera_idx = int(input(f"\nSelect camera index from {available_cameras}: "))
        
        # 2. Capture an image
        print("\n=== Step 2: Image Capture ===")
        input("Press Enter to capture an image...")
        
        frame = capture_image(camera_index=camera_idx, filename="captured_image.jpg")
        print("Image captured successfully!")
        
        # 3. Select crop window
        print("\n=== Step 3: Crop Selection ===")
        crop_coords = select_crop_window(frame)
        
        if crop_coords[2] > 0 and crop_coords[3] > 0:
            print(f"Selected region: x={crop_coords[0]}, y={crop_coords[1]}, "
                  f"width={crop_coords[2]}, height={crop_coords[3]}")
        else:
            print("No region selected, using full image")
            crop_coords = None
        
        # 4. Select binarization threshold
        print("\n=== Step 4: Threshold Selection ===")
        
        # Use cropped image if available for threshold selection
        if crop_coords and crop_coords[2] > 0 and crop_coords[3] > 0:
            x, y, w, h = crop_coords
            threshold_image = frame[y:y+h, x:x+w]
        else:
            threshold_image = frame
        
        threshold = select_threshold(threshold_image)
        print(f"Selected threshold: {threshold}")
        
        # 5. Process and display results
        print("\n=== Step 5: Processing Results ===")
        results = process_image(frame, crop_coords, threshold)
        
        # Save processed images
        for name, img in results.items():
            if name != 'original':
                filename = f"processed_{name}.jpg"
                cv2.imwrite(filename, img)
                print(f"Saved {filename}")
        
        # Display results
        display_results(results)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()
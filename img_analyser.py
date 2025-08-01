import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re

def extract_minute_from_filename(filename):
    """Extract the minute number from filename like 'capture_001.png'"""
    match = re.search(r'capture_(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def calculate_average_pixel_value(image_path):
    """Calculate the average pixel value of a grayscale image"""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return None
    return np.mean(img)

def calculate_white_black_ratio(image_path):
    """Calculate the ratio of white pixels to black pixels in a binary image"""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return None
    
    # Count white pixels (255) and black pixels (0)
    white_pixels = np.sum(img == 255)
    black_pixels = np.sum(img == 0)
    
    # Handle edge case where there are no black pixels
    if black_pixels == 0:
        return float('inf') if white_pixels > 0 else 0
    
    return white_pixels / black_pixels

def calculate_white_percentage(image_path):
    """Calculate the percentage of white pixels in a binary image"""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    total_pixels = img.size
    white_pixels = np.sum(img == 255)
    return (white_pixels / total_pixels) * 100

def read_processing_info(base_folder):
    """Read crop coordinates and threshold value from the processing info files"""
    info = {}
    
    # Try to read crop coordinates
    crop_file = os.path.join(base_folder, 'saved', 'crop_coordinates.txt')
    if os.path.exists(crop_file):
        try:
            with open(crop_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if 'X:' in line:
                        info['crop_x'] = int(line.split(':')[1].strip())
                    elif 'Y:' in line:
                        info['crop_y'] = int(line.split(':')[1].strip())
                    elif 'Width:' in line:
                        info['crop_width'] = int(line.split(':')[1].strip())
                    elif 'Height:' in line:
                        info['crop_height'] = int(line.split(':')[1].strip())
        except Exception as e:
            print(f"Could not read crop coordinates: {e}")
    
    # Try to read threshold value
    threshold_file = os.path.join(base_folder, 'saved', 'threshold_value.txt')
    if os.path.exists(threshold_file):
        try:
            with open(threshold_file, 'r') as f:
                line = f.readline()
                if 'Threshold Value:' in line:
                    info['threshold'] = int(line.split(':')[1].strip())
        except Exception as e:
            print(f"Could not read threshold value: {e}")
    
    return info

def process_images(base_folder="processed_captures"):
    """Process all images and return results"""
    base_path = Path(base_folder)
    grayscale_path = base_path / "grayscale"  # Fixed from "greyscale"
    binarized_path = base_path / "binarized"
    
    # Check if folders exist
    if not grayscale_path.exists():
        raise FileNotFoundError(f"Grayscale folder not found: {grayscale_path}")
    if not binarized_path.exists():
        raise FileNotFoundError(f"Binarized folder not found: {binarized_path}")
    
    # Read processing info if available
    processing_info = read_processing_info(base_folder)
    
    results = []
    
    # Get all image files (assuming common image extensions)
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
    
    grayscale_files = []
    binarized_files = []
    
    for ext in image_extensions:
        grayscale_files.extend(grayscale_path.glob(ext))
        grayscale_files.extend(grayscale_path.glob(ext.upper()))
        binarized_files.extend(binarized_path.glob(ext))
        binarized_files.extend(binarized_path.glob(ext.upper()))
    
    # Remove duplicates and sort
    grayscale_files = sorted(list(set(grayscale_files)))
    binarized_files = sorted(list(set(binarized_files)))
    
    print(f"Found {len(grayscale_files)} grayscale images")
    print(f"Found {len(binarized_files)} binarized images")
    
    if processing_info:
        print("\nProcessing information:")
        if 'crop_x' in processing_info:
            print(f"  Crop region: ({processing_info['crop_x']}, {processing_info['crop_y']}) - "
                  f"{processing_info['crop_width']}x{processing_info['crop_height']}")
        if 'threshold' in processing_info:
            print(f"  Threshold value: {processing_info['threshold']}")
    
    # Process grayscale images
    for grey_file in grayscale_files:
        minute = extract_minute_from_filename(grey_file.name)
        if minute is not None:
            avg_pixel = calculate_average_pixel_value(grey_file)
            if avg_pixel is not None:
                # Find corresponding binarized image
                bin_file = binarized_path / grey_file.name
                if bin_file.exists():
                    ratio = calculate_white_black_ratio(bin_file)
                    white_pct = calculate_white_percentage(bin_file)
                    results.append({
                        'minute': minute,
                        'filename': grey_file.name,
                        'avg_pixel_value': avg_pixel,
                        'white_black_ratio': ratio,
                        'white_percentage': white_pct
                    })
                else:
                    print(f"Warning: No matching binarized image for {grey_file.name}")
    
    return pd.DataFrame(results), processing_info

def save_results_to_csv(df, processing_info, filename="image_analysis_results.csv"):
    """Save the results DataFrame to CSV with metadata"""
    # Add processing info as comments at the top of the CSV
    with open(filename, 'w') as f:
        f.write("# Image Analysis Results\n")
        if processing_info:
            if 'crop_x' in processing_info:
                f.write(f"# Crop Region: ({processing_info['crop_x']}, {processing_info['crop_y']}) - "
                       f"{processing_info['crop_width']}x{processing_info['crop_height']}\n")
            if 'threshold' in processing_info:
                f.write(f"# Threshold Value: {processing_info['threshold']}\n")
        f.write("#\n")
    
    # Append the DataFrame
    df.to_csv(filename, mode='a', index=False)
    print(f"Results saved to {filename}")

def generate_plots(df):
    """Generate plots for average pixel value and white/black ratio"""
    # Sort by minute to ensure proper plotting
    df = df.sort_values('minute')
    
    # Create figure with three subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))
    
    # Plot 1: Average pixel value over time
    ax1.plot(df['minute'], df['avg_pixel_value'], 'b-o', markersize=6, linewidth=2)
    ax1.axvline(x=18, color='red', linestyle='--', linewidth=2, label='Aglomoration Visualy Detected')
    ax1.set_xlabel('Time (minutes)', fontsize=11)
    ax1.set_ylabel('Average Pixel Value (0-255)', fontsize=11)
    ax1.set_title('Average Pixel Value of Grayscale Images Over Time', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 50)
    
    # Add statistics to plot 1
    ax1.text(0.02, 0.98, f'Mean: {df["avg_pixel_value"].mean():.1f}\nStd: {df["avg_pixel_value"].std():.1f}', 
             transform=ax1.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: White to black ratio over time
    # Filter out infinite values for better visualization
    df_filtered = df[~np.isinf(df['white_black_ratio'])]
    
    ax2.plot(df_filtered['minute'], df_filtered['white_black_ratio'], 'r-o', markersize=6, linewidth=2)
    ax2.axvline(x=18, color='red', linestyle='--', linewidth=2, label='Aglomoration Visualy Detected')
    ax2.set_xlabel('Time (minutes)', fontsize=11)
    ax2.set_ylabel('White to Black Pixel Ratio', fontsize=11)
    ax2.set_title('White to Black Pixel Ratio of Binarized Images Over Time', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 0.1)

    
    # Mark any infinite values
    inf_points = df[np.isinf(df['white_black_ratio'])]
    if not inf_points.empty:
        ax2.scatter(inf_points['minute'], [ax2.get_ylim()[1]] * len(inf_points), 
                   color='red', marker='^', s=100, label='All white (inf)')
        ax2.legend()
    
    # Plot 3: White percentage over time
    #ax3.plot(df['minute'], df['white_percentage'], 'g-o', markersize=6, linewidth=2)
    #ax3.set_xlabel('Time (minutes)', fontsize=11)
    #ax3.set_ylabel('White Pixel Percentage (%)', fontsize=11)
    #ax3.set_title('Percentage of White Pixels in Binarized Images Over Time', fontsize=13)
    #ax3.grid(True, alpha=0.3)
    #ax3.set_ylim(0, 10)
    
    plt.tight_layout()
    plt.savefig('image_analysis_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create individual plots with more detail
    # Average pixel value plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['minute'], df['avg_pixel_value'], 'b-o', markersize=8, linewidth=2)
    plt.xlabel('Time (minutes)', fontsize=12)
    plt.ylabel('Average Pixel Value (0-255)', fontsize=12)
    plt.title('Average Pixel Value of Grayscale Images Over Time', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 255)
    
    # Add trend line
    z = np.polyfit(df['minute'], df['avg_pixel_value'], 1)
    p = np.poly1d(z)
    plt.plot(df['minute'], p(df['minute']), "b--", alpha=0.5, label=f'Trend: {z[0]:.2f}x + {z[1]:.1f}')
    plt.legend()
    
    plt.savefig('grayscale_avg_pixel_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # White to black ratio plot
    plt.figure(figsize=(10, 6))
    plt.plot(df_filtered['minute'], df_filtered['white_black_ratio'], 'r-o', markersize=8, linewidth=2)
    plt.xlabel('Time (minutes)', fontsize=12)
    plt.ylabel('White to Black Pixel Ratio', fontsize=12)
    plt.title('White to Black Pixel Ratio of Binarized Images Over Time', fontsize=14)
    plt.ylim(0, 0.1)
    plt.grid(True, alpha=0.3)
    
    # Mark infinite values if any
    if not inf_points.empty:
        plt.scatter(inf_points['minute'], [plt.ylim()[1]] * len(inf_points), 
                   color='red', marker='^', s=100, label='All white (inf)')
        plt.legend()
    
    plt.savefig('binarized_ratio_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # White percentage plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['minute'], df['white_percentage'], 'g-o', markersize=8, linewidth=2)
    plt.xlabel('Time (minutes)', fontsize=12)
    plt.ylabel('White Pixel Percentage (%)', fontsize=12)
    plt.title('Percentage of White Pixels in Binarized Images Over Time', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 10)
    
    # Add horizontal reference lines
    plt.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50%')
    plt.legend()
    
    plt.savefig('white_percentage_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run the analysis"""
    try:
        # Ask user for the processed folder path
        default_path = r"C:\Users\Daniel Meles\Box\experiments\experiment_20250729_1423\processed_captures"
      
        
        # Process images
        print(f"\nProcessing images from: {default_path}")
        df, processing_info = process_images(default_path)
        
        if df.empty:
            print("No images were processed successfully.")
            return
        
        # Save to CSV
        save_results_to_csv(df, processing_info)
        
        # Display results summary
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        print(f"Total images processed: {len(df)}")
        print(f"Time range: {df['minute'].min()} - {df['minute'].max()} minutes")
        print(f"\nGrayscale images:")
        print(f"  Average pixel value range: {df['avg_pixel_value'].min():.2f} - {df['avg_pixel_value'].max():.2f}")
        print(f"  Mean average pixel value: {df['avg_pixel_value'].mean():.2f} ± {df['avg_pixel_value'].std():.2f}")
        
        print(f"\nBinarized images:")
        df_no_inf = df[~np.isinf(df['white_black_ratio'])]
        if not df_no_inf.empty:
            print(f"  White/Black ratio range: {df_no_inf['white_black_ratio'].min():.3f} - {df_no_inf['white_black_ratio'].max():.3f}")
            print(f"  Images with all white pixels: {len(df[np.isinf(df['white_black_ratio'])])}")
        print(f"  White pixel percentage range: {df['white_percentage'].min():.1f}% - {df['white_percentage'].max():.1f}%")
        print(f"  Mean white percentage: {df['white_percentage'].mean():.1f}% ± {df['white_percentage'].std():.1f}%")
        
        # Show first few rows
        print(f"\nFirst 5 measurements:")
        print(df.head().to_string(index=False))
        
        # Generate plots
        print("\nGenerating plots...")
        generate_plots(df)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print("Files created:")
        print("  - image_analysis_results.csv (data with metadata)")
        print("  - image_analysis_plots.png (combined plots)")
        print("  - grayscale_avg_pixel_plot.png (with trend line)")
        print("  - binarized_ratio_plot.png")
        print("  - white_percentage_plot.png")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please check that the folder path is correct and contains 'grayscale' and 'binarized' subfolders.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
"""Debug multi-digit segmentation to see what's happening"""
import cv2
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

def test_segmentation(image_path):
    """Test the segmentation logic on a sample image"""
    # Load image
    img = Image.open(image_path).convert('L')
    img = ImageOps.invert(img)
    arr = np.array(img)
    
    print(f"Original image shape: {arr.shape}")
    print(f"Pixel value range: {arr.min()} to {arr.max()}")
    print(f"Mean pixel value: {arr.mean():.2f}")
    
    # Apply binary threshold
    _, thresh = cv2.threshold(arr, 50, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"\nFound {len(contours)} contours")
    
    # Filter and sort contours
    bounding_boxes = []
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        print(f"Contour {i}: x={x}, y={y}, w={w}, h={h}")
        if w > 10 and h > 15:
            bounding_boxes.append((x, y, w, h))
            print(f"  -> Kept (meets size criteria)")
        else:
            print(f"  -> Filtered out (too small)")
    
    bounding_boxes.sort(key=lambda b: b[0])
    print(f"\nKept {len(bounding_boxes)} digits after filtering")
    
    # Visualize
    fig, axes = plt.subplots(2, max(len(bounding_boxes), 1), figsize=(15, 6))
    if len(bounding_boxes) == 0:
        print("No digits detected!")
        return
    
    if len(bounding_boxes) == 1:
        axes = axes.reshape(-1, 1)
    
    # Show original with bounding boxes
    img_with_boxes = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    for x, y, w, h in bounding_boxes:
        cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Process each digit
    for idx, (x, y, w, h) in enumerate(bounding_boxes):
        # Extract digit with padding
        pad = 5
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(arr.shape[1], x + w + pad)
        y2 = min(arr.shape[0], y + h + pad)
        
        digit = arr[y1:y2, x1:x2]
        
        # Show extracted digit
        axes[0, idx].imshow(digit, cmap='gray')
        axes[0, idx].set_title(f'Digit {idx+1}\nRaw')
        axes[0, idx].axis('off')
        
        # Center and resize like MNIST
        h_digit, w_digit = digit.shape
        max_dim = max(h_digit, w_digit)
        square = np.zeros((max_dim, max_dim), dtype='uint8')
        y_offset = (max_dim - h_digit) // 2
        x_offset = (max_dim - w_digit) // 2
        square[y_offset:y_offset+h_digit, x_offset:x_offset+w_digit] = digit
        
        # Resize to 20x20
        resized = cv2.resize(square, (20, 20), interpolation=cv2.INTER_AREA)
        
        # Center in 28x28
        final = np.zeros((28, 28), dtype='float32')
        final[4:24, 4:24] = resized.astype('float32')
        
        # Show final preprocessed digit
        axes[1, idx].imshow(final, cmap='gray')
        axes[1, idx].set_title(f'Digit {idx+1}\nPreprocessed')
        axes[1, idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('multidigit_debug.png')
    print("\nSaved visualization to multidigit_debug.png")
    print("\nTo use this:")
    print("1. Draw digits in the web app")
    print("2. Save the canvas as an image")
    print("3. Run: python test_multidigit_segmentation.py path/to/your/image.png")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python test_multidigit_segmentation.py <image_path>")
        print("\nCreating a test image...")
        # Create a simple test image with digits
        test_img = np.zeros((150, 400), dtype='uint8')
        # Draw some simple digits (simplified)
        cv2.putText(test_img, '123', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, 255, 5)
        Image.fromarray(test_img).save('test_multidigit.png')
        print("Created test_multidigit.png")
        print("Running test on it...")
        test_segmentation('test_multidigit.png')
    else:
        test_segmentation(sys.argv[1])

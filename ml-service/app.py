import io, base64, os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)
# Enable CORS for all domains and routes
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/', methods=['GET'])
def index():
    return "Digit Prediction API is running. Use POST /predict or /multipredict."

MODEL_PATH = 'model/model.h5'
MODEL_URL = os.getenv('MODEL_URL', None)

# Auto-download model if URL is provided and model doesn't exist
if not os.path.exists(MODEL_PATH) and MODEL_URL:
    import urllib.request
    os.makedirs('model', exist_ok=True)
    print(f'Downloading model from {MODEL_URL}...')
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print('Model downloaded successfully')
    except Exception as e:
        print(f'Failed to download model: {e}')

# Load model
try:
    model = load_model(MODEL_PATH)
    print('Loaded model from', MODEL_PATH)
except Exception as e:
    model = None
    print('Warning: could not load model at startup. Run train_model.py to create model/model.h5. Error:', e)

def preprocess_image(data_url):
    """Preprocess single digit image to match MNIST format"""
    if data_url.startswith('data:'):
        header, data_url = data_url.split(',', 1)
    img_bytes = base64.b64decode(data_url)
    img = Image.open(io.BytesIO(img_bytes)).convert('L')
    # img = ImageOps.invert(img)  # REMOVED: MNIST is white on black, same as canvas
    
    # Convert to numpy array
    arr = np.array(img)
    
    # Find bounding box of the digit to center it like MNIST
    rows = np.any(arr > 30, axis=1)
    cols = np.any(arr > 30, axis=0)
    
    if not rows.any() or not cols.any():
        # No digit drawn, return blank
        arr = np.zeros((28, 28), dtype='float32')
        return arr.reshape(1, 28, 28, 1)
    
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    
    # Crop to bounding box with padding
    digit = arr[ymin:ymax+1, xmin:xmax+1]
    
    # Add padding (20% of max dimension)
    h, w = digit.shape
    max_dim = max(h, w)
    pad = int(max_dim * 0.2)
    digit = np.pad(digit, pad, mode='constant', constant_values=0)
    
    # Resize to 20x20 (like MNIST preprocessing)
    digit_img = Image.fromarray(digit.astype('uint8'))
    digit_img = digit_img.resize((20, 20), Image.LANCZOS)
    
    # Center in 28x28 image
    final = np.zeros((28, 28), dtype='float32')
    final[4:24, 4:24] = np.array(digit_img).astype('float32')
    
    # Normalize
    final = final / 255.0
    return final.reshape(1, 28, 28, 1)

def preprocess_and_segment(data_url):
    """Preprocess and segment multi-digit image with improved logic"""
    if data_url.startswith('data:'):
        header, data_url = data_url.split(',', 1)
    img_bytes = base64.b64decode(data_url)
    img = Image.open(io.BytesIO(img_bytes)).convert('L')
    # img = ImageOps.invert(img)  # REMOVED: MNIST is white on black
    arr = np.array(img)
    
    print(f"[DEBUG] Image shape: {arr.shape}, Min: {arr.min()}, Max: {arr.max()}, Mean: {arr.mean():.2f}")
    
    # Use adaptive thresholding for better segmentation
    # First try Otsu's method
    _, thresh = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    print(f"[DEBUG] After threshold - unique values: {np.unique(thresh)}")
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"[DEBUG] Found {len(contours)} contours")
    
    if not contours:
        print("[DEBUG] No contours found!")
        return []
    
    # Calculate image dimensions to set reasonable filters
    img_height, img_width = arr.shape
    min_digit_width = max(5, img_width // 50)  # At least 2% of image width
    min_digit_height = max(10, img_height // 20)  # At least 5% of image height
    
    print(f"[DEBUG] Min digit size: {min_digit_width}x{min_digit_height}")
    
    # Filter and sort contours by x-position
    bounding_boxes = []
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        print(f"[DEBUG] Contour {i}: x={x}, y={y}, w={w}, h={h}, area={cv2.contourArea(cnt)}")
        
        # Filter: reasonable size and aspect ratio
        if w >= min_digit_width and h >= min_digit_height:
            # Check aspect ratio (digits shouldn't be too wide or too tall)
            aspect_ratio = w / float(h)
            if 0.1 < aspect_ratio < 3.0:  # Reasonable digit proportions
                bounding_boxes.append((x, y, w, h))
                print(f"[DEBUG]   -> Kept (w={w}, h={h}, ratio={aspect_ratio:.2f})")
            else:
                print(f"[DEBUG]   -> Filtered out (bad aspect ratio: {aspect_ratio:.2f})")
        else:
            print(f"[DEBUG]   -> Filtered out (too small)")
    
    if not bounding_boxes:
        print("[DEBUG] No valid digits after filtering!")
        return []
    
    # Sort left to right
    bounding_boxes.sort(key=lambda b: b[0])
    
    print(f"[DEBUG] Processing {len(bounding_boxes)} digits")
    
    # Process each digit
    digits = []
    for idx, (x, y, w, h) in enumerate(bounding_boxes):
        # Extract digit with padding proportional to digit size
        pad = max(5, int(max(w, h) * 0.15))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(arr.shape[1], x + w + pad)
        y2 = min(arr.shape[0], y + h + pad)
        
        digit = arr[y1:y2, x1:x2]
        
        print(f"[DEBUG] Digit {idx}: extracted shape {digit.shape}, values {digit.min()}-{digit.max()}")
        
        # Preprocess like single digit (same as preprocess_image but without bounding box detection)
        # Find actual content within the extracted region
        rows = np.any(digit > 30, axis=1)
        cols = np.any(digit > 30, axis=0)
        
        if not rows.any() or not cols.any():
            print(f"[DEBUG] Digit {idx}: Empty after intensity check, skipping")
            continue
            
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        
        # Crop to actual digit
        digit = digit[ymin:ymax+1, xmin:xmax+1]
        
        # Add padding (20% of max dimension, same as single digit)
        h_digit, w_digit = digit.shape
        max_dim = max(h_digit, w_digit)
        pad_amount = int(max_dim * 0.2)
        digit = np.pad(digit, pad_amount, mode='constant', constant_values=0)
        
        # Resize to 20x20 (like MNIST preprocessing)
        digit_img = Image.fromarray(digit.astype('uint8'))
        digit_img = digit_img.resize((20, 20), Image.LANCZOS)
        
        # Center in 28x28 image
        final = np.zeros((28, 28), dtype='float32')
        final[4:24, 4:24] = np.array(digit_img).astype('float32')
        
        # Normalize
        final = final / 255.0
        
        print(f"[DEBUG] Digit {idx}: final shape {final.shape}, values {final.min():.3f}-{final.max():.3f}")
        
        digits.append(final.reshape(1, 28, 28, 1))
    
    print(f"[DEBUG] Returning {len(digits)} preprocessed digits")
    return digits

@app.route('/predict', methods=['POST'])
def predict():
    """Single digit prediction endpoint"""
    global model
    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded. Run train_model.py to create model/model.h5'}), 500
    data = request.get_json(force=True)
    if not data or 'image' not in data:
        return jsonify({'success': False, 'error': 'No image provided'}), 400
    try:
        x = preprocess_image(data['image'])
        preds = model.predict(x, verbose=0)[0]
        pred_class = int(preds.argmax())
        probs = [float(p) for p in preds]
        return jsonify({'success': True, 'prediction': pred_class, 'probabilities': probs})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/multipredict', methods=['POST'])
def multipredict():
    """Multi-digit prediction endpoint"""
    global model
    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 500
    data = request.get_json(force=True)
    if not data or 'image' not in data:
        return jsonify({'success': False, 'error': 'No image provided'}), 400
    try:
        print("\n[MULTIPREDICT] Starting multi-digit prediction")
        segments = preprocess_and_segment(data['image'])
        
        if not segments:
            print("[MULTIPREDICT] No digits detected after segmentation")
            return jsonify({
                'success': False, 
                'error': 'No digits found. Try:\n- Drawing digits larger\n- Adding more space between digits\n- Drawing with thicker lines\n- Making sure digits are clearly visible'
            }), 400
        
        print(f"[MULTIPREDICT] Predicting {len(segments)} digits")
        predictions = []
        probs_all = []
        
        for i, seg in enumerate(segments):
            preds = model.predict(seg, verbose=0)[0]
            pred_class = int(preds.argmax())
            confidence = float(preds[pred_class])
            predictions.append(pred_class)
            probs_all.append([float(p) for p in preds])
            print(f"[MULTIPREDICT] Digit {i+1}: predicted {pred_class} with confidence {confidence:.3f}")
        
        num_str = ''.join(str(d) for d in predictions)
        print(f"[MULTIPREDICT] Final prediction: {num_str}")
        
        return jsonify({
            'success': True, 
            'prediction': num_str, 
            'digits': predictions, 
            'probabilities': probs_all
        })
    except Exception as e:
        print(f"[MULTIPREDICT] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

import io, base64
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

MODEL_PATH = '../ml-service/model/model.h5'
try:
    model = load_model(MODEL_PATH)
    print('Loaded single-digit model from', MODEL_PATH)
except Exception as e:
    model = None
    print('Warning: could not load model. Train model first in ml-service. Error:', e)

def preprocess_and_segment(data_url):
    if data_url.startswith('data:'):
        header, data_url = data_url.split(',', 1)
    img_bytes = base64.b64decode(data_url)
    img = Image.open(io.BytesIO(img_bytes)).convert('L')
    img = ImageOps.invert(img)
    arr = np.array(img)
    h, w = arr.shape
    target_h = 100
    scale = target_h / float(h)
    new_w = max(100, int(w * scale))
    arr = cv2.resize(arr, (new_w, target_h))
    _, thresh = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digits = []
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 8 and h > 8:
            digit = arr[y:y+h, x:x+w]
            digit = cv2.resize(digit, (28,28))
            digit = digit.astype('float32') / 255.0
            digit = digit.reshape(1,28,28,1)
            digits.append(digit)
            boxes.append(x)
    sorted_digits = [d for _, d in sorted(zip(boxes, digits), key=lambda x:x[0])]
    return sorted_digits

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 500
    data = request.get_json(force=True)
    if not data or 'image' not in data:
        return jsonify({'success': False, 'error': 'No image provided'}), 400
    try:
        segments = preprocess_and_segment(data['image'])
        if not segments:
            return jsonify({'success': False, 'error': 'No digits found'}), 400
        predictions = []
        probs_all = []
        for seg in segments:
            preds = model.predict(seg)[0]
            predictions.append(int(preds.argmax()))
            probs_all.append([float(p) for p in preds])
        num_str = ''.join(str(d) for d in predictions)
        return jsonify({'success': True, 'prediction': num_str, 'digits': predictions, 'probabilities': probs_all})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

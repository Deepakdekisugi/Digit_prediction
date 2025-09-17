import io, base64
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'model/model.h5'
try:
    model = load_model(MODEL_PATH)
    print('Loaded model from', MODEL_PATH)
except Exception as e:
    model = None
    print('Warning: could not load model at startup. Run train_model.py to create model/model.h5. Error:', e)

def preprocess_image(data_url):
    if data_url.startswith('data:'):
        header, data_url = data_url.split(',', 1)
    img_bytes = base64.b64decode(data_url)
    img = Image.open(io.BytesIO(img_bytes)).convert('L')
    img = ImageOps.invert(img)
    img = img.resize((28,28), Image.ANTIALIAS)
    arr = np.array(img).astype('float32') / 255.0
    arr = arr.reshape(1,28,28,1)
    return arr

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded. Run train_model.py to create model/model.h5'}), 500
    data = request.get_json(force=True)
    if not data or 'image' not in data:
        return jsonify({'success': False, 'error': 'No image provided'}), 400
    try:
        x = preprocess_image(data['image'])
        preds = model.predict(x)[0]
        pred_class = int(preds.argmax())
        probs = [float(p) for p in preds]
        return jsonify({'success': True, 'prediction': pred_class, 'probabilities': probs})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

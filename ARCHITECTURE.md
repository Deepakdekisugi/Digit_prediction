# 🏗️ Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         USER BROWSER                         │
│                     https://your-app.vercel.app              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ HTTP/HTTPS
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    REACT FRONTEND                            │
│                      (Vercel CDN)                            │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │   App.js     │  │  Canvas.js   │  │ MultiCanvas.js  │   │
│  │   (Router)   │  │ (Single Dig) │  │ (Multi Digits)  │   │
│  └──────────────┘  └──────────────┘  └─────────────────┘   │
│              HTML5 Canvas Drawing Interface                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ POST /predict
                         │ POST /multipredict
                         │ GET /health
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  FLASK ML SERVICE                            │
│              https://your-app.onrender.com                   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                     app.py                              │ │
│  │  ┌────────────┐  ┌────────────┐  ┌─────────────────┐  │ │
│  │  │  /predict  │  │/multipredict│  │    /health      │  │ │
│  │  │   (POST)   │  │   (POST)    │  │     (GET)       │  │ │
│  │  └────────────┘  └────────────┘  └─────────────────┘  │ │
│  └───────────────────────┬─────────────────────────────────┘ │
│                          │                                    │
│                          ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │            TENSORFLOW KERAS MODEL                        │ │
│  │              model/model.h5                              │ │
│  │         (CNN: 28x28 grayscale → 10 classes)             │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  Dependencies: Flask, TensorFlow, OpenCV, NumPy, Pillow      │
└───────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Single Digit Prediction

```
1. User draws digit on canvas
   └─► Canvas.js captures drawing

2. Canvas converts to data URL
   └─► Base64 encoded image

3. POST to /predict endpoint
   └─► { "image": "data:image/png;base64,..." }

4. ML Service processes:
   a. Decode base64 → PIL Image
   b. Convert to grayscale
   c. Invert colors (white → black)
   d. Resize to 28x28
   e. Normalize (0-255 → 0-1)
   f. Reshape to (1, 28, 28, 1)

5. CNN Model predicts
   └─► Softmax output: [0.01, 0.05, ..., 0.87, ...]

6. Response to frontend
   └─► { "success": true, "prediction": 7, "probabilities": [...] }

7. Display result
   └─► "Prediction: 7" + probability chart
```

### Multi-Digit Prediction

```
1. User draws multiple digits
   └─► MultiCanvas.js captures drawing

2. Canvas converts to data URL
   └─► Base64 encoded image

3. POST to /multipredict endpoint
   └─► { "image": "data:image/png;base64,..." }

4. ML Service processes:
   a. Decode base64 → PIL Image
   b. Convert to grayscale + invert
   c. Resize height to 100px
   d. Apply Otsu thresholding
   e. Find contours (OpenCV)
   f. Sort contours left-to-right
   g. Extract each digit bounding box
   h. Resize each to 28x28
   i. Normalize and reshape

5. Predict each digit
   └─► Loop through segments
   └─► CNN predicts each: [7, 8, 3]

6. Response to frontend
   └─► { "success": true, "prediction": "783", "digits": [7,8,3], ... }

7. Display result
   └─► "You drew: 783"
```

---

## Component Responsibilities

### Frontend (`frontend/`)

**App.js**
- React Router setup
- Navigation between single/multi pages
- Main layout and styling

**Canvas.js**
- HTML5 Canvas for drawing single digit
- Mouse/touch event handlers
- Image preprocessing before API call
- Display prediction results

**MultiCanvas.js**
- HTML5 Canvas for drawing multiple digits
- Mouse/touch event handlers
- Image preprocessing before API call
- Display multi-digit results

**Environment Variables**
- `REACT_APP_ML_API_URL`: ML service endpoint

---

### ML Service (`ml-service/`)

**app.py**
- Flask web server
- CORS configuration
- Model loading (with auto-download)
- Image preprocessing functions
- Prediction endpoints
- Error handling

**train_model.py**
- Loads MNIST dataset
- Defines CNN architecture:
  - Conv2D → MaxPooling → Conv2D → MaxPooling
  - Flatten → Dense(128) → Dropout → Dense(10)
- Trains on 60,000 images
- Validates on 10,000 images
- Saves to `model/model.h5`

**requirements.txt**
- Flask: Web framework
- TensorFlow: Deep learning
- OpenCV: Image processing (segmentation)
- NumPy: Array operations
- Pillow: Image manipulation
- Flask-CORS: Cross-origin requests
- Gunicorn: Production WSGI server

**Environment Variables**
- `MODEL_URL`: (Optional) URL to download model

---

## Model Architecture

```
Input: 28x28x1 grayscale image
  │
  ▼
Conv2D (32 filters, 3x3, ReLU)
  │
  ▼
MaxPooling2D (2x2)
  │
  ▼
Conv2D (64 filters, 3x3, ReLU)
  │
  ▼
MaxPooling2D (2x2)
  │
  ▼
Flatten
  │
  ▼
Dense (128 units, ReLU)
  │
  ▼
Dropout (0.5)
  │
  ▼
Dense (10 units, Softmax)
  │
  ▼
Output: Probabilities for digits 0-9
```

**Training:**
- Dataset: MNIST (70,000 images)
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Accuracy: ~98-99%
- Epochs: 5

---

## Deployment Architecture

### Free Tier Setup

```
┌──────────────────────────────────────────────────────────┐
│                     VERCEL CDN                           │
│                  (Global Edge Network)                   │
│  ┌────────────────────────────────────────────────────┐ │
│  │         React App (Static Build)                   │ │
│  │  • HTML, CSS, JS bundled by Webpack               │ │
│  │  • Served from 100+ edge locations                │ │
│  │  • Free HTTPS with auto-renewal                   │ │
│  │  • 100GB bandwidth/month                          │ │
│  └────────────────────────────────────────────────────┘ │
└────────────────────────┬─────────────────────────────────┘
                         │
                         │ API Calls
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│               RENDER WEB SERVICE                         │
│            (Containerized Python App)                    │
│  ┌────────────────────────────────────────────────────┐ │
│  │     Flask App + TensorFlow + Model                 │ │
│  │  • Runs in Docker container                       │ │
│  │  • 512MB RAM (free tier)                          │ │
│  │  • Sleeps after 15min inactivity                  │ │
│  │  • Cold start: ~30-60 seconds                     │ │
│  │  • Free HTTPS with auto-renewal                   │ │
│  │  • 750 hours/month                                │ │
│  └────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

### Production Considerations

**Scaling:**
- Frontend: Vercel auto-scales (CDN)
- ML Service: Upgrade Render plan for:
  - More RAM (4GB+)
  - No sleep (always-on)
  - Faster CPU
  - Multiple instances

**Monitoring:**
- Vercel: Built-in analytics
- Render: Log streaming
- External: Sentry, LogRocket

**Performance:**
- Frontend: ~50ms (CDN latency)
- ML Service: ~200-500ms (prediction time)
- Cold start: ~30-60s (free tier only)

---

## API Specification

### GET /health
**Description:** Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### POST /predict
**Description:** Predict single digit

**Request:**
```json
{
  "image": "data:image/png;base64,iVBORw0KGgoAAAANSUh..."
}
```

**Response:**
```json
{
  "success": true,
  "prediction": 7,
  "probabilities": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.87, 0.09, 0.00]
}
```

### POST /multipredict
**Description:** Predict multiple digits

**Request:**
```json
{
  "image": "data:image/png;base64,iVBORw0KGgoAAAANSUh..."
}
```

**Response:**
```json
{
  "success": true,
  "prediction": "783",
  "digits": [7, 8, 3],
  "probabilities": [
    [0.01, ..., 0.89, ...],
    [0.02, ..., 0.92, ...],
    [0.03, ..., 0.85, ...]
  ]
}
```

---

## Technology Stack Summary

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | React 18 | UI framework |
| | React Router | Client-side routing |
| | HTML5 Canvas | Drawing interface |
| | Fetch API | HTTP requests |
| **ML Service** | Flask | Web framework |
| | TensorFlow | Deep learning |
| | Keras | Model API |
| | OpenCV | Image segmentation |
| | NumPy | Numerical operations |
| | Pillow | Image processing |
| **Deployment** | Vercel | Frontend hosting |
| | Render | Backend hosting |
| | Git | Version control |
| | GitHub | Code repository |

---

## Security Considerations

✅ **HTTPS Only**: Both Vercel and Render provide automatic HTTPS
✅ **CORS Configured**: Only allows necessary cross-origin requests
✅ **No Auth Required**: Public demo app (appropriate for portfolio)
✅ **Input Validation**: API validates image data before processing
✅ **Error Handling**: Graceful error messages, no sensitive info exposed

**For Production:**
- Add rate limiting (e.g., Flask-Limiter)
- Implement API authentication (e.g., JWT)
- Add request logging and monitoring
- Set up WAF (Web Application Firewall)

---

**This architecture provides a clean, scalable, and free-to-deploy solution for your digit prediction app!**

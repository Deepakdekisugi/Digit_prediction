# 🎨 Digit Prediction - Handwritten Digit Recognition

A full-stack machine learning web app that recognizes handwritten digits using deep learning.

## ✨ Features

- **Single Digit Recognition** - Draw and predict individual digits (0-9)
- **Multi-Digit Recognition** - Draw and predict entire numbers
- **Real-time Predictions** - Instant feedback with probability distributions
- **Modern UI** - Clean, responsive React interface

## 🏗️ Architecture

```
Frontend (React) → ML Service (Flask + TensorFlow)
```

- **Frontend**: React app with canvas drawing interface
- **ML Service**: Unified Flask API serving a trained CNN model for both single and multi-digit predictions

## 🚀 Quick Start

### Prerequisites
- Node.js 16+ and npm
- Python 3.8+
- Git

### Local Development

1. **Train the Model** (first time only):
```bash
cd ml-service
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
python train_model.py  # Creates model/model.h5
```

2. **Start ML Service**:
```bash
cd ml-service
venv\Scripts\activate
python app.py  # Runs on http://localhost:5000
```

3. **Start Frontend** (new terminal):
```bash
cd frontend
npm install
npm start  # Runs on http://localhost:3000
```

4. Open http://localhost:3000 in your browser

## 🌐 Deploy for Free

See [host.md](host.md) for comprehensive deployment guide or [DEPLOY.md](DEPLOY.md) for quick commands.

**Recommended Platforms:**
- **Frontend**: Vercel (free, automatic HTTPS, global CDN)
- **ML Service**: Render (free tier with 750 hours/month)

## 📦 Tech Stack

### Frontend
- React 18
- React Router
- HTML5 Canvas API

### ML Service
- Flask (API framework)
- TensorFlow (deep learning)
- OpenCV (image processing)
- NumPy, Pillow (image manipulation)

## 📝 API Endpoints

- `GET /health` - Health check
- `POST /predict` - Single digit prediction
- `POST /multipredict` - Multi-digit prediction

## 🤝 Contributing

Contributions welcome! Open an issue or submit a PR.

## 📄 License

MIT License

## 🎯 Future Improvements

- [ ] Add confidence threshold
- [ ] Support for mathematical expressions
- [ ] Mobile app version
- [ ] Model fine-tuning on custom datasets
- [ ] Real-time collaborative drawing

---

Built with ❤️ using React and TensorFlow
   ```

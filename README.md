# 🎨 Digit Prediction - Handwritten Digit Recognition

> A full-stack machine learning web app that recognizes handwritten digits (0-9) and multi-digit sequences using deep learning.

## ✨ Features

- **✍️ Single Digit Recognition**: Draw any digit from 0-9 and get instant predictions.
- **🔢 Multi-Digit Recognition**: Scribble multiple digits in a row and watch the model decode the sequence.
- **⚡ Real-time Feedback**: View prediction probabilities and confidence scores.
- **📱 Responsive Design**: Modern, clean UI built with React.

## 🏗️ Architecture

The application follows a decoupled client-server architecture:

- **Frontend**: [React](https://reactjs.org/) application with HTML5 Canvas for drawing.
- **Backend**: [Flask](https://flask.palletsprojects.com/) API serving a trained TensorFlow/Keras CNN model.

## 🚀 Quick Start

### Prerequisites
- Node.js (v14+)
- Python (v3.8+)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/Deepakdekisugi/Digit_prediction.git
cd Digit_prediction
```

### 2. Setup Backend (ML Service)
```bash
cd ml-service
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
# source venv/bin/activate

pip install -r requirements.txt
python train_model.py  # Train the model (creates model/model.h5)
python app.py          # Start the server on http://localhost:5000
```

### 3. Setup Frontend
Open a new terminal:
```bash
cd frontend
npm install
npm start              # Start the React app on http://localhost:3000
```

## 🌐 Deployment

For detailed deployment instructions on **Render** (Backend) and **Vercel** (Frontend), please refer to [DEPLOY.md](DEPLOY.md).

## 📦 Tech Stack

- **Frontend**: React, React Router, Canvas API, CSS3
- **Backend**: Flask, TensorFlow 2.x, OpenCV, NumPy, Pillow
- **Tools**: Git, npm, pip

## 🤝 Contributing

Contributions are welcome!
1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License.

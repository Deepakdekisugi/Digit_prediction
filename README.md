# Digit Recognizer (Single-digit + Multi-digit) - MERN + Python ML Services (UNTRAINED)

This repo contains an untrained, ready-to-run project that separates the ML services from the MERN app.
You must train the ML model locally (instructions below) to create `ml-service/model/model.h5` before running the multi-digit service.

Structure:
- ml-service/      : Flask single-digit MNIST CNN API (port 5000) + train_model.py
- ml-multidigit/   : Flask multi-digit segmentation API (port 5001) that uses ml-service/model.h5
- node-backend/    : Express server exposing /api/predict and /api/multipredict (port 4000)
- react-fronted/   : React app with routes / and /multi

## Quick Setup (UNTRAINED)
Prereqs: Node.js (16+), Python 3.8+, pip, (optional) virtualenv

1. Train model (creates ml-service/model/model.h5)
   ```bash
   cd ml-service
   python -m venv venv
   # activate venv:
   # Linux/macOS: source venv/bin/activate
   # Windows (PowerShell): venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   python train_model.py   # creates model/model.h5
   ```

2. Start ML API (single-digit)
   ```bash
   cd ml-service
   source venv/bin/activate
   python app.py
   # listens: http://localhost:5000
   ```

3. Start Multi-digit API
   ```bash
   cd ml-multidigit
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python app.py
   # listens: http://localhost:5001 (uses ../ml-service/model/model.h5)
   ```

4. Start Node backend (proxy)
   ```bash
   cd node-backend
   npm install
   node index.js
   # listens: http://localhost:4000
   ```

5. Start React frontend
   ```bash
   cd react-frontend
   npm install
   npm start
   # dev server: http://localhost:3000
   ```

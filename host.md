# 🚀 Free Hosting Guide - Digit Prediction Project

Deploy your digit prediction app (React frontend + Python ML service) completely free using modern cloud platforms.

---

## 📁 Project Structure
```
Digit_prediction/
├── frontend/          # React app (single & multi-digit UI)
│   ├── src/
│   │   ├── App.js
│   │   ├── Canvas.js       # Single digit predictor
│   │   └── MultiCanvas.js  # Multi-digit predictor
│   └── package.json
└── ml-service/        # Flask ML API
    ├── app.py              # /predict & /multipredict endpoints
    ├── train_model.py      # Model training script
    └── requirements.txt
```

**Clean Architecture:** Direct frontend → ML service communication (no intermediate Node.js server needed).

---

## ✅ Prerequisites
- GitHub account
- Free accounts on **Vercel** (frontend) and **Render** (ML service)
- Code pushed to a public GitHub repository
- Trained model file (`model/model.h5`)

---

## 🎨 Part 1: Deploy Frontend (React)

### Platform: **Vercel** (Recommended)
**Why:** Instant deployment, automatic HTTPS, perfect for React apps.

### Steps:
1. **Push to GitHub** - Ensure your code is committed and pushed
2. **Sign up at [vercel.com](https://vercel.com)** - Use your GitHub account
3. **Import Repository**:
   - Click "New Project"
   - Select your `Digit_prediction` repository
   - Set **Root Directory**: `frontend`
4. **Configure Build Settings**:
   - Framework Preset: `Create React App`
   - Build Command: `npm run build`
   - Output Directory: `build`
5. **Set Environment Variable**:
   - Key: `REACT_APP_ML_API_URL`
   - Value: (wait for ML service URL in Part 2, then add it here)
6. **Deploy** - Click "Deploy"

### After ML Service is Deployed:
1. Go to your Vercel project settings
2. Add environment variable: `REACT_APP_ML_API_URL` = `https://your-ml-service.onrender.com`
3. Redeploy

**Alternative:** Netlify, Cloudflare Pages, or GitHub Pages (requires manual build)

---

## 🤖 Part 2: Deploy ML Service (Flask)

### Platform: **Render** (Recommended)
**Why:** Free tier, Python support, persistent disk for models, auto-deploy from GitHub.

### Steps:
1. **Upload Model to Cloud** (since `model/` is git-ignored):
   - Option A: Upload `model.h5` to [Hugging Face Hub](https://huggingface.co/)
   - Option B: Use Google Drive or Dropbox with a direct download link
   - Option C: Commit a small model for demo purposes (if <100MB)

2. **Sign up at [render.com](https://render.com)**

3. **Create New Web Service**:
   - Connect your GitHub repository
   - Name: `digit-prediction-ml`
   - Root Directory: `ml-service`
   - Environment: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`

4. **Add to requirements.txt** (if not present):
   ```
   gunicorn
   ```

5. **Environment Variables** (Optional):
   - If downloading model from URL, add:
     - `MODEL_URL` = your model download URL

6. **Deploy** - Render will build and start your service

7. **Get Your Service URL**: `https://digit-prediction-ml.onrender.com`

8. **Update Frontend**: Go back to Vercel and add this URL as `REACT_APP_ML_API_URL`

### Model Download Code (Optional)
If hosting model externally, add this to `app.py` before model loading:

```python
import os
import urllib.request

MODEL_PATH = 'model/model.h5'
MODEL_URL = os.getenv('MODEL_URL', None)

if not os.path.exists(MODEL_PATH) and MODEL_URL:
    os.makedirs('model', exist_ok=True)
    print(f'Downloading model from {MODEL_URL}...')
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print('Model downloaded successfully')
```

---

## 🧪 Part 3: Test Your Deployment

1. Visit your Vercel URL (e.g., `https://your-app.vercel.app`)
2. Test single-digit recognition
3. Test multi-digit recognition
4. Check browser console for any CORS or API errors

**Common Issues:**
- **CORS Error**: Ensure `flask-cors` is installed and `CORS(app)` is in `app.py`
- **Model Not Found**: Check Render logs, ensure model is downloaded or committed
- **Cold Start**: First request may take 30-60 seconds on free tier

---

## 💰 Free Tier Limits

| Platform | Limits |
|----------|--------|
| **Vercel** | 100GB bandwidth/month, unlimited deploys |
| **Render** | 750 hours/month, sleeps after 15min inactivity, 512MB RAM |

**Important:** ML service will "sleep" after inactivity. First request will take 30-60 seconds (cold start).

---

## 🔧 Alternative Platforms

### For ML Service:
- **Railway** - Similar to Render, $5 free credit/month
- **Fly.io** - 3 VMs free, better performance
- **Hugging Face Spaces** - Best for demo/showcase, requires Gradio/Streamlit UI
- **PythonAnywhere** - Free tier available, manual setup

### For Frontend:
- **Netlify** - Similar to Vercel
- **Cloudflare Pages** - Unlimited bandwidth
- **GitHub Pages** - Free, but manual build setup

---

## 📝 Quick Reference

### Local Development:
```bash
# Terminal 1 - ML Service
cd ml-service
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
python train_model.py  # Train model first
python app.py          # Runs on http://localhost:5000

# Terminal 2 - Frontend
cd frontend
npm install
npm start              # Runs on http://localhost:3000
```

### Environment Variables:
- **Frontend**: `REACT_APP_ML_API_URL` - URL of your ML service
- **ML Service**: `MODEL_URL` (optional) - URL to download model

### API Endpoints:
- `GET /health` - Check if ML service is running
- `POST /predict` - Single digit prediction
- `POST /multipredict` - Multi-digit prediction

---

## 🎯 Deployment Checklist

- [ ] Train model locally (`python train_model.py`)
- [ ] Upload model to cloud storage or commit small model
- [ ] Push code to GitHub
- [ ] Deploy ML service to Render
- [ ] Get ML service URL
- [ ] Deploy frontend to Vercel with `REACT_APP_ML_API_URL` set
- [ ] Test both single and multi-digit predictions
- [ ] Share your app! 🎉

---

## 🆘 Troubleshooting

**Frontend can't reach ML service:**
- Check `REACT_APP_ML_API_URL` is set correctly in Vercel
- Ensure ML service is not sleeping (visit `/health` endpoint)
- Check browser console for CORS errors

**Model not loading:**
- Check Render logs for errors
- Verify model file exists and is compatible with TensorFlow version
- Ensure model path is correct (`model/model.h5`)

**Render service timing out:**
- Free tier has 512MB RAM limit
- TensorFlow model may be too large
- Consider using TensorFlow Lite for smaller models

---

## 🌟 Production Tips

For a production deployment:
1. Use a paid tier for no cold starts
2. Add model versioning
3. Implement API rate limiting
4. Add user authentication
5. Use CDN for faster global access
6. Monitor with services like Sentry or LogRocket

---

**Questions? Open an issue on GitHub!**

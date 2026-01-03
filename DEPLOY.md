# 🚀 Deployment Guide

This guide covers how to deploy the **Digit Prediction** application. The project consists of two parts:
1.  **Frontend**: React application
2.  **Backend**: Flask ML Service

## 1. Backend Deployment (ML Service)

We recommend **Render** or **Railway** for the backend as they support Python/Flask and managing dependencies easily.

### Option A: Render (Free Tier Available)

1.  Create a [Render account](https://render.com/).
2.  Click **New +** -> **Web Service**.
3.  Connect your GitHub repository.
4.  Configure the service:
    *   **Root Directory**: `ml-service`
    *   **Runtime**: Python 3
    *   **Build Command**: `pip install -r requirements.txt`
    *   **Start Command**: `gunicorn --workers 1 --threads 1 --timeout 120 app:app`
5.  **Environment Variables**:
    *   Add `PYTHON_VERSION`: `3.10.0` (optional, but good for stability)
6.  Click **Create Web Service**.
7.  Copy the **Service URL** (e.g., `https://digit-prediction-api.onrender.com`). You will need this for the frontend.

### Option B: Heroku

1.  Install Heroku CLI and login: `heroku login`.
2.  Create an app:
    ```bash
    cd ml-service
    heroku create digit-prediction-api
    ```
3.  Deploy:
    ```bash
    git subtree push --prefix ml-service heroku main
    ```

---

## 2. Frontend Deployment

We recommend **Vercel** or **Netlify** for the React frontend.

### Option A: Vercel (Recommended)

1.  Create a [Vercel account](https://vercel.com/).
2.  Install Vercel CLI: `npm i -g vercel` or stick to the web UI.
3.  **Web UI Method**:
    *   Click **Add New...** -> **Project**.
    *   Import your GitHub repository.
    *   Configure the project:
        *   **Framework Preset**: Create React App
        *   **Root Directory**: `frontend`
    *   **Environment Variables**:
        *   `REACT_APP_API_URL`: Paste your Backend Service URL (e.g., `https://digit-prediction-api.onrender.com`)
    *   Click **Deploy**.

### Option B: Netlify

1.  Create a [Netlify account](https://www.netlify.com/).
2.  Click **Add new site** -> **Import from existing project**.
3.  Connect GitHub and select your repo.
4.  Configure:
    *   **Base directory**: `frontend`
    *   **Build command**: `npm run build`
    *   **Publish directory**: `build`
5.  **Environment Variables**:
    *   Click **Show advanced**.
    *   Key: `REACT_APP_API_URL`
    *   Value: Your Backend URL.
6.  Click **Deploy site**.

---

## 🏗️ Local Production Build

To run the production build locally:

1.  **Backend**:
    ```bash
    cd ml-service
    pip install gunicorn
    gunicorn app:app
    ```
2.  **Frontend**:
    ```bash
    cd frontend
    npm run build
    npx serve -s build
    ```

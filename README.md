# 🚦 Traffic Sign Image Classification

A full-stack ML web app that classifies traffic signs in real time using a CNN trained on the GTSRB dataset.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)
![React](https://img.shields.io/badge/React-18-61dafb)
![TailwindCSS](https://img.shields.io/badge/Tailwind-v4-38bdf8)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

## Features
- Drag & drop image upload
- Top-5 predictions with confidence bars
- Grad-CAM heatmap — see what the model focused on
- 43 traffic sign classes · ~95% validation accuracy

## Tech Stack
| Layer | Technology |
|-------|-----------|
| Frontend | React 18 + Vite + Tailwind CSS v4 |
| Backend | FastAPI + Uvicorn |
| ML | TensorFlow / Keras CNN |
| Dataset | GTSRB (German Traffic Sign Recognition Benchmark) |

## Project Structure
```
Traffic-Sign-Image-Classification/
├── backend/
│   ├── app/
│   │   ├── main.py       # FastAPI app + CORS
│   │   ├── model.py      # Model loading + Grad-CAM
│   │   └── predict.py    # /predict endpoint
│   ├── models/           # Saved model (after training)
│   ├── data/             # GTSRB dataset (not in git)
│   └── train.py          # Model training script
└── frontend/
└── src/
├── App.jsx
├── api.js
└── components/
├── UploadZone.jsx
└── ResultPanel.jsx
```

## Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/rajeshM77/Traffic-Sign-Image-Classification.git
cd Traffic-Sign-Image-Classification
```

### 2. Download the dataset
Download GTSRB from Kaggle:
https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

Extract into `backend/data/gtsrb/` so the structure is:

```
backend/data/gtsrb/
├── Train/   (43 subfolders 0–42)
├── Test/
└── Meta/
```

### 3. Backend setup
```bash
cd backend
python -m venv venv
source venv/Scripts/activate   # Windows Git Bash
# source venv/bin/activate     # Mac / Linux

pip install -r requirements.txt
```

### 4. Train the model
```bash
# Inside backend/ with venv active
python train.py
# Takes 15–30 min on CPU
# Saves model to models/traffic_sign_model.h5
```

### 5. Run the backend
```bash
uvicorn app.main:app --reload --port 8000
# API docs available at http://localhost:8000/docs
```

### 6. Frontend setup
```bash
cd ../frontend
npm install
npm run dev
# Opens at http://localhost:5173
```

## Usage
1. Open `http://localhost:5173`
2. Drop any traffic sign image onto the upload zone
3. See the predicted sign name, confidence score, top-5 chart, and Grad-CAM heatmap

## Dataset
GTSRB — German Traffic Sign Recognition Benchmark  
43 classes · 50,000+ images  
Source: https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

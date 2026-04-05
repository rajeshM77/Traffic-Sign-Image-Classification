# Traffic Sign Image Classification

A full-stack ML web app that classifies traffic signs using a CNN trained on the GTSRB dataset.

## Tech Stack
- **Frontend** — React + Vite + Tailwind CSS
- **Backend** — FastAPI + Python
- **ML Model** — TensorFlow / Keras CNN (43 classes, ~95% accuracy)

## Features
- Drag & drop image upload
- Live webcam classification
- Grad-CAM heatmap visualization
- Top-5 confidence chart
- Prediction history log

## Project Structure

Traffic-Sign-Image-Classification/
├── backend/        # FastAPI + TensorFlow
│   ├── app/        # API routes and model logic
│   ├── data/       # GTSRB dataset (not committed)
│   └── models/     # Saved model (not committed)
└── frontend/       # React + Vite
└── src/        # Components

## Setup
See SETUP.md for full instructions.

## Dataset
GTSRB — German Traffic Sign Recognition Benchmark
43 classes, 50,000+ images
Source: https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
EOF

# Titanic Survival Prediction - Machine Learning

Project Machine Learning untuk memprediksi keselamatan penumpang Titanic
menggunakan FastAPI dan model Machine Learning.

## Tools
- Python
- FastAPI
- Scikit-learn
- Uvicorn

## Cara Menjalankan Project
1. Aktifkan virtual environment
2. Install dependency:
   pip install -r requirements.txt
3. Jalankan server:
   uvicorn main:app --reload
4. Buka browser:
   http://127.0.0.1:8000

## Endpoint
- GET /
- POST /predict
- GET /docs

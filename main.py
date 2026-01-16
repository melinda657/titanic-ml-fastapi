# ========== IMPORT DAN SETUP ==========
# Mengimpor library yang diperlukan untuk API
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import joblib
import numpy as np
import pandas as pd

# Memuat model dan encoder yang telah dilatih
model = joblib.load('titanic_model.pkl')
le_embarked = joblib.load('embarked_encoder.pkl')

print(f"Model expects {model.n_features_in_} features")

# Membuat aplikasi FastAPI
app = FastAPI()

from fastapi.responses import FileResponse

@app.get("/")
async def read_root():
    return FileResponse("index.html")

# Mengatur CORS untuk mengizinkan akses dari semua origin
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== MODEL INPUT DATA ==========
# Mendefinisikan struktur data input untuk prediksi survival penumpang
class PassengerData(BaseModel):
    pclass: int
    name: str  # Nama penumpang untuk ekstrak title
    sex: str  # 'male' or 'female'
    age: float
    sibsp: int
    parch: int
    fare: float
    embarked: str  # 'C', 'Q', 'S'

# ========== FUNGSI PEMBUAT FITUR ==========
# Fungsi bantu untuk membuat fitur dari data input pengguna
def create_features(data: PassengerData):
    # Encode Sex
    sex_encoded = 0 if data.sex.lower() == 'male' else 1

    # Family Size
    family_size = data.sibsp + data.parch + 1

    # Is Alone
    is_alone = 1 if family_size == 1 else 0

    # Fare per person
    fare_per_person = data.fare / family_size

    # Age Group
    if data.age <= 12:
        age_group = 0
    elif data.age <= 18:
        age_group = 1
    elif data.age <= 35:
        age_group = 2
    elif data.age <= 60:
        age_group = 3
    else:
        age_group = 4

    # Title from Name
    title = data.name
    if 'Mr.' in title:
        title_encoded = 1
    elif 'Miss.' in title or 'Ms.' in title or 'Mlle.' in title:
        title_encoded = 2
    elif 'Mrs.' in title or 'Mme.' in title:
        title_encoded = 3
    elif 'Master.' in title:
        title_encoded = 4
    else:
        title_encoded = 5  # Rare or others

    # Encode Embarked
    try:
        embarked_encoded = le_embarked.transform([data.embarked])[0]
    except:
        embarked_encoded = 0  # Default

    return np.array([[data.pclass, sex_encoded, data.age, data.sibsp, data.parch, data.fare,
                      embarked_encoded, family_size, is_alone, fare_per_person, age_group, title_encoded]])

# ========== ENDPOINT PREDIKSI ==========
# Mendefinisikan route untuk melakukan prediksi
@app.post("/predict")
def predict_passenger_survival(data: PassengerData):
    # Membuat fitur dari data input
    input_features = create_features(data)

    # Melakukan prediksi
    prediction = model.predict(input_features)[0]
    prediction_proba = model.predict_proba(input_features)[0]

    # Mapping ke bahasa Indonesia
    survival_mapping = {
        0: "Tidak Selamat",
        1: "Selamat"
    }
    predicted_survival = survival_mapping.get(prediction, f"Unknown ({prediction})")

    # Membuat dictionary probabilitas
    prob_dict = {
        "Tidak Selamat": float(prediction_proba[0]),
        "Selamat": float(prediction_proba[1])
    }

    # Mengembalikan hasil prediksi sebagai JSON
    return {
        "predicted_survival": predicted_survival,
        "prediction_probabilities": prob_dict
    }

# ========== CARA MENJALANKAN SERVER ==========
# Jalankan server dengan perintah: uvicorn main:app --reload
# Ini akan memulai server FastAPI pada http://127.0.0.1:8000
# Dokumentasi API tersedia di http://127.0.0.1:8000/docs

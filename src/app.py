
import io
import json
import os
import time
from typing import Annotated

import torch
import torchaudio
from fastapi import FastAPI, File, HTTPException, UploadFile

from src.data import preprocess_audio
from src.model import AudioCNN

app = FastAPI(
    title="Speech Commands API",
    description="API для класифікації голосових команд",
    version="1.0.0"
)

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.pt")
LABELS_PATH = os.getenv("LABELS_PATH", "artifacts/labels.json")
METRICS_PATH = os.getenv("METRICS_PATH", "artifacts/metrics.json")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")

device = torch.device("cpu")

model = None
classes = []
metrics = {}


@app.on_event("startup")
def load_model():
    global model, classes, metrics

    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        classes = json.load(file)

    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    model = AudioCNN(num_classes=len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    print("Модель успішно завантажена", flush=True)
    print("Класи:", classes, flush=True)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.get("/model-info")
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Модель ще не завантажена")

    return {
        "model_name": "AudioCNN",
        "model_version": MODEL_VERSION,
        "classes": classes,
        "accuracy": round(metrics.get("accuracy", 0) * 100, 2)
    }


@app.post("/predict")
async def predict(file: Annotated[UploadFile, File()]):
    if model is None:
        raise HTTPException(status_code=503, detail="Модель ще не завантажена")

    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Потрібен WAV-файл")

    audio_bytes = await file.read()

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Файл порожній")

    start_time = time.perf_counter()

    try:
        waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
        spectrogram = preprocess_audio(waveform, sample_rate)
    except (RuntimeError, ValueError, OSError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Не вдалося прочитати WAV-файл"
        ) from exc

    input_tensor = spectrogram.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_index = torch.max(probabilities, dim=1)

    latency_ms = (time.perf_counter() - start_time) * 1000

    predicted_class = classes[predicted_index.item()]

    probability_dict = {
        class_name: round(probabilities[0][index].item(), 4)
        for index, class_name in enumerate(classes)
    }

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence.item(), 4),
        "probabilities": probability_dict,
        "latency_ms": round(latency_ms, 2),
        "model_version": MODEL_VERSION
    }

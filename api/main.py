"""
FastAPI Gateway для Triton Inference Server (gRPC версия).

Имена слоёв модели определяются автоматически из метаданных Triton
при старте приложения — это позволяет менять модель без правки кода API.
"""
import os
import io
import time
import base64
import asyncio
import logging

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import tritonclient.grpc as grpcclient

# ═══════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ═══════════════════════════════════════════════════════════════════

TRITON_URL = os.getenv("TRITON_URL", "triton:8001")
MODEL_NAME = os.getenv("MODEL_NAME", "image_classifier")
INPUT_SHAPE = (224, 224)

# Классы — порядок ВАЖЕН и должен совпадать с обучением (алфавитный по умолчанию)
CLASS_NAMES = ["cat", "dog", "horse"]

# ImageNet mean для caffe-mode препроцессинга ResNet50 (BGR-порядок)
IMAGENET_MEAN_BGR = np.array([103.939, 116.779, 123.68], dtype=np.float32)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("api")

# ═══════════════════════════════════════════════════════════════════
# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ
# ═══════════════════════════════════════════════════════════════════

triton_client = None   # gRPC-клиент Triton, создаётся в startup()
input_name = None      # Имя входного слоя, извлекается из метаданных Triton
output_name = None     # Имя выходного слоя, извлекается из метаданных Triton

# ═══════════════════════════════════════════════════════════════════
# FASTAPI APP + Prometheus метрики
# ═══════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Animal Classifier API",
    description="Классификация изображений животных (cat / dog / horse) через Triton",
    version="1.0.0",
)

REQUESTS = Counter("api_requests_total", "Total API requests", ["endpoint", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["endpoint"])


# ═══════════════════════════════════════════════════════════════════
# STARTUP / SHUTDOWN
# ═══════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup():
    """Подключение к Triton и автоопределение имён слоёв из метаданных."""
    global triton_client, input_name, output_name

    log.info(f"Запуск API | Triton: {TRITON_URL}")
    triton_client = grpcclient.InferenceServerClient(url=TRITON_URL, verbose=False)

    # Ждём готовности модели (макс 30 попыток × 2 сек)
    for attempt in range(30):
        try:
            if triton_client.is_server_live() and triton_client.is_model_ready(MODEL_NAME):
                metadata = triton_client.get_model_metadata(MODEL_NAME)
                input_name = metadata.inputs[0].name
                output_name = metadata.outputs[0].name
                log.info(f"Модель '{MODEL_NAME}' готова | input='{input_name}' output='{output_name}'")
                return
        except Exception as e:
            log.info(f"Попытка {attempt + 1}/30: {e}")
        await asyncio.sleep(2)

    raise RuntimeError("Не удалось подключиться к Triton за 60 секунд")


@app.on_event("shutdown")
async def shutdown():
    if triton_client:
        triton_client.close()
        log.info("Соединение с Triton закрыто")


# ═══════════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ═══════════════════════════════════════════════════════════════════

def preprocess_image(img: Image.Image) -> np.ndarray:
    """
    Эквивалент tensorflow.keras.applications.resnet50.preprocess_input (mode='caffe'):
      1. Resize до 224x224
      2. RGB → BGR
      3. Вычесть ImageNet mean
    """
    img = img.convert("RGB").resize(INPUT_SHAPE)
    arr = np.array(img, dtype=np.float32)
    arr = arr[..., ::-1]                  # RGB → BGR
    arr -= IMAGENET_MEAN_BGR              # zero-center
    return np.expand_dims(arr, axis=0)    # (1, 224, 224, 3)


def run_inference(data: np.ndarray) -> np.ndarray:
    """Синхронный вызов Triton по gRPC (запускается в thread pool через asyncio.to_thread)."""
    inputs = [grpcclient.InferInput(input_name, data.shape, "FP32")]
    inputs[0].set_data_from_numpy(data)
    outputs = [grpcclient.InferRequestedOutput(output_name)]
    result = triton_client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
    return result.as_numpy(output_name)


def format_result(probs: np.ndarray) -> dict:
    top_idx = int(np.argmax(probs))
    return {
        "class": CLASS_NAMES[top_idx],
        "class_id": top_idx,
        "confidence": float(probs[top_idx]),
        "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
    }


# ═══════════════════════════════════════════════════════════════════
# ЭНДПОИНТЫ
# ═══════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {"message": "Animal Classifier API", "docs": "/docs"}


@app.get("/health")
async def health():
    try:
        return {
            "status": "healthy" if triton_client.is_server_live() else "unhealthy",
            "model_ready": triton_client.is_model_ready(MODEL_NAME),
            "input_name": input_name,
            "output_name": output_name,
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.get("/classes")
async def classes():
    return {"classes": CLASS_NAMES}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start = time.time()
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw))
        tensor = preprocess_image(img)
        # Блокирующий gRPC-вызов уносим в thread pool, чтобы не блокировать event loop
        probs = await asyncio.to_thread(run_inference, tensor)
        result = format_result(probs[0])

        REQUESTS.labels("/predict", "200").inc()
        LATENCY.labels("/predict").observe(time.time() - start)
        return result
    except Exception as e:
        REQUESTS.labels("/predict", "500").inc()
        log.exception("predict error")
        raise HTTPException(500, str(e))


class Base64Request(BaseModel):
    image_base64: str


@app.post("/predict_base64")
async def predict_base64(body: Base64Request):
    start = time.time()
    try:
        raw = base64.b64decode(body.image_base64)
        img = Image.open(io.BytesIO(raw))
        tensor = preprocess_image(img)
        probs = await asyncio.to_thread(run_inference, tensor)
        result = format_result(probs[0])

        REQUESTS.labels("/predict_base64", "200").inc()
        LATENCY.labels("/predict_base64").observe(time.time() - start)
        return result
    except Exception as e:
        REQUESTS.labels("/predict_base64", "500").inc()
        log.exception("predict_base64 error")
        raise HTTPException(500, str(e))


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

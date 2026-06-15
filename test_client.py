"""
Простой тестовый клиент: отправляет случайное изображение в /predict
и /predict_base64 и печатает результат.

Запуск:
    python test_client.py
    python test_client.py path/to/image.jpg
"""

import base64
import io
import sys

import numpy as np
import requests
from PIL import Image

API = "http://localhost:8080"


def make_image(path: str | None) -> bytes:
    if path is None:
        arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
    else:
        img = Image.open(path).convert("RGB").resize((224, 224))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main(path: str | None) -> None:
    print("GET /health  ->", requests.get(f"{API}/health").json())
    print("GET /classes ->", requests.get(f"{API}/classes").json())

    img_bytes = make_image(path)

    # multipart
    r = requests.post(f"{API}/predict", files={"file": ("img.png", img_bytes, "image/png")})
    print("\nPOST /predict (multipart):")
    print(r.json())

    # base64
    r = requests.post(
        f"{API}/predict_base64",
        json={"image": base64.b64encode(img_bytes).decode()},
    )
    print("\nPOST /predict_base64:")
    print(r.json())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

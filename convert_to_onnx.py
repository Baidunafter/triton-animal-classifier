"""
Конвертация best_resnet50.keras → ONNX через SavedModel
(обход бага tf2onnx с Keras 3).

ONNX-модель ожидает на входе УЖЕ препроцессированный тензор
(RGB→BGR + вычитание ImageNet mean). Препроцессинг делается в api/main.py.
"""
import os, sys, shutil, subprocess, tempfile

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
import onnx
import onnxruntime as ort

SRC = "best_resnet50.keras"
DST_DIR = "model_repository/image_classifier/1"
DST = os.path.join(DST_DIR, "model.onnx")
OPSET = 15

os.makedirs(DST_DIR, exist_ok=True)

# 1. Загрузка
model = tf.keras.models.load_model(SRC)
print(f"Input shape: {model.input_shape}, Output shape: {model.output_shape}")

# 2. Экспорт в SavedModel
sm_dir = tempfile.mkdtemp(prefix="resnet_sm_")
model.export(sm_dir)

# 3. SavedModel → ONNX через CLI tf2onnx
subprocess.run([
    sys.executable, "-m", "tf2onnx.convert",
    "--saved-model", sm_dir,
    "--output", DST,
    "--opset", str(OPSET),
], check=True)
shutil.rmtree(sm_dir, ignore_errors=True)

# 4. Имена тензоров
onnx_model = onnx.load(DST)
in_name = onnx_model.graph.input[0].name
out_name = onnx_model.graph.output[0].name
print(f"INPUT_NAME  = {in_name!r}")
print(f"OUTPUT_NAME = {out_name!r}")
print("(Triton сам прочитает эти имена из ONNX — править config.pbtxt не нужно)")

# 5. Сверка Keras vs ONNX
sess = ort.InferenceSession(DST, providers=["CPUExecutionProvider"])
rng = np.random.default_rng(42)
raw = rng.uniform(0, 255, size=(1, 224, 224, 3)).astype(np.float32)
preprocessed = preprocess_input(raw.copy())

keras_out = model.predict(preprocessed, verbose=0)
onnx_out = sess.run([out_name], {in_name: preprocessed})[0]

diff = float(np.max(np.abs(keras_out - onnx_out)))
print(f"max |Keras - ONNX| = {diff:.3e}")
print(f"Размер ONNX: {os.path.getsize(DST) / 1e6:.1f} MB")

# Triton Animal Classifier (cat / dog / horse)

Развёртывание ResNet50-модели классификации животных на NVIDIA Triton Inference Server с FastAPI-шлюзом, мониторингом через Prometheus + Grafana и нагрузочным тестированием с разными конфигурациями Dynamic Batching.

## Архитектура

- **Triton Inference Server 24.05** — обслуживает ONNX-модель через gRPC (порт 8001) и HTTP (8000), метрики на 8002
- **FastAPI Gateway** — REST API для клиентов, обращается к Triton по gRPC; имена входа/выхода определяются автоматически из метаданных модели
- **Prometheus** — собирает метрики с Triton и FastAPI
- **Grafana** — визуализация метрик (auto-provisioned дашборд)
- **`config.pbtxt`** — минимальный (без блоков `input`/`output`, Triton сам читает из ONNX), `kind: KIND_AUTO`
- **Healthcheck для Triton + `depends_on: service_healthy`** для API — корректный порядок запуска

## Быстрый старт

```bash
# 1. Положите свою обученную модель в корень проекта: best_resnet50.keras

# 2. Сконвертируйте в ONNX:
pip install tensorflow tf2onnx onnx onnxruntime
python convert_to_onnx.py

# 3. Поднимите весь стек:
docker compose up -d --build

# 4. Проверьте:
curl http://localhost:8080/health
curl -X POST -F "file=@cat.jpg" http://localhost:8080/predict
```

## Endpoints

- API Swagger: http://localhost:8080/docs
- Grafana: http://localhost:3000 (admin / admin)
- Prometheus: http://localhost:9090
- Triton metrics: http://localhost:8002/metrics

## Нагрузочное тестирование

```bash
pip install aiohttp pillow
python load_test.py                # одиночный прогон
python benchmark_batching.py       # все 4 конфигурации Dynamic Batching
```

## GPU-режим

В `docker-compose.yml` раскомментируйте блок `deploy.resources.reservations.devices` под сервисом `triton`. С `kind: KIND_AUTO` в `config.pbtxt` Triton автоматически переключится на GPU.

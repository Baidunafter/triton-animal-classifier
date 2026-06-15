# 🐾 Triton Animal Classifier

Полнофункциональный пример развёртывания CNN-модели классификации изображений (ResNet50, классы `cat / dog / horse`) на базе NVIDIA Triton Inference Server, FastAPI-шлюза и стека мониторинга Prometheus + Grafana.

---

## 🎯 Преимущества архитектуры

### NVIDIA Triton Inference Server
- **Высокая производительность** — оптимизированный инференс ONNX через ONNX Runtime
- **Динамический батчинг** — автоматическое объединение запросов для увеличения пропускной способности
- **Горячая замена моделей** — обновление без перезапуска сервера
- **Мульти-фреймворк** — поддержка ONNX, TensorFlow, PyTorch, TensorRT и других
- **Auto-config** — `kind: KIND_AUTO` сам выбирает CPU/GPU, имена входа/выхода читаются из ONNX
- **Версионирование** — одновременная работа нескольких версий модели

### FastAPI Gateway
- **gRPC под капотом** — общение с Triton через HTTP/2 (порт 8001), мультиплексирование, нативный для Triton протокол
- **Auto-detection** — имена слоёв определяются автоматически через `get_model_metadata()` при старте
- **Препроцессинг ResNet50** — корректный pipeline (RGB→BGR + ImageNet mean) на стороне API
- **Multipart upload** — приём изображений в стандартном формате `multipart/form-data`
- **Документация** — автогенерация Swagger UI
- **Кастомные метрики** — `api_requests_total`, `api_request_latency_seconds` через `prometheus_client`

### Мониторинг (Prometheus + Grafana)
- **Метрики Triton** — латентность p50/p95/p99, размер очереди, успешные/неуспешные инференсы
- **Метрики API** — RPS, latency по эндпоинтам, статус-коды
- **Auto-provisioning** — Grafana подхватывает datasource и дашборд при первом запуске
- **5 готовых панелей** — RPS, latency percentiles, очередь, счётчики, среднее время в очереди

### Reliability
- **Healthcheck для Triton** + `depends_on: condition: service_healthy` для API — корректный порядок запуска
- **Read-only volumes** для конфигов и моделей
- **`restart: unless-stopped`** для всех сервисов
- **0 отвалов из 12 000 запросов** в стресс-тестах на трёх платформах

---

## 📋 Стек

| Компонент | Версия | Назначение |
|-----------|--------|------------|
| **NVIDIA Triton** | 24.05-py3 | Высокопроизводительный инференс ResNet50 в формате ONNX |
| **FastAPI** | 0.115 | REST API Gateway с gRPC-клиентом к Triton |
| **Prometheus** | 2.54 | Сбор метрик с Triton и FastAPI |
| **Grafana** | 11.2 | Визуализация и дашборды |
| **Streamlit** *(опционально)* | latest | Веб-интерфейс для классификации |

## 📁 Структура проекта

```
triton-animal-classifier/
├── docker-compose.yml                      # Оркестрация всех сервисов
├── convert_to_onnx.py                      # Keras → ONNX через SavedModel
├── load_test.py                            # Нагрузочный тест (aiohttp, асинхронный)
├── benchmark_batching.py                   # Сравнение 4 конфигураций Dynamic Batching
├── README.md
│
├── model_repository/                       # Репозиторий моделей Triton
│   └── image_classifier/                   # Имя директории = имя модели
│       ├── config.pbtxt                    # Минимальный конфиг (KIND_AUTO, batching)
│       └── 1/                              # Версия модели
│           └── model.onnx                  # ResNet50 (~96 МБ), генерируется скриптом
│
├── api/                                    # FastAPI Gateway
│   ├── Dockerfile                          # python:3.11-slim
│   ├── requirements.txt                    # fastapi, tritonclient[grpc], pillow, numpy
│   └── main.py                             # Препроцессинг + gRPC-вызов Triton
│
├── prometheus/
│   └── prometheus.yml                      # Targets: triton:8002, api:8080
│
├── grafana/
│   └── provisioning/
│       ├── datasources/
│       │   └── datasources.yml             # Prometheus как источник данных
│       └── dashboards/
│           ├── dashboards.yml              # Автозагрузка дашбордов
│           └── triton.json                 # 5 панелей: RPS, latency, queue
│
└── streamlit/                              # Опциональный UI
    └── app.py                              # Загрузка изображения + предсказание
```

## ⚡ Быстрый старт

```bash
git clone https://github.com/Baidunafter/triton-animal-classifier.git
cd triton-animal-classifier
docker compose up -d --build
```

## ✅ Проверка

```bash
# Health check (через несколько секунд после старта)
curl http://localhost:8080/health

# Список классов
curl http://localhost:8080/classes
# {"classes":["cat","dog","horse"]}

# Предсказание
curl -X POST -F "file=@cat.jpg" http://localhost:8080/predict
```

Пример ответа:
```json
{
  "class": "cat",
  "class_id": 0,
  "confidence": 0.9999966621398926,
  "probabilities": {
    "cat": 0.9999966621398926,
    "dog": 3.130303e-06,
    "horse": 2.553460e-07
  }
}
```

## 🔌 Сервисы

| Сервис | URL | Описание |
|--------|-----|----------|
| **FastAPI** | http://localhost:8080 | REST API |
| **Swagger UI** | http://localhost:8080/docs | Документация API |
| **Triton HTTP** | http://localhost:8000 | Triton HTTP API |
| **Triton gRPC** | localhost:8001 | Triton gRPC API (используется FastAPI) |
| **Triton Metrics** | http://localhost:8002/metrics | Метрики для Prometheus |
| **Prometheus** | http://localhost:9090 | Web UI и проверка targets |
| **Grafana** | http://localhost:3000 | Дашборды (логин/пароль: admin/admin) |
| **Streamlit** *(опц.)* | http://localhost:8501 | UI для классификации |

## 📡 API Endpoints

| Метод | Endpoint | Описание |
|-------|----------|----------|
| `GET` | `/` | Информация об API |
| `GET` | `/health` | Проверка состояния (Triton + модель) |
| `GET` | `/classes` | Список поддерживаемых классов |
| `GET` | `/docs` | Swagger UI |
| `POST` | `/predict` | Предсказание по `multipart/form-data` файлу |
| `POST` | `/predict_base64` | Предсказание по base64-строке |
| `GET` | `/metrics` | Метрики Prometheus для API |

### Пример запроса `/predict_base64`

```bash
B64=$(base64 -w0 cat.jpg)
curl -X POST http://localhost:8080/predict_base64 \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\": \"$B64\"}"
```

## ⚙️ Конфигурация Triton

```protobuf
name: "image_classifier"
backend: "onnxruntime"
max_batch_size: 32

dynamic_batching {
  preferred_batch_size: [ 4, 8, 16, 32 ]
  max_queue_delay_microseconds: 100000
}

instance_group [{ count: 1, kind: KIND_AUTO }]
```

- **Backend:** `onnxruntime`
- **Max batch size:** `32`
- **Dynamic batching:** preferred `[4, 8, 16, 32]`, max queue delay `100 ms`
- **Execution:** `KIND_AUTO` — GPU если доступен, иначе CPU
- **Input/Output:** не указаны явно — Triton считывает имена и формы из ONNX-файла

## 🧪 Тестирование и бенчмарки

### Одиночный прогон

```bash
pip install aiohttp pillow
python load_test.py  # 1000 запросов, concurrency=10
```

### Сравнение 4 конфигураций Dynamic Batching

```bash
python benchmark_batching.py
```

Скрипт последовательно прогоняет 4 конфигурации с перезапуском Triton+API между ними:

| Конфигурация | preferred_batch_size | max_queue_delay |
|---|---|---|
| `no_batch` | — (батчинг отключён) | — |
| `small` | `[2, 4]` | 50 ms |
| `medium` | `[4, 8, 16]` | 100 ms |
| `large` | `[8, 16, 32]` | 200 ms |

### Результаты тестирования (1000 запросов, concurrency=10)

| Платформа | no_batch | small | medium | large |
|---|---|---|---|---|
| **NVIDIA RTX 4070 Ti** | **71.1 RPS** | 62.5 RPS | 55.4 RPS | 62.4 RPS |
| **AMD Ryzen 7 7800X3D** | 50.7 RPS | 51.0 RPS | 51.5 RPS | **51.6 RPS** |
| **AMD Ryzen AI 7 H350** | **45.6 RPS** | 45.0 RPS | 44.1 RPS | 44.1 RPS |

**Главное наблюдение:** при concurrency=10 Dynamic Batching не даёт ускорения ни на одной платформе. На GPU `no_batch` даже выигрывает у `medium` на 22%, потому что инференс ResNet50 на 4070 Ti настолько быстрый (единицы мс), что задержка очереди в 100 мс становится доминирующим расходом. Подробный разбор — в Jupyter-ноутбуке практической работы.

## 🎮 Поддержка GPU

Раскомментируйте в `docker-compose.yml` блок под сервисом `triton`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

**Предусловия для GPU-режима:**
- Драйвер NVIDIA ≥ 535
- NVIDIA Container Toolkit (`docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi` должно работать)
- На Windows + WSL2 — включить WSL Integration в Docker Desktop

## 📝 Требования

- **Docker** и Docker Compose v2+
- **~6 ГБ RAM** для всего стека в CPU-режиме (Triton сам по себе ест ~3 ГБ)
- **NVIDIA GPU** с CUDA ≥ 12.4 (опционально)
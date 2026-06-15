# 🚀 Triton ML Inference Stack

Полнофункциональный пример развёртывания ML-модели с использованием NVIDIA Triton Inference Server, FastAPI в качестве API-шлюза и стека мониторинга Prometheus + Grafana.

---

> 📚 **Материалы занятия:** [**Google Colab**](https://colab.research.google.com/drive/1pSiIDSUiMMkNlO1gJJdzqNn_HMX2CQiy?usp=sharing)

---

## 🎯 Преимущества архитектуры

### NVIDIA Triton Inference Server
- **Высокая производительность** — оптимизированный инференс с низкой задержкой
- **Динамический батчинг** — автоматическое объединение запросов для увеличения пропускной способности
- **Горячая замена моделей** — обновление без перезапуска сервера
- **Мульти-фреймворк** — поддержка ONNX, TensorFlow, PyTorch, TensorRT и других
- **GPU-ускорение** — эффективное использование видеокарт NVIDIA
- **Версионирование** — одновременная работа нескольких версий модели

### FastAPI Gateway
- **Валидация данных** — автоматическая проверка входных параметров через Pydantic
- **Препроцессинг** — нормализация данных перед отправкой в модель
- **Документация** — автогенерация Swagger UI
- **Простой REST API** — удобный интерфейс для клиентов

### Мониторинг (Prometheus + Grafana)
- **Метрики в реальном времени** — отслеживание latency, throughput, ошибок
- **Готовые дашборды** — визуализация состояния системы
- **Основа для алертинга** — возможность настройки оповещений

---

## 📋 Стек

| Компонент | Назначение |
|-----------|-----------|
| **NVIDIA Triton** | Высокопроизводительный инференс ONNX-модели |
| **FastAPI** | REST API Gateway с валидацией |
| **Prometheus** | Сбор метрик |
| **Grafana** | Визуализация и дашборды |

## 📁 Структура проекта

```
triton-ml-project/
├── docker-compose.yml                      # Оркестрация всех сервисов
├── test_client.py                          # Скрипт стресс-тестирования API
│
├── model_repository/                       # Репозиторий моделей Triton
│   └── california_housing/                 # Директория модели (имя = имя модели)
│       ├── config.pbtxt                    # Конфигурация: батчинг, бэкенд, инстансы
│       ├── scaler.pkl                      # StandardScaler для нормализации признаков
│       └── 1/                              # Версия модели (1, 2, 3...)
│           └── model.onnx                  # Обученная модель в формате ONNX
│
├── api/                                    # FastAPI Gateway
│   ├── Dockerfile                          # Сборка образа на базе python:3.10-slim
│   ├── requirements.txt                    # Зависимости: fastapi, tritonclient, numpy
│   └── main.py                             # Логика API: валидация, препроцессинг, вызов Triton
│
├── prometheus/                             # Система сбора метрик
│   └── prometheus.yml                      # Конфигурация: targets, интервалы scrape
│
└── grafana/                                # Система визуализации
    └── provisioning/
        ├── datasources/
        │   └── datasources.yml             # Подключение Prometheus как источника данных
        └── dashboards/
            ├── dashboards.yml              # Настройка автозагрузки дашбордов
            └── triton.json                 # Дашборд: RPS, latency, счётчики, очередь
```

## ⚡ Быстрый старт

```bash
git clone https://github.com/Alexandre77777/triton-ml-project.git
cd triton-ml-project
docker-compose up -d
```

## ✅ Проверка

```bash
# Health check
curl http://localhost:8080/health

# Предсказание
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"instances": {"MedInc": 8.3, "HouseAge": 41, "AveRooms": 6.9, "AveBedrms": 1.0, "Population": 322, "AveOccup": 2.5, "Latitude": 37.88, "Longitude": -122.23}}'
```

## 🔌 Сервисы

| Сервис | URL | Описание |
|--------|-----|----------|
| **FastAPI** | http://localhost:8080 | REST API |
| **Swagger UI** | http://localhost:8080/docs | Документация API |
| **Triton HTTP** | http://localhost:8000 | Triton HTTP API |
| **Triton gRPC** | localhost:8001 | Triton gRPC API |
| **Triton Metrics** | http://localhost:8002/metrics | Метрики Prometheus |
| **Prometheus** | http://localhost:9090 | Web UI |
| **Grafana** | http://localhost:3000 | Дашборды (admin/admin) |
| **Мониторинг Triton** | http://localhost:3000/d/triton-ml | Дашборд инференса |

## 📡 API Endpoints

| Метод | Endpoint | Описание |
|-------|----------|----------|
| `GET` | `/` | Информация об API |
| `GET` | `/health` | Проверка состояния |
| `GET` | `/docs` | Swagger UI |
| `POST` | `/predict` | Предсказание цены |

### Пример запроса

```json
{
  "instances": [
    {
      "MedInc": 8.3252,
      "HouseAge": 41.0,
      "AveRooms": 6.984,
      "AveBedrms": 1.024,
      "Population": 322.0,
      "AveOccup": 2.556,
      "Latitude": 37.88,
      "Longitude": -122.23
    }
  ]
}
```

## ⚙️ Конфигурация Triton

- **Backend:** `onnxruntime`
- **Max batch size:** `32`
- **Dynamic batching:** `4, 8, 16, 32`
- **Max queue delay:** `100ms`

## 🧪 Тестирование

```bash
python test_client.py
```

## 🎮 Поддержка GPU

Раскомментируйте в `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

## 📝 Требования

- Docker и Docker Compose
- ~4GB RAM (режим CPU)
- NVIDIA GPU + nvidia-docker (опционально)

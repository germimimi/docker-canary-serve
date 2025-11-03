# Canary ASR CPU: NVIDIA Canary 1B V2 HTTP API

**Русский** | [中文](./README.zh.md) | [English](./README.en.md)

Production-ready сервис автоматического распознавания речи (ASR) на базе модели [nvidia/canary-1b-v2](https://huggingface.co/nvidia/canary-1b-v2).

Оптимизирован для работы **только на CPU** с поддержкой **m4a** аудиофайлов. Все входящие файлы автоматически конвертируются в моно WAV 16 кГц через FFmpeg.

## Основные возможности

- ✅ **CPU-only** — не требуется GPU/CUDA
- ✅ **m4a формат** — входные файлы автоматически конвертируются
- ✅ **25 европейских языков** — включая русский
- ✅ **JSON API** — простой HTTP интерфейс
- ✅ **Опциональные timestamps** — временные метки для сегментов
- ✅ **Production-ready** — Docker с non-root user, healthcheck, resource limits

## Поддерживаемые языки (ISO 639-1)

```
bg (Болгарский)    hr (Хорватский)   lt (Литовский)    pt (Португальский)
cs (Чешский)       da (Датский)      lv (Латышский)    ro (Румынский)
nl (Нидерландский) en (Английский)   mt (Мальтийский)  sk (Словацкий)
et (Эстонский)     fi (Финский)      pl (Польский)     sl (Словенский)
fr (Французский)   de (Немецкий)     ru (Русский)      es (Испанский)
el (Греческий)     hu (Венгерский)   sv (Шведский)
ga (Ирландский)    it (Итальянский)
```

## Быстрый старт

### 1. Запуск через Docker Compose

```bash
docker compose -f docker-compose.cpu.yml up --build -d
```

Сервис запустится на `http://localhost:9000`. Первый запуск займет несколько минут для скачивания модели (~3.5 ГБ).

Остановка:

```bash
docker compose -f docker-compose.cpu.yml down
```

### 2. Проверка готовности

```bash
curl http://localhost:9000/health
# => {"status":"ok"}
```

### 3. Транскрипция аудио

**Базовый пример:**

```bash
curl -s http://localhost:9000/inference \
  -F "file=@audio.m4a" \
  -F "source_lang=ru" \
  -F "target_lang=ru"
```

**Ответ:**

```json
{
  "text": "Распознанный текст из аудио"
}
```

**С временными метками:**

```bash
curl -s http://localhost:9000/inference \
  -F "file=@audio.m4a" \
  -F "source_lang=ru" \
  -F "target_lang=ru" \
  -F "timestamps=true"
```

**Ответ:**

```json
{
  "text": "Распознанный текст из аудио",
  "segments": [
    {"start": 0.0, "end": 2.4, "text": "Распознанный"},
    {"start": 2.4, "end": 4.8, "text": "текст из аудио"}
  ]
}
```

## HTTP API

### `POST /inference`

Транскрибирует m4a аудиофайл.

**Параметры (multipart/form-data):**

| Параметр     | Тип    | По умолчанию | Описание                                |
|-------------|--------|-------------|-----------------------------------------|
| file        | file   | (обязательно)| Аудиофайл в формате .m4a               |
| source_lang | string | ru          | ISO 639-1 код языка исходного аудио     |
| target_lang | string | ru          | ISO 639-1 код языка транскрипции        |
| timestamps  | bool   | false       | Включить временные метки в ответе       |

**Ответ:**

```json
{
  "text": "Транскрипция",
  "segments": [...]  // Только если timestamps=true
}
```

**Коды ошибок:**

- `400` — Неверный формат файла, язык или параметры
- `413` — Файл слишком большой (>200MB)
- `500` — Ошибка транскрипции
- `503` — Модель еще загружается

### `GET /health`

Проверка готовности сервиса.

**Ответ:**

```json
{"status": "ok"}
```

## Переменные окружения

| Переменная     | По умолчанию | Описание                              |
|---------------|--------------|---------------------------------------|
| APP_HOST      | 0.0.0.0      | Адрес для прослушивания               |
| APP_PORT      | 9000         | Порт API                              |
| SOURCE_LANG   | ru           | Язык по умолчанию для входного аудио  |
| TARGET_LANG   | ru           | Язык по умолчанию для транскрипции    |
| HF_HOME       | /models      | Директория кэша моделей Hugging Face  |
| MAX_FILE_SIZE | 209715200    | Максимальный размер файла (200MB)     |

Изменить можно в `docker-compose.cpu.yml`.

## Системные требования

- **CPU**: Современный x86_64 с AVX2
- **RAM**: Минимум 6 ГБ (рекомендуется 8 ГБ)
- **Диск**: ~3.5 ГБ для модели в `./models`

> **Важно**: Распознавание на CPU медленнее GPU. Для длинных файлов ожидайте задержки.

## Структура проекта

```
.
├── main.py                 # FastAPI приложение
├── canary_api/
│   ├── __init__.py
│   └── engine.py          # Загрузка и инференс модели
├── Dockerfile.cpu         # Production Dockerfile
├── docker-compose.cpu.yml # Docker Compose конфиг
├── requirements.txt       # Python зависимости
└── README.md
```

## Безопасность

- ✅ Non-root user в контейнере
- ✅ Healthcheck для мониторинга
- ✅ Resource limits (CPU/Memory)
- ✅ Валидация входных данных
- ✅ Ограничение размера файла

## Примеры использования

### Python

```python
import requests

url = "http://localhost:9000/inference"
files = {"file": open("audio.m4a", "rb")}
data = {
    "source_lang": "ru",
    "target_lang": "en",
    "timestamps": "true"
}

response = requests.post(url, files=files, data=data)
print(response.json())
```

### JavaScript

```javascript
const formData = new FormData();
formData.append('file', audioFile);
formData.append('source_lang', 'ru');
formData.append('target_lang', 'ru');
formData.append('timestamps', 'false');

const response = await fetch('http://localhost:9000/inference', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log(result.text);
```

## Troubleshooting

### Модель долго загружается

При первом запуске модель скачивается из Hugging Face (~3.5 ГБ). Проверьте:

```bash
docker compose -f docker-compose.cpu.yml logs -f
```

### Ошибка 503 при запросах

Модель еще не загрузилась. Подождите ~2-5 минут после старта контейнера.

### Медленная транскрипция

На CPU инференс медленнее. Для ускорения рассмотрите:
- Использование GPU версии
- Уменьшение длительности аудио
- Увеличение CPU ресурсов в docker-compose.cpu.yml

## Лицензия

MIT License. Подробности в файле [LICENSE](./LICENSE).

## Ссылки

- [NVIDIA Canary Model](https://huggingface.co/nvidia/canary-1b-v2)
- [NeMo Toolkit](https://github.com/NVIDIA/NeMo)
- [FastAPI](https://fastapi.tiangolo.com/)

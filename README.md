# Руководство по запуску RAG-сервиса

Ниже — минимальный сценарий для локального поднятия БД, подготовки индекса и запуска API.

## Предварительные требования
- Python 3.10+ и `pip`
- Docker + Docker Compose
- [Ollama](https://ollama.com) запущенный локально (порт по умолчанию `11434`)
- Модели Ollama:
  - `ollama pull qwen2.5:1.5b`
  - `ollama pull bge-m3`

## Установка зависимостей
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Подготовка базы данных
Поднимите Postgres с расширением pgvector:
```bash
docker compose up -d db
```
База доступна на `localhost:5433` с пользователем `rag_user` и паролем `password` (см. `docker-compose.yml`).

## Загрузка данных в индекс
Сложите исходные файлы в каталог `./data` (поддерживаются в том числе `.xlsx`).
```bash
python load_data.py
```
Скрипт создаст таблицу `llamaindex_vectors` в `rag_db` и прогонит векторизацию через `bge-m3` из Ollama; для перегенерации индекса повторите команду.

## Запуск API
Поднимите FastAPI через uvicorn:
```bash
uvicorn rag:app --reload --port 8000
```
Эндпоинт поиска:
```bash
curl "http://127.0.0.1:8000/request?query=текст запроса"
```
Ответ вернётся в поле `answer` с учётом гибридного поиска pgvector, rerank-модели `models/cross-encoder-russian-msmarco` и LLM `qwen2.5:1.5b`.

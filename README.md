# Samastroi Scraper

Скрейпер публичных Telegram-каналов для поиска возможного самостроя в МО.
Использует YandexGPT для анализа текста и формирует карточки в Telegram-группу.

## Запуск локально

```bash
python -m venv .venv
source .venv/bin/activate  # или .venv\Scripts\Activate.ps1 в Windows
pip install -r requirements.txt

cp .env.example .env  # заполнить своими значениями
python samastroi_scraper.py
```

## Деплой на Railway

1. Залить репозиторий на GitHub.
2. Создать проект в Railway -> Deploy from GitHub.
3. Заполнить переменные окружения (YAGPT_API_KEY, YAGPT_FOLDER_ID, BOT_TOKEN, TARGET_CHAT_ID, ADMIN_IDS и т.д.).
4. Запустить деплой.

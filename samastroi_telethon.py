import asyncio
import json
import os
from datetime import datetime
from typing import List

import httpx
from dotenv import load_dotenv
from loguru import logger
from telethon import TelegramClient, events

# ------------------ ЗАГРУЗКА НАСТРОЕК (.env) ------------------ #
load_dotenv()

TG_API_ID = int(os.getenv("TG_API_ID", "0"))
TG_API_HASH = os.getenv("TG_API_HASH", "")
SESSION_NAME = os.getenv("SESSION_NAME", "samastroi_telethon")

ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))

YAGPT_API_KEY = os.getenv("YAGPT_API_KEY", "")
YAGPT_FOLDER_ID = os.getenv("YAGPT_FOLDER_ID", "")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Базовые пути
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

GROUPS_FILE = os.path.join(DATA_DIR, "groups.txt")
KEYWORDS_FILE = os.path.join(DATA_DIR, "keywords.txt")
MONITORING_LOG = os.path.join(DATA_DIR, "monitoring.log")
ANALYTICS_LOG = os.path.join(DATA_DIR, "analytics.log")
YAGPT_DATASET = os.path.join(DATA_DIR, "yagpt_dataset.jsonl")

# ------------------ ЛОГИ ------------------ #
logger.remove()
logger.add(
    os.path.join(LOGS_DIR, "bot.log"),
    rotation="10 MB",
    encoding="utf-8",
    level=LOG_LEVEL,
)
logger.add(lambda m: print(m, end=""), level=LOG_LEVEL)


def ensure_file(path: str, default: str = ""):
    """Создать файл с дефолтным содержимым, если он отсутствует."""
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(default)


# Инициализируем файлы данных
for fpath, default in [
    (GROUPS_FILE, "# @username или ID каналов/чатов, по одному в строке\n"),
    (
        KEYWORDS_FILE,
        "самострой\nстройка\nстроительство\nбез разрешения\nучасток\nземельный участок\n",
    ),
    (MONITORING_LOG, ""),
    (ANALYTICS_LOG, ""),
    (YAGPT_DATASET, ""),
]:
    ensure_file(fpath, default)


def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]


def append_line(path: str, text: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{now}] {text}\n")


def append_jsonl(path: str, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ------------------ YANDEX GPT ------------------ #

YAGPT_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"


async def call_yandex_gpt(prompt: str, temperature: float = 0.2) -> str:
    """
    Вызов YandexGPT для простой классификации.
    Возвращает текст ответа (ожидаем 'да' или 'нет').
    """
    if not (YAGPT_API_KEY and YAGPT_FOLDER_ID):
        logger.warning("YAGPT не настроен (нет API_KEY или FOLDER_ID).")
        return "нет"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {YAGPT_API_KEY}",
        "x-folder-id": YAGPT_FOLDER_ID,
    }

    payload = {
        "modelUri": f"gpt://{YAGPT_FOLDER_ID}/yandexgpt/latest",
        "completionOptions": {
            "maxTokens": 64,
            "temperature": temperature,
            "stream": False,
        },
        "messages": [
            {
                "role": "system",
                "text": (
                    "Ты помощник инспектора Главгосстройнадзора Московской области. "
                    "Тебя интересует только самовольное строительство на территории МО. "
                    "Отвечай строго одним словом: 'да' или 'нет'."
                ),
            },
            {
                "role": "user",
                "text": prompt,
            },
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(YAGPT_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.error(f"Ошибка при обращении к YandexGPT: {e}")
        append_line(ANALYTICS_LOG, f"YAGPT_ERROR: {e}")
        return "нет"

    try:
        text = data["result"]["alternatives"][0]["message"]["text"]
        return text.strip().lower()
    except Exception:
        logger.error(f"Неожиданный ответ YAGPT: {data}")
        return "нет"


# ------------------ ПРОВЕРКА ENV ------------------ #

if TG_API_ID == 0 or not TG_API_HASH:
    raise SystemExit("❌ Укажи TG_API_ID и TG_API_HASH в .env")

# ------------------ TELETHON CLIENT ------------------ #

client = TelegramClient(SESSION_NAME, TG_API_ID, TG_API_HASH)


# ------------------ МОНИТОРИНГ СООБЩЕНИЙ ------------------ #

async def process_message_for_monitoring(event: events.NewMessage.Event):
    """
    Логика мониторинга:
      1. Проверяем, из отслеживаемого ли канала/чата.
      2. Ищем ключевые слова.
      3. Если есть — спрашиваем YandexGPT.
      4. Пишем в логи и уведомляем ADMIN_ID при 'да'.
    """
    msg = event.message
    text = (msg.message or "").strip()
    if not text:
        return

    chat = await event.get_chat()
    chat_id = chat.id
    username = getattr(chat, "username", None)
    chat_label = f"@{username}" if username else str(chat_id)

    # 1. проверка по списку групп
    groups = read_lines(GROUPS_FILE)
    if chat_label not in groups and str(chat_id) not in groups:
        return

    # 2. поиск ключевых слов
    keywords = read_lines(KEYWORDS_FILE)
    lower = text.lower()
    matched = [kw for kw in keywords if kw.lower() in lower]
    if not matched:
        return

    logger.info(f"[MATCH] {chat_label}: ключевые слова {matched}")

    # 3. спрашиваем YandexGPT
    prompt = (
        "Текст сообщения:\n"
        f"{text}\n\n"
        "Вопрос: относится ли это сообщение к самовольному строительству "
        "на территории Московской области? Ответь только 'да' или 'нет'."
    )

    verdict = await call_yandex_gpt(prompt)
    is_samostroi = verdict.startswith("да")

    record = {
        "chat": chat_label,
        "chat_id": chat_id,
        "message_id": msg.id,
        "text": text,
        "keywords": matched,
        "verdict": verdict,
        "is_samostroi": is_samostroi,
    }

    append_line(MONITORING_LOG, json.dumps(record, ensure_ascii=False))
    append_line(
        ANALYTICS_LOG,
        f"MONITOR_HIT: {chat_label} msg_id={msg.id} kw={matched} -> {verdict}",
    )

    # 4. уведомление администратора
    if is_samostroi and ADMIN_ID:
        summary = (
            "🦅 Найден возможный самострой\n\n"
            f"Канал/чат: {chat_label}\n"
            f"ID сообщения: {msg.id}\n\n"
            f"{text}\n\n"
            f"🔑 Ключевые слова: {', '.join(matched)}\n"
            f"Ответ YandexGPT: {verdict}"
        )
        await client.send_message(ADMIN_ID, summary)


# ------------------ РЕШЕНИЯ АДМИНА (датасет YAGPT) ------------------ #

async def handle_decision(event: events.NewMessage.Event, label: str):
    """
    Команды администратора:
      .work   -> 'в_работу'
      .wrong  -> 'неверно'
      .attach -> 'привязать'

    Админ пишет команду в ответ на сообщение,
    бот сохраняет пример в датасет YAGPT_DATASET.
    """
    if event.sender_id != ADMIN_ID:
        return

    reply = await event.get_reply_message()
    if not reply:
        await event.reply("Ответь этой командой на сообщение, которое нужно разметить.")
        return

    source_text = reply.message or ""
    rec = {
        "text": source_text,
        "label": label,
        "timestamp": datetime.now().isoformat(),
    }
    append_jsonl(YAGPT_DATASET, rec)
    append_line(ANALYTICS_LOG, f"DECISION: {label}")
    await event.reply(f"✅ Решение зафиксировано как: {label}")


# Команды админа для разметки датасета
@client.on(events.NewMessage(pattern=r"\.work"))
async def cmd_work(event: events.NewMessage.Event):
    await handle_decision(event, "в_работу")


@client.on(events.NewMessage(pattern=r"\.wrong"))
async def cmd_wrong(event: events.NewMessage.Event):
    await handle_decision(event, "неверно")


@client.on(events.NewMessage(pattern=r"\.attach"))
async def cmd_attach(event: events.NewMessage.Event):
    await handle_decision(event, "привязать")


# ------------------ HEALTH CHECK ------------------ #

@client.on(events.NewMessage(pattern=r"/health"))
async def health(event: events.NewMessage.Event):
    """Простая проверка состояния бота (только для ADMIN_ID)."""
    if event.sender_id != ADMIN_ID:
        return

    groups = read_lines(GROUPS_FILE)
    keywords = read_lines(KEYWORDS_FILE)
    txt = (
        "🩺 Health-check\n"
        f"Группы/чаты для мониторинга: {len(groups)}\n"
        f"Ключевых слов: {len(keywords)}\n"
        f"YandexGPT настроен: {'да' if (YAGPT_API_KEY and YAGPT_FOLDER_ID) else 'нет'}\n"
        f"DATA_DIR: {DATA_DIR}\n"
        f"LOGS_DIR: {LOGS_DIR}\n"
    )
    await event.reply(txt)


# ------------------ ОБРАБОТЧИК ВСЕХ НОВЫХ СООБЩЕНИЙ ------------------ #

@client.on(events.NewMessage(incoming=True))
async def all_new_messages(event: events.NewMessage.Event):
    """
    Базовый обработчик всех входящих сообщений.
    Если это канал или группа — отправляем в логику мониторинга.
    """
    if event.is_channel or event.is_group:
        await process_message_for_monitoring(event)


# ------------------ MAIN ------------------ #

async def main():
    logger.info("🚀 Запуск Samastroi Telethon...")
    await client.start()  # при первом запуске спросит телефон и код
    me = await client.get_me()
    logger.info(f"Успешный вход: @{getattr(me, 'username', None)} (id={me.id})")
    append_line(ANALYTICS_LOG, f"STARTED AS id={me.id}")
    await client.run_until_disconnected()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Остановка бота по Ctrl+C")
        append_line(ANALYTICS_LOG, "STOPPED BY KEYBOARD")

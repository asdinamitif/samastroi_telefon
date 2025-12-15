# ================================================================
#   SAMASTROI SCRAPER — ЧАСТЬ 1 / 10
#   Основная структура + постоянное хранилище + админы + логи
# ================================================================

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional

# ---------------------------------------------------------
# ЛОГИРОВАНИЕ
# ---------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("samastroi_scraper")


# ---------------------------------------------------------
# ПУТИ ХРАНИЛИЩА
# ---------------------------------------------------------
DATA_DIR = os.getenv("DATA_DIR", "/app/data")

# Создаём папку, если нет
os.makedirs(DATA_DIR, exist_ok=True)

TRAINING_DATASET = os.path.join(DATA_DIR, "training_dataset.jsonl")
HISTORY_CARDS = os.path.join(DATA_DIR, "history_cards.jsonl")
ADMINS_FILE = os.path.join(DATA_DIR, "admins.json")
CARDS_DIR = os.path.join(DATA_DIR, "cards")
os.makedirs(CARDS_DIR, exist_ok=True)

# ---------------------------------------------------------
# Блокировка polling (чтобы не было 409 Conflict при нескольких инстансах)
# ---------------------------------------------------------
POLL_LOCK_FILE = os.path.join(DATA_DIR, "poll_updates.lock")

def acquire_poll_lock() -> bool:
    """Пытаемся эксклюзивно захватить lock-файл для poll_updates."""
    try:
        fd = os.open(POLL_LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode("utf-8"))
        os.close(fd)
        log.info(f"poll_updates lock захвачен: {POLL_LOCK_FILE}")
        return True
    except FileExistsError:
        log.warning(f"poll_updates lock уже существует — второй инстанс polling не запущен: {POLL_LOCK_FILE}")
        return False
    except Exception as e:
        log.error(f"Не удалось создать poll_updates lock: {e}")
        return False



# ---------------------------------------------------------
# ИНИЦИАЛИЗАЦИЯ ФАЙЛОВ
# ---------------------------------------------------------
def ensure_file(path: str, default_content: Optional[str] = None):
    """Создаёт пустой файл при отсутствии."""
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            if default_content is not None:
                f.write(default_content)
            log.info(f"Создан файл: {path}")


ensure_file(TRAINING_DATASET)
ensure_file(HISTORY_CARDS)
ensure_file(ADMINS_FILE, default_content="[]")


# ---------------------------------------------------------
# СПИСОК АДМИНИСТРАТОРОВ
# ---------------------------------------------------------
DEFAULT_ADMINS = [
    5685586625,
    272923789,
    398960707,
    777464055,
    978125225
]


def load_admins() -> List[int]:
    try:
        with open(ADMINS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list) and all(isinstance(x, int) for x in data) and len(data) > 0:
                log.info(f"Загружено админов: {data}")
                return data
    except Exception as e:
        log.error(f"Ошибка загрузки admins.json: {e}")

    # Восстанавливаем список по умолчанию
    with open(ADMINS_FILE, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_ADMINS, f)
    log.info(f"Восстановлены админы по умолчанию: {DEFAULT_ADMINS}")
    return DEFAULT_ADMINS


ADMINS = load_admins()


def save_admins():
    """Сохраняем список администраторов."""
    with open(ADMINS_FILE, "w", encoding="utf-8") as f:
        json.dump(ADMINS, f)
    log.info(f"Сохранены администраторы: {ADMINS}")


def is_admin(user_id: int) -> bool:
    return user_id in ADMINS


def add_admin(user_id: int):
    if user_id not in ADMINS:
        ADMINS.append(user_id)
        save_admins()
        log.info(f"Добавлен администратор: {user_id}")


def remove_admin(user_id: int):
    if user_id in ADMINS:
        ADMINS.remove(user_id)
        save_admins()
        log.info(f"Удалён администратор: {user_id}")


# ---------------------------------------------------------
# ID группы куда отправляются карточки
# ---------------------------------------------------------
TARGET_CHAT_ID = -1003502443229
log.info(f"Карточки будут отправляться в чат: {TARGET_CHAT_ID}")


# ---------------------------------------------------------
# Запись обучающих событий (в работу / неверно / привязать)
# ---------------------------------------------------------
def log_training_event(card_id: str, label: str, text: str = ""):
    """Записывает событие обучения в training_dataset.jsonl"""
    record = {
        "timestamp": int(time.time()),
        "card_id": card_id,
        "label": label,
        "text": text
    }
    with open(TRAINING_DATASET, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    log.info(f"[TRAIN] {label.upper()} — card_id={card_id}")


log.info("SAMASTROI SCRAPER — ЧАСТЬ 1 загружена успешно.")

# ================================================================
#   SAMASTROI SCRAPER — ЧАСТЬ 2 / 10
#   Сканирование Telegram-каналов + поиск ключевых слов
# ================================================================

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------
# Набор ключевых слов для поиска подозрительных постов
# ---------------------------------------------------------
KEYWORDS = [
    "стройка", "строительство", "самострой", "котлован", "фундамент",
    "арматура", "многоквартирный", "жилой комплекс", "кран", "экскаватор",
    "строители", "проверка", "застройщик", "разрешение", "рнс", "благоустройство",
    "снос", "надзор", "мчс", "инженер", "штраф"
]

KEYWORDS_LOWER = [k.lower() for k in KEYWORDS]


def normalize_text(text: str) -> str:
    """Удаляет мусор, пробелы, ссылки, приводит к нижнему регистру."""
    if not isinstance(text, str):
        return ""
    text = text.replace("\n", " ").replace("\t", " ")
    text = " ".join(text.split())
    return text.lower().strip()



# ---------------------------------------------------------
# Парсинг datetime из Telegram HTML (ISO 8601 -> unix ts)
# ---------------------------------------------------------
def parse_tg_datetime_to_ts(dt_str: str) -> int:
    """Telegram web отдаёт datetime как ISO 8601 (например 2025-12-15T10:20:12+00:00)."""
    if not dt_str:
        return int(time.time())
    try:
        s = str(dt_str).strip().replace("Z", "+00:00")
        return int(datetime.fromisoformat(s).timestamp())
    except Exception:
        try:
            return int(float(str(dt_str).strip()))
        except Exception:
            return int(time.time())
def detect_keywords(text: str) -> List[str]:
    """Возвращает список ключевых слов, найденных в тексте."""
    text_low = text.lower()
    hits = [kw for kw in KEYWORDS_LOWER if kw in text_low]
    return hits


def fetch_channel_page(url: str) -> Optional[str]:
    """
    Загружает контент страницы канала вида https://t.me/s/<channel>.
    Возвращает HTML или None.
    """
    log.info(f"Запрос веб-страницы канала: {url}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    try:
        r = requests.get(url, headers=headers, timeout=10, allow_redirects=False)

        if r.status_code in (301, 302):
            log.error(
                f"Redirect '{r.status_code} Found' for url '{url}', "
                f"Location: '{r.headers.get('Location')}'"
            )
            return None

        if r.status_code != 200:
            log.error(f"Ошибка HTTP {r.status_code} при загрузке {url}")
            return None

        return r.text

    except Exception as e:
        log.error(f"Ошибка запроса {url}: {e}")
        return None


def extract_posts(html: str) -> List[Dict[str, str]]:
    """
    Получает HTML Telegram-канала и извлекает список постов:
    id, текст, ссылки, дата.
    """
    soup = BeautifulSoup(html, "html.parser")
    messages = soup.find_all("div", class_="tgme_widget_message")

    posts = []

    for msg in messages:
        try:
            msg_id = msg.get("data-post", "")

            text_block = msg.find("div", class_="tgme_widget_message_text")
            text = text_block.get_text(" ", strip=True) if text_block else ""

            date_block = msg.find("time", class_="time")
            timestamp = parse_tg_datetime_to_ts(date_block.get("datetime")) if date_block else int(time.time())

            links = []
            for a in msg.find_all("a", href=True):
                if "http" in a["href"]:
                    links.append(a["href"])

            posts.append({
                "id": msg_id,
                "text": text,
                "timestamp": timestamp,
                "links": links
            })

        except Exception as e:
            log.error(f"Ошибка разбора поста: {e}")

    return posts


def process_channel(channel_username: str) -> List[Dict[str, any]]:
    """
    Сканирует один Telegram-канал и возвращает список подозрительных постов.
    """
    url = f"https://t.me/s/{channel_username}"
    html = fetch_channel_page(url)

    if not html:
        log.error(f"Канал @{channel_username} пропущен — нет HTML")
        return []

    posts = extract_posts(html)
    new_posts = []

    for p in posts:
        text_norm = normalize_text(p["text"])
        found = detect_keywords(text_norm)

        if found:
            log.info(f"[MATCH] @{channel_username}: пост {p['id']}, ключевые слова {found}")

            new_posts.append({
                "channel": channel_username,
                "post_id": p["id"],
                "text": p["text"],
                "links": p["links"],
                "timestamp": p["timestamp"],
                "keywords": found
            })

    return new_posts


CHANNEL_LIST = [
    "tipkhimki", "lobnya", "dolgopacity", "vkhimki",
    "podslushanovsolnechnogorske", "klingorod", "mspeaks",
    "pushkino_official", "podmoskow", "trofimovonline",
    "Tipichnoe_Pushkino", "chp_sergiev_posad", "kraftyou",
    "kontext_channel", "podslushano_ivanteevka", "pushkino_live",
    "life_sergiev_posad", "Podslushano_Vidnoe", "novosti_vidnoe",
    "mchs_vidnoe", "mchs_mo", "domodedovop", "bobrovotoday",
    "nedvizha", "developers_policy"
]


def scan_once() -> List[Dict]:
    """Пробегает по списку каналов и собирает подозрительные посты."""
    all_hits = []

    for ch in CHANNEL_LIST:
        try:
            hits = process_channel(ch)
            if hits:
                log.info(f"Найдено новых постов в @{ch}: {len(hits)}")
            else:
                log.info(f"Новых постов в @{ch} не найдено.")
            all_hits.extend(hits)
        except Exception as e:
            log.error(f"Ошибка при обработке канала @{ch}: {e}")

    return all_hits


log.info("SAMASTROI SCRAPER — ЧАСТЬ 2 загружена успешно.")

# ================================================================
#   SAMASTROI SCRAPER — ЧАСТЬ 3 / 10
#   Генерация карточек (текст, структура, сохранение)
# ================================================================

import uuid


def generate_card_id() -> str:
    return str(uuid.uuid4())[:12]  # короткий ID


def build_card_text(card: Dict) -> str:
    """
    Формирует красивый текст карточки для отправки в группу.
    """
    timestamp = datetime.fromtimestamp(card["timestamp"]).strftime("%d.%m.%Y %H:%M")
    keywords = ", ".join(card["keywords"])

    text = f"""
🔎 Обнаружено подозрительное сообщение
Источник: @{card['channel']}
Дата: {timestamp}
ID поста: {card['post_id']}

🔑 Найденные ключевые слова: {keywords}

📝 Текст сообщения:
{card['text']}

📎 Ссылки:
{chr(10).join(card['links']) if card['links'] else "нет ссылок"}

🆔 ID карточки: {card['card_id']}
"""

    return text.strip()


def save_card(card: Dict):
    """
    Каждая карточка сохраняется в:
    /app/data/cards/{card_id}.json
    """
    path = os.path.join(CARDS_DIR, f"{card['card_id']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)
    log.info(f"Карточка сохранена: {path}")


def generate_card(hit: Dict) -> Dict:
    """
    Получает пост из части 2.
    Возвращает готовую карточку.
    """
    card_id = generate_card_id()

    card = {
        "card_id": card_id,
        "channel": hit["channel"],
        "post_id": hit["post_id"],
        "timestamp": hit["timestamp"],
        "text": hit["text"],
        "keywords": hit["keywords"],
        "links": hit["links"],
        "status": "new",   # new / in_work / wrong / bind
        "history": []
    }

    save_card(card)
    return card


def generate_cards_from_hits(hits: List[Dict]) -> List[Dict]:
    """
    Превращает результаты scan_once() в карточки.
    """
    cards = []
    for h in hits:
        try:
            card = generate_card(h)
            cards.append(card)
        except Exception as e:
            log.error(f"Ошибка генерации карточки: {e}")

    log.info(f"Сформировано карточек: {len(cards)}")
    return cards


log.info("SAMASTROI SCRAPER — ЧАСТЬ 3 загружена успешно.")

# ================================================================
#   SAMASTROI SCRAPER — ЧАСТЬ 4 / 10
#   Отправка карточек в Telegram-группу + история отправок
# ================================================================

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
if not BOT_TOKEN:
    log.warning("BOT_TOKEN не задан — отправка карточек работать не будет.")

TELEGRAM_API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}" if BOT_TOKEN else None


def append_history_entry(entry: Dict):
    """
    Любое важное событие по карточке (отправка, смена статуса и т.д.)
    логируем в HISTORY_CARDS в формате JSONL.
    """
    entry = dict(entry)
    entry["ts"] = int(time.time())
    with open(HISTORY_CARDS, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def update_card_file(card: Dict):
    """Перезаписывает файл карточки актуальными данными."""
    path = os.path.join(CARDS_DIR, f"{card['card_id']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)
    log.info(f"Карточка обновлена: {path}")


def send_telegram_message(chat_id: int, text: str) -> Optional[Dict]:
    if not BOT_TOKEN or not TELEGRAM_API_URL:
        log.warning("Попытка отправки сообщения без BOT_TOKEN.")
        return None

    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": False,
    }

    try:
        resp = requests.post(f"{TELEGRAM_API_URL}/sendMessage", json=payload, timeout=10)
        data = resp.json()
        if not data.get("ok"):
            log.error(f"Ошибка sendMessage: {data}")
            return None
        return data["result"]
    except Exception as e:
        log.error(f"Исключение при sendMessage: {e}")
        return None


def send_card_to_group(card: Dict) -> Optional[int]:
    """
    Базовая версия (будет переопределена в Части 6).
    """
    text = build_card_text(card)
    res = send_telegram_message(TARGET_CHAT_ID, text)
    if not res:
        log.error(f"Не удалось отправить карточку {card['card_id']} в чат {TARGET_CHAT_ID}")
        return None

    message_id = res.get("message_id")
    chat_id = res.get("chat", {}).get("id")

    card.setdefault("tg", {})
    card["tg"]["chat_id"] = chat_id
    card["tg"]["message_id"] = message_id
    card["status"] = "sent"
    card.setdefault("history", []).append(
        {
            "event": "sent",
            "chat_id": chat_id,
            "message_id": message_id,
            "ts": int(time.time()),
        }
    )
    update_card_file(card)

    append_history_entry(
        {
            "event": "sent",
            "card_id": card["card_id"],
            "chat_id": chat_id,
            "message_id": message_id,
        }
    )

    log.info(
        f"Карточка {card['card_id']} отправлена в чат {chat_id}, message_id={message_id}"
    )
    return message_id


def send_cards_to_group(cards: List[Dict]) -> int:
    count = 0
    for card in cards:
        mid = send_card_to_group(card)
        if mid:
            count += 1
            time.sleep(0.5)
    log.info(f"Успешно отправлено карточек: {count} из {len(cards)}")
    return count


log.info("SAMASTROI SCRAPER — ЧАСТЬ 4 загружена успешно.")

# ================================================================
#   SAMASTROI SCRAPER — ЧАСТЬ 5 / 10
#   Основной цикл сканирования и отправки карточек
# ================================================================

from time import sleep

SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "300"))


def run_scan_cycle():
    """
    Один цикл:
    1) сканируем каналы,
    2) формируем карточки,
    3) отправляем их в группу.
    """
    log.info("=== НАЧАЛО ЦИКЛА СКАНИРОВАНИЯ ===")

    hits = scan_once()
    if not hits:
        log.info("Подозрительных постов не найдено.")
        return

    log.info(f"Найдено подозрительных постов: {len(hits)}")

    cards = generate_cards_from_hits(hits)
    if not cards:
        log.info("Карточки не сформированы.")
        return

    sent = send_cards_to_group(cards)
    log.info(f"Цикл завершён. Отправлено карточек: {sent}.")


def main_loop():
    """
    Базовый вариант (будет переопределён в Части 6).
    """
    log.info("SAMASTROI SCRAPER — основной цикл запущен.")
    log.info(f"Интервал сканирования: {SCAN_INTERVAL} секунд.")

    while True:
        try:
            run_scan_cycle()
        except Exception as e:
            log.error(f"Ошибка в цикле сканирования: {e}")
        log.info(f"Ожидание {SCAN_INTERVAL} секунд до следующего цикла...")
        sleep(SCAN_INTERVAL)


log.info("SAMASTROI SCRAPER — ЧАСТЬ 5 загружена успешно.")

# ================================================================
#   SAMASTROI SCRAPER — ЧАСТЬ 6 / 10
#   Inline-кнопки карточек + обработка callback + обучение
# ================================================================

import threading


def load_card(card_id: str) -> Optional[Dict]:
    path = os.path.join(CARDS_DIR, f"{card_id}.json")
    if not os.path.exists(path):
        log.error(f"Файл карточки не найден: {path}")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.error(f"Ошибка чтения карточки {card_id}: {e}")
        return None


def build_card_keyboard(card_id: str) -> Dict:
    """
    callback_data: card:<card_id>:<action>, action ∈ {work, wrong, bind}
    """
    return {
        "inline_keyboard": [
            [
                {"text": "✅ В работу", "callback_data": f"card:{card_id}:work"},
                {"text": "❌ Неверно", "callback_data": f"card:{card_id}:wrong"},
            ],
            [
                {"text": "📎 Привязать", "callback_data": f"card:{card_id}:bind"},
            ]
        ]
    }


def answer_callback_query(cb_id: str, text: str = "", show_alert: bool = False):
    if not BOT_TOKEN or not TELEGRAM_API_URL:
        return
    payload = {
        "callback_query_id": cb_id,
        "text": text,
        "show_alert": show_alert,
    }
    try:
        requests.post(f"{TELEGRAM_API_URL}/answerCallbackQuery", json=payload, timeout=10)
    except Exception as e:
        log.error(f"Ошибка answerCallbackQuery: {e}")


def edit_message_reply_markup(chat_id: int, message_id: int, reply_markup: Optional[Dict]):
    if not BOT_TOKEN or not TELEGRAM_API_URL:
        return
    payload = {
        "chat_id": chat_id,
        "message_id": message_id,
        "reply_markup": reply_markup,
    }
    try:
        requests.post(f"{TELEGRAM_API_URL}/editMessageReplyMarkup", json=payload, timeout=10)
    except Exception as e:
        log.error(f"Ошибка editMessageReplyMarkup: {e}")


def send_message_with_keyboard(chat_id: int, text: str, reply_markup: Dict) -> Optional[Dict]:
    if not BOT_TOKEN or not TELEGRAM_API_URL:
        log.warning("Попытка отправки сообщения с клавиатурой без BOT_TOKEN.")
        return None

    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": False,
        "reply_markup": reply_markup,
    }

    try:
        resp = requests.post(f"{TELEGRAM_API_URL}/sendMessage", json=payload, timeout=10)
        data = resp.json()
        if not data.get("ok"):
            log.error(f"Ошибка sendMessage с клавиатурой: {data}")
            return None
        return data["result"]
    except Exception as e:
        log.error(f"Исключение при sendMessage с клавиатурой: {e}")
        return None


def send_card_to_group(card: Dict) -> Optional[int]:
    """
    Переопределённая версия: отправляет карточку с inline-кнопками.
    """
    text = build_card_text(card)
    kb = build_card_keyboard(card["card_id"])
    res = send_message_with_keyboard(TARGET_CHAT_ID, text, kb)
    if not res:
        log.error(f"Не удалось отправить карточку {card['card_id']} в чат {TARGET_CHAT_ID}")
        return None

    message_id = res.get("message_id")
    chat_id = res.get("chat", {}).get("id")

    card.setdefault("tg", {})
    card["tg"]["chat_id"] = chat_id
    card["tg"]["message_id"] = message_id
    card["status"] = "sent"
    card.setdefault("history", []).append(
        {
            "event": "sent",
            "chat_id": chat_id,
            "message_id": message_id,
            "ts": int(time.time()),
        }
    )
    update_card_file(card)

    append_history_entry(
        {
            "event": "sent",
            "card_id": card["card_id"],
            "chat_id": chat_id,
            "message_id": message_id,
        }
    )

    log.info(
        f"Карточка {card['card_id']} отправлена (с кнопками) в чат {chat_id}, message_id={message_id}"
    )
    return message_id


def send_cards_to_group(cards: List[Dict]) -> int:
    count = 0
    for card in cards:
        mid = send_card_to_group(card)
        if mid:
            count += 1
            sleep(0.5)
    log.info(f"Успешно отправлено карточек (с кнопками): {count} из {len(cards)}")
    return count


def apply_card_action(card_id: str, action: str, from_user: int):
    """
    Меняет статус карточки, логирует событие, создаёт запись для обучения.
    action: work / wrong / bind
    """
    card = load_card(card_id)
    if not card:
        log.error(f"Не найдена карточка для действия {action}: {card_id}")
        return "Карточка не найдена."

    old_status = card.get("status", "new")
    if action == "work":
        new_status = "in_work"
        label = "work"
        msg = "Статус карточки: В РАБОТУ ✅"
    elif action == "wrong":
        new_status = "wrong"
        label = "wrong"
        msg = "Статус карточки: НЕВЕРНО ❌"
    elif action == "bind":
        new_status = "bind"
        label = "attach"
        msg = "Статус карточки: ПРИВЯЗАТЬ 📎"
    else:
        log.error(f"Неизвестное действие: {action}")
        return "Неизвестное действие."

    card["status"] = new_status
    card.setdefault("history", []).append(
        {
            "event": f"set_{new_status}",
            "from_user": from_user,
            "ts": int(time.time()),
        }
    )
    update_card_file(card)

    append_history_entry(
        {
            "event": "status_change",
            "card_id": card_id,
            "from_user": from_user,
            "old_status": old_status,
            "new_status": new_status,
        }
    )

    log_training_event(card_id, label, text=card.get("text", ""))

    log.info(f"[ACTION] {action.upper()} — card_id={card_id}, user={from_user}")
    return msg


UPDATE_OFFSET = 0


def handle_callback_query(update: Dict):
    """
    Базовая версия (будет расширена в Части 8).
    """
    cb = update.get("callback_query")
    if not cb:
        return

    cb_id = cb.get("id")
    from_user = cb.get("from", {}).get("id")
    data = cb.get("data", "")
    message = cb.get("message", {})
    chat_id = message.get("chat", {}).get("id")
    message_id = message.get("message_id")

    if not data.startswith("card:"):
        return

    try:
        _, card_id, action = data.split(":", 2)
    except ValueError:
        log.error(f"Некорректный формат callback_data: {data}")
        answer_callback_query(cb_id, "Ошибка формата данных.")
        return

    log.info(f"Callback от {from_user}: card_id={card_id}, action={action}")

    result_msg = apply_card_action(card_id, action, from_user)

    try:
        edit_message_reply_markup(chat_id, message_id, reply_markup=None)
    except Exception as e:
        log.error(f"Ошибка снятия клавиатуры: {e}")

    answer_callback_query(cb_id, result_msg, show_alert=False)


def poll_updates():
    """
    Базовая версия (обрабатывает только callback_query).
    Будет переопределена в Части 8.
    """
    global UPDATE_OFFSET
    if not BOT_TOKEN or not TELEGRAM_API_URL:
        log.warning("BOT_TOKEN не задан — poll_updates не запущен.")
        return

    log.info("Запуск poll_updates (обработка callback_query)...")

    while True:
        try:
            params = {
                "timeout": 25,
                "offset": UPDATE_OFFSET,
                "allowed_updates": ["callback_query"],
            }
            resp = requests.get(f"{TELEGRAM_API_URL}/getUpdates", params=params, timeout=30)
            data = resp.json()

            if not data.get("ok"):
                log.error(f"Ошибка getUpdates: {data}")
                time.sleep(5)
                continue

            updates = data.get("result", [])
            if not updates:
                continue

            for upd in updates:
                UPDATE_OFFSET = max(UPDATE_OFFSET, upd["update_id"] + 1)
                if "callback_query" in upd:
                    handle_callback_query(upd)

        except Exception as e:
            log.error(f"Исключение в poll_updates: {e}")
            time.sleep(5)


def main_loop():
    """
    Переопределённый главный цикл:
    - отдельный поток с poll_updates
    - основной поток — сканирование каналов
    """
    log.info("SAMASTROI SCRAPER — общий main_loop запущен.")
    log.info(f"Интервал сканирования: {SCAN_INTERVAL} секунд.")

    if BOT_TOKEN and TELEGRAM_API_URL:
        t = threading.Thread(target=poll_updates, daemon=True)
        t.start()
        log.info("Поток poll_updates запущен.")
    else:
        log.warning("poll_updates не будет запущен (нет BOT_TOKEN).")

    while True:
        try:
            run_scan_cycle()
        except Exception as e:
            log.error(f"Ошибка в цикле сканирования: {e}")
        log.info(f"Ожидание {SCAN_INTERVAL} секунд до следующего цикла...")
        sleep(SCAN_INTERVAL)


log.info("SAMASTROI SCRAPER — ЧАСТЬ 6 загружена успешно.")

# ================================================================
#   SAMASTROI SCRAPER — ЧАСТЬ 7 / 10
#   Расчёт статистики обучения по training_dataset.jsonl
# ================================================================

TARGET_DATASET_SIZE = int(os.getenv("TARGET_DATASET_SIZE", "5000"))


def compute_training_stats() -> Dict:
    """
    Читает training_dataset.jsonl и считает:
    - total
    - work / wrong / attach
    - model_probability (0–100%)
    - progress (0–100%)
    """
    stats = {
        "total": 0,
        "work": 0,
        "wrong": 0,
        "attach": 0,
        "last_ts": None,
    }

    if not os.path.exists(TRAINING_DATASET):
        return stats

    try:
        with open(TRAINING_DATASET, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                stats["total"] += 1
                label = obj.get("label")
                if label == "work":
                    stats["work"] += 1
                elif label == "wrong":
                    stats["wrong"] += 1
                elif label == "attach":
                    stats["attach"] += 1

                ts = obj.get("timestamp")
                if isinstance(ts, int):
                    if stats["last_ts"] is None or ts > stats["last_ts"]:
                        stats["last_ts"] = ts
    except Exception as e:
        log.error(f"Ошибка чтения {TRAINING_DATASET}: {e}")

    if TARGET_DATASET_SIZE <= 0:
        base_prob = 0.0
        progress = 0.0
    else:
        progress = min(1.0, stats["total"] / TARGET_DATASET_SIZE)
        base_prob = progress

    stats["model_probability"] = round(base_prob * 100.0, 2)
    stats["progress"] = round(progress * 100.0, 2)

    return stats


def format_training_stats(stats: Dict) -> str:
    total = stats.get("total", 0)
    work = stats.get("work", 0)
    wrong = stats.get("wrong", 0)
    attach = stats.get("attach", 0)
    prob = stats.get("model_probability", 0.0)
    prog = stats.get("progress", 0.0)

    last_ts = stats.get("last_ts")
    if last_ts:
        last_dt = datetime.fromtimestamp(last_ts).strftime("%d.%m.%Y %H:%M")
        last_str = f"Последнее обучение: {last_dt}"
    else:
        last_str = "Пока не было ни одного события обучения."

    lines = [
        "📊 Статистика обучения ИИ (YandexGPT):",
        "",
        f"• Всего обучающих событий: {total}",
        f"   ├─ В работу (work): {work}",
        f"   ├─ Неверно (wrong): {wrong}",
        f"   └─ Привязать (attach): {attach}",
        "",
        f"• Текущая условная «уверенность модели»: {prob}%",
        f"• Прогресс к целевому датасету ({TARGET_DATASET_SIZE} примеров): {prog}%",
        "",
        last_str,
    ]

    return "\n".join(lines)


log.info("SAMASTROI SCRAPER — ЧАСТЬ 7 загружена успешно.")

# ================================================================
#   SAMASTROI SCRAPER — ЧАСТЬ 8 / 10
#   Админ-меню, команды и проверка состояния обучения
# ================================================================


def send_plain_message(chat_id: int, text: str):
    send_telegram_message(chat_id, text)


def build_admin_keyboard() -> Dict:
    return {
        "inline_keyboard": [
            [
                {
                    "text": "📊 Проверка состояния обучения",
                    "callback_data": "admin:trainstats",
                }
            ],
            [
                {
                    "text": "👥 Список администраторов",
                    "callback_data": "admin:list_admins",
                }
            ],
        ]
    }


def handle_message(update: Dict):
    """
    Поддерживаем:
    - /admin
    - /trainstats
    - /addadmin <id>
    - /deladmin <id>
    """
    msg = update.get("message")
    if not msg:
        return

    chat_id = msg.get("chat", {}).get("id")
    from_user = msg.get("from", {}).get("id")
    text = msg.get("text", "") or ""

    if not text.startswith("/"):
        return

    cmd, *rest = text.split(" ", 1)
    cmd = cmd.split("@")[0]
    arg = rest[0].strip() if rest else ""

    if cmd == "/admin":
        if not is_admin(from_user):
            send_plain_message(chat_id, "❌ У вас нет доступа к админ-меню.")
            return

        kb = build_admin_keyboard()
        send_message_with_keyboard(
            chat_id,
            "🛠 Админ-панель. Выберите действие:",
            kb,
        )
        return

    if cmd == "/trainstats":
        if not is_admin(from_user):
            send_plain_message(chat_id, "❌ У вас нет доступа к статистике обучения.")
            return

        stats = compute_training_stats()
        txt = format_training_stats(stats)
        send_plain_message(chat_id, txt)
        return

    if cmd == "/addadmin":
        if not is_admin(from_user):
            send_plain_message(chat_id, "❌ Вы не можете добавлять администраторов.")
            return

        if not arg:
            send_plain_message(chat_id, "Использование: /addadmin <telegram_id>")
            return

        try:
            new_admin_id = int(arg)
        except ValueError:
            send_plain_message(chat_id, "ID должен быть числом.")
            return

        if new_admin_id in ADMINS:
            send_plain_message(chat_id, f"👤 {new_admin_id} уже является администратором.")
            return

        add_admin(new_admin_id)
        send_plain_message(chat_id, f"✅ {new_admin_id} добавлен в список администраторов.")
        return

    if cmd == "/deladmin":
        if not is_admin(from_user):
            send_plain_message(chat_id, "❌ Вы не можете удалять администраторов.")
            return

        if not arg:
            send_plain_message(chat_id, "Использование: /deladmin <telegram_id>")
            return

        try:
            del_admin_id = int(arg)
        except ValueError:
            send_plain_message(chat_id, "ID должен быть числом.")
            return

        if del_admin_id not in ADMINS:
            send_plain_message(chat_id, f"👤 {del_admin_id} не найден в списке администраторов.")
            return

        remove_admin(del_admin_id)
        send_plain_message(chat_id, f"🗑 {del_admin_id} удалён из списка администраторов.")
        return

    if is_admin(from_user):
        send_plain_message(chat_id, f"Неизвестная команда: {cmd}")


def handle_callback_query(update: Dict):
    cb = update.get("callback_query")
    if not cb:
        return

    cb_id = cb.get("id")
    from_user = cb.get("from", {}).get("id")
    data = cb.get("data", "")
    message = cb.get("message", {})
    chat_id = message.get("chat", {}).get("id")
    message_id = message.get("message_id")

    if data.startswith("card:"):
        try:
            _, card_id, action = data.split(":", 2)
        except ValueError:
            log.error(f"Некорректный формат callback_data: {data}")
            answer_callback_query(cb_id, "Ошибка формата данных.")
            return

        log.info(f"Callback(card) от {from_user}: card_id={card_id}, action={action}")

        result_msg = apply_card_action(card_id, action, from_user)

        try:
            edit_message_reply_markup(chat_id, message_id, reply_markup=None)
        except Exception as e:
            log.error(f"Ошибка снятия клавиатуры: {e}")

        answer_callback_query(cb_id, result_msg, show_alert=False)
        return

    if data.startswith("admin:"):
        if not is_admin(from_user):
            answer_callback_query(cb_id, "❌ Нет доступа к админ-меню.", show_alert=True)
            return

        action = data.split(":", 1)[1]
        log.info(f"Callback(admin) от {from_user}: action={action}")

        if action == "trainstats":
            stats = compute_training_stats()
            txt = format_training_stats(stats)
            send_plain_message(chat_id, txt)
            answer_callback_query(cb_id, "Статистика обучения обновлена.", show_alert=False)
            return

        if action == "list_admins":
            admins_list = "\n".join(str(a) for a in ADMINS) if ADMINS else "Список пуст."
            send_plain_message(chat_id, "👥 Текущие администраторы:\n" + admins_list)
            answer_callback_query(cb_id, "Список администраторов отправлен.", show_alert=False)
            return

        answer_callback_query(cb_id, "Неизвестное действие админ-меню.", show_alert=False)
        return

    answer_callback_query(cb_id, "", show_alert=False)


UPDATE_OFFSET = 0


def poll_updates():
    """
    long polling:
    - message (команды)
    - callback_query (карточки и админ-меню)
    """
    global UPDATE_OFFSET
    if not BOT_TOKEN or not TELEGRAM_API_URL:
        log.warning("BOT_TOKEN не задан — poll_updates не запущен.")
        return

    log.info("Запуск poll_updates (message + callback_query)...")

    # Сбрасываем webhook, чтобы long polling работал стабильно
    try:
        requests.post(f"{TELEGRAM_API_URL}/deleteWebhook", json={"drop_pending_updates": True}, timeout=10)
    except Exception as e:
        log.warning(f"deleteWebhook не выполнен: {e}")

    while True:
        try:
            params = {
                "timeout": 25,
                "offset": UPDATE_OFFSET,
                "allowed_updates": ["message", "callback_query"],
            }
            resp = requests.get(f"{TELEGRAM_API_URL}/getUpdates", params=params, timeout=30)
            data = resp.json()

            if not data.get("ok"):
                log.error(f"Ошибка getUpdates: {data}")
                time.sleep(5)
                continue

            updates = data.get("result", [])
            if not updates:
                continue

            for upd in updates:
                UPDATE_OFFSET = max(UPDATE_OFFSET, upd["update_id"] + 1)

                if "callback_query" in upd:
                    handle_callback_query(upd)
                elif "message" in upd:
                    handle_message(upd)

        except Exception as e:
            log.error(f"Исключение в poll_updates: {e}")
            time.sleep(5)


log.info("SAMASTROI SCRAPER — ЧАСТЬ 8 загружена успешно.")

# ================================================================
#   SAMASTROI SCRAPER — ЧАСТЬ 9 / 10
#   История карточек + просмотр + ручная смена статуса
# ================================================================

MAX_CARDS_LIST = int(os.getenv("MAX_CARDS_LIST", "20"))
MAX_HISTORY_EVENTS = int(os.getenv("MAX_HISTORY_EVENTS", "30"))


def tail_history_events(limit: int = MAX_HISTORY_EVENTS) -> List[Dict]:
    events: List[Dict] = []
    if not os.path.exists(HISTORY_CARDS):
        return events

    try:
        with open(HISTORY_CARDS, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        log.error(f"Ошибка чтения HISTORY_CARDS: {e}")
        return events

    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            events.append(obj)
        except Exception:
            continue

    return events


def format_history_events(events: List[Dict]) -> str:
    if not events:
        return "📂 История пуста."

    lines = ["📂 Последние события истории карточек:", ""]
    for e in events:
        ts = e.get("ts") or e.get("timestamp")
        if isinstance(ts, int):
            dt = datetime.fromtimestamp(ts).strftime("%d.%m.%Y %H:%M")
        else:
            dt = "—"

        ev = e.get("event", "event")
        cid = e.get("card_id", "—")
        extra = []

        if ev == "sent":
            extra.append(f"chat={e.get('chat_id')}, msg={e.get('message_id')}")
        elif ev == "status_change":
            extra.append(
                f"{e.get('old_status','?')} → {e.get('new_status','?')} (user={e.get('from_user','?')})"
            )

        extra_str = f" [{'; '.join(extra)}]" if extra else ""
        lines.append(f"• {dt} — {ev} — card_id={cid}{extra_str}")

    return "\n".join(lines)


def list_recent_cards(limit: int = MAX_CARDS_LIST) -> List[Dict]:
    files = []
    try:
        for name in os.listdir(CARDS_DIR):
            if not name.endswith(".json"):
                continue
            path = os.path.join(CARDS_DIR, name)
            try:
                mtime = os.path.getmtime(path)
            except Exception:
                mtime = 0
            files.append((mtime, path))
    except Exception as e:
        log.error(f"Ошибка обхода папки с карточками: {e}")
        return []

    files.sort(key=lambda x: x[0], reverse=True)
    files = files[:limit]

    cards: List[Dict] = []
    for _, path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                card = json.load(f)
                cards.append(card)
        except Exception as e:
            log.error(f"Ошибка чтения карточки {path}: {e}")
            continue

    return cards


def format_cards_list(cards: List[Dict]) -> str:
    if not cards:
        return "📂 Карточек пока нет."

    lines = ["📂 Последние карточки:", ""]
    for c in cards:
        cid = c.get("card_id", "—")
        status = c.get("status", "—")
        channel = c.get("channel", "—")
        post_id = c.get("post_id", "—")
        kw = ", ".join(c.get("keywords", [])) or "—"

        ts = c.get("timestamp")
        if isinstance(ts, int):
            dt = datetime.fromtimestamp(ts).strftime("%d.%m.%Y %H:%M")
        else:
            dt = "—"

        lines.append(
            f"• {cid} | статус: {status} | @{channel} #{post_id} | {dt}\n"
            f"   ключевые слова: {kw}"
        )

    return "\n".join(lines)


def handle_message(update: Dict):
    """
    Расширенная версия:
    - /admin
    - /trainstats
    - /addadmin <id>
    - /deladmin <id>
    - /cards
    - /history
    - /setcard <card_id> <status> (work|wrong|bind)
    """
    msg = update.get("message")
    if not msg:
        return

    chat_id = msg.get("chat", {}).get("id")
    from_user = msg.get("from", {}).get("id")
    text = msg.get("text", "") or ""

    if not text.startswith("/"):
        return

    cmd, *rest = text.split(" ", 1)
    cmd = cmd.split("@")[0]
    arg = rest[0].strip() if rest else ""

    if cmd == "/admin":
        if not is_admin(from_user):
            send_plain_message(chat_id, "❌ У вас нет доступа к админ-меню.")
            return

        kb = build_admin_keyboard()
        send_message_with_keyboard(
            chat_id,
            "🛠 Админ-панель. Выберите действие:",
            kb,
        )
        return

    if cmd == "/trainstats":
        if not is_admin(from_user):
            send_plain_message(chat_id, "❌ У вас нет доступа к статистике обучения.")
            return

        stats = compute_training_stats()
        txt = format_training_stats(stats)
        send_plain_message(chat_id, txt)
        return

    if cmd == "/addadmin":
        if not is_admin(from_user):
            send_plain_message(chat_id, "❌ Вы не можете добавлять администраторов.")
            return

        if not arg:
            send_plain_message(chat_id, "Использование: /addadmin <telegram_id>")
            return

        try:
            new_admin_id = int(arg)
        except ValueError:
            send_plain_message(chat_id, "ID должен быть числом.")
            return

        if new_admin_id in ADMINS:
            send_plain_message(chat_id, f"👤 {new_admin_id} уже является администратором.")
            return

        add_admin(new_admin_id)
        send_plain_message(chat_id, f"✅ {new_admin_id} добавлен в список администраторов.")
        return

    if cmd == "/deladmin":
        if not is_admin(from_user):
            send_plain_message(chat_id, "❌ Вы не можете удалять администраторов.")
            return

        if not arg:
            send_plain_message(chat_id, "Использование: /deladmin <telegram_id>")
            return

        try:
            del_admin_id = int(arg)
        except ValueError:
            send_plain_message(chat_id, "ID должен быть числом.")
            return

        if del_admin_id not in ADMINS:
            send_plain_message(chat_id, f"👤 {del_admin_id} не найден в списке администраторов.")
            return

        remove_admin(del_admin_id)
        send_plain_message(chat_id, f"🗑 {del_admin_id} удалён из списка администраторов.")
        return

    if cmd == "/cards":
        if not is_admin(from_user):
            send_plain_message(chat_id, "❌ Команда доступна только администраторам.")
            return

        cards = list_recent_cards()
        txt = format_cards_list(cards)
        send_plain_message(chat_id, txt)
        return

    if cmd == "/history":
        if not is_admin(from_user):
            send_plain_message(chat_id, "❌ Команда доступна только администраторам.")
            return

        events = tail_history_events()
        txt = format_history_events(events)
        send_plain_message(chat_id, txt)
        return

    if cmd == "/setcard":
        if not is_admin(from_user):
            send_plain_message(chat_id, "❌ Команда доступна только администраторам.")
            return

        if not arg:
            send_plain_message(
                chat_id,
                "Использование: /setcard <card_id> <status>\nstatus: work | wrong | bind",
            )
            return

        parts = arg.split()
        if len(parts) != 2:
            send_plain_message(
                chat_id,
                "Использование: /setcard <card_id> <status>\nstatus: work | wrong | bind",
            )
            return

        card_id, status = parts[0], parts[1].lower()
        if status not in ("work", "wrong", "bind"):
            send_plain_message(chat_id, "Статус должен быть одним из: work, wrong, bind")
            return

        result_msg = apply_card_action(card_id, status, from_user)
        send_plain_message(chat_id, result_msg)
        return

    if is_admin(from_user):
        send_plain_message(chat_id, f"Неизвестная команда: {cmd}")


log.info("SAMASTROI SCRAPER — ЧАСТЬ 9 загружена успешно.")

# ================================================================
#   SAMASTROI SCRAPER — ЧАСТЬ 10 / 10
#   Интеграция с YandexGPT: оценка вероятности самостроя
# ================================================================

YAGPT_API_KEY = os.getenv("YAGPT_API_KEY", "").strip()
YAGPT_FOLDER_ID = os.getenv("YAGPT_FOLDER_ID", "").strip()

YAGPT_ENDPOINT = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
YAGPT_MODEL = os.getenv("YAGPT_MODEL", "gpt://{folder_id}/yandexgpt/latest")


def call_yandex_gpt_json(text: str) -> Optional[Dict]:
    """
    Делает запрос к YandexGPT, просит выдать JSON:
    { "probability": 0-100, "comment": "..." }
    """
    if not YAGPT_API_KEY or not YAGPT_FOLDER_ID:
        log.warning("YAGPT не настроен (нет API_KEY или FOLDER_ID).")
        return None

    model_uri = YAGPT_MODEL.format(folder_id=YAGPT_FOLDER_ID)

    prompt = (
        "Ты помощник инспектора строительного надзора.\n"
        "Текст сообщения ниже может относиться к незаконному строительству (самострой), либо быть не связанным.\n\n"
        "1. Оцени вероятность, что сообщение связано с самостроем, в процентах (0-100).\n"
        "2. Дай короткий комментарий для инспектора.\n\n"
        "Ответ верни строго в формате JSON:\n"
        "{\n"
        '  \"probability\": <число от 0 до 100>,\n'
        '  \"comment\": \"краткий комментарий\"\n'
        "}\n\n"
        f"Текст сообщения:\n{text}"
    )

    body = {
        "modelUri": model_uri,
        "completionOptions": {
            "stream": False,
            "temperature": 0.1,
            "maxTokens": 200,
        },
        "messages": [
            {
                "role": "user",
                "text": prompt,
            }
        ],
    }

    headers = {
        "Authorization": f"Api-Key {YAGPT_API_KEY}",
        "x-folder-id": YAGPT_FOLDER_ID,
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(YAGPT_ENDPOINT, headers=headers, json=body, timeout=20)
        data = resp.json()
    except Exception as e:
        log.error(f"Ошибка запроса к YandexGPT: {e}")
        return None

    try:
        alt = data["result"]["alternatives"][0]
        text_out = alt["message"]["text"]
    except Exception as e:
        log.error(f"Не удалось извлечь текст из ответа YandexGPT: {e}, data={data}")
        return None

    try:
        text_out_stripped = text_out.strip()
        if not text_out_stripped.startswith("{"):
            start = text_out_stripped.find("{")
            end = text_out_stripped.rfind("}")
            if start != -1 and end != -1 and end > start:
                text_out_stripped = text_out_stripped[start : end + 1]
        obj = json.loads(text_out_stripped)
        return obj
    except Exception as e:
        log.error(f"Ошибка парсинга JSON из ответа YandexGPT: {e}, text={text_out}")
        return None


def enrich_card_with_yagpt(card: Dict):
    """
    Вызывает YandexGPT для оценки вероятности самостроя.
    Записывает результат в card['ai'].
    """
    if not YAGPT_API_KEY or not YAGPT_FOLDER_ID:
        return

    text = card.get("text", "")
    if not text:
        return

    result = call_yandex_gpt_json(text)
    if not result:
        return

    prob = result.get("probability")
    comment = result.get("comment") or ""

    try:
        if prob is not None:
            prob = float(prob)
            if prob < 0:
                prob = 0.0
            if prob > 100:
                prob = 100.0
    except Exception:
        prob = None

    card.setdefault("ai", {})
    if prob is not None:
        card["ai"]["probability"] = prob
    if comment:
        card["ai"]["comment"] = comment

    log.info(
        f"[YAGPT] card_id={card.get('card_id')} prob={prob} comment={comment[:80]}..."
    )


if "generate_card" in globals():
    _orig_generate_card = generate_card

    def generate_card_with_ai(hit: Dict) -> Dict:
        card = _orig_generate_card(hit)
        try:
            enrich_card_with_yagpt(card)
            update_card_file(card)
        except Exception as e:
            log.error(f"Ошибка enrich_card_with_yagpt: {e}")
        return card

    generate_card = generate_card_with_ai
    log.info("generate_card переопределена: добавлена интеграция с YandexGPT.")


if "build_card_text" in globals():
    _orig_build_card_text = build_card_text

    def build_card_text_with_ai(card: Dict) -> str:
        base_text = _orig_build_card_text(card)

        ai_block_lines = []
        ai = card.get("ai") or {}
        prob = ai.get("probability")
        comment = ai.get("comment")

        if prob is not None:
            ai_block_lines.append(f"🤖 Вероятность самостроя (ИИ): {prob:.1f}%")
        if comment:
            ai_block_lines.append(f"💬 Комментарий ИИ: {comment}")

        if not ai_block_lines:
            return base_text

        return base_text + "\n\n" + "\n".join(ai_block_lines)

    build_card_text = build_card_text_with_ai
    log.info("build_card_text переопределена: добавлен вывод оценки YandexGPT.")


log.info("SAMASTROI SCRAPER — ЧАСТЬ 10 загружена успешно.")

# ================================================================
#   ЗАПУСК МОДУЛЯ
# ================================================================

if __name__ == "__main__":
    log.info("SAMASTROI SCRAPER стартует как самостоятельный процесс.")
    if not BOT_TOKEN:
        log.warning("BOT_TOKEN не задан — карточки НЕ будут отправляться в Telegram.")

    main_loop()

import asyncio
import json
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv
from loguru import logger
from urllib.parse import quote_plus

# ------------------ ЗАГРУЗКА НАСТРОЕК (.env) ------------------ #

BASE_DIR = os.path.dirname(__file__)
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH)

# Telegram Bot API
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
TARGET_CHAT_ID = int(os.getenv("TARGET_CHAT_ID", "0") or "0")

# Администраторы (для /risk и служебных команд)
ADMIN_IDS: List[int] = []
_raw_admin_ids = os.getenv("ADMIN_IDS", "").strip()
if _raw_admin_ids:
    for part in _raw_admin_ids.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ADMIN_IDS.append(int(part))
        except ValueError:
            logger.warning(f"Не могу распарсить ADMIN_ID '{part}'")

# YandexGPT настройки
YAGPT_API_KEY = os.getenv("YAGPT_API_KEY", "").strip()
YAGPT_FOLDER_ID = os.getenv("YAGPT_FOLDER_ID", "").strip()

# Яндекс Геокодер (для адрес -> координаты)
YANDEX_GEOCODER_KEY = os.getenv("YANDEX_GEOCODER_KEY", "").strip()

# Уровень логирования
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Порог вероятности самостроя по умолчанию
DEFAULT_MIN_RISK_PROBABILITY = int(os.getenv("MIN_RISK_PROBABILITY", "10") or "10")

# Настройки Telethon для работы с @rs_search_bot
TG_API_ID = int(os.getenv("TG_API_ID", "0") or "0")
TG_API_HASH = os.getenv("TG_API_HASH", "").strip()
SESSION_NAME = os.getenv("SESSION_NAME", "samastroi_rs_session").strip()

# ID бота Росреестра
RS_SEARCH_BOT = "rs_search_bot"  # @rs_search_bot

# ------------------ ДИРЕКТОРИИ И ФАЙЛЫ ------------------ #

DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

GROUPS_FILE = os.path.join(DATA_DIR, "groups.txt")
KEYWORDS_FILE = os.path.join(DATA_DIR, "keywords.txt")
STATE_FILE = os.path.join(DATA_DIR, "state.json")
MONITORING_LOG = os.path.join(DATA_DIR, "monitoring.log")
ANALYTICS_LOG = os.path.join(DATA_DIR, "analytics.log")
YAGPT_DATASET = os.path.join(DATA_DIR, "yagpt_dataset.jsonl")
NEWS_FILE = os.path.join(DATA_DIR, "news.jsonl")
ONZS_DIR = os.path.join(DATA_DIR, "onzs")
os.makedirs(ONZS_DIR, exist_ok=True)

# ------------------ ЛОГИ ------------------ #

logger.remove()
logger.add(
    os.path.join(LOGS_DIR, "scraper.log"),
    rotation="10 MB",
    encoding="utf-8",
    level=LOG_LEVEL,
)
logger.add(lambda m: print(m, end=""), level=LOG_LEVEL)


def ensure_file(path: str, default: str = ""):
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(default)


for fpath, default in [
    (GROUPS_FILE, "# @username каналов, по одному в строке\n@podmoskow\n"),
    (
        KEYWORDS_FILE,
        "самострой\nстройка\nстроительство\nнадзор\nштраф\nразрешение на строительство\nразрешение на ввод\nучасток\nземельный участок\n",
    ),
    (MONITORING_LOG, ""),
    (ANALYTICS_LOG, ""),
    (YAGPT_DATASET, ""),
    (NEWS_FILE, ""),
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


# ------------------ СОСТОЯНИЕ (state.json) ------------------ #


@dataclass
class BotState:
    last_post_ids: Dict[str, int]
    user_subscriptions: Dict[str, List[int]]  # user_id -> [1..12]
    user_paused: Dict[str, bool]
    min_risk_probability: int

    @staticmethod
    def default() -> "BotState":
        return BotState(
            last_post_ids={},
            user_subscriptions={},
            user_paused={},
            min_risk_probability=DEFAULT_MIN_RISK_PROBABILITY,
        )


def load_state() -> BotState:
    if not os.path.exists(STATE_FILE):
        return BotState.default()
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return BotState(
            last_post_ids=data.get("last_post_ids", {}),
            user_subscriptions=data.get("user_subscriptions", {}),
            user_paused=data.get("user_paused", {}),
            min_risk_probability=int(
                data.get("min_risk_probability", DEFAULT_MIN_RISK_PROBABILITY)
            ),
        )
    except Exception as e:
        logger.error(f"Ошибка чтения {STATE_FILE}: {e}")
        return BotState.default()


def save_state(state: BotState):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(asdict(state), f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Ошибка записи {STATE_FILE}: {e}")


STATE = load_state()

# ------------------ YANDEX GPT ------------------ #

YAGPT_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"


async def call_yandex_gpt_json(prompt: str, temperature: float = 0.2) -> Optional[Dict[str, Any]]:
    """
    Вызывает YandexGPT и пытается распарсить JSON из ответа (внутри ```json ... ```).
    Ожидаемый формат:
    {
      "object_type": "...",
      "violation_type": "...",
      "address": "...",
      "okrug_city": "...",
      "cadastral_number": "...",
      "risk_probability": 0-100,
      "risk_score": 0-100,
      "risk_level": "низкий/средний/высокий",
      "summary": "..."
    }
    """
    if not (YAGPT_API_KEY and YAGPT_FOLDER_ID):
        logger.warning("YAGPT не настроен (нет API_KEY или FOLDER_ID).")
        return None

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {YAGPT_API_KEY}",
        "x-folder-id": YAGPT_FOLDER_ID,
    }

    system_prompt = (
        "Ты помощник инспектора Главгосстройнадзора Московской области. "
        "По тексту сообщения из Telegram нужно понять, есть ли признаки самовольного строительства "
        "и, если да, структурировать данные в JSON: объект, адрес, кадастровый номер, риск и краткое описание."
        "Отвечай строго одним JSON-объектом, без комментариев, без текста до и после."
    )

    payload = {
        "modelUri": f"gpt://{YAGPT_FOLDER_ID}/yandexgpt/latest",
        "completionOptions": {
            "maxTokens": 400,
            "temperature": temperature,
            "stream": False,
        },
        "messages": [
            {"role": "system", "text": system_prompt},
            {"role": "user", "text": prompt},
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=40) as client:
            resp = await client.post(YAGPT_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.error(f"Ошибка при обращении к YandexGPT: {e}")
        append_line(ANALYTICS_LOG, f"YAGPT_ERROR: {e}")
        return None

    try:
        text = data["result"]["alternatives"][0]["message"]["text"]
        text = text.strip()
        # Срезаем ```json ... ``` если есть
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*", "", text)
            text = re.sub(r"```$", "", text).strip()
        obj = json.loads(text)
        return obj
    except Exception as e:
        logger.error(f"Не удалось распарсить JSON из YAGPT: {e}; raw={data}")
        append_line(ANALYTICS_LOG, f"YAGPT_JSON_PARSE_ERROR: {e}")
        return None


# ------------------ КООРДИНАТЫ / РОСРЕЕСТР ------------------ #

def extract_coords(text: str) -> Optional[Tuple[float, float]]:
    """
    Ищем в тексте координаты вида 56.054712 37.148884 или 56.054712, 37.148884.
    """
    pattern = r"(\d{1,2}\.\d{5,})[,\s]+(\d{1,2}\.\d{5,})"
    m = re.search(pattern, text)
    if not m:
        return None
    try:
        lat = float(m.group(1))
        lon = float(m.group(2))
        return lat, lon
    except ValueError:
        return None


async def geocode_address(address: str) -> Optional[Tuple[float, float]]:
    """
    Преобразует адрес в координаты через Яндекс Геокодер.
    """
    if not (YANDEX_GEOCODER_KEY and address):
        return None

    url = "https://geocode-maps.yandex.ru/1.x/"
    params = {
        "apikey": YANDEX_GEOCODER_KEY,
        "format": "json",
        "geocode": address,
        "lang": "ru_RU",
    }
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        member = (
            data["response"]["GeoObjectCollection"]["featureMember"][0]["GeoObject"]
        )
        pos = member["Point"]["pos"]  # "37.620393 55.75396"
        lon_str, lat_str = pos.split()
        return float(lat_str), float(lon_str)
    except Exception as e:
        logger.error(f"Ошибка геокодера для '{address}': {e}")
        return None


# ------------------ TELETHON для @rs_search_bot ------------------ #

from telethon import TelegramClient, events

if TG_API_ID == 0 or not TG_API_HASH:
    logger.warning("TG_API_ID/TG_API_HASH не заданы — интеграция с @rs_search_bot работать не будет.")
    RS_CLIENT: Optional[TelegramClient] = None
else:
    RS_CLIENT = TelegramClient(SESSION_NAME, TG_API_ID, TG_API_HASH)


async def ensure_rs_client_started():
    if RS_CLIENT is None:
        return
    if not RS_CLIENT.is_connected():
        await RS_CLIENT.start()


async def query_rs_search_bot_by_coords(lat: float, lon: float) -> Optional[str]:
    """
    Отправляет координаты в @rs_search_bot и возвращает его ответ (текст).
    Формат координат: "56.007403 37.869397".
    """
    if RS_CLIENT is None:
        return None

    await ensure_rs_client_started()

    coords_text = f"{lat:.6f} {lon:.6f}"
    try:
        bot_entity = await RS_CLIENT.get_entity(RS_SEARCH_BOT)
        await RS_CLIENT.send_message(bot_entity, coords_text)

        @RS_CLIENT.on(events.NewMessage(from_users=bot_entity))
        async def handler(event):
            pass

        # Ждем ответа 15 секунд
        resp = await RS_CLIENT.wait_for(
            events.NewMessage(from_users=bot_entity), timeout=15
        )
        return resp.raw_text
    except Exception as e:
        logger.error(f"Ошибка запроса к @rs_search_bot: {e}")
        return None


def extract_rosreestr_block(text: str) -> Optional[str]:
    """
    Из ответа @rs_search_bot вынимаем две ключевые строки:
    - 'Кад. номер ЗУ ...'
    - 'RU.. от ....'
    И формируем блок:
      Кад. номер ЗУ ...
      
      RU...
    """
    if not text:
        return None

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    kad_line = None
    ru_line = None
    for line in lines:
        if kad_line is None and line.startswith("Кад. номер"):
            kad_line = line
        if ru_line is None and line.startswith("RU"):
            ru_line = line
        if kad_line and ru_line:
            break

    if not kad_line and not ru_line:
        return None

    parts = []
    if kad_line:
        parts.append(kad_line)
    if ru_line:
        if parts:
            parts.append("")
        parts.append(ru_line)
    return "\n".join(parts)


# ------------------ ONZS МАППИНГ ------------------ #

ONZS_MAPPING: Dict[int, List[str]] = {
    1: [
        "одинцовский",
        "наро-фоминский",
        "власиха",
        "краснознаменск",
        "можайск",
    ],
    2: [
        "красногорск",
        "истра",
        "восход",
        "волоколамск",
        "лотошино",
        "руза",
        "шаховская",
    ],
    3: [
        "химки",
        "солнечногорск",
        "долгопрудный",
        "лобня",
        "клин",
    ],
    4: [
        "мытищи",
        "королев",
    ],
    5: [
        "пушкинский",
        "сергиево-посад",
    ],
    6: [
        "подольск",
        "серпухов",
        "чехов",
    ],
    7: [
        "домодедово",
        "ленинский",
    ],
    8: [
        "щелково",
        "звездный городок",
        "лосино-петровский",
        "фрязино",
        "черноголовка",
        "электросталь",
    ],
    9: [
        "люберц",
        "котельник",
        "лыткарин",
        "балаших",
        "реутов",
    ],
    10: [
        "коломн",
        "воскресенск",
        "зарайск",
        "кашира",
        "луховиц",
        "раменск",
        "бронниц",
        "жуковск",
        "серебряные пруды",
        "ступино",
    ],
    11: [
        "дмитров",
        "дубна",
        "талдом",
    ],
    12: [
        "орехово-зуево",
        "егорьевск",
        "павлово-посад",
        "шатур",
    ],
}


def detect_onzs_by_text(text: str) -> int:
    """
    Простейшее определение ОНзС по тексту (адрес, округ, город).
    """
    t = text.lower()
    for onzs, patterns in ONZS_MAPPING.items():
        for p in patterns:
            if p in t:
                return onzs
    return 0


# ------------------ КОНСТРУКЦИЯ КАРТОЧЕК ------------------ #


def build_card_text(card: Dict[str, Any]) -> str:
    """
    Формируем текст карточки по данным, которые вернул YandexGPT + Росреестр.
    """
    channel = card.get("channel", "-")
    post_id = card.get("post_id", "-")
    original_url = card.get("original_url", "-")

    object_type = card.get("object_type") or "-"
    violation_type = card.get("violation_type") or "-"
    address = card.get("address") or "-"
    okrug_city = card.get("okrug_city") or "-"
    cadastral_number = card.get("cadastral_number") or card.get("rosreestr_kad") or "-"
    risk_probability = card.get("risk_probability")
    risk_score = card.get("risk_score")
    risk_level = card.get("risk_level") or "-"
    summary = card.get("summary") or "-"
    rosreestr_block = card.get("rosreestr_block") or "-"

    if risk_probability is None:
        risk_probability = 0
    if risk_score is None:
        risk_score = risk_probability

    text_lines = []

    text_lines.append(f"Найдено в {channel}")
    text_lines.append("")
    text_lines.append("🏗 Объект и нарушение")
    text_lines.append(f"• Тип объекта: {object_type}")
    text_lines.append(f"• Тип нарушения: {violation_type}")
    text_lines.append(f"• Адрес: {address}")
    text_lines.append(f"• Округ/город: {okrug_city}")
    text_lines.append(f"• Кадастровый номер: {cadastral_number}")
    text_lines.append(
        f"📈 Вероятность самостроя: {risk_probability}%"
    )
    text_lines.append(f"🧠 Итоговый риск ИИ: {risk_level} ({risk_score} из 100)")
    text_lines.append("")
    text_lines.append("📝 Кратко по сути:")
    text_lines.append(summary)
    text_lines.append("")

    text_lines.append("📑 Данные Росреестра")
    text_lines.append(rosreestr_block if rosreestr_block != "-" else "нет данных")
    text_lines.append("")

    text_lines.append(f"🔗 Открыть оригинал сообщения ({original_url})")
    text_lines.append("")
    text_lines.append(
        "🧠 Обучение: ответь на эту карточку словами «в работу», «неверно» или «привязать» "
        "или нажми соответствующую кнопку под карточкой."
    )

    text_lines.append("")
    text_lines.append(card.get("source_excerpt", "Фрагмент исходного сообщения недоступен."))

    return "\n".join(text_lines)


def build_inline_keyboard(card: Dict[str, Any], channel: str, post_id: int) -> Dict[str, Any]:
    """
    Инлайн-клавиатура:
      • в работу / неверно / привязать
      • 📍 Посмотреть на карте (если есть адрес)
    """
    card_key = f"{channel}:{post_id}"

    keyboard: List[List[Dict[str, Any]]] = [
        [
            {
                "text": "в работу",
                "callback_data": f"train:work:{card_key}",
            },
            {
                "text": "неверно",
                "callback_data": f"train:wrong:{card_key}",
            },
            {
                "text": "привязать",
                "callback_data": f"train:attach:{card_key}",
            },
        ]
    ]

    address = (card.get("address") or "").strip()
    if address and address != "-":
        try:
            query = quote_plus(address)
            map_url = f"https://yandex.ru/maps/?text={query}"
            keyboard.append(
                [
                    {
                        "text": "📍 Посмотреть на карте",
                        "url": map_url,
                    }
                ]
            )
        except Exception as e:
            logger.error(f"Ошибка при формировании ссылки на карту: {e}")

    return {"inline_keyboard": keyboard}


# ------------------ Telegram Bot API (отправка сообщений) ------------------ #

TG_API_BASE = "https://api.telegram.org"


async def tg_request(method: str, data: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{TG_API_BASE}/bot{BOT_TOKEN}/{method}"
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(url, json=data)
        r.raise_for_status()
        return r.json()


async def send_card_to_tg_group(card: Dict[str, Any]) -> Optional[int]:
    """
    Отправляем карточку в целевой чат.
    """
    if not BOT_TOKEN:
        logger.error("BOT_TOKEN не задан — не могу отправлять карточки в Telegram.")
        return None
    if TARGET_CHAT_ID == 0:
        logger.error("TARGET_CHAT_ID не задан — некуда отправлять карточки.")
        return None

    text = build_card_text(card)
    channel = card.get("channel", "-")
    post_id = card.get("post_id", 0)
    markup = build_inline_keyboard(card, channel, post_id)

    data: Dict[str, Any] = {
        "chat_id": TARGET_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
        "reply_markup": markup,
    }

    try:
        resp = await tg_request("sendMessage", data)
        if not resp.get("ok"):
            logger.error(f"Telegram API error: {resp}")
            return None
        message_id = resp["result"]["message_id"]
        return message_id
    except Exception as e:
        logger.error(f"Ошибка отправки карточки в Telegram: {e}")
        return None


# ------------------ ПОДПИСКИ ПОЛЬЗОВАТЕЛЕЙ НА ОНзС ------------------ #

async def send_tg_message(chat_id: int, text: str, reply_markup: Optional[Dict[str, Any]] = None):
    if not BOT_TOKEN:
        return
    data: Dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }
    if reply_markup:
        data["reply_markup"] = reply_markup
    try:
        await tg_request("sendMessage", data)
    except Exception as e:
        logger.error(f"Ошибка send_tg_message: {e}")


def build_onzs_keyboard(selected: List[int]) -> Dict[str, Any]:
    buttons = []
    row = []
    for i in range(1, 13):
        text = f"{i} {'✅' if i in selected else ''}"
        row.append({"text": text, "callback_data": f"onzs:{i}"})
        if len(row) == 4:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    buttons.append(
        [
            {
                "text": "Все ОНзС",
                "callback_data": "onzs:all",
            }
        ]
    )
    return {"inline_keyboard": buttons}


async def broadcast_card_to_subscribers(card: Dict[str, Any], main_message_id: Optional[int] = None):
    """
    Заготовка под рассылку подписчикам по ОНзС.
    Сейчас не используется — только общий чат.
    """
    return


# ------------------ ОБРАБОТКА ПУБЛИЧНЫХ КАНАЛОВ (WEB SCRAPING) ------------------ #

TELEGRAM_WEB_BASE = "https://t.me"


async def fetch_channel_page(username: str) -> str:
    """
    Забираем HTML публичного канала через веб-страницу /s/<username>.
    """
    url = f"{TELEGRAM_WEB_BASE}/s/{username.lstrip('@')}"
    logger.info(f"Запрос веб-страницы канала: {url}")
    async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
        r = await client.get(url)
        if r.status_code in (301, 302, 303, 307, 308):
            logger.error(
                f"Redirect response '{r.status_code} {r.reason_phrase}' for url '{url}'\n"
                f"Redirect location: '{r.headers.get('Location')}'"
            )
            raise RuntimeError(f"Redirect for {url}")
        r.raise_for_status()
        return r.text


def parse_posts_from_html(html: str) -> List[Tuple[int, str]]:
    """
    Упрощенный парсер: ищем блоки data-post="channel/12345" и рядом текст.
    """
    posts: List[Tuple[int, str]] = []

    for m in re.finditer(r'data-post="[^/]+/(\d+)"', html):
        msg_id = int(m.group(1))
        # Грубый захват фрагмента текста вокруг ID
        start = max(0, m.start() - 2000)
        end = min(len(html), m.end() + 2000)
        snippet = html[start:end]
        snippet = re.sub(r"<[^>]+>", " ", snippet)  # вырезаем теги
        snippet = re.sub(r"\s+", " ", snippet)
        posts.append((msg_id, snippet))

    # Убираем дубликаты и сортируем
    unique: Dict[int, str] = {}
    for msg_id, text in posts:
        if msg_id not in unique:
            unique[msg_id] = text
    result = sorted(unique.items(), key=lambda x: x[0])
    return result


async def analyze_case_with_yagpt(channel: str, msg_id: int, text: str, original_url: str) -> Optional[Dict[str, Any]]:
    """
    Постобработка одного кандидата:
      1) Вызов YandexGPT -> JSON-структура
      2) Попытка найти координаты / адрес и запросить @rs_search_bot
      3) Определение ОНзС
    """
    prompt = (
        f"Канал: {channel}\n"
        f"ID сообщения: {msg_id}\n\n"
        f"Текст:\n{text}\n\n"
        "Сформируй JSON-объект с полями:\n"
        "{\n"
        '  "object_type": "тип объекта",\n'
        '  "violation_type": "тип нарушения (если есть признаки самостроя)",\n'
        '  "address": "адрес (если указан)",\n'
        '  "okrug_city": "муниципалитет/город (если указан)",\n'
        '  "cadastral_number": "кадастровый номер (если есть)",\n'
        '  "risk_probability": 0-100,\n'
        '  "risk_score": 0-100,\n'
        '  "risk_level": "низкий/средний/высокий",\n'
        '  "summary": "краткое описание ситуации"\n'
        "}"
    )

    yagpt_data = await call_yandex_gpt_json(prompt)
    if not yagpt_data:
        return None

    card: Dict[str, Any] = {
        "channel": channel,
        "post_id": msg_id,
        "original_url": original_url,
        "source_excerpt": text[:500] + ("..." if len(text) > 500 else ""),
    }

    for key in [
        "object_type",
        "violation_type",
        "address",
        "okrug_city",
        "cadastral_number",
        "risk_probability",
        "risk_score",
        "risk_level",
        "summary",
    ]:
        if key in yagpt_data:
            card[key] = yagpt_data[key]

    # Порог по вероятности
    rp = int(card.get("risk_probability") or 0)
    if rp < STATE.min_risk_probability:
        logger.info(
            f"Карточка отклонена по порогу вероятности: {rp}% < {STATE.min_risk_probability}%"
        )
        return None

    # Координаты -> @rs_search_bot
    rosreestr_block = None
    coords = extract_coords(text)
    if not coords and card.get("address"):
        coords = await geocode_address(card["address"])

    if coords:
        lat, lon = coords
        rs_resp = await query_rs_search_bot_by_coords(lat, lon)
        rosreestr_block = extract_rosreestr_block(rs_resp or "")
    card["rosreestr_block"] = rosreestr_block or "-"

    # ОНзС по адресу/муниципалитету
    onzs_text_source = f"{card.get('address', '')} {card.get('okrug_city', '')}"
    card["onzs"] = detect_onzs_by_text(onzs_text_source)

    return card


async def process_public_post(channel: str, msg_id: int, text: str):
    """
    Обработка одного поста в публичном канале:
      - Поиск ключевых слов
      - Вызов YandexGPT
      - Сохранение карточки
      - Отправка в Telegram
    """
    keywords = read_lines(KEYWORDS_FILE)
    lower = text.lower()
    matched = [kw for kw in keywords if kw.lower() in lower]
    if not matched:
        return

    logger.info(f"[MATCH] @{channel}: пост {msg_id}, ключевые слова {matched}")

    original_url = f"https://t.me/{channel}/{msg_id}"

    card = await analyze_case_with_yagpt(
        channel=f"@{channel}", msg_id=msg_id, text=text, original_url=original_url
    )
    if not card:
        return

    append_jsonl(NEWS_FILE, card)
    onzs = int(card.get("onzs") or 0)
    if onzs in range(1, 13):
        onzs_file = os.path.join(ONZS_DIR, f"onzs_{onzs}.jsonl")
        append_jsonl(onzs_file, card)

    append_line(
        MONITORING_LOG,
        json.dumps(
            {
                "channel": channel,
                "msg_id": msg_id,
                "keywords": matched,
                "card": card,
            },
            ensure_ascii=False,
        ),
    )

    msg_id_sent = await send_card_to_tg_group(card)
    await broadcast_card_to_subscribers(card, msg_id_sent)


async def scan_once():
    """
    Один проход по списку каналов из groups.txt.
    """
    groups = read_lines(GROUPS_FILE)
    for raw in groups:
        username = raw.lstrip("@")
        try:
            html = await fetch_channel_page(username)
            posts = parse_posts_from_html(html)
            last_seen = int(STATE.last_post_ids.get(username, 0))
            new_posts = [(mid, txt) for (mid, txt) in posts if mid > last_seen]
            if not new_posts:
                logger.info(f"Новых постов в @{username} не найдено.")
                continue

            logger.info(f"Найдено новых постов в @{username}: {len(new_posts)}")

            for mid, txt in new_posts:
                await process_public_post(username, mid, txt)
                if mid > last_seen:
                    last_seen = mid

            STATE.last_post_ids[username] = last_seen
            save_state(STATE)

        except Exception as e:
            logger.error(f"Ошибка при обработке канала @{username}: {e}")


# ------------------ ОБРАБОТКА CALLBACK (ОБУЧЕНИЕ YAGPT) ------------------ #

async def handle_callback_query(callback_query: Dict[str, Any]):
    """
    Обработка callback_data формата train:<action>:<channel>:<post_id>
    и onzs:<номер>
    """
    data = callback_query.get("data", "")
    from_id = callback_query.get("from", {}).get("id")
    message = callback_query.get("message", {})
    message_id = message.get("message_id")
    chat_id = message.get("chat", {}).get("id")

    if data.startswith("train:"):
        _, action, key = data.split(":", 2)
        channel, post_id_str = key.split(":", 1)
        label_map = {
            "work": "в_работу",
            "wrong": "неверно",
            "attach": "привязать",
        }
        label = label_map.get(action, action)
        rec = {
            "text": message.get("text", ""),
            "label": label,
            "timestamp": datetime.now().isoformat(),
            "from_id": from_id,
        }
        append_jsonl(YAGPT_DATASET, rec)
        append_line(ANALYTICS_LOG, f"DECISION: {label} by {from_id}")

        # Убираем кнопки с карточки для всех
        try:
            await tg_request(
                "editMessageReplyMarkup",
                {
                    "chat_id": chat_id,
                    "message_id": message_id,
                    "reply_markup": {"inline_keyboard": []},
                },
            )
        except Exception as e:
            logger.error(f"Ошибка editMessageReplyMarkup: {e}")

        # Ответ пользователю (notification)
        await tg_request(
            "answerCallbackQuery",
            {
                "callback_query_id": callback_query.get("id"),
                "text": f"Решение зафиксировано: {label}",
                "show_alert": False,
            },
        )

    elif data.startswith("onzs:"):
        val = data.split(":", 1)[1]
        user_key = str(from_id)
        if val == "all":
            STATE.user_subscriptions[user_key] = list(range(1, 13))
        else:
            try:
                onzs_num = int(val)
            except ValueError:
                return
            subs = STATE.user_subscriptions.get(user_key, [])
            if onzs_num in subs:
                subs.remove(onzs_num)
            else:
                subs.append(onzs_num)
            STATE.user_subscriptions[user_key] = sorted(subs)

        save_state(STATE)
        new_kb = build_onzs_keyboard(STATE.user_subscriptions.get(user_key, []))
        await tg_request(
            "editMessageReplyMarkup",
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "reply_markup": new_kb,
            },
        )
        await tg_request(
            "answerCallbackQuery",
            {
                "callback_query_id": callback_query.get("id"),
                "text": "Настройки ОНзС обновлены",
                "show_alert": False,
            },
        )


# ------------------ ПРИЁМ UPDATE'ов от Telegram (WEBHOOK/POLLING) ------------------ #

OFFSET = 0


async def poll_updates():
    global OFFSET
    if not BOT_TOKEN:
        logger.warning("BOT_TOKEN не задан — часть функционала (управление ботом) не будет работать.")
        return

    while True:
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.get(
                    f"{TG_API_BASE}/bot{BOT_TOKEN}/getUpdates",
                    params={"offset": OFFSET, "timeout": 30},
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            logger.error(f"Ошибка getUpdates: {e}")
            await asyncio.sleep(5)
            continue

        if not data.get("ok"):
            await asyncio.sleep(5)
            continue

        for update in data.get("result", []):
            OFFSET = update["update_id"] + 1
            await handle_update(update)


async def handle_update(update: Dict[str, Any]):
    if "message" in update:
        await handle_message(update["message"])
    if "callback_query" in update:
        await handle_callback_query(update["callback_query"])


async def handle_message(message: Dict[str, Any]):
    chat_id = message.get("chat", {}).get("id")
    from_id = message.get("from", {}).get("id")
    text = message.get("text", "") or ""

    if not text:
        return

    if text.startswith("/"):
        cmd, *args = text.split()
        if cmd == "/start":
            await cmd_start(chat_id, from_id)
        elif cmd == "/stop":
            await cmd_stop(chat_id, from_id)
        elif cmd == "/risk":
            await cmd_risk(chat_id, from_id, args)
        elif cmd == "/chatid":
            # Универсальная команда: вернуть ID текущего чата (личный, группа, супергруппа)
            await send_tg_message(chat_id, f"Chat ID: {chat_id}")
        return

    # Обучение по тексту-ответу на карточку
    reply_to = message.get("reply_to_message")
    if reply_to and reply_to.get("text"):
        lower = text.strip().lower()
        if lower in ("в работу", "в_работу", "работа"):
            label = "в_работу"
        elif lower in ("неверно", "не относится", "не относится.", "не наш"):
            label = "неверно"
        elif lower in ("привязать", "привязка"):
            label = "привязать"
        else:
            return

        rec = {
            "text": reply_to.get("text", ""),
            "label": label,
            "timestamp": datetime.now().isoformat(),
            "from_id": from_id,
        }
        append_jsonl(YAGPT_DATASET, rec)
        append_line(ANALYTICS_LOG, f"DECISION_REPLY: {label} by {from_id}")
        await send_tg_message(chat_id, f"✅ Решение зафиксировано: {label}")


async def cmd_start(chat_id: int, user_id: int):
    user_key = str(user_id)
    subs = STATE.user_subscriptions.get(user_key, [])
    kb = build_onzs_keyboard(subs)
    STATE.user_paused[user_key] = False
    save_state(STATE)
    text = (
        "👋 Привет! Это бот мониторинга самовольного строительства.\n\n"
        "Ниже выбери, по каким ОНзС ты хочешь получать карточки.\n"
        "Можно отметить несколько, либо нажать «Все ОНзС»."
    )
    await send_tg_message(chat_id, text, kb)


async def cmd_stop(chat_id: int, user_id: int):
    user_key = str(user_id)
    STATE.user_paused[user_key] = True
    save_state(STATE)
    await send_tg_message(chat_id, "⏸ Показ карточек для тебя приостановлен. Чтобы возобновить — набери /start.")


async def cmd_risk(chat_id: int, user_id: int, args: List[str]):
    if user_id not in ADMIN_IDS:
        await send_tg_message(chat_id, "Эта команда доступна только администраторам.")
        return

    if not args:
        await send_tg_message(
            chat_id,
            f"Текущий порог вероятности самостроя: {STATE.min_risk_probability}%.\n"
            f"Измени командой: /risk 25 (от 0 до 100).",
        )
        return

    try:
        val = int(args[0])
    except ValueError:
        await send_tg_message(chat_id, "Укажи число от 0 до 100, например: /risk 15")
        return

    if not (0 <= val <= 100):
        await send_tg_message(chat_id, "Число должно быть от 0 до 100.")
        return

    STATE.min_risk_probability = val
    save_state(STATE)
    await send_tg_message(chat_id, f"✅ Новый порог вероятности самостроя: {val}%.")


# ------------------ MAIN ------------------ #

async def main():
    logger.info("🚀 Запуск Samastroi Scraper (public channels via web + rs_search_bot + карта)...")

    # Проверка критических параметров
    if not BOT_TOKEN:
        logger.error("BOT_TOKEN не задан в .env")
    if TARGET_CHAT_ID == 0:
        logger.error("TARGET_CHAT_ID не задан в .env")

    # Запускаем Telethon-клиент для @rs_search_bot
    if RS_CLIENT is not None:
        await RS_CLIENT.start()
        me = await RS_CLIENT.get_me()
        logger.info(f"Telethon-сессия активна: {me}")

    # Параллельно: опрос публичных каналов и опрос Bot API (getUpdates)
    await asyncio.gather(
        scanner_loop(),
        poll_updates(),
    )


async def scanner_loop():
    while True:
        try:
            await scan_once()
        except Exception as e:
            logger.error(f"Ошибка в scanner_loop: {e}")
        await asyncio.sleep(180)  # интервал между полными проходами по каналам


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Остановка бота по Ctrl+C")
        append_line(ANALYTICS_LOG, "STOPPED BY KEYBOARD")

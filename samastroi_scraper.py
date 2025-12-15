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

# Telegram BOT API (бот @samastroq_MO_bot)
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
TARGET_CHAT_ID = int(os.getenv("TARGET_CHAT_ID", "0") or "0")

# Начальный список администраторов (через .env, через запятую)
ENV_ADMIN_IDS: List[int] = []
_raw_admin_ids = os.getenv("ADMIN_IDS", "").strip()
if _raw_admin_ids:
    for part in _raw_admin_ids.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ENV_ADMIN_IDS.append(int(part))
        except ValueError:
            logger.warning(f"Не могу распарсить ADMIN_ID '{part}'")

# YandexGPT
YAGPT_API_KEY = os.getenv("YAGPT_API_KEY", "").strip()
YAGPT_FOLDER_ID = os.getenv("YAGPT_FOLDER_ID", "").strip()

# Яндекс Геокодер (опционально)
YANDEX_GEOCODER_KEY = os.getenv("YANDEX_GEOCODER_KEY", "").strip()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Порог вероятности самостроя по умолчанию
DEFAULT_MIN_RISK_PROBABILITY = int(os.getenv("MIN_RISK_PROBABILITY", "10") or "10")

# Telethon для работы с @rs_search_bot (опционально)
TG_API_ID = int(os.getenv("TG_API_ID", "0") or "0")
TG_API_HASH = os.getenv("TG_API_HASH", "").strip()
SESSION_NAME = os.getenv("SESSION_NAME", "samastroi_rs_session").strip()
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
HISTORY_CARDS = os.path.join(DATA_DIR, "history_cards.jsonl")
os.makedirs(ONZS_DIR, exist_ok=True)


# ------------------ ЛОГИ ------------------ #

logger.remove()
logger.add(
    os.path.join(LOGS_DIR, "samastroi_telethon.log"),
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
    (GROUPS_FILE, "# @username каналов, по одному в строке\n"),
    (
        KEYWORDS_FILE,
        "самострой\nстройка\nстроительство\nнадзор\nштраф\nразрешение на строительство\nразрешение на ввод\nучасток\nземельный участок\n",
    ),
    (MONITORING_LOG, ""),
    (ANALYTICS_LOG, ""),
    (YAGPT_DATASET, ""),
    (NEWS_FILE, ""),
    (HISTORY_CARDS, ""),
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
    user_subscriptions: Dict[str, List[int]]
    user_paused: Dict[str, bool]
    min_risk_probability: int
    bot_admin_ids: List[int]

    @staticmethod
    def default() -> "BotState":
        return BotState(
            last_post_ids={},
            user_subscriptions={},
            user_paused={},
            min_risk_probability=DEFAULT_MIN_RISK_PROBABILITY,
            bot_admin_ids=ENV_ADMIN_IDS.copy(),
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
            bot_admin_ids=data.get("bot_admin_ids", ENV_ADMIN_IDS.copy()),
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
        "и, если да, структурировать данные в JSON: объект, адрес, кадастровый номер, риск и краткое описание. "
        "Отвечай строго одним JSON-объектом, без комментариев."
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
        pos = member["Point"]["pos"]
        lon_str, lat_str = pos.split()
        return float(lat_str), float(lon_str)
    except Exception as e:
        logger.error(f"Ошибка геокодера для '{address}': {e}")
        return None


# ------------------ Telethon для @rs_search_bot (опционально) ------------------ #

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
    if RS_CLIENT is None:
        return None

    await ensure_rs_client_started()

    coords_text = f"{lat:.6f} {lon:.6f}"
    try:
        bot_entity = await RS_CLIENT.get_entity(RS_SEARCH_BOT)
        await RS_CLIENT.send_message(bot_entity, coords_text)
        resp = await RS_CLIENT.wait_for(
            events.NewMessage(from_users=bot_entity), timeout=15
        )
        return resp.raw_text
    except Exception as e:
        logger.error(f"Ошибка запроса к @rs_search_bot: {e}")
        return None


def extract_rosreestr_block(text: str) -> Optional[str]:
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


# ------------------ ОНзС ------------------ #

ONZS_MAPPING: Dict[int, List[str]] = {
    1: ["одинцовский", "наро-фоминский", "власиха", "краснознаменск", "можайск"],
    2: ["красногорск", "истра", "восход", "волоколамск", "лотошино", "руза", "шаховская"],
    3: ["химки", "солнечногорск", "долгопрудный", "лобня", "клин"],
    4: ["мытищи", "королев"],
    5: ["пушкинский", "сергиево-посад"],
    6: ["подольск", "серпухов", "чехов"],
    7: ["домодедово", "ленинский"],
    8: ["щелково", "звездный городок", "лосино-петровский", "фрязино", "черноголовка", "электросталь"],
    9: ["люберц", "котельник", "лыткарин", "балаших", "реутов"],
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
    11: ["дмитров", "дубна", "талдом"],
    12: ["орехово-зуево", "егорьевск", "павлово-посад", "шатур"],
}


def detect_onzs_by_text(text: str) -> int:
    t = text.lower()
    for onzs, patterns in ONZS_MAPPING.items():
        for p in patterns:
            if p in t:
                return onzs
    return 0


# ------------------ КАРТОЧКИ ------------------ #

def build_card_text(card: Dict[str, Any]) -> str:
    channel = card.get("channel", "-")
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

    text_lines: List[str] = []
    text_lines.append(f"Найдено в {channel}")
    text_lines.append("")
    text_lines.append("🏗 Объект и нарушение")
    text_lines.append(f"• Тип объекта: {object_type}")
    text_lines.append(f"• Тип нарушения: {violation_type}")
    text_lines.append(f"• Адрес: {address}")
    text_lines.append(f"• Округ/город: {okrug_city}")
    text_lines.append(f"• Кадастровый номер: {cadastral_number}")
    text_lines.append(f"📈 Вероятность самостроя: {risk_probability}%")
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
    card_key = f"{channel}:{post_id}"

    keyboard: List[List[Dict[str, Any]]] = [
        [
            {"text": "в работу", "callback_data": f"train:work:{card_key}"},
            {"text": "неверно", "callback_data": f"train:wrong:{card_key}"},
            {"text": "привязать", "callback_data": f"train:attach:{card_key}"},
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


# ------------------ Telegram Bot API ------------------ #

TG_API_BASE = "https://api.telegram.org"


async def tg_request(method: str, data: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{TG_API_BASE}/bot{BOT_TOKEN}/{method}"
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(url, json=data)
        r.raise_for_status()
        return r.json()


async def send_card_to_tg_group(card: Dict[str, Any]) -> Optional[int]:
    if not BOT_TOKEN:
        logger.error("BOT_TOKEN не задан — не могу отправлять карточки.")
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
        logger.info(f"Отправка карточки в chat_id={TARGET_CHAT_ID}")
        resp = await tg_request("sendMessage", data)
        if not resp.get("ok"):
            logger.error(f"Telegram API error: {resp}")
            return None
        message_id = resp["result"]["message_id"]
        return message_id
    except Exception as e:
        logger.error(f"Ошибка отправки карточки: {e}")
        return None


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


# ------------------ ПОДПИСКИ НА ОНзС ------------------ #

def build_onzs_keyboard(selected: List[int]) -> Dict[str, Any]:
    buttons: List[List[Dict[str, Any]]] = []
    row: List[Dict[str, Any]] = []
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
    # пока отключено, вся рассылка идёт только в общий чат TARGET_CHAT_ID
    return


# ------------------ СКРАПЕР Telegram Web ------------------ #

TELEGRAM_WEB_BASE = "https://t.me"


async def fetch_channel_page(username: str) -> str:
    url = f"{TELEGRAM_WEB_BASE}/s/{username.lstrip('@')}"
    logger.info(f"Запрос веб-страницы канала: {url}")
    async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
        r = await client.get(url)
        if r.status_code in (301, 302, 303, 307, 308):
            logger.error(
                f"Redirect '{r.status_code} {r.reason_phrase}' for url '{url}', "
                f"Location: '{r.headers.get('Location')}'"
            )
            raise RuntimeError(f"Redirect for {url}")
        r.raise_for_status()
        return r.text


def parse_posts_from_html(html: str) -> List[Tuple[int, str]]:
    posts: List[Tuple[int, str]] = []
    for m in re.finditer(r'data-post="[^/]+/(\d+)"', html):
        msg_id = int(m.group(1))
        start = max(0, m.start() - 2000)
        end = min(len(html), m.end() + 2000)
        snippet = html[start:end]
        snippet = re.sub(r"<[^>]+>", " ", snippet)
        snippet = re.sub(r"\s+", " ", snippet)
        posts.append((msg_id, snippet))

    unique: Dict[int, str] = {}
    for mid, txt in posts:
        if mid not in unique:
            unique[mid] = txt
    return sorted(unique.items(), key=lambda x: x[0])


async def analyze_case_with_yagpt(channel: str, msg_id: int, text: str, original_url: str) -> Optional[Dict[str, Any]]:
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

    rp = int(card.get("risk_probability") or 0)
    if rp < STATE.min_risk_probability:
        logger.info(
            f"Карточка отклонена по порогу вероятности: {rp}% < {STATE.min_risk_probability}%"
        )
        return None

    rosreestr_block = None
    coords = extract_coords(text)
    if not coords and card.get("address"):
        coords = await geocode_address(card["address"])

    if coords:
        lat, lon = coords
        rs_resp = await query_rs_search_bot_by_coords(lat, lon)
        rosreestr_block = extract_rosreestr_block(rs_resp or "")
    card["rosreestr_block"] = rosreestr_block or "-"

    onzs_text_source = f"{card.get('address', '')} {card.get('okrug_city', '')}"
    card["onzs"] = detect_onzs_by_text(onzs_text_source)

    return card


async def process_public_post(username: str, msg_id: int, text: str):
    keywords = read_lines(KEYWORDS_FILE)
    lower = text.lower()
    matched = [kw for kw in keywords if kw.lower() in lower]
    if not matched:
        return

    logger.info(f"[MATCH] @{username}: пост {msg_id}, ключевые слова {matched}")
    original_url = f"https://t.me/{username}/{msg_id}"

    card = await analyze_case_with_yagpt(
        channel=f"@{username}", msg_id=msg_id, text=text, original_url=original_url
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
                "channel": username,
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


# ------------------ РОЛИ АДМИНИСТРАТОРА ------------------ #

async def is_group_admin(chat_id: int, user_id: int) -> bool:
    if chat_id is None or chat_id > 0:
        return False
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{TG_API_BASE}/bot{BOT_TOKEN}/getChatMember",
                params={"chat_id": chat_id, "user_id": user_id},
            )
            data = resp.json()
            if not data.get("ok"):
                return False
            member = data["result"]
            status = member.get("status")
            return status in ("administrator", "creator")
    except Exception as e:
        logger.error(f"Ошибка проверки администратора группы: {e}")
        return False


async def is_bot_admin(user_id: int, chat_id: Optional[int] = None) -> bool:
    if user_id in STATE.bot_admin_ids:
        return True
    if user_id in ENV_ADMIN_IDS:
        return True
    if chat_id:
        if await is_group_admin(chat_id, user_id):
            return True
    return False


# ------------------ СТАТИСТИКА ОБУЧЕНИЯ ------------------ #

def compute_training_stats() -> Dict[str, Any]:
    total = 0
    by_label = {"в_работу": 0, "неверно": 0, "привязать": 0, "other": 0}
    last_for_text: Dict[str, str] = {}

    if not os.path.exists(YAGPT_DATASET):
        return {"total": 0, "by_label": by_label, "effective_total": 0}

    with open(YAGPT_DATASET, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            text = obj.get("text", "")
            label = obj.get("label", "other")
            total += 1
            if label not in by_label:
                by_label["other"] += 1
            else:
                by_label[label] += 1
            if text:
                last_for_text[text] = label

    effective_total = len(last_for_text)
    return {"total": total, "by_label": by_label, "effective_total": effective_total}


def build_training_stats_text() -> str:
    stats = compute_training_stats()
    total = stats["total"]
    eff = stats["effective_total"]
    by_label = stats["by_label"]

    target = 1000  # условная цель
    remaining = max(0, target - eff)

    lines: List[str] = []
    lines.append("📊 Состояние обучения YandexGPT")
    lines.append("")
    lines.append(f"Всего решений (записей): {total}")
    lines.append(f"Уникальных примеров (по последнему статусу): {eff}")
    lines.append("")
    lines.append("Разметка по статусам (последние решения):")
    lines.append(f"• В работу: {by_label.get('в_работу', 0)}")
    lines.append(f"• Неверно: {by_label.get('неверно', 0)}")
    lines.append(f"• Привязать: {by_label.get('привязать', 0)}")
    if by_label.get("other", 0):
        lines.append(f"• Прочие: {by_label.get('other', 0)}")
    lines.append("")
    lines.append(f"Текущий порог вероятности самостроя: {STATE.min_risk_probability}%")
    lines.append("")
    lines.append(f"До условного 'идеала' (цель {target} примеров) осталось ~{remaining}.")
    lines.append(
        "Чтобы изменить статус конкретной карточки — ещё раз ответь на неё "
        "словом «в работу», «неверно» или «привязать». Последнее решение считается актуальным."
    )
    return "\n".join(lines)


# ------------------ CALLBACK (кнопки) ------------------ #

async def handle_callback_query(callback_query: Dict[str, Any]):
    data = callback_query.get("data", "")
    from_id = callback_query.get("from", {}).get("id")
    message = callback_query.get("message", {})
    message_id = message.get("message_id")
    chat_id = message.get("chat", {}).get("id")

    # Открыть админ-панель по кнопке "Админ"
    if data == "admin:open":
        if not await is_bot_admin(from_id, chat_id):
            await tg_request(
                "answerCallbackQuery",
                {
                    "callback_query_id": callback_query.get("id"),
                    "text": "Только администраторы бота могут открыть админ-панель.",
                    "show_alert": True,
                },
            )
            return
        await cmd_admin(chat_id, from_id)
        await tg_request(
            "answerCallbackQuery",
            {
                "callback_query_id": callback_query.get("id"),
                "text": "Админ-панель открыта.",
                "show_alert": False,
            },
        )
        return

    # Статистика обучения
    if data == "admin:stats":
        if not await is_bot_admin(from_id, chat_id):
            await tg_request(
                "answerCallbackQuery",
                {
                    "callback_query_id": callback_query.get("id"),
                    "text": "Только администраторы бота могут смотреть статистику.",
                    "show_alert": True,
                },
            )
            return
        text = build_training_stats_text()
        await send_tg_message(chat_id, text)
        await tg_request(
            "answerCallbackQuery",
            {
                "callback_query_id": callback_query.get("id"),
                "text": "Статистика отправлена.",
                "show_alert": False,
            },
        )
        return

    # Кнопки обучения
    if data.startswith("train:"):
        if not await is_bot_admin(from_id, chat_id):
            await tg_request(
                "answerCallbackQuery",
                {
                    "callback_query_id": callback_query.get("id"),
                    "text": "Только администраторы могут обучать ИИ.",
                    "show_alert": True,
                },
            )
            return

        _, action, key = data.split(":", 2)
        channel, post_id_str = key.split(":", 1)

        label_map = {
            "work": "в_работу",
            "wrong": "неверно",
            "attach": "привязать",
        }
        label = label_map.get(action, action)
        card_text = message.get("text", "")

        rec = {
            "text": card_text,
            "label": label,
            "timestamp": datetime.now().isoformat(),
            "from_id": from_id,
            "chat_id": chat_id,
            "message_id": message_id,
            "source": "callback",
        }
        append_jsonl(YAGPT_DATASET, rec)
        append_jsonl(HISTORY_CARDS, rec)
        append_line(ANALYTICS_LOG, f"DECISION: {label} by {from_id}")

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


# ------------------ ПОЛЛИНГ getUpdates ------------------ #

OFFSET = 0


async def poll_updates():
    global OFFSET
    if not BOT_TOKEN:
        logger.warning("BOT_TOKEN не задан — управление ботом работать не будет.")
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

    # Команды
    if text.startswith("/"):
        cmd, *args = text.split()
        if cmd == "/start":
            await cmd_start(chat_id, from_id)
        elif cmd == "/stop":
            await cmd_stop(chat_id, from_id)
        elif cmd == "/risk":
            await cmd_risk(chat_id, from_id, args)
        elif cmd == "/chatid":
            await send_tg_message(chat_id, f"Chat ID: {chat_id}")
        elif cmd == "/admin":
            await cmd_admin(chat_id, from_id)
        elif cmd == "/addadmin":
            await cmd_add_admin(chat_id, from_id, args)
        elif cmd == "/deladmin":
            await cmd_del_admin(chat_id, from_id, args)
        elif cmd == "/trainstats":
            await cmd_train_stats(chat_id, from_id)
        return

    # Обучение по ответу на карточку (только для админов)
    reply_to = message.get("reply_to_message")
    if reply_to and reply_to.get("text"):
        if not await is_bot_admin(from_id, chat_id):
            return

        lower = text.strip().lower()
        if lower in ("в работу", "в_работу", "работа"):
            label = "в_работу"
        elif lower in ("неверно", "не относится", "не наш"):
            label = "неверно"
        elif lower in ("привязать", "привязка"):
            label = "привязать"
        else:
            return

        base_text = reply_to.get("text", "")
        rec = {
            "text": base_text,
            "label": label,
            "timestamp": datetime.now().isoformat(),
            "from_id": from_id,
            "chat_id": chat_id,
            "message_id": reply_to.get("message_id"),
            "source": "reply",
        }
        append_jsonl(YAGPT_DATASET, rec)
        append_jsonl(HISTORY_CARDS, rec)
        append_line(ANALYTICS_LOG, f"DECISION_REPLY: {label} by {from_id}")
        await send_tg_message(chat_id, f"✅ Решение зафиксировано: {label}")


# ------------------ КОМАНДЫ ------------------ #

async def cmd_start(chat_id: int, user_id: int):
    user_key = str(user_id)
    subs = STATE.user_subscriptions.get(user_key, [])
    kb = build_onzs_keyboard(subs)
    inline_kb = kb["inline_keyboard"]

    # для администратора добавляем кнопку "Админ"
    if await is_bot_admin(user_id, chat_id):
        inline_kb.append(
            [
                {
                    "text": "🛠 Админ",
                    "callback_data": "admin:open",
                }
            ]
        )

    STATE.user_paused[user_key] = False
    save_state(STATE)

    text = (
        "👋 Привет! Это бот мониторинга самовольного строительства.\n\n"
        "Ниже выбери, по каким ОНзС ты хочешь получать карточки (на будущее, для личных рассылок).\n"
        "Сейчас все карточки приходят в общий групповой чат.\n\n"
        "Для администратора доступна кнопка «Админ» и команда /admin."
    )
    await send_tg_message(chat_id, text, {"inline_keyboard": inline_kb})


async def cmd_stop(chat_id: int, user_id: int):
    user_key = str(user_id)
    STATE.user_paused[user_key] = True
    save_state(STATE)
    await send_tg_message(
        chat_id,
        "⏸ Показ карточек для тебя приостановлен (для личных рассылок). Чтобы возобновить — /start.",
    )


async def cmd_risk(chat_id: int, user_id: int, args: List[str]):
    if not await is_bot_admin(user_id, chat_id):
        await send_tg_message(chat_id, "Эта команда доступна только администраторам бота.")
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


async def cmd_admin(chat_id: int, user_id: int):
    if not await is_bot_admin(user_id, chat_id):
        await send_tg_message(chat_id, "Эта панель доступна только администраторам бота.")
        return

    all_admins = sorted(set(STATE.bot_admin_ids + ENV_ADMIN_IDS))
    admins_str = ", ".join(str(a) for a in all_admins) if all_admins else "нет"

    text = (
        "🛠 Админ-панель бота\n\n"
        "Администраторы могут обучать YandexGPT (кнопки и ответы), менять порог /risk,\n"
        "просматривать статистику и управлять списком админов.\n\n"
        f"Текущий список администраторов (ID): {admins_str}\n\n"
        "Команды:\n"
        "• /addadmin <id> — добавить администратора\n"
        "• /deladmin <id> — удалить администратора\n"
        "• /risk — посмотреть/изменить порог вероятности\n"
        "• /trainstats — посмотреть статистику обучения\n\n"
        "Чтобы поменять статус карточки — ещё раз ответь на неё текстом «в работу», "
        "«неверно» или «привязать». Последнее решение будет учтено."
    )

    kb = {
        "inline_keyboard": [
            [{"text": "📊 Проверка состояния обучения", "callback_data": "admin:stats"}],
        ]
    }
    await send_tg_message(chat_id, text, kb)


async def cmd_add_admin(chat_id: int, user_id: int, args: List[str]):
    if not await is_bot_admin(user_id, chat_id):
        await send_tg_message(chat_id, "Добавлять администраторов может только администратор бота.")
        return

    if not args:
        await send_tg_message(chat_id, "Использование: /addadmin <telegram_id>")
        return

    try:
        new_id = int(args[0])
    except ValueError:
        await send_tg_message(chat_id, "ID должен быть числом.")
        return

    if new_id in STATE.bot_admin_ids:
        await send_tg_message(chat_id, f"ID {new_id} уже в списке администраторов.")
        return

    STATE.bot_admin_ids.append(new_id)
    save_state(STATE)
    await send_tg_message(chat_id, f"✅ ID {new_id} добавлен в администраторы бота.")


async def cmd_del_admin(chat_id: int, user_id: int, args: List[str]):
    if not await is_bot_admin(user_id, chat_id):
        await send_tg_message(chat_id, "Удалять администраторов может только администратор бота.")
        return

    if not args:
        await send_tg_message(chat_id, "Использование: /deladmin <telegram_id>")
        return

    try:
        del_id = int(args[0])
    except ValueError:
        await send_tg_message(chat_id, "ID должен быть числом.")
        return

    if del_id in STATE.bot_admin_ids:
        STATE.bot_admin_ids.remove(del_id)
        save_state(STATE)
        await send_tg_message(chat_id, f"✅ ID {del_id} удалён из администраторов бота.")
    else:
        await send_tg_message(chat_id, f"ID {del_id} не найден в списке администраторов.")


async def cmd_train_stats(chat_id: int, user_id: int):
    if not await is_bot_admin(user_id, chat_id):
        await send_tg_message(chat_id, "Статистику обучения могут смотреть только администраторы.")
        return
    text = build_training_stats_text()
    await send_tg_message(chat_id, text)


# ------------------ MAIN ------------------ #

async def scanner_loop():
    while True:
        try:
            await scan_once()
        except Exception as e:
            logger.error(f"Ошибка в scanner_loop: {e}")
        await asyncio.sleep(180)


async def main():
    logger.info("🚀 Запуск samastroi_telethon...")

    if not BOT_TOKEN:
        logger.error("BOT_TOKEN не задан в .env")
    if TARGET_CHAT_ID == 0:
        logger.error("TARGET_CHAT_ID не задан в .env")

    if RS_CLIENT is not None:
        await RS_CLIENT.start()
        me = await RS_CLIENT.get_me()
        logger.info(f"Telethon-сессия для @rs_search_bot активна: {me}")

    await asyncio.gather(
        scanner_loop(),
        poll_updates(),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Остановка бота по Ctrl+C")
        append_line(ANALYTICS_LOG, "STOPPED BY KEYBOARD")

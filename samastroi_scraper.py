import asyncio
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from loguru import logger

# ---------------------------------------------------------------------
#  ЗАГРУЗКА .env и БАЗОВЫЕ ПУТИ
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
ONZS_DIR = DATA_DIR / "onzs"

DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
ONZS_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(BASE_DIR / ".env")

# ---------------------------------------------------------------------
#  КОНФИГ И ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ
# ---------------------------------------------------------------------

YAGPT_API_KEY = os.getenv("YAGPT_API_KEY", "").strip()
YAGPT_FOLDER_ID = os.getenv("YAGPT_FOLDER_ID", "").strip()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

TELECOM_LOCATOR_BASE_URL = os.getenv("TELECOM_LOCATOR_BASE_URL", "").strip()
TELECOM_LOCATOR_API_KEY = os.getenv("TELECOM_LOCATOR_API_KEY", "").strip()

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
TARGET_CHAT_ID = int(os.getenv("TARGET_CHAT_ID", "0") or "0")
NEWS_THREAD_ID = int(os.getenv("NEWS_THREAD_ID", "0") or "0")

SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "600") or "600")

# Список админов для обучения (через запятую: ADMIN_IDS=111,222,333)
ADMIN_IDS_ENV = os.getenv("ADMIN_IDS", "").strip()
if ADMIN_IDS_ENV:
    ADMIN_IDS: Set[int] = {
        int(x)
        for x in ADMIN_IDS_ENV.replace(" ", "").split(",")
        if x
    }
else:
    ADMIN_IDS = set()  # пусто → обучать могут все

TELEGRAM_API_URL = (
    f"https://api.telegram.org/bot{BOT_TOKEN}" if BOT_TOKEN else ""
)

# ---------------------------------------------------------------------
#  ЛОГИРОВАНИЕ
# ---------------------------------------------------------------------

logger.remove()
logger.add(
    LOGS_DIR / "samastroi_scraper.log",
    rotation="10 MB",
    encoding="utf-8",
    level=LOG_LEVEL,
)
logger.add(lambda m: print(m, end=""), level=LOG_LEVEL)

# ---------------------------------------------------------------------
#  ФАЙЛЫ ДАННЫХ
# ---------------------------------------------------------------------

GROUPS_FILE = DATA_DIR / "groups.txt"
KEYWORDS_FILE = DATA_DIR / "keywords.txt"
STATE_FILE = DATA_DIR / "state.json"
CARDS_FILE = DATA_DIR / "cards.jsonl"
YAGPT_DATASET_FILE = DATA_DIR / "yagpt_dataset.jsonl"
SUBSCRIBERS_FILE = DATA_DIR / "subscribers.json"
MONITORING_LOG = DATA_DIR / "monitoring.log"
ANALYTICS_LOG = DATA_DIR / "analytics.log"

for path, default in [
    (GROUPS_FILE, "# @username каналов/чатов, по одному в строке\n"),
    (
        KEYWORDS_FILE,
        "самострой\nстройка\nстроительство\nстройплощадка\nнадзор\nзастройщик\nразрешение на строительство\nввод в эксплуатацию\n",
    ),
    (STATE_FILE, "{}"),
    (CARDS_FILE, ""),
    (YAGPT_DATASET_FILE, ""),
    (SUBSCRIBERS_FILE, "[]"),
    (MONITORING_LOG, ""),
    (ANALYTICS_LOG, ""),
]:
    if not path.exists():
        path.write_text(default, encoding="utf-8")

# ---------------------------------------------------------------------
#  ОНЗС: СПРАВОЧНИК
# ---------------------------------------------------------------------

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
    3: ["химки", "солнечногорск", "долгопрудный", "лобня", "клин"],
    4: ["мытищи", "королев"],
    5: ["пушкинский", "сергиево-посад", "сергиев посад"],
    6: ["подольск", "серпухов", "чехов"],
    7: ["домодедово", "ленинский"],
    8: [
        "щелково",
        "звездный городок",
        "лосино-петровский",
        "фрязино",
        "черноголовка",
        "электросталь",
    ],
    9: ["люберцы", "котельники", "лыткарино", "балашиха", "реутов"],
    10: [
        "коломна",
        "воскресенск",
        "зарайск",
        "кашир",
        "луховиц",
        "раменск",
        "бронницы",
        "жуковский",
        "серебряные пруды",
        "ступино",
    ],
    11: ["дмитров", "дубна", "талдом"],
    12: ["орехово-зуев", "егорьевск", "павлово-посад", "шатура"],
}

ONZS_PROMPT_TEXT = """
ОНзС 1 - Одинцовский г.о.; Наро-Фоминский г.о. (Одинцовский г.о., г.о. Власиха, г.о. Краснознаменск, Наро-Фоминский г.о., г.о. Можайск)
ОНзС 2 - г.о. Красногорск; м.о. Истра (г.о. Красногорск, м.о. Истра, г.о. Восход, Волоколамский м.о., м.о. Лотошино, Рузский м.о., м.о. Шаховская)
ОНзС 3 - г.о. Химки; г.о. Солнечногорск (г.о. Химки, г.о. Солнечногорск, г.о. Долгопрудный, г.о. Лобня, г.о. Клин)
ОНзС 4 - г.о. Мытищи (г.о. Мытищи, г.о. Королев)
ОНзС 5 - г.о. Пушкинский (г.о. Пушкинский, Сергиево-Посадский г.о.)
ОНзС 6 - г.о. Подольск (г.о. Подольск, г.о. Серпухов, м.о. Чехов)
ОНзС 7 - г.о. Домодедово (г.о. Домодедово, Ленинский г.о.)
ОНзС 8 - г.о. Щелково (г.о. Щелково, г.о. Звездный городок, г.о. Лосино-Петровский, г.о. Фрязино, г.о. Черноголовка, г.о. Электросталь)
ОНзС 9 - г.о. Люберцы; г.о. Балашиха (г.о. Люберцы, г.о. Котельники, г.о. Лыткарино, г.о. Балашиха, г.о. Реутов)
ОНзС 10 - г.о. Коломна; г.о. Ступино (г.о. Коломна, г.о. Воскресенск, м.о. Зарайск, г.о. Кашира, м.о. Луховицы, Раменский м.о., г.о. Бронницы, г.о. Жуковский, м.о. Серебряные Пруды, г.о. Ступино)
ОНзС 11 - Дмитровский м.о. (Дмитровский м.о., г.о. Дубна, Талдомский г.о.)
ОНзС 12 - Орехово-Зуевский г.о. (Орехово-Зуевский г.о., м.о. Егорьевск, Павлово-Посадский г.о., м.о. Шатура)
"""

# ---------------------------------------------------------------------
#  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ---------------------------------------------------------------------


def read_lines(path: Path) -> List[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def append_line(path: Path, text: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{now}] {text}\n")


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_state() -> Dict[str, Any]:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(state: Dict[str, Any]) -> None:
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def load_subscribers() -> Set[int]:
    try:
        data = json.loads(SUBSCRIBERS_FILE.read_text(encoding="utf-8"))
        return {int(x) for x in data}
    except Exception:
        return set()


def save_subscribers(subs: Set[int]) -> None:
    SUBSCRIBERS_FILE.write_text(
        json.dumps(sorted(list(subs))), encoding="utf-8"
    )


SUBSCRIBERS: Set[int] = load_subscribers()

# ---------------------------------------------------------------------
#  YANDEX GPT
# ---------------------------------------------------------------------

YAGPT_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"


async def call_yandex_gpt(prompt: str, temperature: float = 0.2) -> Optional[str]:
    if not (YAGPT_API_KEY and YAGPT_FOLDER_ID):
        logger.warning("YandexGPT не настроен (нет API_KEY или FOLDER_ID).")
        return None

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {YAGPT_API_KEY}",
        "x-folder-id": YAGPT_FOLDER_ID,
    }

    payload = {
        "modelUri": f"gpt://{YAGPT_FOLDER_ID}/yandexgpt/latest",
        "completionOptions": {
            "maxTokens": 512,
            "temperature": temperature,
            "stream": False,
        },
        "messages": [
            {
                "role": "system",
                "text": (
                    "Ты помощник инспектора Главгосстройнадзора Московской области. "
                    "Твоя задача — находить признаки САМОСТРОЯ и структурировать информацию.\n\n"
                    "Классифицируй сообщение и ответь СТРОГО в виде JSON БЕЗ ``` и комментариев.\n"
                    "Структура JSON:\n"
                    "{\n"
                    '  "object_type": "тип объекта",\n'
                    '  "violation_type": "тип нарушения",\n'
                    '  "address": "полный адрес, если есть",\n'
                    '  "okrug_city": "муниципалитет/город",\n'
                    '  "cadastral_number": "кадастровый номер или пусто",\n'
                    '  "risk_probability": 0-100,\n'
                    '  "risk_score": 0-100,\n'
                    '  "risk_level": "низкий/средний/высокий",\n'
                    '  "summary": "кратко по сути на русском",\n'
                    '  "municipality": "муниципалитет (например, Химки, Подольск и т.п.)",\n'
                    '  "onzs_number": 0-12  // номер ОНзС, 0 если определить нельзя\n'
                    "}\n\n"
                    "Информация об округах и ОНзС:\n"
                    f"{ONZS_PROMPT_TEXT}\n"
                    "Если это явно не про строительство в Московской области, ставь risk_probability и risk_score близко к 0, "
                    "onzs_number = 0, summary всё равно сформулируй."
                ),
            },
            {
                "role": "user",
                "text": prompt,
            },
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=40) as client:
            resp = await client.post(YAGPT_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            text = data["result"]["alternatives"][0]["message"]["text"]
            return text
    except Exception as e:
        logger.error(f"Ошибка при обращении к YandexGPT: {e}")
        append_line(ANALYTICS_LOG, f"YAGPT_ERROR: {e}")
        return None


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    YandexGPT иногда заворачивает JSON в ``` ... ```.
    Пытаемся вытащить подстроку между первым '{' и последней '}'.
    """
    if not text:
        return None
    try:
        # если вдруг уже голый JSON
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except Exception:
        logger.error(f"Не удалось распарсить JSON из YAGPT: raw={text!r}")
        return None


def infer_onzs_from_location(location_text: str) -> int:
    """
    Простая эвристика по подстрокам, если YAGPT не смог.
    """
    if not location_text:
        return 0
    lower = location_text.lower()
    for num, subs in ONZS_MAPPING.items():
        for s in subs:
            if s in lower:
                return num
    return 0


async def analyze_case_with_yagpt(
    channel: str, text: str, post_url: str
) -> Optional[Dict[str, Any]]:
    prompt = (
        f"Канал: @{channel}\n"
        f"Ссылка на сообщение: {post_url}\n\n"
        f"Текст сообщения:\n{text}\n"
    )

    raw = await call_yandex_gpt(prompt)
    if raw is None:
        return None

    data = extract_json_from_text(raw)
    if data is None:
        return None

    object_type = data.get("object_type") or ""
    violation_type = data.get("violation_type") or ""
    address = data.get("address") or ""
    okrug_city = data.get("okrug_city") or ""
    cadastral_number = data.get("cadastral_number") or ""
    risk_probability = int(float(data.get("risk_probability") or 0))
    risk_score = int(float(data.get("risk_score") or 0))
    risk_level = (data.get("risk_level") or "").strip() or "не определён"
    summary = data.get("summary") or ""
    municipality = data.get("municipality") or okrug_city or address

    onzs_number = int(data.get("onzs_number") or 0)
    if onzs_number not in range(1, 13):
        onzs_number = infer_onzs_from_location(municipality)

    result: Dict[str, Any] = {
        "object_type": object_type,
        "violation_type": violation_type,
        "address": address,
        "okrug_city": okrug_city,
        "cadastral_number": cadastral_number,
        "risk_probability": risk_probability,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "summary": summary,
        "municipality": municipality,
        "onzs_number": onzs_number,
        "source_channel": channel,
        "post_url": post_url,
        "created_at": datetime.utcnow().isoformat(),
        "raw_model_json": data,
    }
    return result


# ---------------------------------------------------------------------
#  ФОРМИРОВАНИЕ КАРТОЧКИ И СОХРАНЕНИЕ
# ---------------------------------------------------------------------


def build_card_text(card: Dict[str, Any]) -> str:
    ch = card.get("source_channel", "")
    post_url = card.get("post_url", "")
    object_type = card.get("object_type") or "-"
    violation_type = card.get("violation_type") or "-"
    address = card.get("address") or "-"
    okrug_city = card.get("okrug_city") or "-"
    cadastral_number = card.get("cadastral_number") or "-"
    risk_probability = card.get("risk_probability", 0)
    risk_score = card.get("risk_score", 0)
    risk_level = card.get("risk_level") or "-"
    summary = card.get("summary") or "-"
    onzs_number = card.get("onzs_number") or 0

    lines = []
    lines.append(f"🔍 Найдено в @{ch}")
    lines.append("")
    lines.append("🏗 Объект и нарушение")
    lines.append(f"• Тип объекта: {object_type}")
    lines.append(f"• Тип нарушения: {violation_type}")
    lines.append(f"• Адрес: {address}")
    lines.append(f"• Округ/город: {okrug_city}")
    lines.append(f"• Кадастровый номер: {cadastral_number or '-'}")
    lines.append(f"• ОНзС: {onzs_number or '-'}")
    lines.append(f"📈 Вероятность самостроя: {risk_probability}%")
    lines.append(f"🧠 Итоговый риск ИИ: {risk_level} ({risk_score} из 100)")
    lines.append("")
    lines.append("📝 Кратко по сути:")
    lines.append(summary)
    lines.append("")
    lines.append("📑 Данные Росреестра")
    lines.append("📘 Данные НСПД (nspd.gov.ru)")
    lines.append("• Вид объекта недвижимости: -")
    lines.append("• Вид земельного участка: -")
    lines.append("• Дата присвоения: -")
    lines.append("• Кадастровый номер: -")
    lines.append("• Кадастровый квартал: -")
    lines.append("• Адрес: -")
    lines.append("• Площадь уточненная: -")
    lines.append("• Статус: -")
    lines.append("• Категория земель: -")
    lines.append("• Вид разрешенного использования: -")
    lines.append("• Форма собственности: -")
    lines.append("• Кадастровая стоимость: -")
    lines.append("• Удельный показатель кадастровой стоимости: -")
    lines.append("")
    if post_url:
        lines.append("🔗 Открыть оригинал сообщения")
        lines.append(post_url)
    lines.append("")
    lines.append(
        "🧠 Обучение: ответь на эту карточку кнопками «в работу», «неверно» или «привязать»."
    )
    return "\n".join(lines)


def save_card_to_onzs_files(card: Dict[str, Any]) -> None:
    append_jsonl(CARDS_FILE, card)

    all_file = ONZS_DIR / "onzs_all.jsonl"
    append_jsonl(all_file, card)

    onzs_number = int(card.get("onzs_number") or 0)
    if onzs_number in range(1, 13):
        onzs_file = ONZS_DIR / f"onzs_{onzs_number}.jsonl"
        append_jsonl(onzs_file, card)


def build_inline_keyboard(channel: str, post_id: int) -> Dict[str, Any]:
    """
    callback_data формата: train:<action>:<channel>:<post_id>
    """
    card_key = f"{channel}:{post_id}"
    return {
        "inline_keyboard": [
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
    }


def build_reply_keyboard() -> Dict[str, Any]:
    return {
        "keyboard": [
            [{"text": "Старт"}, {"text": "Стоп"}],
            [{"text": "ОНзС 1"}, {"text": "ОНзС 2"}, {"text": "ОНзС 3"}],
            [{"text": "ОНзС 4"}, {"text": "ОНзС 5"}, {"text": "ОНзС 6"}],
            [{"text": "ОНзС 7"}, {"text": "ОНзС 8"}, {"text": "ОНзС 9"}],
            [{"text": "ОНзС 10"}, {"text": "ОНзС 11"}, {"text": "ОНзС 12"}],
        ],
        "resize_keyboard": True,
        "is_persistent": True,
    }


# ---------------------------------------------------------------------
#  TELEGRAM API
# ---------------------------------------------------------------------


async def tg_request(
    method: str, payload: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    if not TELEGRAM_API_URL:
        return None
    url = f"{TELEGRAM_API_URL}/{method}"
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(url, json=payload)
            data = resp.json()
            if not data.get("ok"):
                logger.error(f"Telegram API error {method}: {data}")
            return data
    except Exception as e:
        logger.error(f"Telegram API exception {method}: {e}")
        return None


async def send_card_to_tg_group(
    card: Dict[str, Any], channel: str, post_id: int
) -> Optional[int]:
    text = build_card_text(card)
    payload: Dict[str, Any] = {
        "chat_id": TARGET_CHAT_ID,
        "text": text,
        "disable_web_page_preview": False,
        "reply_markup": build_inline_keyboard(channel, post_id),
    }
    if NEWS_THREAD_ID:
        payload["message_thread_id"] = NEWS_THREAD_ID

    resp = await tg_request("sendMessage", payload)
    if resp and resp.get("ok"):
        msg_id = resp["result"]["message_id"]
        return msg_id
    return None


async def send_card_to_subscribers(
    card: Dict[str, Any], channel: str, post_id: int
) -> None:
    """
    Отправляет карточку всем, кто нажал Старт — с теми же кнопками обучения.
    """
    if not SUBSCRIBERS or not BOT_TOKEN:
        return

    text = build_card_text(card)
    markup = build_inline_keyboard(channel, post_id)

    async with httpx.AsyncClient(timeout=20) as client:
        to_remove: Set[int] = set()
        for user_id in list(SUBSCRIBERS):
            try:
                resp = await client.post(
                    f"{TELEGRAM_API_URL}/sendMessage",
                    json={
                        "chat_id": user_id,
                        "text": text,
                        "reply_markup": markup,
                    },
                )
                data = resp.json()
                if not data.get("ok") and data.get("error_code") == 403:
                    logger.warning(
                        f"Пользователь {user_id} заблокировал бота — удаляю из подписчиков."
                    )
                    to_remove.add(user_id)
            except Exception as e:
                logger.error(f"Ошибка отправки карточки подписчику {user_id}: {e}")

        if to_remove:
            for uid in to_remove:
                SUBSCRIBERS.discard(uid)
            save_subscribers(SUBSCRIBERS)


async def send_control_keyboard() -> None:
    if not BOT_TOKEN or not TARGET_CHAT_ID:
        return
    await tg_request(
        "sendMessage",
        {
            "chat_id": TARGET_CHAT_ID,
            "text": "Панель Samostroi Scraper:",
            "reply_markup": build_reply_keyboard(),
        },
    )


# ---------------------------------------------------------------------
#  ОБРАБОТКА TELEGRAM-ОБНОВЛЕНИЙ (Старт/Стоп, ОНзС, КНОПКИ)
# ---------------------------------------------------------------------

LAST_UPDATE_ID: int = 0


async def handle_message_update(update: Dict[str, Any]) -> None:
    msg = update["message"]
    chat = msg["chat"]
    chat_id = chat["id"]
    text = (msg.get("text") or "").strip()
    if not text:
        return

    from_user = msg.get("from") or {}
    user_id = msg.get("from", {}).get("id")

    lower = text.lower()

    start_triggers = {"старт", "/start"}
    stop_triggers = {"стоп", "/stop"}
    if lower in start_triggers:
        SUBSCRIBERS.add(int(user_id))
        save_subscribers(SUBSCRIBERS)
        await tg_request(
            "sendMessage",
            {
                "chat_id": chat_id,
                "text": "✅ Вы подписались: бот будет присылать вам карточки в личные сообщения.",
                "reply_markup": build_reply_keyboard(),
            },
        )
        return
    if lower in stop_triggers:
        if int(user_id) in SUBSCRIBERS:
            SUBSCRIBERS.discard(int(user_id))
            save_subscribers(SUBSCRIBERS)
            text_resp = "⏸ Вы отписались: бот больше не будет присылать вам карточки в личку."
        else:
            text_resp = "Вы и так не подписаны на личную рассылку."
        await tg_request(
            "sendMessage",
            {
                "chat_id": chat_id,
                "text": text_resp,
                "reply_markup": build_reply_keyboard(),
            },
        )
        return

    # Кнопки "ОНзС N"
    m = re.match(r"онзс\s*(\d+)", lower)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 12:
            file_path = ONZS_DIR / f"onzs_{n}.jsonl"
            if file_path.exists():
                count = sum(1 for _ in file_path.open("r", encoding="utf-8"))
            else:
                count = 0
            await tg_request(
                "sendMessage",
                {
                    "chat_id": chat_id,
                    "text": f"📂 В папке ОНзС {n} сохранено карточек: {count}.",
                    "reply_markup": build_reply_keyboard(),
                },
            )
        else:
            await tg_request(
                "sendMessage",
                {
                    "chat_id": chat_id,
                    "text": "Номер ОНзС должен быть от 1 до 12.",
                    "reply_markup": build_reply_keyboard(),
                },
            )
        return

    # Любое другое сообщение — просто обновляем клавиатуру
    await tg_request(
        "sendMessage",
        {
            "chat_id": chat_id,
            "text": "Используйте клавиатуру ниже для управления ботом.",
            "reply_markup": build_reply_keyboard(),
        },
    )


async def handle_callback_update(update: Dict[str, Any]) -> None:
    cb = update["callback_query"]
    data = cb.get("data") or ""
    from_user = cb.get("from") or {}
    user_id = int(from_user.get("id"))
    callback_id = cb.get("id")

    message = cb.get("message") or {}
    chat_id = message.get("chat", {}).get("id")
    message_id = message.get("message_id")

    # Разбираем data
    parts = data.split(":")
    if len(parts) < 3 or parts[0] != "train":
        # Просто ответим, чтобы кнопка "крутилась" недолго
        await tg_request(
            "answerCallbackQuery",
            {"callback_query_id": callback_id, "text": "Ок"},
        )
        return

    action = parts[1]  # work / wrong / attach
    card_key = ":".join(parts[2:])
    label_map = {
        "work": "в_работу",
        "wrong": "неверно",
        "attach": "привязать",
    }
    label = label_map.get(action, action)

    # Проверка прав: если ADMIN_IDS не пустой — только они
    if ADMIN_IDS:
        can_train = user_id in ADMIN_IDS
    else:
        can_train = True

    if can_train:
        rec = {
            "card_key": card_key,
            "label": label,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
        }
        append_jsonl(YAGPT_DATASET_FILE, rec)
        append_line(ANALYTICS_LOG, f"TRAIN: {rec}")
        answer_text = "✅ Решение зафиксировано и добавлено в датасет."
    else:
        answer_text = "ℹ️ Обучение учитывается только от администратора."

    # Убираем кнопки под этим сообщением для всех
    if chat_id and message_id:
        await tg_request(
            "editMessageReplyMarkup",
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "reply_markup": {"inline_keyboard": []},
            },
        )

    await tg_request(
        "answerCallbackQuery",
        {"callback_query_id": callback_id, "text": answer_text, "show_alert": False},
    )


async def poll_updates_loop() -> None:
    global LAST_UPDATE_ID
    if not BOT_TOKEN:
        return

    logger.info("Запускаю цикл получения обновлений Telegram (getUpdates)...")
    while True:
        try:
            params: Dict[str, Any] = {"timeout": 30}
            if LAST_UPDATE_ID:
                params["offset"] = LAST_UPDATE_ID + 1

            async with httpx.AsyncClient(timeout=35) as client:
                resp = await client.get(
                    f"{TELEGRAM_API_URL}/getUpdates", params=params
                )
                data = resp.json()
        except Exception as e:
            logger.error(f"Ошибка getUpdates: {e}")
            await asyncio.sleep(5)
            continue

        if not data.get("ok"):
            logger.error(f"Ошибка getUpdates: {data}")
            await asyncio.sleep(5)
            continue

        for upd in data.get("result", []):
            LAST_UPDATE_ID = upd["update_id"]
            if "message" in upd:
                await handle_message_update(upd)
            elif "callback_query" in upd:
                await handle_callback_update(upd)


# ---------------------------------------------------------------------
#  WEB-SCRAPING TELEGRAM PUBLIC CHANNELS
# ---------------------------------------------------------------------


async def fetch_channel_page(channel: str) -> Optional[str]:
    url = f"https://t.me/s/{channel}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0 Safari/537.36"
        )
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers=headers, follow_redirects=False)
            if resp.status_code == 302:
                logger.warning(
                    f"Redirect при обращении к {url}: {resp.headers.get('Location')}"
                )
                return None
            resp.raise_for_status()
            return resp.text
    except Exception as e:
        logger.error(f"Ошибка загрузки канала {channel}: {e}")
        return None


def parse_public_posts(
    channel: str, html: str
) -> List[Tuple[int, str, str]]:
    """
    Возвращает список (post_id, text, url).
    """
    soup = BeautifulSoup(html, "html.parser")
    results: List[Tuple[int, str, str]] = []

    for msg in soup.select("div.tgme_widget_message_wrap"):
        date_a = msg.select_one("a.tgme_widget_message_date")
        if not date_a:
            continue
        href = date_a.get("href", "")
        m = re.search(r"/(\d+)$", href)
        if not m:
            continue
        post_id = int(m.group(1))
        text_div = msg.select_one("div.tgme_widget_message_text")
        text = text_div.get_text("\n", strip=True) if text_div else ""
        if not text:
            continue
        results.append((post_id, text, href))

    results.sort(key=lambda x: x[0])
    return results


async def process_public_post(
    channel: str,
    post_id: int,
    text: str,
    post_url: str,
    keywords: List[str],
) -> None:
    lower = text.lower()
    matched = [kw for kw in keywords if kw.lower() in lower]
    if not matched:
        return

    logger.info(f"[MATCH] @{channel}: пост {post_id}, ключевые слова {matched}")
    append_line(
        MONITORING_LOG,
        json.dumps(
            {
                "channel": channel,
                "post_id": post_id,
                "keywords": matched,
                "text": text[:500],
            },
            ensure_ascii=False,
        ),
    )

    analysis = await analyze_case_with_yagpt(channel, text, post_url)
    if not analysis:
        logger.warning(f"YAGPT вернул пустой результат для @{channel}/{post_id}")
        return

    card = analysis.copy()
    card["telegram_post_id"] = post_id

    save_card_to_onzs_files(card)

    # В группу
    await send_card_to_tg_group(card, channel, post_id)
    # Подписчикам
    await send_card_to_subscribers(card, channel, post_id)


async def scan_once(state: Dict[str, Any]) -> None:
    groups_raw = read_lines(GROUPS_FILE)
    if not groups_raw:
        logger.warning("В groups.txt нет ни одного канала для мониторинга.")
        return

    keywords = read_lines(KEYWORDS_FILE)
    if not keywords:
        logger.warning("В keywords.txt нет ключевых слов.")
        return

    for grp in groups_raw:
        channel = grp.lstrip("@")
        logger.info(f"Обработка канала @{channel} ...")
        html = await fetch_channel_page(channel)
        if not html:
            continue

        posts = parse_public_posts(channel, html)
        if not posts:
            logger.info(f"В @{channel} нет постов или не удалось распарсить.")
            continue

        last_processed = int(state.get(channel, 0) or 0)
        new_posts = [p for p in posts if p[0] > last_processed]
        if not new_posts:
            logger.info(f"Новых постов в @{channel} не найдено.")
            continue

        logger.info(f"Найдено новых постов в @{channel}: {len(new_posts)}")
        for post_id, text, url in new_posts:
            await process_public_post(channel, post_id, text, url, keywords)
            state[channel] = max(state.get(channel, 0), post_id)

        save_state(state)


async def scan_loop() -> None:
    state = load_state()
    while True:
        try:
            await scan_once(state)
        except Exception as e:
            logger.error(f"Ошибка в scan_once: {e}")
        await asyncio.sleep(SCAN_INTERVAL)


# ---------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------


async def main() -> None:
    if not BOT_TOKEN:
        logger.error("BOT_TOKEN не задан в .env")
        return
    if not TARGET_CHAT_ID:
        logger.error("TARGET_CHAT_ID не задан в .env")
        return

    logger.info("🚀 Запуск Samastroi Scraper (public channels via web)...")

    # Показать клавиатуру в группе
    await send_control_keyboard()

    await asyncio.gather(
        scan_loop(),
        poll_updates_loop(),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Остановка Samastroi Scraper по Ctrl+C")

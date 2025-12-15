#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAMASTROI SCRAPER (single-file)
- Telegram web-scrape of public channels (t.me/s/<channel>)
- Card generation + persistent storage (/app/data)
- Inline buttons: ✅ В работу / ❌ Неверно / 📎 Привязать (admins only)
- Training dataset JSONL + history JSONL (never deleted by code changes if DATA_DIR is persistent)
- Admin menu (/admin): stats, plot, threshold, admins management
- YandexGPT probability enrichment + adaptive (rule-based) prior + few-shot prompt steering
- Auto-filter cards below probability threshold
"""

import os
import json
import time
import uuid
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple

import requests
from bs4 import BeautifulSoup


# ==========================================================
# LOGGING
# ==========================================================
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
log = logging.getLogger("samastroi_scraper")


# ==========================================================
# STORAGE PATHS (Persistent if DATA_DIR is mounted)
# ==========================================================
DATA_DIR = os.getenv("DATA_DIR", "/app/data")
os.makedirs(DATA_DIR, exist_ok=True)

CARDS_DIR = os.path.join(DATA_DIR, "cards")
os.makedirs(CARDS_DIR, exist_ok=True)

TRAINING_DATASET = os.path.join(DATA_DIR, "training_dataset.jsonl")  # immutable append-only log
HISTORY_CARDS = os.path.join(DATA_DIR, "history_cards.jsonl")        # immutable append-only log
ADMINS_FILE = os.path.join(DATA_DIR, "admins.json")
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")
ADAPTIVE_RULES_FILE = os.path.join(DATA_DIR, "adaptive_rules.json")


def ensure_file(path: str, default_content: Optional[str] = None) -> None:
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            if default_content is not None:
                f.write(default_content)
        log.info(f"Created file: {path}")


ensure_file(TRAINING_DATASET)
ensure_file(HISTORY_CARDS)
ensure_file(ADMINS_FILE, default_content="[]")
ensure_file(SETTINGS_FILE, default_content="{}")
ensure_file(ADAPTIVE_RULES_FILE, default_content="{}")


# ==========================================================
# DEFAULT ADMINS + ADMIN STORAGE
# ==========================================================
DEFAULT_ADMINS = [
    5685586625,
    272923789,
    398960707,
    777464055,
    978125225,
]


def load_admins() -> List[int]:
    try:
        with open(ADMINS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            admins = []
            for x in data:
                try:
                    admins.append(int(x))
                except Exception:
                    continue
            # If file is empty list — restore defaults (common Railway first-run case)
            if not admins:
                admins = list(DEFAULT_ADMINS)
                with open(ADMINS_FILE, "w", encoding="utf-8") as wf:
                    json.dump(admins, wf, ensure_ascii=False)
            log.info(f"Loaded admins: {admins}")
            return admins
    except Exception as e:
        log.error(f"Failed to load admins.json: {e}")

    # fallback
    try:
        with open(ADMINS_FILE, "w", encoding="utf-8") as wf:
            json.dump(DEFAULT_ADMINS, wf, ensure_ascii=False)
    except Exception:
        pass
    log.info(f"Restored default admins: {DEFAULT_ADMINS}")
    return list(DEFAULT_ADMINS)


ADMINS: List[int] = load_admins()


def save_admins() -> None:
    with open(ADMINS_FILE, "w", encoding="utf-8") as f:
        json.dump(ADMINS, f, ensure_ascii=False)
    log.info(f"Saved admins: {ADMINS}")


def is_admin(user_id: int) -> bool:
    return int(user_id) in set(ADMINS)


def add_admin(user_id: int) -> bool:
    uid = int(user_id)
    if uid in ADMINS:
        return False
    ADMINS.append(uid)
    save_admins()
    return True


def remove_admin(user_id: int) -> bool:
    uid = int(user_id)
    if uid not in ADMINS:
        return False
    ADMINS.remove(uid)
    save_admins()
    return True


# ==========================================================
# SETTINGS (threshold etc.)
# ==========================================================
def load_settings() -> Dict[str, Any]:
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {}


def save_settings(settings: Dict[str, Any]) -> None:
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)


SETTINGS = load_settings()
DEFAULT_THRESHOLD = int(os.getenv("PROB_THRESHOLD", str(SETTINGS.get("prob_threshold", 0) or 0)))


def get_threshold() -> int:
    v = SETTINGS.get("prob_threshold", DEFAULT_THRESHOLD)
    try:
        v = int(v)
    except Exception:
        v = DEFAULT_THRESHOLD
    if v < 0:
        v = 0
    if v > 100:
        v = 100
    return v


def set_threshold(v: int) -> int:
    v = int(v)
    if v < 0:
        v = 0
    if v > 100:
        v = 100
    SETTINGS["prob_threshold"] = v
    save_settings(SETTINGS)
    return v


# ==========================================================
# TARGET CHAT / BOT API
# ==========================================================
TARGET_CHAT_ID = int(os.getenv("TARGET_CHAT_ID", "-1003502443229"))

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
TELEGRAM_API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}" if BOT_TOKEN else None

SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "300"))
MAX_CARDS_LIST = int(os.getenv("MAX_CARDS_LIST", "20"))
MAX_HISTORY_EVENTS = int(os.getenv("MAX_HISTORY_EVENTS", "50"))
TARGET_DATASET_SIZE = int(os.getenv("TARGET_DATASET_SIZE", "5000"))

if not BOT_TOKEN:
    log.warning("BOT_TOKEN is not set. Sending cards and admin controls will not work.")
log.info(f"Cards will be sent to chat_id: {TARGET_CHAT_ID}")
log.info(f"DATA_DIR={DATA_DIR} (must be persistent volume if you want training/history to survive redeploys)")


# ==========================================================
# KEYWORDS
# ==========================================================
KEYWORDS = [
    "стройка", "строительство", "самострой", "котлован", "фундамент",
    "арматура", "многоквартирный", "жилой комплекс", "кран", "экскаватор",
    "строители", "проверка", "застройщик", "разрешение", "рнс", "благоустройство",
    "снос", "надзор", "мчс", "инженер", "штраф",
]
KEYWORDS_LOWER = [k.lower() for k in KEYWORDS]


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.replace("\n", " ").replace("\t", " ")
    t = " ".join(t.split())
    return t.lower().strip()


def detect_keywords(text: str) -> List[str]:
    t = text.lower()
    return [kw for kw in KEYWORDS_LOWER if kw in t]


# ==========================================================
# ADAPTIVE RULES (rule-based "fine-tuning")
# ==========================================================
def load_adaptive_rules() -> Dict[str, Any]:
    try:
        with open(ADAPTIVE_RULES_FILE, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def save_adaptive_rules(obj: Dict[str, Any]) -> None:
    with open(ADAPTIVE_RULES_FILE, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


ADAPTIVE = load_adaptive_rules()
# Structure:
# {
#   "keywords": {"стройка": {"work": 12, "wrong": 3}, ...},
#   "channels": {"tipkhimki": {"work": 2, "wrong": 7}, ...}
# }


def _inc_counter(d: Dict[str, Any], key: str, label: str) -> None:
    bucket = d.setdefault(key, {"work": 0, "wrong": 0, "attach": 0})
    if label not in bucket:
        bucket[label] = 0
    bucket[label] += 1


def update_adaptive_from_training(label: str, text: str, channel: str = "") -> None:
    label = label.lower()
    if label not in ("work", "wrong", "attach"):
        return

    # only work/wrong affect "probability", attach is neutral
    if label in ("work", "wrong"):
        kws = detect_keywords(normalize_text(text))
        kw_stats = ADAPTIVE.setdefault("keywords", {})
        for kw in kws:
            _inc_counter(kw_stats, kw, label)

        if channel:
            ch_stats = ADAPTIVE.setdefault("channels", {})
            _inc_counter(ch_stats, channel, label)

        save_adaptive_rules(ADAPTIVE)


def compute_rule_prior(text: str, channel: str = "") -> Optional[float]:
    """
    Returns prior probability 0..100 derived from adaptive keyword/channel stats.
    If no adaptive data yet, returns None.
    """
    kw_stats = (ADAPTIVE.get("keywords") or {})
    ch_stats = (ADAPTIVE.get("channels") or {})

    signals: List[float] = []
    t_norm = normalize_text(text)
    kws = detect_keywords(t_norm)

    def score_bucket(bucket: Dict[str, Any]) -> Optional[float]:
        try:
            w = int(bucket.get("work", 0))
            r = int(bucket.get("wrong", 0))
            n = w + r
            if n <= 0:
                return None
            # score in [-1, +1]
            s = (w - r) / n
            return float(s)
        except Exception:
            return None

    for kw in kws:
        b = kw_stats.get(kw)
        if isinstance(b, dict):
            s = score_bucket(b)
            if s is not None:
                signals.append(s)

    if channel and channel in ch_stats and isinstance(ch_stats[channel], dict):
        s = score_bucket(ch_stats[channel])
        if s is not None:
            signals.append(s)

    if not signals:
        return None

    # Aggregate and map to [0..100]
    avg = sum(signals) / len(signals)
    prior = (avg + 1.0) * 50.0
    if prior < 0:
        prior = 0.0
    if prior > 100:
        prior = 100.0
    return prior


# ==========================================================
# TRAINING / HISTORY LOGS (append-only)
# ==========================================================
def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    obj = dict(obj)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def log_training_event(card_id: str, label: str, from_user: int, text: str = "", channel: str = "") -> None:
    rec = {
        "timestamp": int(time.time()),
        "card_id": card_id,
        "label": label,
        "from_user": int(from_user),
        "text": text,
        "channel": channel,
    }
    append_jsonl(TRAINING_DATASET, rec)
    update_adaptive_from_training(label, text=text, channel=channel)
    log.info(f"[TRAIN] {label.upper()} card_id={card_id} user={from_user}")


def append_history(event: Dict[str, Any]) -> None:
    event = dict(event)
    event["ts"] = int(time.time())
    append_jsonl(HISTORY_CARDS, event)


# ==========================================================
# TELEGRAM WEB SCRAPE
# ==========================================================
def fetch_channel_page(url: str) -> Optional[str]:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200:
            log.error(f"HTTP {r.status_code} for {url}")
            return None
        return r.text
    except Exception as e:
        log.error(f"Request error {url}: {e}")
        return None


def iso_to_epoch_seconds(iso_str: str) -> Optional[int]:
    """
    Telegram page uses <time datetime="2025-12-15T10:22:08+00:00">.
    Convert to epoch seconds.
    """
    if not iso_str or not isinstance(iso_str, str):
        return None
    try:
        # Python 3.11: fromisoformat supports "+00:00"
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        return None


def extract_posts(html: str) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    messages = soup.find_all("div", class_="tgme_widget_message")
    posts: List[Dict[str, Any]] = []

    for msg in messages:
        try:
            msg_id = msg.get("data-post", "")  # e.g. "channel/123"
            text_block = msg.find("div", class_="tgme_widget_message_text")
            text = text_block.get_text(" ", strip=True) if text_block else ""

            date_block = msg.find("time", class_="time")
            ts = None
            if date_block and date_block.get("datetime"):
                ts = iso_to_epoch_seconds(date_block.get("datetime"))
            if ts is None:
                ts = int(time.time())

            links = []
            for a in msg.find_all("a", href=True):
                href = a["href"]
                if href.startswith("http"):
                    links.append(href)

            posts.append({"id": msg_id, "text": text, "timestamp": ts, "links": links})
        except Exception as e:
            log.error(f"Post parse error: {e}")

    return posts


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


def process_channel(channel_username: str) -> List[Dict[str, Any]]:
    url = f"https://t.me/s/{channel_username}"
    html = fetch_channel_page(url)
    if not html:
        return []

    posts = extract_posts(html)
    hits: List[Dict[str, Any]] = []

    for p in posts:
        t_norm = normalize_text(p["text"])
        found = detect_keywords(t_norm)
        if found:
            hits.append({
                "channel": channel_username,
                "post_id": p["id"],
                "text": p["text"],
                "links": p["links"],
                "timestamp": p["timestamp"],
                "keywords": found,
            })

    return hits


def scan_once() -> List[Dict[str, Any]]:
    all_hits: List[Dict[str, Any]] = []
    for ch in CHANNEL_LIST:
        try:
            hits = process_channel(ch)
            if hits:
                log.info(f"@{ch}: hits={len(hits)}")
            all_hits.extend(hits)
        except Exception as e:
            log.error(f"Channel @{ch} error: {e}")
    return all_hits


# ==========================================================
# CARD MODEL
# ==========================================================
def generate_card_id() -> str:
    return str(uuid.uuid4())[:12]


def card_path(card_id: str) -> str:
    return os.path.join(CARDS_DIR, f"{card_id}.json")


def save_card(card: Dict[str, Any]) -> None:
    with open(card_path(card["card_id"]), "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)


def load_card(card_id: str) -> Optional[Dict[str, Any]]:
    p = card_path(card_id)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def build_card_text(card: Dict[str, Any]) -> str:
    ts = card.get("timestamp")
    dt = datetime.fromtimestamp(ts).strftime("%d.%m.%Y %H:%M") if isinstance(ts, int) else "—"
    keywords = ", ".join(card.get("keywords", [])) or "—"
    links = card.get("links") or []
    links_str = "\n".join(links) if links else "нет ссылок"

    lines = [
        "🔎 Обнаружено подозрительное сообщение",
        f"Источник: @{card.get('channel','—')}",
        f"Дата: {dt}",
        f"ID поста: {card.get('post_id','—')}",
        "",
        f"🔑 Ключевые слова: {keywords}",
        "",
        "📝 Текст сообщения:",
        card.get("text", "") or "—",
        "",
        "📎 Ссылки:",
        links_str,
        "",
        f"🆔 ID карточки: {card.get('card_id')}",
    ]

    # AI block
    ai = card.get("ai") or {}
    prob = ai.get("probability_final")
    comment = ai.get("comment")
    if prob is not None or comment:
        lines.append("")
        lines.append(f"🤖 Вероятность самостроя: {float(prob):.1f}%" if prob is not None else "🤖 Вероятность самостроя: —")
        if comment:
            lines.append(f"💬 Комментарий ИИ: {comment}")

    return "\n".join(lines)


# ==========================================================
# YANDEXGPT INTEGRATION (prompt steering with few-shot)
# ==========================================================
YAGPT_API_KEY = os.getenv("YAGPT_API_KEY", "").strip()
YAGPT_FOLDER_ID = os.getenv("YAGPT_FOLDER_ID", "").strip()
YAGPT_ENDPOINT = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
YAGPT_MODEL = os.getenv("YAGPT_MODEL", "gpt://{folder_id}/yandexgpt/latest")


def tail_training_examples(limit: int = 6) -> List[Dict[str, Any]]:
    if not os.path.exists(TRAINING_DATASET):
        return []
    try:
        with open(TRAINING_DATASET, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    for ln in lines[::-1]:
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        label = (obj.get("label") or "").lower()
        txt = (obj.get("text") or "").strip()
        if label in ("work", "wrong") and txt:
            out.append({"label": label, "text": txt[:800]})
            if len(out) >= limit:
                break
    return out[::-1]


def build_few_shot_block() -> str:
    examples = tail_training_examples(limit=6)
    if not examples:
        return ""
    parts = ["Примеры решений инспектора (для калибровки):"]
    for ex in examples:
        lab = "САМOСТРОЙ (в работу)" if ex["label"] == "work" else "НЕ САМOСТРОЙ (неверно)"
        parts.append(f"- Пример: {lab}\n  Текст: {ex['text']}")
    return "\n".join(parts) + "\n\n"


def call_yandex_gpt_json(text: str) -> Optional[Dict[str, Any]]:
    if not YAGPT_API_KEY or not YAGPT_FOLDER_ID:
        return None

    model_uri = YAGPT_MODEL.format(folder_id=YAGPT_FOLDER_ID)
    few_shot = build_few_shot_block()

    prompt = (
        "Ты помощник инспектора строительного надзора.\n"
        "Ниже дан текст сообщения из Telegram.\n"
        "Твоя задача: оценить вероятность, что сообщение связано с самовольным/незаконным строительством (самострой).\n\n"
        f"{few_shot}"
        "Правила ответа:\n"
        "1) probability: число 0..100\n"
        "2) comment: короткое пояснение (1-2 предложения)\n"
        "Верни строго JSON без лишнего текста.\n\n"
        f"Текст сообщения:\n{text}"
    )

    body = {
        "modelUri": model_uri,
        "completionOptions": {"stream": False, "temperature": 0.1, "maxTokens": 220},
        "messages": [{"role": "user", "text": prompt}],
    }
    headers = {
        "Authorization": f"Api-Key {YAGPT_API_KEY}",
        "x-folder-id": YAGPT_FOLDER_ID,
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(YAGPT_ENDPOINT, headers=headers, json=body, timeout=25)
        data = resp.json()
    except Exception as e:
        log.error(f"YandexGPT request error: {e}")
        return None

    try:
        out_text = data["result"]["alternatives"][0]["message"]["text"]
    except Exception:
        log.error(f"YandexGPT unexpected response: {data}")
        return None

    # extract JSON object safely
    s = (out_text or "").strip()
    if not s.startswith("{"):
        a = s.find("{")
        b = s.rfind("}")
        if a != -1 and b != -1 and b > a:
            s = s[a:b+1]

    try:
        return json.loads(s)
    except Exception:
        log.error(f"YandexGPT JSON parse error. raw={out_text}")
        return None


def enrich_card_with_ai(card: Dict[str, Any]) -> None:
    text = card.get("text") or ""
    channel = card.get("channel") or ""
    prior = compute_rule_prior(text, channel=channel)  # may be None

    yagpt_prob = None
    yagpt_comment = ""

    obj = call_yandex_gpt_json(text) if (YAGPT_API_KEY and YAGPT_FOLDER_ID) else None
    if isinstance(obj, dict):
        try:
            yagpt_prob = float(obj.get("probability"))
        except Exception:
            yagpt_prob = None
        yagpt_comment = (obj.get("comment") or "").strip()

    # combine
    final_prob = None
    if yagpt_prob is not None and prior is not None:
        # Blend AI + adaptive prior
        final_prob = 0.75 * yagpt_prob + 0.25 * prior
    elif yagpt_prob is not None:
        final_prob = yagpt_prob
    elif prior is not None:
        final_prob = prior

    # clamp
    if final_prob is not None:
        if final_prob < 0:
            final_prob = 0.0
        if final_prob > 100:
            final_prob = 100.0

    card.setdefault("ai", {})
    if yagpt_prob is not None:
        card["ai"]["probability_yagpt"] = yagpt_prob
    if prior is not None:
        card["ai"]["probability_prior"] = prior
    if final_prob is not None:
        card["ai"]["probability_final"] = float(final_prob)
    if yagpt_comment:
        card["ai"]["comment"] = yagpt_comment


def generate_card(hit: Dict[str, Any]) -> Dict[str, Any]:
    cid = generate_card_id()
    card = {
        "card_id": cid,
        "channel": hit["channel"],
        "post_id": hit["post_id"],
        "timestamp": hit["timestamp"],
        "text": hit["text"],
        "keywords": hit["keywords"],
        "links": hit["links"],
        "status": "new",          # new/sent/filtered/in_work/wrong/bind
        "history": [],
    }
    enrich_card_with_ai(card)
    save_card(card)
    return card


# ==========================================================
# TELEGRAM SENDING + INLINE KEYBOARD
# ==========================================================
def build_card_keyboard(card_id: str) -> Dict[str, Any]:
    return {
        "inline_keyboard": [
            [
                {"text": "✅ В работу", "callback_data": f"card:{card_id}:work"},
                {"text": "❌ Неверно", "callback_data": f"card:{card_id}:wrong"},
            ],
            [{"text": "📎 Привязать", "callback_data": f"card:{card_id}:bind"}],
        ]
    }


def send_telegram(method: str, payload: Dict[str, Any], timeout: int = 20) -> Optional[Dict[str, Any]]:
    if not TELEGRAM_API_URL:
        return None
    try:
        resp = requests.post(f"{TELEGRAM_API_URL}/{method}", json=payload, timeout=timeout)
        data = resp.json()
        if not data.get("ok"):
            log.error(f"Telegram API error {method}: {data}")
            return None
        return data.get("result")
    except Exception as e:
        log.error(f"Telegram API exception {method}: {e}")
        return None


def send_plain_message(chat_id: int, text: str) -> None:
    send_telegram("sendMessage", {"chat_id": chat_id, "text": text, "disable_web_page_preview": True})


def send_message_with_keyboard(chat_id: int, text: str, reply_markup: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": False,
        "reply_markup": reply_markup,
    }
    return send_telegram("sendMessage", payload)


def edit_message_reply_markup(chat_id: int, message_id: int, reply_markup: Optional[Dict[str, Any]]) -> None:
    payload = {"chat_id": chat_id, "message_id": message_id, "reply_markup": reply_markup}
    send_telegram("editMessageReplyMarkup", payload)


def answer_callback_query(cb_id: str, text: str = "", show_alert: bool = False) -> None:
    payload = {"callback_query_id": cb_id, "text": text, "show_alert": show_alert}
    send_telegram("answerCallbackQuery", payload)


def should_send_card(card: Dict[str, Any]) -> Tuple[bool, Optional[float]]:
    thr = get_threshold()
    ai = card.get("ai") or {}
    prob = ai.get("probability_final")
    if prob is None:
        # if no AI/probability available, treat as 0 (or always send if threshold=0)
        prob = 0.0
    try:
        p = float(prob)
    except Exception:
        p = 0.0
    return (p >= thr), p


def send_card_to_group(card: Dict[str, Any]) -> Optional[int]:
    ok_send, p = should_send_card(card)
    if not ok_send:
        card["status"] = "filtered"
        card.setdefault("history", []).append({"event": "filtered", "threshold": get_threshold(), "prob": p, "ts": int(time.time())})
        save_card(card)
        append_history({"event": "filtered", "card_id": card["card_id"], "threshold": get_threshold(), "prob": p})
        log.info(f"Filtered card {card['card_id']} prob={p} < thr={get_threshold()}")
        return None

    text = build_card_text(card)
    kb = build_card_keyboard(card["card_id"])
    res = send_message_with_keyboard(TARGET_CHAT_ID, text, kb)
    if not res:
        return None

    message_id = res.get("message_id")
    chat_id = res.get("chat", {}).get("id")

    card.setdefault("tg", {})
    card["tg"]["chat_id"] = chat_id
    card["tg"]["message_id"] = message_id
    card["status"] = "sent"
    card.setdefault("history", []).append({"event": "sent", "chat_id": chat_id, "message_id": message_id, "ts": int(time.time())})
    save_card(card)

    append_history({"event": "sent", "card_id": card["card_id"], "chat_id": chat_id, "message_id": message_id})
    return message_id


def send_cards_to_group(cards: List[Dict[str, Any]]) -> int:
    cnt = 0
    for c in cards:
        mid = send_card_to_group(c)
        if mid:
            cnt += 1
            time.sleep(0.35)
    return cnt


# ==========================================================
# CARD ACTIONS (admins only)
# ==========================================================
def apply_card_action(card_id: str, action: str, from_user: int) -> str:
    card = load_card(card_id)
    if not card:
        return "Карточка не найдена."

    action = action.lower().strip()
    if action == "work":
        new_status, label, msg = "in_work", "work", "Статус карточки: В РАБОТУ ✅"
    elif action == "wrong":
        new_status, label, msg = "wrong", "wrong", "Статус карточки: НЕВЕРНО ❌"
    elif action == "bind":
        new_status, label, msg = "bind", "attach", "Статус карточки: ПРИВЯЗАТЬ 📎"
    else:
        return "Неизвестное действие."

    old_status = card.get("status", "new")
    card["status"] = new_status
    card.setdefault("history", []).append({"event": f"set_{new_status}", "from_user": int(from_user), "ts": int(time.time())})
    save_card(card)

    append_history({"event": "status_change", "card_id": card_id, "from_user": int(from_user), "old_status": old_status, "new_status": new_status})
    log_training_event(card_id, label, from_user=from_user, text=card.get("text", ""), channel=card.get("channel", ""))

    return msg


# ==========================================================
# TRAINING STATS + PLOT
# ==========================================================
def compute_training_stats() -> Dict[str, Any]:
    stats = {
        "total": 0, "work": 0, "wrong": 0, "attach": 0,
        "by_admin": {},  # {user_id: {"total":..,"work":..,"wrong":..,"attach":..}}
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
                label = (obj.get("label") or "").lower()
                if label == "work":
                    stats["work"] += 1
                elif label == "wrong":
                    stats["wrong"] += 1
                elif label == "attach":
                    stats["attach"] += 1

                uid = obj.get("from_user")
                try:
                    uid = int(uid)
                except Exception:
                    uid = None
                if uid is not None:
                    ba = stats["by_admin"].setdefault(str(uid), {"total": 0, "work": 0, "wrong": 0, "attach": 0})
                    ba["total"] += 1
                    if label in ba:
                        ba[label] += 1

                ts = obj.get("timestamp")
                if isinstance(ts, int):
                    if stats["last_ts"] is None or ts > stats["last_ts"]:
                        stats["last_ts"] = ts
    except Exception as e:
        log.error(f"Training stats read error: {e}")

    progress = min(1.0, stats["total"] / max(1, TARGET_DATASET_SIZE))
    stats["progress"] = round(progress * 100.0, 2)
    stats["confidence"] = round(progress * 100.0, 2)  # surrogate
    return stats


def format_training_stats(stats: Dict[str, Any]) -> str:
    total = stats.get("total", 0)
    work = stats.get("work", 0)
    wrong = stats.get("wrong", 0)
    attach = stats.get("attach", 0)

    last_ts = stats.get("last_ts")
    last_str = "Пока не было ни одного события обучения."
    if isinstance(last_ts, int):
        last_str = f"Последнее обучение: {datetime.fromtimestamp(last_ts).strftime('%d.%m.%Y %H:%M')}"

    lines = [
        "📊 Статистика обучения ИИ (YandexGPT):",
        "",
        f"• Всего обучающих событий: {total}",
        f"   ├─ В работу: {work}",
        f"   ├─ Неверно: {wrong}",
        f"   └─ Привязать: {attach}",
        "",
        f"• Порог авто-фильтра: {get_threshold()}%",
        f"• Прогресс к целевому датасету ({TARGET_DATASET_SIZE}): {stats.get('progress',0)}%",
        "",
        "👥 По администраторам:",
    ]

    by_admin = stats.get("by_admin") or {}
    if not by_admin:
        lines.append("• данных пока нет")
    else:
        for uid, s in by_admin.items():
            lines.append(f"• {uid}: всего={s['total']} | work={s['work']} | wrong={s['wrong']} | bind={s['attach']}")

    lines.extend(["", last_str])
    return "\n".join(lines)


def build_training_plot_png(path: str) -> bool:
    """
    Creates a PNG plot of cumulative training events over time.
    Returns True if created.
    """
    # read dataset
    points: List[Tuple[int, int]] = []
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
                ts = obj.get("timestamp")
                if isinstance(ts, int):
                    points.append((ts, 1))
    except Exception:
        points = []

    if not points:
        return False

    points.sort(key=lambda x: x[0])
    xs: List[float] = []
    ys: List[int] = []
    cum = 0
    for ts, one in points:
        cum += one
        xs.append(ts)
        ys.append(cum)

    # optional matplotlib
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xs, ys)
    ax.set_title("График обучения (накопление примеров)")
    ax.set_xlabel("Время (epoch)")
    ax.set_ylabel("Количество примеров (cumulative)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def send_photo(chat_id: int, png_path: str, caption: str = "") -> None:
    if not TELEGRAM_API_URL:
        return
    try:
        with open(png_path, "rb") as f:
            files = {"photo": f}
            data = {"chat_id": str(chat_id), "caption": caption}
            resp = requests.post(f"{TELEGRAM_API_URL}/sendPhoto", data=data, files=files, timeout=30)
            try:
                j = resp.json()
                if not j.get("ok"):
                    log.error(f"sendPhoto error: {j}")
            except Exception:
                pass
    except Exception as e:
        log.error(f"send_photo exception: {e}")


# ==========================================================
# ADMIN MENU
# ==========================================================
def build_admin_keyboard() -> Dict[str, Any]:
    return {
        "inline_keyboard": [
            [{"text": "📊 Статистика обучения", "callback_data": "admin:trainstats"}],
            [{"text": "📈 График обучения", "callback_data": "admin:trainplot"}],
            [{"text": "🎚 Порог вероятности", "callback_data": "admin:threshold"}],
            [{"text": "👥 Управление администраторами", "callback_data": "admin:admins"}],
        ]
    }


def build_threshold_keyboard() -> Dict[str, Any]:
    thr = get_threshold()
    return {
        "inline_keyboard": [
            [{"text": f"Текущий порог: {thr}%", "callback_data": "admin:noop"}],
            [
                {"text": "−10", "callback_data": "admin:thr:-10"},
                {"text": "+10", "callback_data": "admin:thr:+10"},
            ],
            [
                {"text": "0%", "callback_data": "admin:thr:0"},
                {"text": "50%", "callback_data": "admin:thr:50"},
                {"text": "80%", "callback_data": "admin:thr:80"},
                {"text": "100%", "callback_data": "admin:thr:100"},
            ],
            [{"text": "↩️ Назад", "callback_data": "admin:back"}],
        ]
    }


# ==========================================================
# UPDATES (long polling)
# ==========================================================
UPDATE_OFFSET = 0


def handle_message(update: Dict[str, Any]) -> None:
    msg = update.get("message")
    if not msg:
        return

    chat_id = msg.get("chat", {}).get("id")
    from_user = msg.get("from", {}).get("id")
    text = (msg.get("text") or "").strip()
    if not text.startswith("/"):
        return

    cmd, *rest = text.split(" ", 1)
    cmd = cmd.split("@")[0]
    arg = rest[0].strip() if rest else ""

    if cmd == "/admin":
        if not is_admin(from_user):
            send_plain_message(chat_id, "❌ Доступ запрещён.")
            return
        send_message_with_keyboard(chat_id, "🛠 Админ-панель:", build_admin_keyboard())
        return

    if cmd == "/trainstats":
        if not is_admin(from_user):
            send_plain_message(chat_id, "❌ Доступ запрещён.")
            return
        stats = compute_training_stats()
        send_plain_message(chat_id, format_training_stats(stats))
        return

    if cmd == "/setthreshold":
        if not is_admin(from_user):
            send_plain_message(chat_id, "❌ Доступ запрещён.")
            return
        if not arg:
            send_plain_message(chat_id, "Использование: /setthreshold <0..100>")
            return
        try:
            v = int(arg)
        except Exception:
            send_plain_message(chat_id, "Порог должен быть числом 0..100.")
            return
        v = set_threshold(v)
        send_plain_message(chat_id, f"✅ Порог установлен: {v}%")
        return

    if cmd == "/addadmin":
        if not is_admin(from_user):
            send_plain_message(chat_id, "❌ Доступ запрещён.")
            return
        try:
            uid = int(arg)
        except Exception:
            send_plain_message(chat_id, "Использование: /addadmin <telegram_id>")
            return
        if add_admin(uid):
            send_plain_message(chat_id, f"✅ Добавлен админ: {uid}")
        else:
            send_plain_message(chat_id, f"👤 {uid} уже админ.")
        return

    if cmd == "/deladmin":
        if not is_admin(from_user):
            send_plain_message(chat_id, "❌ Доступ запрещён.")
            return
        try:
            uid = int(arg)
        except Exception:
            send_plain_message(chat_id, "Использование: /deladmin <telegram_id>")
            return
        if remove_admin(uid):
            send_plain_message(chat_id, f"🗑 Удалён админ: {uid}")
        else:
            send_plain_message(chat_id, f"👤 {uid} не найден.")
        return

    if cmd == "/cards":
        if not is_admin(from_user):
            send_plain_message(chat_id, "❌ Доступ запрещён.")
            return
        cards = list_recent_cards(limit=MAX_CARDS_LIST)
        send_plain_message(chat_id, format_cards_list(cards))
        return

    if cmd == "/history":
        if not is_admin(from_user):
            send_plain_message(chat_id, "❌ Доступ запрещён.")
            return
        events = tail_history_events(limit=MAX_HISTORY_EVENTS)
        send_plain_message(chat_id, format_history_events(events))
        return

    send_plain_message(chat_id, f"Неизвестная команда: {cmd}")


def handle_callback_query(update: Dict[str, Any]) -> None:
    cb = update.get("callback_query")
    if not cb:
        return

    cb_id = cb.get("id")
    from_user = cb.get("from", {}).get("id")
    data = cb.get("data", "") or ""
    message = cb.get("message", {}) or {}
    chat_id = message.get("chat", {}).get("id")
    message_id = message.get("message_id")

    # card actions
    if data.startswith("card:"):
        if not is_admin(from_user):
            answer_callback_query(cb_id, "❌ Только администраторы могут менять статус.", show_alert=True)
            return

        try:
            _, card_id, action = data.split(":", 2)
        except Exception:
            answer_callback_query(cb_id, "Ошибка формата.", show_alert=True)
            return

        result = apply_card_action(card_id, action, from_user=int(from_user))

        # remove buttons for everyone after any admin action
        try:
            edit_message_reply_markup(chat_id, message_id, reply_markup=None)
        except Exception:
            pass

        answer_callback_query(cb_id, result, show_alert=False)
        return

    # admin menu
    if data.startswith("admin:"):
        if not is_admin(from_user):
            answer_callback_query(cb_id, "❌ Доступ запрещён.", show_alert=True)
            return

        action = data.split(":", 1)[1]

        if action == "trainstats":
            stats = compute_training_stats()
            send_plain_message(chat_id, format_training_stats(stats))
            answer_callback_query(cb_id, "Готово.", show_alert=False)
            return

        if action == "trainplot":
            png = os.path.join(DATA_DIR, "training_plot.png")
            ok = build_training_plot_png(png)
            if ok:
                send_photo(chat_id, png, caption="📈 График обучения (накопление примеров)")
                answer_callback_query(cb_id, "График отправлен.", show_alert=False)
            else:
                answer_callback_query(cb_id, "График недоступен (нет данных или matplotlib).", show_alert=True)
            return

        if action == "threshold":
            send_message_with_keyboard(chat_id, "🎚 Настройка порога вероятности:", build_threshold_keyboard())
            answer_callback_query(cb_id, "Ок.", show_alert=False)
            return

        if action.startswith("thr:"):
            val = action.split(":", 1)[1]
            cur = get_threshold()
            try:
                if val in ("+10", "-10"):
                    delta = int(val)
                    cur = set_threshold(cur + delta)
                else:
                    cur = set_threshold(int(val))
                answer_callback_query(cb_id, f"Порог: {cur}%", show_alert=False)
                # update the same message keyboard
                try:
                    edit_message_reply_markup(chat_id, message_id, build_threshold_keyboard())
                except Exception:
                    pass
            except Exception:
                answer_callback_query(cb_id, "Ошибка установки порога.", show_alert=True)
            return

        if action == "admins":
            admins_text = "\n".join(str(a) for a in ADMINS) if ADMINS else "пусто"
            send_plain_message(
                chat_id,
                "👥 Администраторы:\n"
                f"{admins_text}\n\n"
                "Команды:\n"
                "/addadmin <id>\n"
                "/deladmin <id>"
            )
            answer_callback_query(cb_id, "Ок.", show_alert=False)
            return

        if action == "back":
            send_message_with_keyboard(chat_id, "🛠 Админ-панель:", build_admin_keyboard())
            answer_callback_query(cb_id, "Ок.", show_alert=False)
            return

        # noop / unknown
        answer_callback_query(cb_id, "", show_alert=False)
        return

    answer_callback_query(cb_id, "", show_alert=False)


def poll_updates() -> None:
    global UPDATE_OFFSET
    if not TELEGRAM_API_URL:
        log.warning("poll_updates is disabled (no BOT_TOKEN).")
        return

    log.info("poll_updates started (message + callback_query).")

    while True:
        try:
            params = {
                "timeout": 25,
                "offset": UPDATE_OFFSET,
                "allowed_updates": ["message", "callback_query"],
            }
            resp = requests.get(f"{TELEGRAM_API_URL}/getUpdates", params=params, timeout=35)
            data = resp.json()

            if not data.get("ok"):
                # Common: 409 Conflict when two instances run simultaneously
                desc = data.get("description", "")
                log.error(f"getUpdates error: {data}")
                if "Conflict" in desc or data.get("error_code") == 409:
                    time.sleep(10)
                else:
                    time.sleep(5)
                continue

            updates = data.get("result", [])
            if not updates:
                continue

            for upd in updates:
                UPDATE_OFFSET = max(UPDATE_OFFSET, upd.get("update_id", 0) + 1)
                if "callback_query" in upd:
                    handle_callback_query(upd)
                elif "message" in upd:
                    handle_message(upd)

        except Exception as e:
            log.error(f"poll_updates exception: {e}")
            time.sleep(5)


# ==========================================================
# HISTORY / CARDS LIST
# ==========================================================
def tail_history_events(limit: int = 50) -> List[Dict[str, Any]]:
    if not os.path.exists(HISTORY_CARDS):
        return []
    try:
        with open(HISTORY_CARDS, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    for ln in lines[-limit:]:
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out


def format_history_events(events: List[Dict[str, Any]]) -> str:
    if not events:
        return "📂 История пуста."
    lines = ["📂 Последние события:", ""]
    for e in events:
        ts = e.get("ts")
        dt = datetime.fromtimestamp(ts).strftime("%d.%m.%Y %H:%M") if isinstance(ts, int) else "—"
        ev = e.get("event", "event")
        cid = e.get("card_id", "—")
        extra = []
        if ev == "sent":
            extra.append(f"msg={e.get('message_id')}")
        if ev == "filtered":
            extra.append(f"thr={e.get('threshold')}, prob={e.get('prob')}")
        if ev == "status_change":
            extra.append(f"{e.get('old_status')}→{e.get('new_status')} user={e.get('from_user')}")
        lines.append(f"• {dt} — {ev} — {cid}" + (f" ({'; '.join(extra)})" if extra else ""))
    return "\n".join(lines)


def list_recent_cards(limit: int = 20) -> List[Dict[str, Any]]:
    files = []
    try:
        for name in os.listdir(CARDS_DIR):
            if name.endswith(".json"):
                p = os.path.join(CARDS_DIR, name)
                try:
                    m = os.path.getmtime(p)
                except Exception:
                    m = 0
                files.append((m, p))
    except Exception:
        return []
    files.sort(key=lambda x: x[0], reverse=True)
    files = files[:limit]
    cards = []
    for _, p in files:
        try:
            with open(p, "r", encoding="utf-8") as f:
                cards.append(json.load(f))
        except Exception:
            continue
    return cards


def format_cards_list(cards: List[Dict[str, Any]]) -> str:
    if not cards:
        return "📂 Карточек пока нет."
    lines = ["📂 Последние карточки:", ""]
    for c in cards:
        cid = c.get("card_id", "—")
        st = c.get("status", "—")
        ch = c.get("channel", "—")
        pid = c.get("post_id", "—")
        ts = c.get("timestamp")
        dt = datetime.fromtimestamp(ts).strftime("%d.%m.%Y %H:%M") if isinstance(ts, int) else "—"
        prob = (c.get("ai") or {}).get("probability_final")
        prob_str = f"{float(prob):.1f}%" if prob is not None else "—"
        lines.append(f"• {cid} | {st} | @{ch} | {dt} | p={prob_str} | post={pid}")
    return "\n".join(lines)


# ==========================================================
# MAIN SCAN LOOP
# ==========================================================
def run_scan_cycle() -> None:
    log.info("=== SCAN CYCLE START ===")
    hits = scan_once()
    if not hits:
        log.info("No hits.")
        return

    cards = [generate_card(h) for h in hits]
    sent = send_cards_to_group(cards) if TELEGRAM_API_URL else 0
    log.info(f"Cycle done. cards={len(cards)} sent={sent} thr={get_threshold()}%")


def main_loop() -> None:
    # callbacks in separate thread
    if TELEGRAM_API_URL:
        t = threading.Thread(target=poll_updates, daemon=True)
        t.start()
        log.info("poll_updates thread started.")

    while True:
        try:
            run_scan_cycle()
        except Exception as e:
            log.error(f"scan loop error: {e}")
        time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    log.info("SAMASTROI SCRAPER starting.")
    main_loop()

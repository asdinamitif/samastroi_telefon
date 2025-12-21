import os
import re
import json
import time
import math
import random
import sqlite3
import logging
import hashlib
from io import BytesIO
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta

import requests
import pandas as pd

# ----------------------------- LOGGING -----------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("samastroi")


def get_sender_user_id(update: dict) -> int:
    """Returns Telegram user id of the sender for messages/callbacks."""
    try:
        if update.get("message") and update["message"].get("from"):
            return int(update["message"]["from"].get("id"))
        if update.get("callback_query") and update["callback_query"].get("from"):
            return int(update["callback_query"]["from"].get("id"))
    except Exception:
        pass
    return 0

# ----------------------------- CONFIG -----------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
TARGET_CHAT_ID = int(os.getenv("TARGET_CHAT_ID", "0") or "0")

DATA_DIR = os.getenv("DATA_DIR", "/data").strip()
os.makedirs(DATA_DIR, exist_ok=True)

SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "300") or "300")
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "15") or "15")

# IMPORTANT: AI gate; if YandexGPT is unavailable or refuses, card is not sent
MIN_AI_GATE = float(os.getenv("MIN_AI_GATE", "5"))

# YandexGPT
YAGPT_API_KEY = os.getenv("YAGPT_API_KEY", "").strip()
YAGPT_FOLDER_ID = os.getenv("YAGPT_FOLDER_ID", "").strip()
YAGPT_MODEL = os.getenv("YAGPT_MODEL", f"gpt://{YAGPT_FOLDER_ID}/yandexgpt/latest").strip()

# Telegram
TELEGRAM_API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}" if BOT_TOKEN else ""

# Roles (supports legacy env names: ADMIN_IDS/MODERATOR_IDS/LEAD_IDS)
def _parse_ids(*names: str, default: str = "") -> list[int]:
    """Parse comma-separated telegram user IDs from the first non-empty env var name."""
    for n in names:
        v = os.getenv(n, "")
        if v and v.strip():
            return [int(x) for x in v.split(",") if x.strip().isdigit()]
    return [int(x) for x in default.split(",") if x.strip().isdigit()]

# Keep defaults for safety (won't be used if env vars are set)
ADMINS = _parse_ids("ADMINS", "ADMIN_IDS", default="272923789,398960707")
MODERATORS = _parse_ids("MODERATORS", "MODERATOR_IDS", default="777464055,978125225")
LEADERSHIP = _parse_ids("LEADERSHIP", "LEAD_IDS", default="5685586625")

# Persistent files
SCRAPER_DB = os.path.join(DATA_DIR, "scraper.db")
HISTORY_CARDS = os.path.join(DATA_DIR, "training_dataset.jsonl")
CHANNEL_BIAS_FILE = os.path.join(DATA_DIR, "channel_bias.json")
KEYWORD_BIAS_FILE = os.path.join(DATA_DIR, "keyword_bias.json")

# ONZS files
ONZS_XLSX = os.getenv("ONZS_XLSX", "Номера ОНзС.xlsx").strip()

# Base dir / path resolver (Railway may run with different CWD)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_path(p: str) -> str:
    """Resolve relative paths against common locations.
    Order: BASE_DIR -> CWD -> DATA_DIR.
    """
    if not p:
        return p
    if os.path.isabs(p):
        return p

    # 1) Next to this script
    cand1 = os.path.join(BASE_DIR, p)
    if os.path.exists(cand1):
        return cand1

    # 2) Current working directory
    cand2 = os.path.join(os.getcwd(), p)
    if os.path.exists(cand2):
        return cand2

    # 3) Data dir volume
    cand3 = os.path.join(DATA_DIR, os.path.basename(p))
    if os.path.exists(cand3):
        return cand3

    return p

def debug_paths_for_file(p: str):
    """Log helpful diagnostics to understand why a file is not found."""
    try:
        log.error(f"[PATH] missing file: {p}")
        log.error(f"[PATH] CWD={os.getcwd()} | BASE_DIR={BASE_DIR} | DATA_DIR={DATA_DIR}")
        for label, d in [("CWD", os.getcwd()), ("BASE_DIR", BASE_DIR), ("DATA_DIR", DATA_DIR)]:
            try:
                items = os.listdir(d)
                # don't spam: show only first 50
                shown = items[:50]
                log.error(f"[PATH] {label} list (first {len(shown)}/{len(items)}): {shown}")
            except Exception as e:
                log.error(f"[PATH] {label} list error: {e}")
    except Exception:
        pass
ONZS_TRAIN_FILE = os.path.join(DATA_DIR, "onzs_training.jsonl")

# Lock file to avoid 409 / multi pollers
LOCK_FILE = os.path.join(DATA_DIR, ".poller.lock")

# ----------------------------- STOP TOPICS (anti-news/politics noise) -----------------------------
STOP_TOPICS = [
    "путин", "украин", "войн", "политик", "санкц", "выбор", "митинг", "байден",
    "трамп", "хамас", "израил", "террор", "дрон", "ракет", "фронт", "мобилиз",
    "наступлен", "обстрел", "всу", "армия", "нато", "премьер", "президент",
]

# Construction signal words (soft allow)
CONSTR_HINTS = [
    "строй", "строит", "строител", "самостро", "котлован", "фундамент", "арматур",
    "бетон", "плита", "монолит", "перекрыт", "этаж", "кран", "опалуб", "забор",
    "пристрой", "надстрой", "реконструкц", "капремонт", "разрешен", "рнс", "гпзу",
]

# ----------------------------- UTIL -----------------------------
def now_ts() -> int:
    return int(time.time())

def is_admin(uid: int) -> bool:
    return uid in ADMINS

def is_moderator(uid: int) -> bool:
    return uid in MODERATORS or uid in ADMINS

def is_lead(uid: int) -> bool:
    return uid in LEADERSHIP or uid in ADMINS

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)

def append_jsonl(path: str, record: Dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def read_last_jsonl(path: str, limit: int = 50) -> List[Dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()[-limit:]
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out

# ----------------------------- DB -----------------------------
def init_db():
    con = sqlite3.connect(SCRAPER_DB)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS cards (
            card_id TEXT PRIMARY KEY,
            created_ts INTEGER,
            payload TEXT
        )
        """
    )
    con.commit()
    con.close()

def save_card(card: Dict):
    con = sqlite3.connect(SCRAPER_DB)
    cur = con.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO cards(card_id, created_ts, payload) VALUES (?, ?, ?)",
        (card["card_id"], int(card.get("created_ts", now_ts())), json.dumps(card, ensure_ascii=False)),
    )
    con.commit()
    con.close()

def load_card(card_id: str) -> Optional[Dict]:
    con = sqlite3.connect(SCRAPER_DB)
    cur = con.cursor()
    cur.execute("SELECT payload FROM cards WHERE card_id=?", (card_id,))
    row = cur.fetchone()
    con.close()
    if not row:
        return None
    try:
        return json.loads(row[0])
    except Exception:
        return None

# ----------------------------- BIAS (channel/keyword) -----------------------------
def load_json_file(path: str, default: Any):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json_file(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

CHANNEL_BIAS = load_json_file(CHANNEL_BIAS_FILE, {})
KEYWORD_BIAS = load_json_file(KEYWORD_BIAS_FILE, {})

def get_channel_bias(channel: str) -> float:
    try:
        return float(CHANNEL_BIAS.get(channel, 0.0))
    except Exception:
        return 0.0

def get_keyword_bias_points(text: str) -> float:
    total = 0.0
    low = text.lower()
    for k, v in KEYWORD_BIAS.items():
        if k and k in low:
            try:
                total += float(v)
            except Exception:
                pass
    return total

# ----------------------------- ONZS CATALOG -----------------------------
ONZS_MAP: Dict[int, str] = {}

def _to_onzs_int(v) -> Optional[int]:
    """Extract ONZS number 1..12 from a cell value (robust to '1.0', '1 – ...', etc.)."""
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none"):
        return None
    m = re.search(r"(\d+)", s)
    if not m:
        return None
    try:
        n = int(m.group(1))
    except Exception:
        return None
    return n if 1 <= n <= 12 else None

def load_onzs_catalog():
    """Load ONZS catalog from Excel.
    Robust mode:
      - reads all sheets
      - no header dependency
      - finds numbers 1..12 in any column
      - extracts description from right neighbor or first meaningful text in row
    """
    global ONZS_MAP
    try:
        path = resolve_path(ONZS_XLSX)
        log.info(f"[ONZS] trying path: {path}")
        if not os.path.exists(path):
            debug_paths_for_file(path)

        sheets = pd.read_excel(path, sheet_name=None, header=None)
        mp: Dict[int, str] = {}

        for sheet_name, df in (sheets or {}).items():
            if df is None or df.empty:
                continue

            for _, row in df.iterrows():
                cells = row.tolist()

                found_n = None
                found_idx = None
                for i, c in enumerate(cells):
                    n = _to_onzs_int(c)
                    if n is not None:
                        found_n = n
                        found_idx = i
                        break
                if found_n is None:
                    continue

                desc = ""

                # Prefer the right neighbor of the ONZS number cell
                if found_idx is not None and found_idx + 1 < len(cells):
                    v = cells[found_idx + 1]
                    if v is not None:
                        sv = str(v).strip()
                        if sv and sv.lower() not in ("nan", "none"):
                            desc = sv

                # Otherwise: first non-empty non-numeric text in the row excluding the number cell
                if not desc:
                    for j, v in enumerate(cells):
                        if j == found_idx:
                            continue
                        if v is None:
                            continue
                        sv = str(v).strip()
                        if not sv or sv.lower() in ("nan", "none"):
                            continue
                        if re.fullmatch(r"\d+(?:\.\d+)?", sv):
                            continue
                        desc = sv
                        break

                if desc:
                    mp[found_n] = desc

        ONZS_MAP = mp
        log.info(f"[ONZS] catalog loaded: {len(ONZS_MAP)} items")
        if not ONZS_MAP:
            log.error("[ONZS] loaded 0 items: check Excel structure (numbers/description).")

    except Exception as e:
        ONZS_MAP = {}
        log.error(f"[ONZS] catalog load error: {e}")

# ----------------------------- TELEGRAM API HELPERS -----------------------------
def tg_get(method: str, params: Dict, timeout_override: Optional[int] = None) -> Optional[Dict]:
    if not TELEGRAM_API_URL:
        return None
    # For long-polling getUpdates, Telegram can hold the request up to `timeout` seconds.
    # We must set HTTP client timeout > Telegram timeout to avoid premature Read timed out.
    try:
        tg_timeout = int((params or {}).get("timeout", 0) or 0)
    except Exception:
        tg_timeout = 0
    effective_timeout = timeout_override if timeout_override is not None else HTTP_TIMEOUT
    # Make sure effective timeout safely exceeds long-poll timeout
    effective_timeout = max(int(effective_timeout or 0), tg_timeout + 10, 30)

    try:
        r = requests.get(f"{TELEGRAM_API_URL}/{method}", params=params, timeout=effective_timeout)
        return r.json()
    except Exception as e:
        log.error(f"Telegram GET {method} error: {e}")
        return None

def tg_post(method: str, payload: Dict) -> Optional[Dict]:
    if not TELEGRAM_API_URL:
        return None
    try:
        r = requests.post(f"{TELEGRAM_API_URL}/{method}", json=payload, timeout=HTTP_TIMEOUT)
        return r.json()
    except Exception as e:
        log.error(f"Telegram POST {method} error: {e}")
        return None

def answer_callback(callback_query_id: str, text: str = "", show_alert: bool = False):
    tg_post("answerCallbackQuery", {"callback_query_id": callback_query_id, "text": text, "show_alert": show_alert})

def edit_reply_markup(chat_id: int, message_id: int, reply_markup: Optional[Dict] = None):
    payload = {"chat_id": chat_id, "message_id": message_id}
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup
    resp = tg_post("editMessageReplyMarkup", payload)
    if resp and not resp.get("ok", True):
        log.error(f"editMessageReplyMarkup failed: {resp}")
    return resp

def edit_message_text(chat_id: int, message_id: int, text: str, reply_markup: Optional[Dict] = None):
    payload = {"chat_id": chat_id, "message_id": message_id, "text": text, "disable_web_page_preview": False}
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup
    resp = tg_post("editMessageText", payload)
    if resp and not resp.get("ok", True):
        log.error(f"editMessageText failed: {resp}")
    return resp

def send_message(chat_id: int, text: str, reply_markup: Optional[Dict] = None):
    # Telegram limit ~4096; chunk to avoid 400
    max_len = 3900
    chunks = [text[i:i + max_len] for i in range(0, len(text), max_len)] or [""]
    last = None
    for idx, ch in enumerate(chunks):
        payload = {"chat_id": chat_id, "text": ch, "disable_web_page_preview": False}
        if reply_markup is not None and idx == len(chunks) - 1:
            payload["reply_markup"] = reply_markup
        last = tg_post("sendMessage", payload)
        if last and not last.get("ok", True):
            log.error(f"sendMessage failed: {last}")
    return last

# ----------------------------- CARD UI -----------------------------
def build_card_keyboard(card_id: str) -> Dict:
    return {
        "inline_keyboard": [
            [{"text": "✅ В работу", "callback_data": f"card:{card_id}:work"},
             {"text": "❌ Неверно", "callback_data": f"card:{card_id}:wrong"}],
            [{"text": "📎 Привязать", "callback_data": f"card:{card_id}:attach"}],
            [{"text": "✏️ Изменить ОНзС", "callback_data": f"onzs:edit:{card_id}"},
             {"text": "✅ Подтвердить ОНзС", "callback_data": f"onzs:confirm:{card_id}"}],
        ]
    }

def build_onzs_pick_keyboard(card_id: str) -> Dict:
    rows = []
    row = []
    for n in range(1, 13):
        row.append({"text": str(n), "callback_data": f"onzs:set:{card_id}:{n}"})
        if len(row) == 6:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([{"text": "⬅️ Назад к карточке", "callback_data": f"onzs:back:{card_id}"}])
    return {"inline_keyboard": rows}

# ----------------------------- TEXT NORMALIZATION -----------------------------
URL_RE = re.compile(r"(https?://\S+)")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")

def build_admin_keyboard() -> Dict:
    return {
        "inline_keyboard": [
            [
                {"text": "📊 Статистика ОНзС", "callback_data": "admin:onzs_stats"},
                {"text": "🔄 Перезагрузить ОНзС", "callback_data": "admin:reload_onzs"},
            ],
            [
                {"text": "🧪 Тест YandexGPT", "callback_data": "admin:test_yagpt"},
            ],
        ]
    }


def clean_text_for_ai(text: str) -> str:
    t = (text or "").strip()
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = HASHTAG_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    # keep bounded
    if len(t) > 2500:
        t = t[:2500] + "…"
    return t

def is_stop_topic(text: str) -> bool:
    low = (text or "").lower()
    if any(w in low for w in STOP_TOPICS):
        # Allow if strong construction hints exist (rare but possible)
        if any(h in low for h in CONSTR_HINTS):
            return False
        return True
    return False

# ----------------------------- YANDEX GPT -----------------------------
def call_yandex_gpt_raw(messages: List[Dict]) -> Tuple[Optional[str], Optional[Dict]]:
    if not (YAGPT_API_KEY and YAGPT_FOLDER_ID and YAGPT_MODEL):
        return None, {"error": "YandexGPT not configured"}
    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    headers = {"Authorization": f"Api-Key {YAGPT_API_KEY}"}
    payload = {
        "modelUri": YAGPT_MODEL,
        "completionOptions": {"stream": False, "temperature": 0.2, "maxTokens": 400},
        "messages": messages,
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=20)
        txt = r.text
        if not r.ok:
            return None, {"status": r.status_code, "text": txt}
        js = r.json()
        try:
            out = js["result"]["alternatives"][0]["message"]["text"]
        except Exception:
            out = txt
        return out, js
    except Exception as e:
        return None, {"error": str(e)}

JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

def extract_json_from_text(text: str) -> Optional[Dict]:
    if not text:
        return None
    # Try direct json
    t = text.strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    m = JSON_OBJ_RE.search(t)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def call_yandex_gpt_json(prompt: str, channel: Optional[str] = None) -> Optional[Dict]:
    cleaned = clean_text_for_ai(prompt)
    # hard skip for stop topics (prevents refusals)
    if is_stop_topic(cleaned):
        return {"probability": 0, "comment": "Отсев: новость/политика (стоп-тема)."}
    # Few-shot from history: last N labeled cards for relevance
    few = read_last_jsonl(HISTORY_CARDS, limit=12)
    few_block = ""
    if few:
        lines = []
        for ex in few[-10:]:
            t = clean_text_for_ai(ex.get("text", ""))
            lbl = ex.get("label", "")
            hint = ex.get("reason", "")
            if not t or not lbl:
                continue
            lines.append(f"- Метка={lbl} (ориентир {hint}). Текст: {t}")
        few_block = "\n" + "\n".join(lines) + "\n"

    # our own bias to stabilize decisions (channel/keyword learning)
    bias_ch = get_channel_bias(channel) if channel else 0.0
    bias_kw = get_keyword_bias_points(cleaned)
    bias_total = bias_ch + bias_kw

    prompt_full = (
        "Ты помощник инспектора строительного надзора.\n"
        "Твоя задача: оценить вероятность (0-100), что текст относится к самовольному строительству/нарушениям на стройке.\n"
        "Запрещено: новости, политика, общие рассуждения.\n"
        "Если текст содержит запрещённые/неподходящие темы, всё равно верни JSON с probability=0 и кратким comment.\n"
        "Ответь СТРОГО одним JSON без пояснений и без Markdown.\n"
        'Формат: {"probability": <0-100>, "comment": "..."}\n'
        f"Калибровка (bias, прибавь к вероятности): {bias_total:+.1f}\n"
        + few_block +
        "\nТекст:\n" + cleaned
    )

    messages = [
        {"role": "system", "text": "Отвечай строго JSON."},
        {"role": "user", "text": prompt_full},
    ]

    # retries for stability
    for attempt in range(4):
        out_text, meta = call_yandex_gpt_raw(messages)
        if out_text is None:
            # retry on transient
            time.sleep(0.8 * (attempt + 1))
            continue
        js = extract_json_from_text(out_text)
        if js is None:
            # if model refused, do not crash; return minimal
            if "не могу обсуждать" in out_text.lower():
                return {"probability": 0, "comment": "Отказ модели: неподходящая тема."}
            # retry once
            time.sleep(0.6 * (attempt + 1))
            continue
        if "probability" not in js:
            # allow legacy keys
            if "prob" in js:
                js["probability"] = js["prob"]
        return js
    return None

# ----------------------------- ONZS DETECTION -----------------------------
def detect_onzs_with_yagpt(text: str, channel: Optional[str] = None) -> Optional[Dict]:
    if not ONZS_MAP:
        return None

    cleaned = clean_text_for_ai(text)
    if is_stop_topic(cleaned):
        return {"onzs": None, "confidence": 0.0, "reason": "Отсев: стоп-тема."}

    catalog = "\n".join([f"{k}: {v}" for k, v in sorted(ONZS_MAP.items())])

    prompt = (
        "Ты инспектор строительного надзора.\n"
        "Определи номер ОНзС (1–12) по описанию. Если нельзя определить — onzs=null.\n"
        "Ответь СТРОГО одним JSON без пояснений.\n"
        'Формат: {"onzs": <1-12|null>, "confidence": <0-1>, "reason": "..."}\n\n'
        "Классификатор ОНзС:\n"
        f"{catalog}\n\n"
        "Текст:\n"
        f"{cleaned}"
    )

    messages = [
        {"role": "system", "text": "Отвечай строго JSON."},
        {"role": "user", "text": prompt},
    ]

    for attempt in range(4):
        out_text, meta = call_yandex_gpt_raw(messages)
        if out_text is None:
            time.sleep(0.8 * (attempt + 1))
            continue
        js = extract_json_from_text(out_text)
        if js is None:
            if "не могу обсуждать" in out_text.lower():
                return {"onzs": None, "confidence": 0.0, "reason": "Отказ модели."}
            time.sleep(0.6 * (attempt + 1))
            continue
        return js
    return None

def save_onzs_training(text: str, onzs: int, confirmed: bool):
    rec = {"text": clean_text_for_ai(text), "onzs": int(onzs), "confirmed": bool(confirmed), "ts": now_ts()}
    append_jsonl(ONZS_TRAIN_FILE, rec)

def build_onzs_stats() -> str:
    if not os.path.exists(ONZS_TRAIN_FILE):
        return "Нет данных по ОНзС."
    stats: Dict[int, Dict[str, int]] = {}
    with open(ONZS_TRAIN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            o = r.get("onzs")
            if o is None:
                continue
            try:
                o = int(o)
            except Exception:
                continue
            stats.setdefault(o, {"ok": 0, "all": 0})
            stats[o]["all"] += 1
            if r.get("confirmed"):
                stats[o]["ok"] += 1

    total_all = sum(v["all"] for v in stats.values()) or 0
    total_ok = sum(v["ok"] for v in stats.values()) or 0
    acc_total = int(100 * total_ok / total_all) if total_all else 0

    out = [f"🎯 Точность ИИ по ОНзС: {acc_total}% (верно {total_ok}/{total_all})"]
    for o in sorted(stats.keys()):
        s = stats[o]
        acc = int(100 * s["ok"] / s["all"]) if s["all"] else 0
        out.append(f"ОНзС-{o}: {acc}% ({s['ok']}/{s['all']})")
    return "\n".join(out)

# ----------------------------- CARD PIPELINE -----------------------------
def build_card_id(channel: str, post_id: Any) -> str:
    return sha1(f"{channel}:{post_id}")

def extract_links(text: str) -> List[str]:
    return URL_RE.findall(text or "")

def extract_keywords_hit(text: str, keywords: List[str]) -> List[str]:
    low = (text or "").lower()
    hits = []
    for k in keywords:
        if k and k.lower() in low:
            hits.append(k)
    return hits

DEFAULT_KEYWORDS = [
    "самострой", "стройка", "строительство", "котлован", "фундамент", "бетон", "арматура",
    "кран", "опалубка", "забор", "пристройка", "надстройка", "реконструкция",
    "этаж", "плита", "перекрытие"
]

def classify_with_ai(text: str, channel: str) -> Optional[Dict]:
    # AI gate mandatory; returns {"probability":..., "comment":...} or None
    return call_yandex_gpt_json(text, channel=channel)

def create_card(channel: str, post_id: Any, text: str) -> Optional[Dict]:
    # stop-topic prefilter
    if is_stop_topic(text):
        return None

    keywords_hit = extract_keywords_hit(text, DEFAULT_KEYWORDS)
    # soft filter: require at least one keyword OR strong construction hint
    if not keywords_hit and not any(h in (text or "").lower() for h in CONSTR_HINTS):
        return None

    ai = classify_with_ai(text, channel)
    if ai is None:
        # strict: no AI -> no card
        log.info("Skip card: YandexGPT unavailable")
        return None

    try:
        prob = float(ai.get("probability", 0))
    except Exception:
        prob = 0.0

    if prob < MIN_AI_GATE:
        return None

    card_id = build_card_id(channel, post_id)
    card = {
        "card_id": card_id,
        "created_ts": now_ts(),
        "timestamp": now_ts(),
        "channel": channel,
        "post_id": post_id,
        "text": text,
        "keywords": keywords_hit,
        "links": extract_links(text),
        "ai": {
            "probability": prob,
            "comment": ai.get("comment", ""),
        },
        "onzs": {},
    }

    # ONZS detect AFTER passing AI gate
    onzs = detect_onzs_with_yagpt(text, channel=channel)
    if onzs and onzs.get("onzs"):
        try:
            o = int(onzs.get("onzs"))
        except Exception:
            o = None
        if o and 1 <= o <= 12:
            card["onzs"] = {
                "ai": o,
                "confidence": float(onzs.get("confidence", 0) or 0),
                "reason": (onzs.get("reason") or "").strip(),
                "source": "ai",
                "confirmed": False,
            }

    save_card(card)
    return card

def build_card_text(card: Dict) -> str:
    ts = int(card.get("timestamp", now_ts()))
    dt = datetime.fromtimestamp(ts).strftime("%d.%m.%Y %H:%M")
    kw = ", ".join(card.get("keywords", [])) or "—"
    links = card.get("links") or []
    links_str = "\n".join(links) if links else "нет ссылок"

    ai = card.get("ai") or {}
    prob = ai.get("probability")
    comment = ai.get("comment")

    ai_lines: List[str] = []
    if prob is not None:
        try:
            p = float(prob)
        except Exception:
            p = None
        if p is not None:
            ai_lines.append(f"🤖 Вероятность самостроя (ИИ): {p:.1f}%")
    if comment:
        ai_lines.append(f"💬 Комментарий ИИ: {comment}")

    base = (
        "🔎 Обнаружено подозрительное сообщение\n"
        f"Источник: @{card.get('channel','—')}\n"
        f"Дата: {dt}\n"
        f"ID поста: {card.get('post_id','—')}\n\n"
        f"🔑 Ключевые слова: {kw}\n\n"
        "📝 Текст:\n"
        f"{card.get('text','')}\n\n"
        "📎 Ссылки:\n"
        f"{links_str}\n\n"
        f"🆔 ID карточки: {card.get('card_id','—')}"
    )

    if ai_lines:
        base += "\n\n" + "\n".join(ai_lines)

    # ONZS block
    oz = card.get("onzs") or {}
    val = oz.get("value") if oz.get("value") else oz.get("ai")
    if val:
        src = oz.get("source") or ("ai" if oz.get("ai") else "manual")
        conf = oz.get("confidence")
        confirmed = oz.get("confirmed")
        line = f"🏗 ОНзС: {val}"
        if src == "ai" and conf is not None:
            try:
                line += f" ({int(float(conf)*100)}%)"
            except Exception:
                pass
        if confirmed:
            line += " ✅ подтверждено"
        base += "\n\n" + line

        reason = (oz.get("reason") or "").strip()
        if src == "ai" and reason:
            base += "\n" + f"📌 Причина: {reason}"

    return base

def append_history(entry: Dict):
    entry = dict(entry)
    entry["ts"] = now_ts()
    append_jsonl(HISTORY_CARDS, entry)

# ----------------------------- CALLBACK HANDLER -----------------------------
def handle_callback_query(upd: Dict):
    cb = upd.get("callback_query") or {}
    data = cb.get("data") or ""
    cb_id = cb.get("id") or ""
    msg = cb.get("message") or {}
    from_user = (cb.get("from") or {}).get("id")
    chat_id = (msg.get("chat") or {}).get("id")
    message_id = msg.get("message_id")

    if not from_user:
        answer_callback(cb_id, "Ошибка", show_alert=True)
        return

    # -------------------- ADMIN ACTIONS --------------------
    if data.startswith("admin:"):
        # доступ: админ/модератор/руководство
        if not (is_admin(from_user) or is_moderator(from_user) or is_lead(from_user)):
            answer_callback(cb_id, "Нет доступа", show_alert=True)
            return

        op = data.split(":", 1)[1]

        if op == "onzs_stats":
            if chat_id:
                send_message(chat_id, build_onzs_stats())
            answer_callback(cb_id, "Готово")
            return

        if op == "reload_onzs":
            load_onzs_catalog()
            if chat_id:
                send_message(chat_id, f"🔄 Каталог ОНзС перезагружен: {len(ONZS_MAP)} элементов")
            answer_callback(cb_id, "Перезагружено")
            return

        if op == "test_yagpt":
            ok = False
            detail = ""
            try:
                out_text, meta = call_yandex_gpt_raw([
                    {"role": "system", "text": "Отвечай одной строкой: OK."},
                    {"role": "user", "text": "Ответь одной строкой: OK"},
                ])
                if isinstance(out_text, str) and "OK" in out_text.upper():
                    ok = True
                else:
                    detail = (out_text or "")[:200]
            except Exception as e:
                detail = str(e)[:200]

            if chat_id:
                send_message(chat_id, "✅ YandexGPT: OK" if ok else f"⚠️ YandexGPT: нет ответа. {detail}")
            answer_callback(cb_id, "OK" if ok else "Проблема")
            return

        answer_callback(cb_id, "Неизвестная команда", show_alert=True)
        return

    # -------------------- ONZS ACTIONS --------------------
    if data.startswith("onzs:"):
        if not is_moderator(from_user):
            answer_callback(cb_id, "❌ Нет доступа.", show_alert=True)
            return

        parts = data.split(":")
        op = parts[1] if len(parts) > 1 else ""

        if op == "edit" and len(parts) == 3:
            card_id = parts[2]
            if chat_id and message_id:
                edit_reply_markup(chat_id, message_id, reply_markup=build_onzs_pick_keyboard(card_id))
            answer_callback(cb_id, "Выбери ОНзС (1–12)")
            return

        if op == "set" and len(parts) == 4:
            card_id = parts[2]
            try:
                n = int(parts[3])
            except Exception:
                n = 0
            if n < 1 or n > 12:
                answer_callback(cb_id, "ОНзС должен быть 1–12", show_alert=True)
                return

            card = load_card(card_id)
            if not card:
                answer_callback(cb_id, "Карточка не найдена", show_alert=True)
                return

            card.setdefault("onzs", {})
            card["onzs"]["value"] = n
            card["onzs"]["source"] = "manual"
            card["onzs"]["confirmed"] = False
            card["onzs"]["updated_by"] = from_user
            card["onzs"]["updated_ts"] = now_ts()
            save_card(card)

            # learning: manual correction is also a training signal (confirmed=False)
            save_onzs_training(card.get("text", ""), n, confirmed=False)

            if chat_id and message_id:
                edit_message_text(chat_id, message_id, build_card_text(card), reply_markup=build_card_keyboard(card_id))
            answer_callback(cb_id, f"ОНзС установлен: {n}")
            return

        if op == "confirm" and len(parts) == 3:
            card_id = parts[2]
            card = load_card(card_id)
            if not card:
                answer_callback(cb_id, "Карточка не найдена", show_alert=True)
                return

            oz = card.get("onzs") or {}
            val = oz.get("value") if oz.get("value") else oz.get("ai")
            if not val:
                answer_callback(cb_id, "ОНзС ещё не определён", show_alert=True)
                return

            card.setdefault("onzs", {})
            card["onzs"]["confirmed"] = True
            card["onzs"]["confirmed_by"] = from_user
            card["onzs"]["confirmed_ts"] = now_ts()
            save_card(card)

            # learning: confirmation is a positive example
            save_onzs_training(card.get("text", ""), int(val), confirmed=True)

            if chat_id and message_id:
                edit_message_text(chat_id, message_id, build_card_text(card), reply_markup=build_card_keyboard(card_id))
            answer_callback(cb_id, "ОНзС подтверждён")
            return

        if op == "back" and len(parts) == 3:
            card_id = parts[2]
            if chat_id and message_id:
                edit_reply_markup(chat_id, message_id, reply_markup=build_card_keyboard(card_id))
            answer_callback(cb_id, "Ок")
            return

        answer_callback(cb_id, "Неизвестная команда ОНзС", show_alert=True)
        return

    # -------------------- CARD ACTIONS --------------------
    if data.startswith("card:"):
        parts = data.split(":")
        if len(parts) != 3:
            answer_callback(cb_id, "Ошибка", show_alert=True)
            return
        card_id, action = parts[1], parts[2]
        if not is_moderator(from_user):
            answer_callback(cb_id, "❌ Нет доступа.", show_alert=True)
            return

        card = load_card(card_id)
        if not card:
            answer_callback(cb_id, "Карточка не найдена", show_alert=True)
            return

        label = None
        if action == "work":
            label = "work"
            answer_callback(cb_id, "Принято: В работу")
        elif action == "wrong":
            label = "wrong"
            answer_callback(cb_id, "Принято: Неверно")
        elif action == "attach":
            label = "attach"
            answer_callback(cb_id, "Принято: Привязать")
        else:
            answer_callback(cb_id, "Неизвестное действие", show_alert=True)
            return

        append_history({"text": card.get("text", ""), "label": label, "channel": card.get("channel", ""), "reason": "user_action"})

        # Remove buttons after action
        if chat_id and message_id:
            edit_reply_markup(chat_id, message_id, reply_markup={"inline_keyboard": []})
        return

    answer_callback(cb_id, "OK")

# ----------------------------- COMMANDS -----------------------------
def handle_message(upd: Dict):
    msg = upd.get("message") or {}
    text = (msg.get("text") or "").strip()
    chat_id = (msg.get("chat") or {}).get("id")
    from_user = (msg.get("from") or {}).get("id")
    if not chat_id or not from_user:
        return

    if text == "/admin":
        uid = get_sender_user_id(update)
        if not (is_admin(from_user) or is_moderator(from_user) or is_lead(from_user)):
            send_message(chat_id, "❌ Нет доступа.")
            return

        onzs_cnt = len(ONZS_MAP) if isinstance(ONZS_MAP, dict) else 0
        yagpt_enabled = bool(YAGPT_API_KEY and YAGPT_FOLDER_ID)
        info = []
        info.append("🛠 Админ-панель")
        info.append(f"ID: {from_user}")
        info.append(f"YandexGPT: {'ON' if yagpt_enabled else 'OFF'} | model={YAGPT_MODEL}")
        info.append(f"AI-gate: {MIN_AI_GATE}% | HTTP_TIMEOUT={HTTP_TIMEOUT}s")
        info.append(f"ОНзС каталог: {onzs_cnt} | файл: {ONZS_XLSX}")
        info.append(f"Admins: {len(ADMINS)} | Moderators: {len(MODERATORS)} | Leadership: {len(LEADERSHIP)}")
        send_message(chat_id, "\n".join(info), reply_markup=build_admin_keyboard())
        return

    if text == "/onzs_ai_stats":
        uid = get_sender_user_id(update)
        if not (is_admin(from_user) or is_moderator(from_user) or is_lead(from_user)):
            send_message(chat_id, "❌ Нет доступа.")
            return
        send_message(chat_id, build_onzs_stats())
        return

    if text == "/start":
        send_message(chat_id, "Бот запущен.")
        return

# ----------------------------- GETUPDATES LOOP -----------------------------
def acquire_lock() -> bool:
    try:
        if os.path.exists(LOCK_FILE):
            # stale lock check: 10 minutes
            if now_ts() - int(os.path.getmtime(LOCK_FILE)) > 600:
                os.remove(LOCK_FILE)
            else:
                return False
        with open(LOCK_FILE, "w", encoding="utf-8") as f:
            f.write(str(now_ts()))
        return True
    except Exception:
        return True

def touch_lock():
    try:
        with open(LOCK_FILE, "w", encoding="utf-8") as f:
            f.write(str(now_ts()))
    except Exception:
        pass

def run_poller():
    offset = 0
    backoff = 2
    max_backoff = 30

    while True:
        touch_lock()

        # Use long-polling (timeout=45) and ensure HTTP timeout exceeds it
        resp = tg_get("getUpdates", {"timeout": 45, "offset": offset}, timeout_override=max(HTTP_TIMEOUT, 60))

        if not resp or not resp.get("ok"):
            # Network is often unstable on some hosts; apply exponential backoff
            time.sleep(backoff)
            backoff = min(max_backoff, backoff * 2)
            continue

        # Successful response -> reset backoff
        backoff = 2

        updates = resp.get("result", [])
        for u in updates:
            offset = max(offset, (u.get("update_id", 0) + 1))
            if "callback_query" in u:
                handle_callback_query(u)
            elif "message" in u:
                handle_message(u)

# ----------------------------- MAIN -----------------------------
def main():
    init_db()
    load_onzs_catalog()
    log.info('=== VERSION: ONZS + AI-GATE + BUTTONS + STATS ===')

    if YAGPT_API_KEY and YAGPT_FOLDER_ID:
        log.info(f"[YAGPT] enabled | folder={YAGPT_FOLDER_ID} | model={YAGPT_MODEL}")
    else:
        log.warning("[YAGPT] disabled (missing key/folder)")

    log.info("SAMASTROI SCRAPER starting...")
    log.info(f"DATA_DIR={DATA_DIR}")
    log.info(f"TARGET_CHAT_ID={TARGET_CHAT_ID}")
    log.info(f"SCAN_INTERVAL={SCAN_INTERVAL}")
    log.info(f"Admins: {ADMINS}")
    log.info(f"Moderators: {MODERATORS}")
    log.info(f"Leadership: {LEADERSHIP}")
    log.info(f"Prob threshold: {MIN_AI_GATE}%")

    if not acquire_lock():
        log.error("Lock exists: another poller is running. Exiting.")
        return

    log.info(f"Lock acquired: {LOCK_FILE}")
    log.info("Starting getUpdates poller...")
    run_poller()

if __name__ == "__main__":
    main()

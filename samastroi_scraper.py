#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAMASTROI SCRAPER (Railway-ready, OFFICIAL)
- Web-scrapes t.me/s/<channel> pages
- Builds "cards", enriches with YandexGPT probability (JSON response)
- Auto-filters by probability threshold
- Sends cards to a Telegram group with inline action buttons
- Global decision lock: once any admin clicks ("В работу / Неверно / Привязать") buttons disappear for everyone
- Training log (JSONL) + decisions & daily aggregates (SQLite) stored on Railway Volume
- Admin panel: threshold, role management, stats, log, PNG plot, XLSX/PDF reports, KPI dashboard
- Protection from double poller: single-instance lock on shared volume (prevents 409 getUpdates conflict)
- "Self-training" (practical): few-shot retrieval + adaptive calibration weights (per-channel bias) stored in DB
"""

import os
import re
import json
import time
import uuid
import sqlite3
import logging
import threading
from datetime import datetime, timezone, date, timedelta
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

# matplotlib (PNG plot)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reports
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from openpyxl import Workbook
from openpyxl.utils import get_column_letter

# ----------------------------- LOGGING -----------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("samastroi_scraper")

# ----------------------------- ENV / PATHS -----------------------------
DATA_DIR = os.getenv("DATA_DIR", "/data")
os.makedirs(DATA_DIR, exist_ok=True)

CARDS_DIR = os.path.join(DATA_DIR, "cards")
os.makedirs(CARDS_DIR, exist_ok=True)

REPORTS_DIR = os.path.join(DATA_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

TRAINING_DATASET = os.path.join(DATA_DIR, "training_dataset.jsonl")
HISTORY_CARDS = os.path.join(DATA_DIR, "history_cards.jsonl")
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")
DB_PATH = os.path.join(DATA_DIR, "scraper.db")
LOCK_PATH = os.path.join(DATA_DIR, ".poller.lock")

def _ensure_file(path: str, default: str = ""):
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(default)

_ensure_file(TRAINING_DATASET)
_ensure_file(HISTORY_CARDS)
_ensure_file(SETTINGS_FILE, "{}")

# ----------------------------- TELEGRAM -----------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
TELEGRAM_API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}" if BOT_TOKEN else ""

TARGET_CHAT_ID = int(os.getenv("TARGET_CHAT_ID", "-1003502443229"))  # group/channel for cards

# ----------------------------- ROLES (DEFAULTS) -----------------------------
# from user:
DEFAULT_LEADERSHIP = [5685586625]
DEFAULT_ADMINS = [272923789, 398960707]
DEFAULT_MODERATORS = [978125225, 777464055]

# ----------------------------- YANDEX GPT -----------------------------
YAGPT_API_KEY = os.getenv("YAGPT_API_KEY", "").strip()
YAGPT_FOLDER_ID = os.getenv("YAGPT_FOLDER_ID", "").strip()
YAGPT_ENDPOINT = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
YAGPT_MODEL = os.getenv("YAGPT_MODEL", "gpt://{folder_id}/yandexgpt/latest")

# log YandexGPT readiness
if not YAGPT_API_KEY or not YAGPT_FOLDER_ID:
    log.warning("[YAGPT] disabled: missing YAGPT_API_KEY or YAGPT_FOLDER_ID")
else:
    try:
        log.info(f"[YAGPT] enabled | folder={YAGPT_FOLDER_ID} | model={YAGPT_MODEL.format(folder_id=YAGPT_FOLDER_ID)}")
    except Exception:
        log.info(f"[YAGPT] enabled | folder={YAGPT_FOLDER_ID} | model={YAGPT_MODEL}")


# ----------------------------- SCRAPER SETTINGS -----------------------------
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "300"))

MIN_AI_GATE = float(os.getenv(\"MIN_AI_GATE\", \"5\"))  # минимальный порог ИИ для отправки карточки
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "15"))
MAX_TRAIN_LOG = int(os.getenv("MAX_TRAIN_LOG", "50"))
TARGET_DATASET_SIZE = int(os.getenv("TARGET_DATASET_SIZE", "5000"))
DEFAULT_THRESHOLD = int(os.getenv("DEFAULT_THRESHOLD", "0"))

DEFAULT_CHANNELS = [
    "tipkhimki", "lobnya", "dolgopacity", "vkhimki",
    "podslushanovsolnechnogorske", "klingorod", "mspeaks",
    "pushkino_official", "podmoskow", "trofimovonline",
    "Tipichnoe_Pushkino", "chp_sergiev_posad", "kraftyou",
    "kontext_channel", "podslushano_ivanteevka", "pushkino_live",
    "life_sergiev_posad"
]
CHANNEL_LIST = [c.strip() for c in os.getenv("CHANNEL_LIST", "").split(",") if c.strip()] or DEFAULT_CHANNELS

KEYWORDS = [
    "стройка", "строительство", "самострой", "котлован", "фундамент",
    "арматура", "многоквартирный", "жилой комплекс", "кран", "экскаватор",
    "строители", "проверка", "застройщик", "разрешение", "рнс",
    "снос", "надзор", "инженер", "штраф"
]
KEYWORDS_LOWER = [k.lower() for k in KEYWORDS]
# ----------------------------- CONTEXT FILTERS -----------------------------
# Политика/ЧП/война и прочий "шум" — не самострой. Сразу отсекаем.
STOP_TOPICS = [
    "путин", "украин", "войн", "сво", "нато", "санкц", "президент", "госдума",
    "выбор", "митинг", "протест", "миграц", "теракт", "убийств", "дтп", "пожар",
    "мобилизац", "фронт", "обстрел", "ракет", "дрон", "армия"
]

AMBIGUOUS_KEYWORDS = {"кран"}  # часто встречается в быту (кухонный кран и т.п.)
CONSTRUCTION_CONTEXT = [
    "строй", "строит", "строитель", "котлован", "фундамент", "арматур", "бетон",
    "плита", "опалуб", "этаж", "перекрыт", "забор", "огражден", "площадк",
    "экскават", "самосвал", "рнс", "разрешени", "застройщик", "генподряд",
    "монолит", "кладк", "кирпич", "панел", "балк", "сваи", "бурен", "скважин"
]

def is_noise_topic(text: str) -> bool:
    low = (text or "").lower()
    return any(w in low for w in STOP_TOPICS)

def has_construction_context(text: str) -> bool:
    low = (text or "").lower()
    return any(w in low for w in CONSTRUCTION_CONTEXT)

def is_relevant_hit(text: str, found_keywords: List[str]) -> bool:
    # жёсткий отсев полит/ЧП
    if is_noise_topic(text):
        return False
    # если нашли только неоднозначные ключи (например, "кран") — нужен строительный контекст
    found_set = set([k.lower() for k in (found_keywords or [])])
    if found_set and found_set.issubset(AMBIGUOUS_KEYWORDS):
        return has_construction_context(text)
    return True


# ----------------------------- DB -----------------------------
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def init_db():
    conn = db()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS seen_posts (
            channel TEXT NOT NULL,
            post_id TEXT NOT NULL,
            first_seen_ts INTEGER NOT NULL,
            PRIMARY KEY (channel, post_id)
        );
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS card_decisions (
            card_id TEXT PRIMARY KEY,
            decision TEXT NOT NULL,
            decided_by INTEGER NOT NULL,
            decided_ts INTEGER NOT NULL
        );
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS train_daily (
            day TEXT PRIMARY KEY,
            total INTEGER NOT NULL,
            work INTEGER NOT NULL,
            wrong INTEGER NOT NULL,
            attach INTEGER NOT NULL
        );
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_roles (
            user_id INTEGER PRIMARY KEY,
            role TEXT NOT NULL
        );
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS model_params (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL
        );
    """)

    # seed roles if empty
    cnt = int(conn.execute("SELECT COUNT(*) FROM user_roles;").fetchone()[0] or 0)
    if cnt == 0:
        for uid in DEFAULT_LEADERSHIP:
            conn.execute("INSERT OR REPLACE INTO user_roles(user_id, role) VALUES (?, ?);", (int(uid), "leadership"))
        for uid in DEFAULT_ADMINS:
            conn.execute("INSERT OR REPLACE INTO user_roles(user_id, role) VALUES (?, ?);", (int(uid), "admin"))
        for uid in DEFAULT_MODERATORS:
            conn.execute("INSERT OR REPLACE INTO user_roles(user_id, role) VALUES (?, ?);", (int(uid), "moderator"))

    conn.execute(
        "INSERT OR IGNORE INTO model_params(key, value_json) VALUES (?, ?);",
        ("threshold", json.dumps({"value": DEFAULT_THRESHOLD}, ensure_ascii=False))
    )
    # weights: per-channel bias in probability points ([-25..25]) + label weights for aggregation
    conn.execute(
        "INSERT OR IGNORE INTO model_params(key, value_json) VALUES (?, ?);",
        ("weights", json.dumps({"channels": {}, "label_weights": {"work": 1.0, "wrong": 1.0, "attach": 1.0}}, ensure_ascii=False))
    )

    conn.commit()
    conn.close()

init_db()

# ----------------------------- SETTINGS -----------------------------
def load_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path: str, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def load_settings() -> Dict:
    return load_json(SETTINGS_FILE, {})

def save_settings(s: Dict):
    save_json(SETTINGS_FILE, s)

def get_prob_threshold() -> int:
    s = load_settings()
    try:
        v = int(s.get("prob_threshold", DEFAULT_THRESHOLD))
        return max(0, min(100, v))
    except Exception:
        return DEFAULT_THRESHOLD

def set_prob_threshold(v: int):
    v = max(0, min(100, int(v)))
    s = load_settings()
    s["prob_threshold"] = v
    save_settings(s)

def get_update_offset() -> int:
    s = load_settings()
    try:
        return int(s.get("update_offset", 0))
    except Exception:
        return 0

def set_update_offset(v: int):
    s = load_settings()
    s["update_offset"] = int(v)
    save_settings(s)

# ----------------------------- ROLES API -----------------------------
def get_role(user_id: int) -> Optional[str]:
    conn = db()
    row = conn.execute("SELECT role FROM user_roles WHERE user_id=?;", (int(user_id),)).fetchone()
    conn.close()
    return row[0] if row else None

def is_admin(user_id: int) -> bool:
    return get_role(int(user_id)) == "admin"

def is_moderator(user_id: int) -> bool:
    return get_role(int(user_id)) == "moderator"

def is_leadership(user_id: int) -> bool:
    return get_role(int(user_id)) == "leadership"

def list_users_by_role(role: str) -> List[int]:
    conn = db()
    rows = conn.execute("SELECT user_id FROM user_roles WHERE role=? ORDER BY user_id;", (role,)).fetchall()
    conn.close()
    return [int(r[0]) for r in rows]

def upsert_role(user_id: int, role: str) -> None:
    conn = db()
    conn.execute("INSERT OR REPLACE INTO user_roles(user_id, role) VALUES (?, ?);", (int(user_id), role))
    conn.commit()
    conn.close()

def remove_role(user_id: int) -> None:
    conn = db()
    conn.execute("DELETE FROM user_roles WHERE user_id=?;", (int(user_id),))
    conn.commit()
    conn.close()

def add_admin(user_id: int): upsert_role(int(user_id), "admin")
def add_moderator(user_id: int): upsert_role(int(user_id), "moderator")
def add_leadership(user_id: int): upsert_role(int(user_id), "leadership")

def remove_admin(user_id: int): remove_role(int(user_id))
def remove_moderator(user_id: int): remove_role(int(user_id))
def remove_leadership(user_id: int): remove_role(int(user_id))

# ----------------------------- SINGLE INSTANCE LOCK -----------------------------
def acquire_lock_or_exit() -> None:
    """
    Prevent multiple instances from running getUpdates poller on Railway volume.
    """
    try:
        fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
        log.info(f"Lock acquired: {LOCK_PATH}")
    except FileExistsError:
        log.error("Another instance holds poller lock. Exiting (prevents 409 getUpdates conflict).")
        raise SystemExit(0)

def release_lock():
    try:
        if os.path.exists(LOCK_PATH):
            os.remove(LOCK_PATH)
            log.info("Lock released.")
    except Exception:
        pass

# ----------------------------- UTIL -----------------------------
def now_ts() -> int:
    return int(time.time())

def append_jsonl(path: str, obj: Dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\n", " ").replace("\t", " ")
    text = " ".join(text.split())
    return text.strip()

def detect_keywords(text: str) -> List[str]:
    low = text.lower()
    return [kw for kw in KEYWORDS_LOWER if kw in low]

def parse_tg_datetime(dt_str: str) -> int:
    """
    Telegram embeds: <time datetime="2025-12-14T16:27:14+00:00">
    Convert to unix ts.
    """
    if not dt_str:
        return now_ts()
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        return now_ts()

# ----------------------------- MODEL PARAMS (WEIGHTS) -----------------------------
def _get_model_param(key: str, default_obj: Dict) -> Dict:
    conn = db()
    row = conn.execute("SELECT value_json FROM model_params WHERE key=?;", (key,)).fetchone()
    conn.close()
    if not row:
        return default_obj
    try:
        return json.loads(row[0])
    except Exception:
        return default_obj

def _set_model_param(key: str, obj: Dict) -> None:
    conn = db()
    conn.execute("INSERT OR REPLACE INTO model_params(key, value_json) VALUES (?, ?);", (key, json.dumps(obj, ensure_ascii=False)))
    conn.commit()
    conn.close()

def get_channel_bias(channel: str) -> float:
    w = _get_model_param("weights", {"channels": {}, "label_weights": {}})
    try:
        return float(w.get("channels", {}).get(channel, 0.0))
    except Exception:
        return 0.0

def update_channel_bias(channel: str, label: str) -> None:
    """
    Adaptive calibration:
    - If admins often mark channel posts as "work/attach" -> increase bias (more aggressive)
    - If often "wrong" -> decrease bias
    Stored as +/- probability points.
    """
    w = _get_model_param("weights", {"channels": {}, "label_weights": {"work": 1.0, "wrong": 1.0, "attach": 1.0}})
    ch = w.setdefault("channels", {})
    cur = float(ch.get(channel, 0.0) or 0.0)

    step = 1.5  # points per decision
    if label in ("work", "attach"):
        cur += step
    elif label == "wrong":
        cur -= step

    cur = max(-25.0, min(25.0, cur))
    ch[channel] = round(cur, 2)
    _set_model_param("weights", w)


# ----------------------------- KEYWORD LEARNING (ONLINE) -----------------------------
KW_STATS_KEY = "kw_stats"

def _get_kw_stats() -> Dict:
    # {"kw": {"work": n, "wrong": n, "attach": n, "total": n}}
    return _get_model_param(KW_STATS_KEY, {"kw": {}})

def _set_kw_stats(obj: Dict) -> None:
    _set_model_param(KW_STATS_KEY, obj)

def update_keyword_stats(text: str, label: str) -> None:
    """Online learning: собираем статистику по ключевым словам из разметки."""
    if label not in ("work", "wrong", "attach"):
        return
    kws = detect_keywords((text or "").lower())
    if not kws:
        return

    st = _get_kw_stats()
    kw_map = st.setdefault("kw", {})

    for kw in kws:
        rec = kw_map.setdefault(kw, {"work": 0, "wrong": 0, "attach": 0, "total": 0})
        rec["total"] = int(rec.get("total", 0)) + 1
        rec[label] = int(rec.get(label, 0)) + 1

    _set_kw_stats(st)

def get_keyword_bias_points(text: str) -> float:
    """Возвращает поправку (-15..+15) к вероятности на основе истории разметки."""
    kws = detect_keywords((text or "").lower())
    if not kws:
        return 0.0

    st = _get_kw_stats()
    kw_map = (st.get("kw") or {})
    score = 0.0
    used = 0

    for kw in kws:
        rec = kw_map.get(kw)
        if not rec:
            continue
        total = float(rec.get("total", 0) or 0)
        if total < 5:
            continue  # мало данных по ключу — не влияем
        pos = float((rec.get("work", 0) or 0) + (rec.get("attach", 0) or 0))
        neg = float(rec.get("wrong", 0) or 0)
        val = (pos - neg) / max(1.0, total)  # (-1..+1)
        score += val
        used += 1

    if used == 0:
        return 0.0

    score = score / used
    points = score * 10.0  # масштаб (8..12)
    return max(-15.0, min(15.0, points))

# ----------------------------- DEDUPE & DECISIONS -----------------------------
def mark_seen(channel: str, post_id: str, ts: int) -> bool:
    conn = db()
    try:
        conn.execute("INSERT OR IGNORE INTO seen_posts(channel, post_id, first_seen_ts) VALUES(?,?,?)", (channel, post_id, ts))
        changed = conn.execute("SELECT changes()").fetchone()[0] == 1
        return changed
    finally:
        conn.close()

def decision_exists(card_id: str) -> Optional[Tuple[str, int, int]]:
    conn = db()
    try:
        row = conn.execute("SELECT decision, decided_by, decided_ts FROM card_decisions WHERE card_id=?", (card_id,)).fetchone()
        return row if row else None
    finally:
        conn.close()

def set_decision(card_id: str, decision: str, user_id: int) -> bool:
    """
    Idempotent global decision: only first admin click is accepted.
    """
    conn = db()
    try:
        conn.execute("BEGIN IMMEDIATE;")
        row = conn.execute("SELECT card_id FROM card_decisions WHERE card_id=?", (card_id,)).fetchone()
        if row:
            conn.execute("COMMIT;")
            return False
        conn.execute(
            "INSERT INTO card_decisions(card_id, decision, decided_by, decided_ts) VALUES(?,?,?,?)",
            (card_id, decision, int(user_id), now_ts()),
        )
        conn.execute("COMMIT;")
        return True
    except Exception:
        conn.execute("ROLLBACK;")
        raise
    finally:
        conn.close()

# ----------------------------- TRAINING LOGIC -----------------------------
def update_train_daily(label: str):
    d = date.today().isoformat()
    conn = db()
    try:
        row = conn.execute("SELECT total, work, wrong, attach FROM train_daily WHERE day=?", (d,)).fetchone()
        if row:
            total, work, wrong, attach = row
        else:
            total = work = wrong = attach = 0
        total += 1
        if label == "work":
            work += 1
        elif label == "wrong":
            wrong += 1
        elif label == "attach":
            attach += 1
        conn.execute(
            "INSERT OR REPLACE INTO train_daily(day,total,work,wrong,attach) VALUES(?,?,?,?,?)",
            (d, total, work, wrong, attach),
        )
    finally:
        conn.close()

def log_training_event(card_id: str, label: str, text: str, channel: str, admin_id: int):
    rec = {
        "timestamp": now_ts(),
        "card_id": card_id,
        "label": label,
        "admin_id": int(admin_id),
        "channel": channel,
        "text": (text or "")[:5000],
    }
    append_jsonl(TRAINING_DATASET, rec)
    update_train_daily(label)
    update_channel_bias(channel, label)
    update_keyword_stats(text, label)

def compute_training_stats() -> Dict:
    conn = db()
    rows = conn.execute("SELECT total, work, wrong, attach FROM train_daily").fetchall()
    conn.close()

    total = sum(r[0] for r in rows) if rows else 0
    work = sum(r[1] for r in rows) if rows else 0
    wrong = sum(r[2] for r in rows) if rows else 0
    attach = sum(r[3] for r in rows) if rows else 0

    last_ts = None
    try:
        with open(TRAINING_DATASET, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size > 0:
                f.seek(max(0, size - 8192), os.SEEK_SET)
                chunk = f.read().decode("utf-8", errors="ignore")
                lines = [ln for ln in chunk.splitlines() if ln.strip()]
                for ln in reversed(lines):
                    try:
                        obj = json.loads(ln)
                        ts = obj.get("timestamp")
                        if isinstance(ts, int):
                            last_ts = ts
                            break
                    except Exception:
                        continue
    except Exception:
        pass

    prog = 0.0 if TARGET_DATASET_SIZE <= 0 else min(1.0, total / TARGET_DATASET_SIZE)
    return {
        "total": total,
        "work": work,
        "wrong": wrong,
        "attach": attach,
        "progress": round(prog * 100.0, 2),
        "confidence": round(prog * 100.0, 2),
        "last_ts": last_ts,
        "target": TARGET_DATASET_SIZE,
    }

def tail_training_log(limit: int = MAX_TRAIN_LOG) -> List[Dict]:
    if not os.path.exists(TRAINING_DATASET):
        return []
    try:
        with open(TRAINING_DATASET, "r", encoding="utf-8") as f:
            lines = f.readlines()
        out = []
        for ln in lines[-limit:]:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except Exception:
                pass
        return out
    except Exception:
        return []

def sparkline(values: List[int]) -> str:
    if not values:
        return "—"
    blocks = "▁▂▃▄▅▆▇█"
    mn, mx = min(values), max(values)
    if mx == mn:
        return blocks[0] * len(values)
    out = []
    for v in values:
        idx = int((v - mn) * (len(blocks) - 1) / (mx - mn))
        out.append(blocks[idx])
    return "".join(out)

def training_plot_text(days: int = 14) -> str:
    rows = _fetch_train_daily_last(days)
    if not rows:
        return "📊 График роста обучения: данных пока нет."
    labels = [r[0][5:] for r in rows]
    totals = [int(r[1]) for r in rows]
    return "📊 График роста обучения (событий в день):\n" + sparkline(totals) + "\n" + " | ".join(f"{labels[i]}:{totals[i]}" for i in range(len(labels)))

# ----------------------------- YANDEXGPT SELF-TRAINING (PRACTICAL) -----------------------------
def select_few_shot_examples(text: str, k: int = 3) -> List[Dict]:
    """
    Retrieves recent labeled examples by keyword overlap for few-shot calibration.
    """
    keys = set(detect_keywords((text or "").lower()))
    if not keys:
        return []
    events = tail_training_log(limit=250)
    scored = []
    for e in events:
        t = (e.get("text") or "").lower()
        if not t:
            continue
        ekeys = set(detect_keywords(t))
        score = len(keys & ekeys)
        if score > 0:
            scored.append((score, e))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:k]]

def _sanitize_for_llm(text: str, max_chars: int = 1800) -> str:
    """Light cleanup to reduce safety-triggering noise and keep request small."""
    if not text:
        return ""
    t = str(text)
    # drop URLs
    t = re.sub(r"https?://\S+", " ", t)
    # drop @mentions and hashtags (often noisy)
    t = re.sub(r"[@#][\w_]+", " ", t)
    # normalize whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t[:max_chars]


# ----------------------------- LLM SAFETY FILTER (skip obvious off-topic / high-risk) -----------------------------
# YandexGPT may refuse some topics and return non-JSON. We avoid sending such texts.
LLM_SKIP_PATTERNS = [
    r"\b(война|сво|армия|фронт|удар|ракет|дрон|террор|теракт)\b",
    r"\b(путин|зеленск|байден|выбор|голосован|партия|госдума|санкци)\b",
    r"\b(убийств|суицид|самоубийств|наркотик|героин|кокаин)\b",
    r"\b(порно|эротик|18\+)\b",
]

def llm_should_skip(text: str) -> bool:
    t = (text or "").lower()
    for pat in LLM_SKIP_PATTERNS:
        try:
            if re.search(pat, t):
                return True
        except re.error:
            continue
    return False

def _extract_json_object(s: str) -> Optional[str]:
    if not s:
        return None
    s = s.strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    # Try to extract first {...} block
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        return m.group(0)
    return None


def call_yandex_gpt_json(text: str, channel: str = "") -> Dict:
    """Call YandexGPT and force a JSON-like decision; never return None."""
    if not YAGPT_API_KEY or not YAGPT_FOLDER_ID:
        return {"probability": 0, "comment": "YandexGPT not configured"}

    model_uri = YAGPT_MODEL.format(folder_id=YAGPT_FOLDER_ID)

    cleaned = _sanitize_for_llm(text)

    # If content is obviously off-topic/high-risk, do not call LLM (it may refuse).
    if llm_should_skip(cleaned):
        hits = len(detect_keywords(cleaned.lower()))
        return {"probability": 0, "comment": "Skipped LLM (policy/off-topic)"}

    few = select_few_shot_examples(cleaned, k=3)

    few_block = ""
    if few:
        lines = ["Примеры разметки (для калибровки):"]
        for ex in few:
            lbl = ex.get("label")
            t = _sanitize_for_llm(ex.get("text") or "", max_chars=240)
            hint = "70-100" if lbl == "work" else ("0-30" if lbl == "wrong" else "40-70")
            lines.append(f"- Метка={lbl} (ориентир {hint}). Текст: {t}")
        few_block = "\n" + "\n".join(lines) + "\n"

    # our own bias to stabilize decisions (channel/keyword learning)
    bias_ch = get_channel_bias(channel) if channel else 0.0
    bias_kw = get_keyword_bias_points(cleaned)
    bias_total = bias_ch + bias_kw

    prompt = (
        "Ты помощник инспектора строительного надзора.\n"
        "Твоя задача: оценить вероятность (0-100), что текст относится к самовольному строительству/нарушениям на стройке.\n"
        "Запрещено: новости, политика, общие рассуждения.\n"
        "Если текст содержит запрещённые/неподходящие темы, всё равно верни JSON с probability=0 и кратким comment.\n"
        "Ответь СТРОГО одним JSON без пояснений и без Markdown.\n"
        "Формат: {\"probability\": <0-100>, \"comment\": \"...\"}\n"
        f"Калибровка (bias, прибавь к вероятности): {bias_total:+.1f}\n"
        + few_block +
        "\nТекст:\n" + cleaned
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

    # Retry on transient errors (429/5xx). Do not spam: max 4 attempts with backoff.
    data = None
    last_status = None
    for attempt in range(4):
        try:
            resp = requests.post(YAGPT_ENDPOINT, headers=headers, json=body, timeout=25)
            last_status = resp.status_code
            if resp.ok:
                data = resp.json()
                break
            # transient
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = min(10 * (2 ** attempt), 60)
                log.warning(f"YandexGPT HTTP {resp.status_code}; retry in {wait}s (attempt {attempt+1}/4). Body: {resp.text[:300]}")
                time.sleep(wait)
                continue
            # non-transient
            log.error(f"YandexGPT HTTP {resp.status_code}: {resp.text[:800]}")
            return {"probability": 0, "comment": f"YandexGPT HTTP {resp.status_code}"}
        except Exception as e:
            wait = min(5 * (2 ** attempt), 30)
            log.warning(f"YandexGPT request error: {e}; retry in {wait}s (attempt {attempt+1}/4)")
            time.sleep(wait)
            continue

    if data is None:
        return {"probability": 0, "comment": f"YandexGPT unavailable ({last_status})"}

    try:
        text_out = data["result"]["alternatives"][0]["message"]["text"]
    except Exception as e:
        log.error(f"YandexGPT response parse error: {e}; data={str(data)[:800]}")
        return {"probability": 0, "comment": "YandexGPT response parse error"}

    raw = (text_out or "").strip()

    # If model refused (common phrase), do not treat as error; fallback gracefully.
    if "не могу обсуждать" in raw.lower() or "давайте поговорим" in raw.lower():
        hits = len(detect_keywords(cleaned.lower()))
        return {"probability": 0, "comment": "YandexGPT refused"}

    candidate = _extract_json_object(raw) or raw

    try:
        obj = json.loads(candidate)
        p = float(obj.get("probability", 0))
        p = max(0.0, min(100.0, p))
        obj["probability"] = int(round(p))
        obj["comment"] = str(obj.get("comment", "")).strip()[:400]
        return obj
    except Exception as e:
        # Model refused or returned non-JSON.
        log.error(f"YandexGPT JSON parse error: {e}; text={raw[:300]}")
        # Heuristic fallback: if keywords hit, not zero.
        hits = len(detect_keywords(cleaned.lower()))
        return {"probability": 0, "comment": "YandexGPT invalid/refused"}


def enrich_card_with_yagpt(card: Dict) -> None:
    t = (card.get("text") or "").strip()
    if not t:
        return
    res = call_yandex_gpt_json(t, channel=card.get('channel',''))
    if not res:
        return
    prob = res.get("probability")
    comment = (res.get("comment") or "").strip()

    prob_f = None
    try:
        prob_f = float(prob)
    except Exception:
        prob_f = None

    if prob_f is not None:
        prob_f = max(0.0, min(100.0, prob_f))
        # calibration bias (channel + keyword learning)
        bias_ch = get_channel_bias(card.get("channel", ""))
        bias_kw = get_keyword_bias_points(t)
        bias = bias_ch + bias_kw

        prob_adj = max(0.0, min(100.0, prob_f + bias))
        card.setdefault("ai", {})
        card["ai"]["probability_raw"] = round(prob_f, 1)
        card["ai"]["bias"] = round(bias, 1)
        card["ai"]["bias_ch"] = round(bias_ch, 1)
        card["ai"]["bias_kw"] = round(bias_kw, 1)
        card["ai"]["probability"] = round(prob_adj, 1)

    if comment:
        card.setdefault("ai", {})
        card["ai"]["comment"] = comment[:600]


# ----------------------------- ONZS (1–12) -----------------------------
ONZS_XLSX = os.getenv("ONZS_XLSX", "Номера ОНзС.xlsx")
ONZS_CATALOG: Dict[int, str] = {}

def load_onzs_catalog() -> None:
    global ONZS_CATALOG
    try:
        if not os.path.exists(ONZS_XLSX):
            log.warning(f"[ONZS] catalog file not found: {ONZS_XLSX}")
            ONZS_CATALOG = {}
            return
        df = pd.read_excel(ONZS_XLSX)
        cat = {}
        for _, row in df.iterrows():
            try:
                n = int(float(str(row.iloc[0]).strip()))
            except Exception:
                continue
            if 1 <= n <= 12:
                desc = str(row.iloc[1]).strip() if len(row) > 1 else ""
                if desc:
                    cat[n] = desc
        ONZS_CATALOG = cat
        log.info(f"[ONZS] catalog loaded: {len(ONZS_CATALOG)} items from {ONZS_XLSX}")
    except Exception as e:
        log.error(f"[ONZS] catalog load error: {e}")
        ONZS_CATALOG = {}

load_onzs_catalog()

ONZS_TRAIN_FILE = os.path.join(DATA_DIR, "onzs_training.jsonl")

def call_yagpt_raw(prompt: str, max_tokens: int = 220, temperature: float = 0.1) -> Optional[str]:
    if not YAGPT_API_KEY or not YAGPT_FOLDER_ID:
        return None
    model_uri = YAGPT_MODEL.format(folder_id=YAGPT_FOLDER_ID)
    body = {
        "modelUri": model_uri,
        "completionOptions": {"stream": False, "temperature": float(temperature), "maxTokens": int(max_tokens)},
        "messages": [{"role": "user", "text": prompt}],
    }
    headers = {
        "Authorization": f"Api-Key {YAGPT_API_KEY}",
        "x-folder-id": YAGPT_FOLDER_ID,
        "Content-Type": "application/json",
    }
    last_text = None
    for attempt in range(4):
        try:
            resp = requests.post(YAGPT_ENDPOINT, headers=headers, json=body, timeout=30)
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(1.2 * (attempt + 1))
                continue
            if not resp.ok:
                log.error(f"[YAGPT] HTTP {resp.status_code}: {resp.text[:300]}")
                return None
            j = resp.json()
            alts = (((j.get("result") or {}).get("alternatives")) or [])
            if not alts:
                return None
            last_text = ((alts[0].get("message") or {}).get("text") or "").strip()
            return last_text or None
        except Exception as e:
            log.warning(f"[YAGPT] raw call error (attempt {attempt+1}): {e}")
            time.sleep(1.2 * (attempt + 1))
    return last_text

def save_onzs_training(text: str, onzs: int, confirmed: bool, ai_onzs: Optional[int] = None) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    rec = {
        "ts": now_ts(),
        "text": _sanitize_for_llm(text or "", max_chars=800),
        "onzs": int(onzs),
        "confirmed": bool(confirmed),
        "ai_onzs": int(ai_onzs) if ai_onzs else None,
    }
    try:
        with open(ONZS_TRAIN_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        log.error(f"[ONZS] training save error: {e}")

def build_onzs_stats_text() -> str:
    if not os.path.exists(ONZS_TRAIN_FILE):
        return "🎯 Точность ИИ по ОНзС\nНет данных."
    per = {}
    total = ok = 0
    with open(ONZS_TRAIN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            onzs = int(r.get("onzs") or 0)
            confirmed = bool(r.get("confirmed"))
            ai_onzs = r.get("ai_onzs")
            # если ai_onzs задан, то confirmed=True означает "ИИ был прав", confirmed=False означает "исправили"
            if ai_onzs is None:
                continue
            per.setdefault(onzs, {"all": 0, "ok": 0})
            per[onzs]["all"] += 1
            total += 1
            if confirmed:
                per[onzs]["ok"] += 1
                ok += 1
    acc = int(round((ok / total) * 100)) if total else 0
    lines = [f"🎯 Точность ИИ по ОНзС: {acc}% (верно {ok}/{total})"]
    for n in range(1, 13):
        if n in per and per[n]["all"] > 0:
            a = int(round((per[n]["ok"] / per[n]["all"]) * 100))
            lines.append(f"ОНзС-{n}: {a}% ({per[n]['ok']}/{per[n]['all']})")
    return "\n".join(lines)

def detect_onzs_with_yagpt(text: str) -> Optional[Dict]:
    if not ONZS_CATALOG:
        return None
    catalog = "\n".join([f"{k}: {v}" for k, v in sorted(ONZS_CATALOG.items())])
    prompt = (
        "Ты инспектор строительного надзора Московской области.\n"
        "Определи номер ОНзС (1–12) по описанию.\n"
        "Классификатор ОНзС:\n"
        f"{catalog}\n\n"
        "Верни ТОЛЬКО JSON без лишнего текста:\n"
        '{"onzs": 7, "confidence": 0.82, "reason": "кратко"}\n\n'
        "Текст:\n"
        f"{_sanitize_for_llm(text or '', max_chars=1200)}"
    )
    raw = call_yagpt_raw(prompt, max_tokens=200, temperature=0.1)
    if not raw:
        return None
    cand = _extract_json_object(raw) or raw
    try:
        obj = json.loads(cand)
        onzs = obj.get("onzs")
        if onzs is None:
            return None
        onzs = int(float(onzs))
        if onzs < 1 or onzs > 12:
            return None
        conf = float(obj.get("confidence", 0))
        conf = max(0.0, min(1.0, conf))
        reason = str(obj.get("reason", "")).strip()[:220]
        return {"onzs": onzs, "confidence": conf, "reason": reason}
    except Exception:
        return None

def enrich_card_with_onzs(card: Dict) -> None:
    ai = card.get("ai") or {}
    try:
        p = float(ai.get("probability"))
    except Exception:
        return
    # ОНзС имеет смысл только для вероятных случаев самостроя
    if p < max(MIN_AI_GATE, 10.0):
        return
    res = detect_onzs_with_yagpt(card.get("text") or "")
    if not res:
        return
    card["onzs"] = {
        "ai": res["onzs"],
        "confidence": round(res["confidence"], 3),
        "reason": res.get("reason", ""),
        "value": None,           # ручное значение
        "source": "ai",
        "confirmed": False,
    }

# ----------------------------- CARDS -----------------------------
def generate_card_id() -> str:
    return str(uuid.uuid4())[:12]

def save_card(card: Dict) -> str:
    path = os.path.join(CARDS_DIR, f"{card['card_id']}.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    return path

def load_card(card_id: str) -> Optional[Dict]:
    path = os.path.join(CARDS_DIR, f"{card_id}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def build_card_text(card: Dict) -> str:
    ts = int(card.get("timestamp", now_ts()))
    dt = datetime.fromtimestamp(ts).strftime("%d.%m.%Y %H:%M")
    kw = ", ".join(card.get("keywords", [])) or "—"
    links = card.get("links") or []
    links_str = "\n".join(links) if links else "нет ссылок"

    ai = card.get("ai") or {}
    prob = ai.get("probability")
    raw = ai.get("probability_raw")
    bias = ai.get("bias")
    comment = ai.get("comment")

    ai_lines = []
    if prob is not None:
        if raw is not None and bias is not None:
            ai_lines.append(f"🤖 Вероятность самостроя (ИИ): {prob:.1f}% (raw {raw:.1f}%, bias {bias:+.1f})")
        else:
            ai_lines.append(f"🤖 Вероятность самостроя (ИИ): {float(prob):.1f}%")
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
    base += "

" + "
".join(ai_lines)

# ONZS block
oz = card.get("onzs") or {}
# value preference: manual value if set, else ai
val = oz.get("value") if oz.get("value") else oz.get("ai")
if val:
    conf = oz.get("confidence")
    src = oz.get("source") or ("ai" if oz.get("ai") else "manual")
    confirmed = oz.get("confirmed")
    line = f"🏗 ОНзС: {val}"
    if src == "ai" and conf is not None:
        line += f" ({int(float(conf)*100)}%)"
    if confirmed:
        line += " ✅ подтверждено"
    base += "

" + line
    reason = (oz.get("reason") or "").strip()
    if src == "ai" and reason:
        base += "
" + f"📌 Причина: {reason}"

return base

def append_history(entry: Dict):
    entry = dict(entry)
    entry["ts"] = now_ts()
    append_jsonl(HISTORY_CARDS, entry)

# ----------------------------- TELEGRAM API HELPERS -----------------------------
def tg_get(method: str, params: Dict) -> Optional[Dict]:
    if not TELEGRAM_API_URL:
        return None
    try:
        r = requests.get(f"{TELEGRAM_API_URL}/{method}", params=params, timeout=HTTP_TIMEOUT)
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

def send_message(chat_id: int, text: str, reply_markup: Optional[Dict] = None) -> Optional[Dict]:
    """Send text message; split into chunks to satisfy Telegram 4096-char limit."""
    if text is None:
        text = ""
    limit = 3500  # safer margin under Telegram 4096 (UTF-8, markup)
    parts = [text[i:i+limit] for i in range(0, len(text), limit)] or [""]
    last_resp = None
    for idx, part in enumerate(parts):
        payload = {"chat_id": chat_id, "text": part, "disable_web_page_preview": False}
        # attach markup only to last part
        if reply_markup is not None and idx == len(parts) - 1:
            payload["reply_markup"] = reply_markup
        last_resp = tg_post("sendMessage", payload)
        if last_resp and last_resp.get("ok") is False:
            log.error(f"sendMessage failed: {last_resp}")
            break
    return last_resp

def edit_reply_markup(chat_id: int, message_id: int, reply_markup: Optional[Dict]):
    payload = {"chat_id": chat_id, "message_id": message_id}
    # To remove inline keyboard for everyone, omit reply_markup field.
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
    rows.append([{"text": "⬅️ Назад", "callback_data": f"onzs:back:{card_id}"}])
    return {"inline_keyboard": rows}


def answer_callback(cb_id: str, text: str = "", show_alert: bool = False):
    return tg_post("answerCallbackQuery", {"callback_query_id": cb_id, "text": text, "show_alert": show_alert})

def send_document(chat_id: int, file_path: str, filename: Optional[str] = None, caption: str = ""):
    if not BOT_TOKEN:
        return
    filename = filename or os.path.basename(file_path)
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument"
    with open(file_path, "rb") as f:
        files = {"document": (filename, f)}
        data = {"chat_id": chat_id, "caption": caption}
        r = requests.post(url, data=data, files=files, timeout=HTTP_TIMEOUT)
        if not r.ok:
            log.error(f"sendDocument failed: {r.text}")

def send_photo(chat_id: int, file_path: str, caption: str = ""):
    if not BOT_TOKEN:
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    with open(file_path, "rb") as f:
        files = {"photo": (os.path.basename(file_path), f)}
        data = {"chat_id": chat_id, "caption": caption}
        r = requests.post(url, data=data, files=files, timeout=HTTP_TIMEOUT)
        if not r.ok:
            log.error(f"sendPhoto failed: {r.text}")

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

# ----------------------------- ADMIN UI -----------------------------
ADMIN_STATE: Dict[int, str] = {}  # user_id -> pending_action

def build_admin_keyboard() -> Dict:
    thr = get_prob_threshold()
    return {
        "inline_keyboard": [
            [{"text": f"🎯 Порог вероятности: {thr}%", "callback_data": "admin:threshold:menu"}],
            [{"text": "📊 Статистика обучения", "callback_data": "admin:trainstats"}],
            [{"text": "📈 График роста обучения (текст)", "callback_data": "admin:trainplot:text"}],
            [{"text": "🖼 PNG график обучения", "callback_data": "admin:trainplot:png"}],
            [{"text": "🗂 Журнал обучения", "callback_data": "admin:trainlog"}],
            [{"text": "👥 Управление администраторами", "callback_data": "admin:admins:menu"}],
            [{"text": "🧑‍⚖️ Управление модераторами", "callback_data": "admin:mods:menu"}],
            [{"text": "🏛 Управление руководством", "callback_data": "admin:leaders:menu"}],
            [{"text": "📄 Отчёт XLSX", "callback_data": "admin:report:xlsx"}],
            [{"text": "🧾 Отчёт PDF", "callback_data": "admin:report:pdf"}],
            [{"text": "📊 Дашборд KPI", "callback_data": "admin:kpi"}],
        ]
    }

def build_threshold_keyboard() -> Dict:
    presets = [0, 20, 40, 60, 70, 80, 90]
    rows, row = [], []
    for p in presets:
        row.append({"text": f"{p}%", "callback_data": f"admin:threshold:set:{p}"})
        if len(row) == 4:
            rows.append(row); row = []
    if row:
        rows.append(row)
    rows.append([{"text": "✍️ Ввести вручную (0-100)", "callback_data": "admin:threshold:manual"}])
    rows.append([{"text": "⬅️ Назад", "callback_data": "admin:menu"}])
    return {"inline_keyboard": rows}

def build_users_keyboard(kind: str) -> Dict:
    # kind in admins/mods/leaders
    mapping = {
        "admins": ("администратора", "admin"),
        "mods": ("модератора", "moderator"),
        "leaders": ("руководство", "leadership"),
    }
    title, role = mapping[kind]
    return {
        "inline_keyboard": [
            [{"text": f"➕ Добавить {title}", "callback_data": f"admin:{kind}:add"}],
            [{"text": f"➖ Удалить {title}", "callback_data": f"admin:{kind}:del"}],
            [{"text": "📋 Показать список", "callback_data": f"admin:{kind}:list"}],
            [{"text": "⬅️ Назад", "callback_data": "admin:menu"}],
        ]
    }

# ----------------------------- SCRAPER -----------------------------
def fetch_channel_page(url: str) -> Optional[str]:
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Safari/537.36"}
    try:
        r = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT, allow_redirects=True)
        if r.status_code != 200:
            log.error(f"HTTP {r.status_code} for {url}")
            return None
        return r.text
    except Exception as e:
        log.error(f"fetch_channel_page error {url}: {e}")
        return None

def extract_posts(html: str) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    messages = soup.find_all("div", class_="tgme_widget_message")
    posts = []
    for msg in messages:
        try:
            msg_id = msg.get("data-post", "")  # "channel/123"
            text_block = msg.find("div", class_="tgme_widget_message_text")
            text = text_block.get_text(" ", strip=True) if text_block else ""
            time_tag = msg.find("time")
            ts = parse_tg_datetime(time_tag.get("datetime") if time_tag else "")
            links = []
            for a in msg.find_all("a", href=True):
                href = a["href"]
                if href.startswith("http"):
                    links.append(href)
            posts.append({"id": msg_id, "text": text, "timestamp": ts, "links": links})
        except Exception as e:
            log.error(f"extract_posts error: {e}")
    return posts

def process_channel(channel_username: str) -> List[Dict]:
    url = f"https://t.me/s/{channel_username}"
    html = fetch_channel_page(url)
    if not html:
        return []
    posts = extract_posts(html)
    hits = []
    for p in posts:
        text = normalize_text(p["text"])
        found = detect_keywords(text)
        if not found:
            continue
        if not is_relevant_hit(text, found):
            continue
        if not mark_seen(channel_username, p["id"], p["timestamp"]):
            continue
        hits.append({
            "channel": channel_username,
            "post_id": p["id"],
            "text": p["text"],
            "timestamp": p["timestamp"],
            "links": p.get("links", []),
            "keywords": found,
        })
    return hits

def scan_once() -> List[Dict]:
    all_hits: List[Dict] = []
    for ch in CHANNEL_LIST:
        try:
            hits = process_channel(ch)
            if hits:
                log.info(f"@{ch}: hits={len(hits)}")
            all_hits.extend(hits)
        except Exception as e:
            log.error(f"scan channel @{ch} error: {e}")
    return all_hits

def generate_card(hit: Dict) -> Dict:
    cid = generate_card_id()
    card = {
        "card_id": cid,
        "channel": hit["channel"],
        "post_id": hit["post_id"],
        "timestamp": hit["timestamp"],
        "text": hit["text"],
        "keywords": hit["keywords"],
        "links": hit.get("links", []),
        "status": "new",
        "history": [],
    }
    try:
        enrich_card_with_yagpt(card)
        # ONZS classification (only for relevant cases)
        enrich_card_with_onzs(card)
    except Exception as e:
        log.error(f"enrich_card_with_yagpt/onqs error: {e}")
    save_card(card)
    return card

def send_card_to_group(card: Dict) -> Optional[int]:
    thr = get_prob_threshold()
    ai = card.get("ai") or {}
    prob = None
    try:
        prob = float(ai.get("probability"))
    except Exception:
        prob = None

    # If YandexGPT is not configured / not available — do not spam the group.
    if not YAGPT_API_KEY or not YAGPT_FOLDER_ID or (ai.get("comment") == "YandexGPT not configured"):
        card["status"] = "skipped_no_ai"
        card.setdefault("history", []).append({"event": "skipped_no_ai", "ts": now_ts()})
        save_card(card)
        return None
    if prob is None:
        card["status"] = "skipped_no_ai"
        card.setdefault("history", []).append({"event": "skipped_no_ai", "ts": now_ts()})
        save_card(card)
        return None

    eff_thr = max(float(thr), float(MIN_AI_GATE))
    if prob < eff_thr:
        card["status"] = "filtered"
        card.setdefault("history", []).append({"event": "filtered", "threshold": eff_thr, "ts": now_ts()})
        save_card(card)
        append_history({"event": "filtered", "card_id": card["card_id"], "threshold": eff_thr, "prob": prob})
        return None



    res = send_message(TARGET_CHAT_ID, build_card_text(card), reply_markup=build_card_keyboard(card["card_id"]))
    if not res or not res.get("ok"):
        log.error(f"sendMessage failed: {res}")
        return None

    msg = res["result"]
    card.setdefault("tg", {})
    card["tg"]["chat_id"] = msg["chat"]["id"]
    card["tg"]["message_id"] = msg["message_id"]
    card["status"] = "sent"
    card.setdefault("history", []).append({"event": "sent", "ts": now_ts(), "chat_id": card["tg"]["chat_id"], "message_id": card["tg"]["message_id"]})
    save_card(card)
    append_history({"event": "sent", "card_id": card["card_id"], "chat_id": card["tg"]["chat_id"], "message_id": card["tg"]["message_id"]})
    return msg["message_id"]

# ----------------------------- CARD ACTIONS -----------------------------
def apply_card_action(card_id: str, action: str, from_user: int) -> Tuple[str, bool]:
    """
    Returns (message, decided_now).
    decided_now=True only for the first admin that made the decision.
    """
    existing = decision_exists(card_id)
    if existing:
        dec, by, ts = existing
        dt = datetime.fromtimestamp(ts).strftime("%d.%m.%Y %H:%M")
        return (f"Уже обработано: {dec} (админ {by}, {dt})", False)

    if action not in ("work", "wrong", "attach"):
        return ("Неизвестное действие.", False)

    card = load_card(card_id)
    if not card:
        return ("Карточка не найдена.", False)

    wrote = set_decision(card_id, action, from_user)
    if not wrote:
        return ("Уже обработано другим администратором.", False)

    old_status = card.get("status", "new")
    if action == "work":
        new_status, label, msg = "in_work", "work", "Статус: В РАБОТУ ✅"
    elif action == "wrong":
        new_status, label, msg = "wrong", "wrong", "Статус: НЕВЕРНО ❌"
    else:
        new_status, label, msg = "bind", "attach", "Статус: ПРИВЯЗАТЬ 📎"

    card["status"] = new_status
    card.setdefault("history", []).append({"event": f"set_{new_status}", "from_user": int(from_user), "ts": now_ts()})
    save_card(card)

    append_history({"event": "status_change", "card_id": card_id, "from_user": int(from_user), "old_status": old_status, "new_status": new_status})
    log_training_event(card_id, label, card.get("text", ""), card.get("channel", ""), admin_id=int(from_user))
    return (msg, True)

# ----------------------------- REPORTS / KPI -----------------------------
def _fetch_train_daily_last(days: int = 30):
    conn = db()
    rows = conn.execute("SELECT day, total, work, wrong, attach FROM train_daily ORDER BY day DESC LIMIT ?;", (int(days),)).fetchall()
    conn.close()
    return list(reversed(rows))

def build_kpi_text() -> str:
    rows = _fetch_train_daily_last(30)
    if not rows:
        return "📊 KPI: данных обучения пока нет."
    total = sum(int(r[1]) for r in rows)
    work = sum(int(r[2]) for r in rows)
    wrong = sum(int(r[3]) for r in rows)
    attach = sum(int(r[4]) for r in rows)
    acc = ((work + attach) / total * 100.0) if total > 0 else 0.0
    last_day = rows[-1][0]
    return (
        "📊 KPI (самострой-контроль)\n"
        f"Период: последние {len(rows)} дн. (до {last_day})\n\n"
        f"Всего решений: {total}\n"
        f"В работу: {work}\n"
        f"Неверно: {wrong}\n"
        f"Привязать: {attach}\n"
        f"Доля полезных (в работу+привязать): {acc:.1f}%\n"
    )

def build_report_xlsx() -> str:
    out_path = os.path.join(REPORTS_DIR, f"report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.xlsx")
    wb = Workbook()
    ws = wb.active
    ws.title = "KPI"
    ws.append(["Показатель", "Значение"])
    for line in build_kpi_text().splitlines()[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            ws.append([k.strip(), v.strip()])

    ws2 = wb.create_sheet("TrainingDaily")
    ws2.append(["day", "total", "work", "wrong", "attach"])
    for r in _fetch_train_daily_last(90):
        ws2.append(list(r))

    ws3 = wb.create_sheet("ChannelBias")
    ws3.append(["channel", "bias_points"])
    w = _get_model_param("weights", {"channels": {}})
    for ch, b in sorted((w.get("channels") or {}).items(), key=lambda x: x[0]):
        ws3.append([ch, b])

    for wsx in [ws, ws2, ws3]:
        for col in range(1, wsx.max_column + 1):
            wsx.column_dimensions[get_column_letter(col)].width = 28

    wb.save(out_path)
    return out_path

def build_report_pdf() -> str:
    out_path = os.path.join(REPORTS_DIR, f"report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf")
    c = canvas.Canvas(out_path, pagesize=A4)
    width, height = A4
    text = c.beginText(40, height - 60)
    text.setFont("Helvetica", 12)
    for line in build_kpi_text().splitlines():
        text.textLine(line)
    c.drawText(text)
    c.showPage()
    c.save()
    return out_path

def build_trainplot_png(days: int = 60) -> str:
    out_path = os.path.join(REPORTS_DIR, f"trainplot_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png")
    rows = _fetch_train_daily_last(days)
    plt.figure(figsize=(10, 4))
    if not rows:
        plt.title("Training (no data)")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        return out_path

    days_list = [r[0] for r in rows]
    total = [int(r[1]) for r in rows]
    work = [int(r[2]) for r in rows]
    wrong = [int(r[3]) for r in rows]
    attach = [int(r[4]) for r in rows]

    plt.plot(days_list, total, label="total")
    plt.plot(days_list, work, label="work")
    plt.plot(days_list, wrong, label="wrong")
    plt.plot(days_list, attach, label="attach")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path

def get_all_report_recipients() -> List[int]:
    ids = set()
    for role in ("leadership", "admin", "moderator"):
        for uid in list_users_by_role(role):
            ids.add(int(uid))
    return sorted(ids)

def daily_reports_worker():
    # Daily at 09:00 Moscow
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("Europe/Moscow")
    except Exception:
        tz = None

    while True:
        now = datetime.now(tz) if tz else datetime.now()
        target = now.replace(hour=9, minute=0, second=0, microsecond=0)
        if target <= now:
            target = target + timedelta(days=1)
        time.sleep(max(5, int((target - now).total_seconds())))

        try:
            kpi = build_kpi_text()
            xlsx = build_report_xlsx()
            pdf = build_report_pdf()
            png = build_trainplot_png()

            for uid in get_all_report_recipients():
                send_message(uid, kpi)
                send_document(uid, xlsx, caption="📄 Ежедневный отчёт (XLSX)")
                send_document(uid, pdf, caption="🧾 Ежедневный отчёт (PDF)")
                send_photo(uid, png, caption="📈 График обучения")
        except Exception as e:
            log.exception(f"daily_reports_worker error: {e}")

# ----------------------------- TELEGRAM UPDATES -----------------------------
UPDATE_OFFSET = get_update_offset()

def handle_callback_query(upd: Dict):
    cb = upd.get("callback_query") or {}
    cb_id = cb.get("id")
    from_user = int((cb.get("from") or {}).get("id", 0))
    data = (cb.get("data") or "").strip()
    msg_obj = cb.get("message") or {}
    chat_id = (msg_obj.get("chat") or {}).get("id")
    message_id = msg_obj.get("message_id")

    role = get_role(from_user)  # may be None


# ONZS actions
if data.startswith("onzs:"):
    if not (is_admin(from_user) or is_moderator(from_user)):
        answer_callback(cb_id, "❌ Нет доступа.", show_alert=True)
        return

    parts = data.split(":")
    op = parts[1] if len(parts) > 1 else ""

    # onzs:edit:<card_id> -> показать выбор 1..12
    if op == "edit" and len(parts) == 3:
        card_id = parts[2]
        if chat_id is not None and message_id is not None:
            edit_reply_markup(chat_id, message_id, reply_markup=build_onzs_pick_keyboard(card_id))
        answer_callback(cb_id, "Выбери ОНзС (1–12)")
        return

    # onzs:set:<card_id>:<n> -> установить ручной ОНзС
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

        # обучение: фиксируем, что ИИ (если был) ошибся, и правильный номер n
        ai_onzs = None
        if (card.get("onzs") or {}).get("ai"):
            ai_onzs = int((card.get("onzs") or {}).get("ai"))
        save_onzs_training(card.get("text") or "", n, confirmed=False, ai_onzs=ai_onzs)

        card.setdefault("onzs", {})
        card["onzs"]["value"] = n
        card["onzs"]["source"] = "manual"
        card["onzs"]["confirmed"] = True
        save_card(card)

        if chat_id is not None and message_id is not None:
            edit_message_text(chat_id, message_id, build_card_text(card), reply_markup=build_card_keyboard(card_id))
        answer_callback(cb_id, f"ОНзС установлен: {n}")
        return

    # onzs:confirm:<card_id> -> подтвердить ОНзС от ИИ
    if op == "confirm" and len(parts) == 3:
        card_id = parts[2]
        card = load_card(card_id)
        if not card:
            answer_callback(cb_id, "Карточка не найдена", show_alert=True)
            return

        oz = card.get("onzs") or {}
        val = oz.get("value") or oz.get("ai")
        if not val:
            answer_callback(cb_id, "ОНзС не определён", show_alert=True)
            return

        # обучение: если подтверждаем ai — значит ai был прав
        ai_onzs = oz.get("ai")
        if ai_onzs:
            save_onzs_training(card.get("text") or "", int(ai_onzs), confirmed=True, ai_onzs=int(ai_onzs))

        card.setdefault("onzs", {})
        card["onzs"]["confirmed"] = True
        save_card(card)

        if chat_id is not None and message_id is not None:
            edit_message_text(chat_id, message_id, build_card_text(card), reply_markup=build_card_keyboard(card_id))
        answer_callback(cb_id, "ОНзС подтверждён")
        return

    # onzs:back:<card_id>
    if op == "back" and len(parts) == 3:
        card_id = parts[2]
        if chat_id is not None and message_id is not None:
            edit_reply_markup(chat_id, message_id, reply_markup=build_card_keyboard(card_id))
        answer_callback(cb_id, "Ок")
        return

    answer_callback(cb_id, "Неизвестная команда ОНзС", show_alert=True)
    return

    # Card actions
    if data.startswith("card:"):
        if not is_admin(from_user):
            answer_callback(cb_id, "Только администраторы могут менять статус.", show_alert=True)
            return

        try:
            _, card_id, action = data.split(":", 2)
        except ValueError:
            answer_callback(cb_id, "Ошибка формата данных.", show_alert=True)
            return

        result, decided_now = apply_card_action(card_id, action, from_user)

        # Always try to remove keyboard; even if already decided (for cleanliness)
        try:
            if chat_id is not None and message_id is not None:
                edit_reply_markup(chat_id, message_id, reply_markup=None)
        except Exception:
            pass

        answer_callback(cb_id, result, show_alert=False)
        return

    # Admin panel
    if data.startswith("admin:"):
        if not is_admin(from_user):
            answer_callback(cb_id, "❌ Нет доступа.", show_alert=True)
            return

        parts = data.split(":")

        if data == "admin:menu":
            send_message(chat_id, "🛠 Админ-панель:", reply_markup=build_admin_keyboard())
            answer_callback(cb_id, "Ок"); return

        # Threshold
        if data == "admin:threshold:menu":
            send_message(chat_id, "🎯 Настройка порога вероятности (0–100):", reply_markup=build_threshold_keyboard())
            answer_callback(cb_id, "Ок"); return

        if len(parts) == 4 and parts[1] == "threshold" and parts[2] == "set":
            try: v = int(parts[3])
            except Exception: v = DEFAULT_THRESHOLD
            set_prob_threshold(v)
            send_message(chat_id, f"✅ Порог установлен: {get_prob_threshold()}%", reply_markup=build_admin_keyboard())
            answer_callback(cb_id, "Сохранено"); return

        if data == "admin:threshold:manual":
            ADMIN_STATE[from_user] = "await_threshold"
            send_message(chat_id, "Введите порог числом 0–100 (сообщением).")
            answer_callback(cb_id, "Ожидаю ввод"); return

        # Users management
        if data == "admin:admins:menu":
            send_message(chat_id, "👥 Управление администраторами:", reply_markup=build_users_keyboard("admins"))
            answer_callback(cb_id, "Ок"); return
        if data == "admin:mods:menu":
            send_message(chat_id, "🧑‍⚖️ Управление модераторами:", reply_markup=build_users_keyboard("mods"))
            answer_callback(cb_id, "Ок"); return
        if data == "admin:leaders:menu":
            send_message(chat_id, "🏛 Управление руководством:", reply_markup=build_users_keyboard("leaders"))
            answer_callback(cb_id, "Ок"); return

        # list/add/del handlers
        if len(parts) == 3 and parts[2] == "list" and parts[1] in ("admins","mods","leaders"):
            role_map = {"admins":"admin","mods":"moderator","leaders":"leadership"}
            role_key = role_map[parts[1]]
            ids = list_users_by_role(role_key)
            txt = "\n".join(str(i) for i in ids) if ids else "Список пуст."
            send_message(chat_id, f"Список ({role_key}):\n{txt}")
            answer_callback(cb_id, "Ок"); return

        if len(parts) == 3 and parts[2] in ("add","del") and parts[1] in ("admins","mods","leaders"):
            op = parts[2]
            ADMIN_STATE[from_user] = f"await_{op}_{parts[1]}"
            send_message(chat_id, "Отправьте Telegram ID (числом) следующим сообщением.")
            answer_callback(cb_id, "Ожидаю ID"); return

        # Reports & KPI
        if data == "admin:report:xlsx":
            p = build_report_xlsx()
            send_document(chat_id, p, caption="📄 Отчёт (XLSX)")
            answer_callback(cb_id, "Готово"); return

        if data == "admin:report:pdf":
            p = build_report_pdf()
            send_document(chat_id, p, caption="🧾 Отчёт (PDF)")
            answer_callback(cb_id, "Готово"); return

        if data == "admin:kpi":
            send_message(chat_id, build_kpi_text())
            answer_callback(cb_id, "Ок"); return

        # Training info
        if data == "admin:trainstats":
            st = compute_training_stats()
            last = st["last_ts"]
            last_s = datetime.fromtimestamp(last).strftime("%d.%m.%Y %H:%M") if last else "—"
            send_message(
                chat_id,
                "📊 Статистика обучения (агрегация по всем админам):\n\n"
                f"• Всего событий: {st['total']}\n"
                f"   ├─ В работу: {st['work']}\n"
                f"   ├─ Неверно: {st['wrong']}\n"
                f"   └─ Привязать: {st['attach']}\n\n"
                f"• Прогресс к цели ({st['target']}): {st['progress']}%\n"
                f"• Условная уверенность: {st['confidence']}%\n"
                f"• Последнее событие: {last_s}\n"
            )
            answer_callback(cb_id, "Ок"); return

        if data == "admin:trainplot:text":
            send_message(chat_id, training_plot_text(days=14))
            answer_callback(cb_id, "Ок"); return

        if data == "admin:trainplot:png":
            p = build_trainplot_png()
            send_photo(chat_id, p, caption="📈 График обучения (PNG)")
            answer_callback(cb_id, "Ок"); return

        if data == "admin:trainlog":
            events = tail_training_log(limit=MAX_TRAIN_LOG)
            if not events:
                send_message(chat_id, "🗂 Журнал обучения пуст.")
                answer_callback(cb_id, "Ок"); return
            lines = ["🗂 Последние события обучения:"]
            for e in events[-MAX_TRAIN_LOG:]:
                ts = e.get("timestamp")
                dt = datetime.fromtimestamp(int(ts)).strftime("%d.%m %H:%M") if isinstance(ts, int) else "—"
                lbl = e.get("label", "—")
                adm = e.get("admin_id", "—")
                cid = e.get("card_id", "—")
                ch = e.get("channel", "—")
                lines.append(f"• {dt} | {lbl} | @{ch} | admin={adm} | card={cid}")
            send_message(chat_id, "\n".join(lines))
            answer_callback(cb_id, "Ок"); return

        answer_callback(cb_id, "Неизвестное действие.", show_alert=False)
        return

    answer_callback(cb_id, "")

def handle_message(upd: Dict):
    msg = upd.get("message") or {}
    chat_id = (msg.get("chat") or {}).get("id")
    from_user = int((msg.get("from") or {}).get("id", 0))
    text = (msg.get("text") or "").strip()

    # ONZS AI stats
    if text.startswith("/onzs_ai_stats"):
        if not (is_admin(from_user) or is_moderator(from_user)):
            send_message(chat_id, "❌ Нет доступа.")
            return
        send_message(chat_id, build_onzs_stats_text())
        return

    # stateful admin inputs
    if is_admin(from_user) and from_user in ADMIN_STATE and not text.startswith("/"):
        st = ADMIN_STATE.pop(from_user, "")

        if st == "await_threshold":
            m = re.findall(r"-?\d+", text)
            if not m:
                send_message(chat_id, "❌ Не распознал число. Введите 0–100.")
                ADMIN_STATE[from_user] = "await_threshold"
                return
            set_prob_threshold(int(m[0]))
            send_message(chat_id, f"✅ Порог установлен: {get_prob_threshold()}%", reply_markup=build_admin_keyboard())
            return

        # user role operations
        m = re.findall(r"\d+", text)
        if not m:
            send_message(chat_id, "❌ Не распознал ID. Отправьте число.")
            ADMIN_STATE[from_user] = st
            return
        uid = int(m[0])

        if st == "await_add_admins":
            add_admin(uid); send_message(chat_id, f"✅ Администратор добавлен: {uid}", reply_markup=build_admin_keyboard()); return
        if st == "await_del_admins":
            if uid == from_user:
                send_message(chat_id, "❌ Нельзя удалить самого себя через меню. Используйте другого админа."); return
            remove_admin(uid); send_message(chat_id, f"🗑 Администратор удалён: {uid}", reply_markup=build_admin_keyboard()); return

        if st == "await_add_mods":
            add_moderator(uid); send_message(chat_id, f"✅ Модератор добавлен: {uid}", reply_markup=build_admin_keyboard()); return
        if st == "await_del_mods":
            remove_moderator(uid); send_message(chat_id, f"🗑 Модератор удалён: {uid}", reply_markup=build_admin_keyboard()); return

        if st == "await_add_leaders":
            add_leadership(uid); send_message(chat_id, f"✅ Добавлено в руководство: {uid}", reply_markup=build_admin_keyboard()); return
        if st == "await_del_leaders":
            remove_leadership(uid); send_message(chat_id, f"🗑 Удалено из руководства: {uid}", reply_markup=build_admin_keyboard()); return

        # unknown state
        send_message(chat_id, "⚠️ Неизвестная операция. /admin")
        return

    if not text.startswith("/"):
        return

    cmd = text.split()[0].split("@")[0]

    if cmd == "/admin":
        if not is_admin(from_user):
            send_message(chat_id, "❌ Команда /admin доступна только администраторам.")
            return
        send_message(chat_id, "🛠 Админ-панель:", reply_markup=build_admin_keyboard())
        return

    if cmd == "/dashboard":
        if not (is_admin(from_user) or is_leadership(from_user)):
            send_message(chat_id, "❌ Нет доступа.")
            return
        send_message(chat_id, build_kpi_text())
        p = build_trainplot_png()
        send_photo(chat_id, p, caption="📈 График обучения (PNG)")
        return

    if cmd == "/trainstats":
        if not is_admin(from_user):
            send_message(chat_id, "❌ Команда доступна только администраторам.")
            return
        st = compute_training_stats()
        last = st["last_ts"]
        last_s = datetime.fromtimestamp(last).strftime("%d.%m.%Y %H:%M") if last else "—"
        send_message(chat_id, f"Всего={st['total']} (work={st['work']}, wrong={st['wrong']}, attach={st['attach']}), последн={last_s}")
        return

def poll_updates_loop():
    global UPDATE_OFFSET
    if not TELEGRAM_API_URL:
        log.warning("Telegram API not configured; poller not started.")
        return

    try:
        tg_post("deleteWebhook", {"drop_pending_updates": False})
    except Exception:
        pass

    log.info("Starting getUpdates poller...")
    while True:
        try:
            params = {"timeout": 25, "offset": UPDATE_OFFSET, "allowed_updates": ["message", "callback_query"]}
            data = tg_get("getUpdates", params=params)
            if not data:
                time.sleep(2); continue

            if not data.get("ok"):
                if data.get("error_code") == 409:
                    log.error("getUpdates conflict (409). Another instance is polling this BOT_TOKEN. Backing off 60s.")
                    time.sleep(60)
                    continue
                log.error(f"getUpdates error: {data}")
                time.sleep(3); continue

            updates = data.get("result", []) or []
            if not updates:
                continue

            for upd in updates:
                UPDATE_OFFSET = max(UPDATE_OFFSET, int(upd["update_id"]) + 1)
                if "callback_query" in upd:
                    handle_callback_query(upd)
                elif "message" in upd:
                    handle_message(upd)

            # persist offset (so restart doesn't replay)
            set_update_offset(UPDATE_OFFSET)

        except SystemExit:
            raise
        except Exception as e:
            log.error(f"poll_updates exception: {e}")
            time.sleep(3)

# ----------------------------- MAIN LOOP -----------------------------
def run_scan_cycle() -> int:
    hits = scan_once()
    if not hits:
        return 0
    sent_count = 0
    for h in hits:
        card = generate_card(h)
        mid = send_card_to_group(card)
        if mid:
            sent_count += 1
            time.sleep(0.4)
    return sent_count

def main():
    log.info("SAMASTROI SCRAPER starting...")
    log.info(f"DATA_DIR={DATA_DIR}")
    log.info(f"TARGET_CHAT_ID={TARGET_CHAT_ID}")
    log.info(f"SCAN_INTERVAL={SCAN_INTERVAL}")
    log.info(f"Admins: {list_users_by_role('admin')}")
    log.info(f"Moderators: {list_users_by_role('moderator')}")
    log.info(f"Leadership: {list_users_by_role('leadership')}")
    log.info(f"Prob threshold: {get_prob_threshold()}%")

    acquire_lock_or_exit()

    try:
        # poller + daily reports in daemon threads
        threading.Thread(target=poll_updates_loop, daemon=True).start()
        threading.Thread(target=daily_reports_worker, daemon=True).start()

        while True:
            try:
                sent = run_scan_cycle()
                if sent:
                    log.info(f"Cycle done: sent={sent}")
            except Exception as e:
                log.error(f"scan cycle error: {e}")
            time.sleep(SCAN_INTERVAL)
    finally:
        release_lock()

if __name__ == "__main__":
    main()

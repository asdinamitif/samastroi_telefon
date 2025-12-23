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

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests
import pandas as pd

# ----------------------------- LOGGING -----------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("samastroi")



# ----------------- SINGLE INSTANCE LOCK -----------------
def acquire_lock() -> bool:
    """Create a lock file in DATA_DIR to prevent running multiple pollers."""
    try:
        data_dir = globals().get("DATA_DIR") or os.getenv("DATA_DIR", "/data")
        os.makedirs(data_dir, exist_ok=True)
        lock_path = os.path.join(data_dir, ".poller.lock")

        # stale lock: 10 minutes
        if os.path.exists(lock_path):
            try:
                if (time.time() - os.path.getmtime(lock_path)) > 600:
                    os.remove(lock_path)
                else:
                    log.error("Lock exists: another poller is running. Exiting.")
                    return False
            except Exception:
                log.error("Lock exists: another poller is running. Exiting.")
                return False

        # atomic create
        try:
            with open(lock_path, "x", encoding="utf-8") as f:
                f.write(str(os.getpid()))
        except FileExistsError:
            log.error("Lock exists: another poller is running. Exiting.")
            return False

        log.info(f"Lock acquired: {lock_path}")
        return True
    except Exception as e:
        log.error(f"Lock acquire error: {e}")
        return True  # best-effort: do not hard fail


def release_lock():
    try:
        data_dir = globals().get("DATA_DIR") or os.getenv("DATA_DIR", "/data")
        lock_path = os.path.join(data_dir, ".poller.lock")
        if os.path.exists(lock_path):
            os.remove(lock_path)
    except Exception:
        pass
# --------------------------------------------------------

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

# --------------------------------------------
#          ROLES PERSISTENCE (/data)
# --------------------------------------------
ROLES_PATH = os.path.join(DATA_DIR, "roles.json")

def _parse_ids_csv(v: str):
    if not v:
        return []
    out = []
    for x in str(v).split(","):
        x = x.strip()
        if not x:
            continue
        if re.fullmatch(r"-?\d+", x):
            try:
                out.append(int(x))
            except Exception:
                pass
    return out


def load_roles() -> dict:
    """Load roles (admins/moderators/leadership/report_targets) with persistence.

    Priority logic (safe & predictable):
    1) Start from ENV lists (ADMIN_IDS/MODERATOR_IDS/LEADERSHIP etc).
    2) If /data/roles.json exists, MERGE it with ENV (additive). It must not delete ENV access.
    This prevents the classic situation where an old roles.json blocks access after redeploy.
    """
    base = {
        "admins": _parse_ids_csv(os.getenv("ADMIN_IDS", "")) or _parse_ids_csv(os.getenv("ADMINS", "")),
        "moderators": _parse_ids_csv(os.getenv("MODERATOR_IDS", "")) or _parse_ids_csv(os.getenv("MODERATORS", "")),
        "leadership": _parse_ids_csv(os.getenv("LEAD_IDS", "")) or _parse_ids_csv(os.getenv("LEADERSHIP", "")),
        "report_targets": _parse_ids_csv(os.getenv("REPORT_TARGETS", "")),
    }

    try:
        if os.path.exists(ROLES_PATH):
            with open(ROLES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
            for k in list(base.keys()):
                file_list = data.get(k, [])
                if isinstance(file_list, list):
                    merged = []
                    seen = set()
                    for src in (base.get(k, []), file_list):
                        for i in src:
                            try:
                                ii = int(i)
                            except Exception:
                                continue
                            if ii not in seen:
                                seen.add(ii)
                                merged.append(ii)
                    base[k] = merged
    except Exception as e:
        log.error(f"[ROLES] load error: {e}")

    # Persist merged result (so that UI changes are remembered)
    save_roles(base)
    return base




def save_roles(data: dict) -> None:
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(ROLES_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        log.info(f"[ROLES] saved: {ROLES_PATH}")
    except Exception as e:
        log.error(f"[ROLES] save error: {e}")

ROLES = load_roles()

def roles_refresh_globals():
    global ADMINS, MODERATORS, LEADERSHIP
    ADMINS = ROLES.get("admins", [])
    MODERATORS = ROLES.get("moderators", [])
    LEADERSHIP = ROLES.get("leadership", [])

roles_refresh_globals()

def is_privileged(uid: int) -> bool:
    return uid in ROLES.get("admins", []) or uid in ROLES.get("moderators", []) or uid in ROLES.get("leadership", [])

def is_admin(uid: int) -> bool:
    return uid in ROLES.get("admins", [])
os.makedirs(DATA_DIR, exist_ok=True)

SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "300") or "300")
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "15") or "15")

# IMPORTANT: AI gate; if YandexGPT is unavailable or refuses, card is not sent
MIN_AI_GATE = float(os.getenv("MIN_AI_GATE", "5"))
# Allow runtime override from /data/config.json (0..100 percent)
try:
    MIN_AI_GATE = float(max(0.0, min(100.0, get_cfg_float("min_ai_gate", MIN_AI_GATE))))
except Exception:
    pass


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
ONZS_XLSX = os.getenv("ONZS_XLSX", "ÐÐ¾Ð¼ÐµÑÐ° ÐÐÐ·Ð¡.xlsx").strip()

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
    "Ð¿ÑÑÐ¸Ð½", "ÑÐºÑÐ°Ð¸Ð½", "Ð²Ð¾Ð¹Ð½", "Ð¿Ð¾Ð»Ð¸ÑÐ¸Ðº", "ÑÐ°Ð½ÐºÑ", "Ð²ÑÐ±Ð¾Ñ", "Ð¼Ð¸ÑÐ¸Ð½Ð³", "Ð±Ð°Ð¹Ð´ÐµÐ½",
    "ÑÑÐ°Ð¼Ð¿", "ÑÐ°Ð¼Ð°Ñ", "Ð¸Ð·ÑÐ°Ð¸Ð»", "ÑÐµÑÑÐ¾Ñ", "Ð´ÑÐ¾Ð½", "ÑÐ°ÐºÐµÑ", "ÑÑÐ¾Ð½Ñ", "Ð¼Ð¾Ð±Ð¸Ð»Ð¸Ð·",
    "Ð½Ð°ÑÑÑÐ¿Ð»ÐµÐ½", "Ð¾Ð±ÑÑÑÐµÐ»", "Ð²ÑÑ", "Ð°ÑÐ¼Ð¸Ñ", "Ð½Ð°ÑÐ¾", "Ð¿ÑÐµÐ¼ÑÐµÑ", "Ð¿ÑÐµÐ·Ð¸Ð´ÐµÐ½Ñ",
]

# Construction signal words (soft allow)
CONSTR_HINTS = [
    "ÑÑÑÐ¾Ð¹", "ÑÑÑÐ¾Ð¸Ñ", "ÑÑÑÐ¾Ð¸ÑÐµÐ»", "ÑÐ°Ð¼Ð¾ÑÑÑÐ¾", "ÐºÐ¾ÑÐ»Ð¾Ð²Ð°Ð½", "ÑÑÐ½Ð´Ð°Ð¼ÐµÐ½Ñ", "Ð°ÑÐ¼Ð°ÑÑÑ",
    "Ð±ÐµÑÐ¾Ð½", "Ð¿Ð»Ð¸ÑÐ°", "Ð¼Ð¾Ð½Ð¾Ð»Ð¸Ñ", "Ð¿ÐµÑÐµÐºÑÑÑ", "ÑÑÐ°Ð¶", "ÐºÑÐ°Ð½", "Ð¾Ð¿Ð°Ð»ÑÐ±", "Ð·Ð°Ð±Ð¾Ñ",
    "Ð¿ÑÐ¸ÑÑÑÐ¾Ð¹", "Ð½Ð°Ð´ÑÑÑÐ¾Ð¹", "ÑÐµÐºÐ¾Ð½ÑÑÑÑÐºÑ", "ÐºÐ°Ð¿ÑÐµÐ¼Ð¾Ð½Ñ", "ÑÐ°Ð·ÑÐµÑÐµÐ½", "ÑÐ½Ñ", "Ð³Ð¿Ð·Ñ",
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
    """Extract ONZS number 1..12 from a cell value (robust to '1.0', '1 â ...', etc.)."""
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
    """POST wrapper with basic resiliency (timeouts/resets/429) and safe JSON parsing."""
    if not TELEGRAM_API_URL:
        return None

    max_attempts = int(os.getenv("TG_HTTP_RETRIES", "5"))
    base_sleep = float(os.getenv("TG_HTTP_RETRY_SLEEP", "1.0"))  # seconds
    last_err: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.post(
                f"{TELEGRAM_API_URL}/{method}",
                json=payload,
                timeout=HTTP_TIMEOUT,
            )

            # Telegram sometimes rate-limits (429) with retry_after in response JSON.
            if r.status_code == 429:
                try:
                    j = r.json()
                    retry_after = float(j.get("parameters", {}).get("retry_after", 1))
                except Exception:
                    retry_after = 1.0
                sleep_s = max(1.0, retry_after)
                log.warning(f"Telegram POST {method} rate-limited (429). Sleeping {sleep_s:.1f}s (attempt {attempt}/{max_attempts})")
                time.sleep(sleep_s)
                continue

            # Non-200: still try to parse JSON for diagnostics, but retry on 5xx.
            if r.status_code >= 500:
                log.warning(f"Telegram POST {method} server error {r.status_code}. Attempt {attempt}/{max_attempts}")
                time.sleep(base_sleep * attempt)
                continue

            try:
                return r.json()
            except Exception:
                # Not JSON (rare). Log a short snippet and do not crash.
                snippet = (r.text or "")[:300]
                log.error(f"Telegram POST {method} non-JSON response (status={r.status_code}): {snippet}")
                return None

        except Exception as e:
            last_err = e
            log.error(f"Telegram POST {method} error: {e!r} (attempt {attempt}/{max_attempts})")
            time.sleep(base_sleep * attempt)

    if last_err:
        log.error(f"Telegram POST {method} failed after {max_attempts} attempts: {last_err!r}")
    return None

def answer_callback(callback_query_id: str, text: str = "", show_alert: bool = False):
    tg_post("answerCallbackQuery", {"callback_query_id": callback_query_id, "text": text, "show_alert": show_alert})

# Backward-compatible alias used by some handlers
def answer_callback_query(callback_query_id: str, text: str = "", show_alert: bool = False):
    return answer_callback(callback_query_id, text=text, show_alert=show_alert)

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
        desc = str(resp.get("description", ""))
        # Telegram returns this when trying to set the same text/markup; not a real failure.
        if "message is not modified" in desc:
            log.debug(f"editMessageText not modified: {desc}")
        else:
            log.error(f"editMessageText failed: {resp}")
    return resp


# --------------------------------------------
#               ADMIN PANEL
# --------------------------------------------
def admin_menu_text() -> str:
    return (
        "ð  ÐÐ´Ð¼Ð¸Ð½-Ð¿Ð°Ð½ÐµÐ»Ñ\n"
        f"â¢ Admins: {len(ROLES.get('admins',[]))}\n"
        f"â¢ Moderators: {len(ROLES.get('moderators',[]))}\n"
        f"â¢ Leadership: {len(ROLES.get('leadership',[]))}\n"
        f"â¢ Reports targets: {len(ROLES.get('report_targets',[]))}\n"
        "\nÐÑÐ±ÐµÑÐ¸ÑÐµ ÑÐ°Ð·Ð´ÐµÐ»:"
    )

def admin_menu_kb():
    return {"inline_keyboard": [
        [{"text":"ð¥ Ð Ð¾Ð»Ð¸", "callback_data":"admin:roles"}],
        [{"text":"ð Ð¡ÑÐ°ÑÐ¸ÑÑÐ¸ÐºÐ°", "callback_data":"admin:stats"}],
        [{"text":"ð§¾ ÐÑÑÑÑÑ", "callback_data":"admin:reports"}],
        [{"text":"âï¸ ÐÐ°ÑÑÑÐ¾Ð¹ÐºÐ¸", "callback_data":"admin:settings"}],
    ]}

def admin_roles_kb():
    return {"inline_keyboard": [
        [{"text":"â ÐÐ¾Ð±Ð°Ð²Ð¸ÑÑ Ð°Ð´Ð¼Ð¸Ð½Ð°", "callback_data":"admin:add_admin"}],
        [{"text":"â Ð£Ð´Ð°Ð»Ð¸ÑÑ Ð°Ð´Ð¼Ð¸Ð½Ð°", "callback_data":"admin:del_admin"}],
        [{"text":"â ÐÐ¾Ð±Ð°Ð²Ð¸ÑÑ Ð¼Ð¾Ð´ÐµÑÐ°ÑÐ¾ÑÐ°", "callback_data":"admin:add_mod"}],
        [{"text":"â Ð£Ð´Ð°Ð»Ð¸ÑÑ Ð¼Ð¾Ð´ÐµÑÐ°ÑÐ¾ÑÐ°", "callback_data":"admin:del_mod"}],
        [{"text":"â ÐÐ¾Ð±Ð°Ð²Ð¸ÑÑ ÑÑÐºÐ¾Ð²Ð¾Ð´ÑÑÐ²Ð¾", "callback_data":"admin:add_lead"}],
        [{"text":"â Ð£Ð´Ð°Ð»Ð¸ÑÑ ÑÑÐºÐ¾Ð²Ð¾Ð´ÑÑÐ²Ð¾", "callback_data":"admin:del_lead"}],
        [{"text":"ð ÐÐ¾ÐºÐ°Ð·Ð°ÑÑ ÑÐ¾Ð»Ð¸", "callback_data":"admin:list_roles"}],
        [{"text":"â¬ï¸ ÐÐ°Ð·Ð°Ð´", "callback_data":"admin:back"}],
    ]}

def admin_reports_kb():
    return {"inline_keyboard": [
        [{"text":"ð¤ ÐÑÑÑÑ Ð·Ð° ÑÑÑÐºÐ¸", "callback_data":"admin:report_day"}],
        [{"text":"ð¬ ÐÐ¾Ð»ÑÑÐ°ÑÐµÐ»Ð¸ Ð¾ÑÑÑÑÐ¾Ð²", "callback_data":"admin:report_targets"}],
        [{"text":"â ÐÐ¾Ð±Ð°Ð²Ð¸ÑÑ Ð¿Ð¾Ð»ÑÑÐ°ÑÐµÐ»Ñ", "callback_data":"admin:add_report_target"}],
        [{"text":"â Ð£Ð´Ð°Ð»Ð¸ÑÑ Ð¿Ð¾Ð»ÑÑÐ°ÑÐµÐ»Ñ", "callback_data":"admin:del_report_target"}],
        [{"text":"â¬ï¸ ÐÐ°Ð·Ð°Ð´", "callback_data":"admin:back"}],
    ]}

def admin_settings_kb():
    return {"inline_keyboard": [
        [ {"text": f"ð ÐÐ¾ÑÐ¾Ð³ AI-gate: {MIN_AI_GATE:.1f}%", "callback_data": "admin:set_aigate"} ],
        [{"text":"ð ÐÐµÑÐµÐ·Ð°Ð³ÑÑÐ·Ð¸ÑÑ ÐÐÐ·Ð¡", "callback_data":"admin:reload_onzs"}],
        [{"text":"ð§ª Ð¢ÐµÑÑ YandexGPT", "callback_data":"admin:test_yagpt"}],
        [{"text":"â¬ï¸ ÐÐ°Ð·Ð°Ð´", "callback_data":"admin:back"}],
    ]}

ADMIN_STATE_PATH = os.path.join(DATA_DIR, "admin_state.json")
# --------------------------------------------
#               PERSISTENT CONFIG (DATA_DIR)
# --------------------------------------------
CONFIG_PATH = os.path.join(DATA_DIR, "config.json")

def load_config() -> dict:
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f) or {}
    except Exception:
        pass
    return {}

def save_config(cfg: dict) -> None:
    try:
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg or {}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def get_cfg_float(key: str, default: float) -> float:
    cfg = load_config()
    v = cfg.get(key, default)
    try:
        return float(str(v).replace(",", "."))
    except Exception:
        return float(default)

def set_cfg_value(key: str, value) -> None:
    cfg = load_config()
    cfg[key] = value
    save_config(cfg)


def load_admin_state():
    try:
        if os.path.exists(ADMIN_STATE_PATH):
            with open(ADMIN_STATE_PATH, "r", encoding="utf-8") as f:
                return json.load(f) or {}
    except Exception:
        pass
    return {}

def save_admin_state(st):
    try:
        with open(ADMIN_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(st, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

ADMIN_STATE = load_admin_state()

def set_admin_mode(*args):
    """Set per-user admin mode.
    Supports signatures:
      set_admin_mode(uid, mode)
      set_admin_mode(chat_id, uid, mode)  # chat_id is ignored (kept for backward compatibility)
    """
    if len(args) == 2:
        uid, mode = args
    elif len(args) == 3:
        _chat_id, uid, mode = args
    else:
        raise TypeError("set_admin_mode expected (uid, mode) or (chat_id, uid, mode)")
    try:
        uid = int(uid)
    except Exception:
        uid = int(str(uid).strip())
    ADMIN_STATE[str(uid)] = {"mode": str(mode), "ts": int(time.time())}

def clear_admin_mode(*args):
    """Clear per-user admin mode.
    Supports signatures:
      clear_admin_mode(uid)
      clear_admin_mode(uid)  # chat_id is ignored
    """
    if len(args) == 1:
        uid = args[0]
    elif len(args) == 2:
        _chat_id, uid = args
    else:
        raise TypeError("clear_admin_mode expected (uid) or (chat_id, uid)")
    try:
        uid = int(uid)
    except Exception:
        uid = int(str(uid).strip())
    ADMIN_STATE.pop(str(uid), None)
    save_admin_state(ADMIN_STATE)

def pop_admin_mode(uid:int):
    ADMIN_STATE.pop(str(uid), None)
    save_admin_state(ADMIN_STATE)

def get_admin_mode(uid:int):
    v = ADMIN_STATE.get(str(uid))
    return v.get("mode") if v else None

def _roles_add(key:str, uid:int):
    arr = ROLES.get(key, [])
    if uid not in arr:
        arr.append(uid)
    ROLES[key] = sorted(list(dict.fromkeys(arr)))
    save_roles(ROLES)
    roles_refresh_globals()

def _roles_del(key:str, uid:int):
    arr = [x for x in ROLES.get(key, []) if x != uid]
    ROLES[key] = arr
    save_roles(ROLES)
    roles_refresh_globals()

def build_roles_text():
    def fmt(lst):
        return ", ".join(str(x) for x in lst) if lst else "â"
    return (
        "ð¥ Ð¢ÐµÐºÑÑÐ¸Ðµ ÑÐ¾Ð»Ð¸\n"
        f"Admins: {fmt(ROLES.get('admins',[]))}\n"
        f"Moderators: {fmt(ROLES.get('moderators',[]))}\n"
        f"Leadership: {fmt(ROLES.get('leadership',[]))}\n"
        f"Report targets: {fmt(ROLES.get('report_targets',[]))}"
    )
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
            [{"text": "â Ð ÑÐ°Ð±Ð¾ÑÑ", "callback_data": f"card:{card_id}:work"},
             {"text": "â ÐÐµÐ²ÐµÑÐ½Ð¾", "callback_data": f"card:{card_id}:wrong"}],
            [{"text": "ð ÐÑÐ¸Ð²ÑÐ·Ð°ÑÑ", "callback_data": f"card:{card_id}:attach"}],
            [{"text": "âï¸ ÐÐ·Ð¼ÐµÐ½Ð¸ÑÑ ÐÐÐ·Ð¡", "callback_data": f"onzs:edit:{card_id}"},
             {"text": "â ÐÐ¾Ð´ÑÐ²ÐµÑÐ´Ð¸ÑÑ ÐÐÐ·Ð¡", "callback_data": f"onzs:confirm:{card_id}"}],
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
    rows.append([{"text": "â¬ï¸ ÐÐ°Ð·Ð°Ð´ Ðº ÐºÐ°ÑÑÐ¾ÑÐºÐµ", "callback_data": f"onzs:back:{card_id}"}])
    return {"inline_keyboard": rows}

# ----------------------------- TEXT NORMALIZATION -----------------------------
URL_RE = re.compile(r"(https?://\S+)")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")

def build_admin_keyboard() -> Dict:
    return {
        "inline_keyboard": [
            [
                {"text": "ð Ð¡ÑÐ°ÑÐ¸ÑÑÐ¸ÐºÐ° ÐÐÐ·Ð¡", "callback_data": "admin:onzs_stats"},
                {"text": "ð ÐÐµÑÐµÐ·Ð°Ð³ÑÑÐ·Ð¸ÑÑ ÐÐÐ·Ð¡", "callback_data": "admin:reload_onzs"},
            ],
            [
                {"text": "ð§ª Ð¢ÐµÑÑ YandexGPT", "callback_data": "admin:test_yagpt"},
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
        t = t[:2500] + "â¦"
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
        return {"probability": 0, "comment": "ÐÑÑÐµÐ²: Ð½Ð¾Ð²Ð¾ÑÑÑ/Ð¿Ð¾Ð»Ð¸ÑÐ¸ÐºÐ° (ÑÑÐ¾Ð¿-ÑÐµÐ¼Ð°)."}
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
            lines.append(f"- ÐÐµÑÐºÐ°={lbl} (Ð¾ÑÐ¸ÐµÐ½ÑÐ¸Ñ {hint}). Ð¢ÐµÐºÑÑ: {t}")
        few_block = "\n" + "\n".join(lines) + "\n"

    # our own bias to stabilize decisions (channel/keyword learning)
    bias_ch = get_channel_bias(channel) if channel else 0.0
    bias_kw = get_keyword_bias_points(cleaned)
    bias_total = bias_ch + bias_kw

    prompt_full = (
        "Ð¢Ñ Ð¿Ð¾Ð¼Ð¾ÑÐ½Ð¸Ðº Ð¸Ð½ÑÐ¿ÐµÐºÑÐ¾ÑÐ° ÑÑÑÐ¾Ð¸ÑÐµÐ»ÑÐ½Ð¾Ð³Ð¾ Ð½Ð°Ð´Ð·Ð¾ÑÐ°.\n"
        "Ð¢Ð²Ð¾Ñ Ð·Ð°Ð´Ð°ÑÐ°: Ð¾ÑÐµÐ½Ð¸ÑÑ Ð²ÐµÑÐ¾ÑÑÐ½Ð¾ÑÑÑ (0-100), ÑÑÐ¾ ÑÐµÐºÑÑ Ð¾ÑÐ½Ð¾ÑÐ¸ÑÑÑ Ðº ÑÐ°Ð¼Ð¾Ð²Ð¾Ð»ÑÐ½Ð¾Ð¼Ñ ÑÑÑÐ¾Ð¸ÑÐµÐ»ÑÑÑÐ²Ñ/Ð½Ð°ÑÑÑÐµÐ½Ð¸ÑÐ¼ Ð½Ð° ÑÑÑÐ¾Ð¹ÐºÐµ.\n"
        "ÐÐ°Ð¿ÑÐµÑÐµÐ½Ð¾: Ð½Ð¾Ð²Ð¾ÑÑÐ¸, Ð¿Ð¾Ð»Ð¸ÑÐ¸ÐºÐ°, Ð¾Ð±ÑÐ¸Ðµ ÑÐ°ÑÑÑÐ¶Ð´ÐµÐ½Ð¸Ñ.\n"
        "ÐÑÐ»Ð¸ ÑÐµÐºÑÑ ÑÐ¾Ð´ÐµÑÐ¶Ð¸Ñ Ð·Ð°Ð¿ÑÐµÑÑÐ½Ð½ÑÐµ/Ð½ÐµÐ¿Ð¾Ð´ÑÐ¾Ð´ÑÑÐ¸Ðµ ÑÐµÐ¼Ñ, Ð²ÑÑ ÑÐ°Ð²Ð½Ð¾ Ð²ÐµÑÐ½Ð¸ JSON Ñ probability=0 Ð¸ ÐºÑÐ°ÑÐºÐ¸Ð¼ comment.\n"
        "ÐÑÐ²ÐµÑÑ Ð¡Ð¢Ð ÐÐÐ Ð¾Ð´Ð½Ð¸Ð¼ JSON Ð±ÐµÐ· Ð¿Ð¾ÑÑÐ½ÐµÐ½Ð¸Ð¹ Ð¸ Ð±ÐµÐ· Markdown.\n"
        'Ð¤Ð¾ÑÐ¼Ð°Ñ: {"probability": <0-100>, "comment": "..."}\n'
        f"ÐÐ°Ð»Ð¸Ð±ÑÐ¾Ð²ÐºÐ° (bias, Ð¿ÑÐ¸Ð±Ð°Ð²Ñ Ðº Ð²ÐµÑÐ¾ÑÑÐ½Ð¾ÑÑÐ¸): {bias_total:+.1f}\n"
        + few_block +
        "\nÐ¢ÐµÐºÑÑ:\n" + cleaned
    )

    messages = [
        {"role": "system", "text": "ÐÑÐ²ÐµÑÐ°Ð¹ ÑÑÑÐ¾Ð³Ð¾ JSON."},
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
            if "Ð½Ðµ Ð¼Ð¾Ð³Ñ Ð¾Ð±ÑÑÐ¶Ð´Ð°ÑÑ" in out_text.lower():
                return {"probability": 0, "comment": "ÐÑÐºÐ°Ð· Ð¼Ð¾Ð´ÐµÐ»Ð¸: Ð½ÐµÐ¿Ð¾Ð´ÑÐ¾Ð´ÑÑÐ°Ñ ÑÐµÐ¼Ð°."}
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
        return {"onzs": None, "confidence": 0.0, "reason": "ÐÑÑÐµÐ²: ÑÑÐ¾Ð¿-ÑÐµÐ¼Ð°."}

    catalog = "\n".join([f"{k}: {v}" for k, v in sorted(ONZS_MAP.items())])

    prompt = (
        "Ð¢Ñ Ð¸Ð½ÑÐ¿ÐµÐºÑÐ¾Ñ ÑÑÑÐ¾Ð¸ÑÐµÐ»ÑÐ½Ð¾Ð³Ð¾ Ð½Ð°Ð´Ð·Ð¾ÑÐ°.\n"
        "ÐÐ¿ÑÐµÐ´ÐµÐ»Ð¸ Ð½Ð¾Ð¼ÐµÑ ÐÐÐ·Ð¡ (1â12) Ð¿Ð¾ Ð¾Ð¿Ð¸ÑÐ°Ð½Ð¸Ñ. ÐÑÐ»Ð¸ Ð½ÐµÐ»ÑÐ·Ñ Ð¾Ð¿ÑÐµÐ´ÐµÐ»Ð¸ÑÑ â onzs=null.\n"
        "ÐÑÐ²ÐµÑÑ Ð¡Ð¢Ð ÐÐÐ Ð¾Ð´Ð½Ð¸Ð¼ JSON Ð±ÐµÐ· Ð¿Ð¾ÑÑÐ½ÐµÐ½Ð¸Ð¹.\n"
        'Ð¤Ð¾ÑÐ¼Ð°Ñ: {"onzs": <1-12|null>, "confidence": <0-1>, "reason": "..."}\n\n'
        "ÐÐ»Ð°ÑÑÐ¸ÑÐ¸ÐºÐ°ÑÐ¾Ñ ÐÐÐ·Ð¡:\n"
        f"{catalog}\n\n"
        "Ð¢ÐµÐºÑÑ:\n"
        f"{cleaned}"
    )

    messages = [
        {"role": "system", "text": "ÐÑÐ²ÐµÑÐ°Ð¹ ÑÑÑÐ¾Ð³Ð¾ JSON."},
        {"role": "user", "text": prompt},
    ]

    for attempt in range(4):
        out_text, meta = call_yandex_gpt_raw(messages)
        if out_text is None:
            time.sleep(0.8 * (attempt + 1))
            continue
        js = extract_json_from_text(out_text)
        if js is None:
            if "Ð½Ðµ Ð¼Ð¾Ð³Ñ Ð¾Ð±ÑÑÐ¶Ð´Ð°ÑÑ" in out_text.lower():
                return {"onzs": None, "confidence": 0.0, "reason": "ÐÑÐºÐ°Ð· Ð¼Ð¾Ð´ÐµÐ»Ð¸."}
            time.sleep(0.6 * (attempt + 1))
            continue
        return js
    return None

def save_onzs_training(text: str, onzs: int, confirmed: bool):
    rec = {"text": clean_text_for_ai(text), "onzs": int(onzs), "confirmed": bool(confirmed), "ts": now_ts()}
    append_jsonl(ONZS_TRAIN_FILE, rec)


def build_daily_report_text() -> str:
    today = datetime.now().date().isoformat()
    lines = ["ð§¾ ÐÑÑÑÑ Ð·Ð° ÑÑÑÐºÐ¸", f"ÐÐ°ÑÐ°: {today}"]
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        if not tables:
            conn.close()
            lines.append("ÐÐµÑ ÑÐ°Ð±Ð»Ð¸Ñ Ð² ÐÐ.")
            return "\n".join(lines)
        table = "cards" if "cards" in tables else tables[0]
        try:
            cur.execute(f"SELECT status, COUNT(*) FROM {table} WHERE date(ts)=? GROUP BY status", (today,))
            rows = cur.fetchall()
            if rows:
                lines.append("Ð¡ÑÐ°ÑÑÑÑ:")
                for st,cnt in rows:
                    lines.append(f"â¢ {st}: {cnt}")
        except Exception:
            pass
        try:
            cur.execute(f"SELECT COUNT(*) FROM {table} WHERE date(ts)=? AND onzs_final IS NOT NULL", (today,))
            n = cur.fetchone()[0]
            lines.append(f"ÐÐ¾Ð´ÑÐ²ÐµÑÐ¶Ð´ÑÐ½Ð½ÑÐµ ÐÐÐ·Ð¡: {n}")
        except Exception:
            pass
        conn.close()
    except Exception as e:
        lines.append(f"ÐÑÐ¸Ð±ÐºÐ° Ð¾ÑÑÑÑÐ°: {e}")
    return "\n".join(lines)
def build_onzs_stats() -> str:
    if not os.path.exists(ONZS_TRAIN_FILE):
        return "ÐÐµÑ Ð´Ð°Ð½Ð½ÑÑ Ð¿Ð¾ ÐÐÐ·Ð¡."
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

    out = [f"ð¯ Ð¢Ð¾ÑÐ½Ð¾ÑÑÑ ÐÐ Ð¿Ð¾ ÐÐÐ·Ð¡: {acc_total}% (Ð²ÐµÑÐ½Ð¾ {total_ok}/{total_all})"]
    for o in sorted(stats.keys()):
        s = stats[o]
        acc = int(100 * s["ok"] / s["all"]) if s["all"] else 0
        out.append(f"ÐÐÐ·Ð¡-{o}: {acc}% ({s['ok']}/{s['all']})")
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
    "ÑÐ°Ð¼Ð¾ÑÑÑÐ¾Ð¹", "ÑÑÑÐ¾Ð¹ÐºÐ°", "ÑÑÑÐ¾Ð¸ÑÐµÐ»ÑÑÑÐ²Ð¾", "ÐºÐ¾ÑÐ»Ð¾Ð²Ð°Ð½", "ÑÑÐ½Ð´Ð°Ð¼ÐµÐ½Ñ", "Ð±ÐµÑÐ¾Ð½", "Ð°ÑÐ¼Ð°ÑÑÑÐ°",
    "ÐºÑÐ°Ð½", "Ð¾Ð¿Ð°Ð»ÑÐ±ÐºÐ°", "Ð·Ð°Ð±Ð¾Ñ", "Ð¿ÑÐ¸ÑÑÑÐ¾Ð¹ÐºÐ°", "Ð½Ð°Ð´ÑÑÑÐ¾Ð¹ÐºÐ°", "ÑÐµÐºÐ¾Ð½ÑÑÑÑÐºÑÐ¸Ñ",
    "ÑÑÐ°Ð¶", "Ð¿Ð»Ð¸ÑÐ°", "Ð¿ÐµÑÐµÐºÑÑÑÐ¸Ðµ"
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
    kw = ", ".join(card.get("keywords", [])) or "â"
    links = card.get("links") or []
    links_str = "\n".join(links) if links else "Ð½ÐµÑ ÑÑÑÐ»Ð¾Ðº"

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
            ai_lines.append(f"ð¤ ÐÐµÑÐ¾ÑÑÐ½Ð¾ÑÑÑ ÑÐ°Ð¼Ð¾ÑÑÑÐ¾Ñ (ÐÐ): {p:.1f}%")
    if comment:
        ai_lines.append(f"ð¬ ÐÐ¾Ð¼Ð¼ÐµÐ½ÑÐ°ÑÐ¸Ð¹ ÐÐ: {comment}")

    base = (
        "ð ÐÐ±Ð½Ð°ÑÑÐ¶ÐµÐ½Ð¾ Ð¿Ð¾Ð´Ð¾Ð·ÑÐ¸ÑÐµÐ»ÑÐ½Ð¾Ðµ ÑÐ¾Ð¾Ð±ÑÐµÐ½Ð¸Ðµ\n"
        f"ÐÑÑÐ¾ÑÐ½Ð¸Ðº: @{card.get('channel','â')}\n"
        f"ÐÐ°ÑÐ°: {dt}\n"
        f"ID Ð¿Ð¾ÑÑÐ°: {card.get('post_id','â')}\n\n"
        f"ð ÐÐ»ÑÑÐµÐ²ÑÐµ ÑÐ»Ð¾Ð²Ð°: {kw}\n\n"
        "ð Ð¢ÐµÐºÑÑ:\n"
        f"{card.get('text','')}\n\n"
        "ð Ð¡ÑÑÐ»ÐºÐ¸:\n"
        f"{links_str}\n\n"
        f"ð ID ÐºÐ°ÑÑÐ¾ÑÐºÐ¸: {card.get('card_id','â')}"
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
        line = f"ð ÐÐÐ·Ð¡: {val}"
        if src == "ai" and conf is not None:
            try:
                line += f" ({int(float(conf)*100)}%)"
            except Exception:
                pass
        if confirmed:
            line += " â Ð¿Ð¾Ð´ÑÐ²ÐµÑÐ¶Ð´ÐµÐ½Ð¾"
        base += "\n\n" + line

        reason = (oz.get("reason") or "").strip()
        if src == "ai" and reason:
            base += "\n" + f"ð ÐÑÐ¸ÑÐ¸Ð½Ð°: {reason}"

    return base

def append_history(entry: Dict):
    entry = dict(entry)
    entry["ts"] = now_ts()
    append_jsonl(HISTORY_CARDS, entry)

# ----------------------------- CALLBACK HANDLER -----------------------------
def handle_message(upd: Dict):
    """ÐÐ±ÑÐ°Ð±Ð°ÑÑÐ²Ð°ÐµÑ Ð²ÑÐ¾Ð´ÑÑÐ¸Ðµ ÑÐ¾Ð¾Ð±ÑÐµÐ½Ð¸Ñ Ð¸ Ð¿Ð¾ÑÑÑ.

    ÐÐ¾Ð´Ð´ÐµÑÐ¶Ð¸Ð²Ð°ÐµÑ update-Ð¿Ð¾Ð»Ñ: message / edited_message / channel_post / edited_channel_post.
    ÐÐ»Ñ Ð¼ÐµÐ´Ð¸Ð° (ÑÐ¾ÑÐ¾/Ð²Ð¸Ð´ÐµÐ¾/Ð´Ð¾ÐºÑÐ¼ÐµÐ½Ñ) Ð¸ÑÐ¿Ð¾Ð»ÑÐ·ÑÐµÑ caption.
    """
    msg = (
        upd.get("message")
        or upd.get("edited_message")
        or upd.get("channel_post")
        or upd.get("edited_channel_post")
        or {}
    )
    chat = msg.get("chat") or {}
    chat_id = int(chat.get("id", 0) or 0)

    text = (msg.get("text") or msg.get("caption") or "").strip()
    if not text:
        return

    # --- ADMIN MODE INPUT (role management) ---
    uid = get_sender_user_id(upd)
    mode = get_admin_mode(uid)
    # ÐÑÐ»Ð¸ Ð°Ð´Ð¼Ð¸Ð½ Ð² ÑÐµÐ¶Ð¸Ð¼Ðµ Ð²Ð²Ð¾Ð´Ð° (Ð½Ð°Ð¿ÑÐ¸Ð¼ÐµÑ, Ð¿Ð¾ÑÐ¾Ð³ AI-gate), Ð½Ð¾ Ð¿ÑÐ¸ÑÐ»Ð°Ð» ÐºÐ¾Ð¼Ð°Ð½Ð´Ñ (/admin Ð¸ Ñ.Ð¿.),
    # Ð²ÑÑÐ¾Ð´Ð¸Ð¼ Ð¸Ð· ÑÐµÐ¶Ð¸Ð¼Ð° Ð¸ Ð´Ð°ÑÐ¼ Ð¾Ð±ÑÐ°Ð±Ð¾ÑÐ°ÑÑÑÑ ÐºÐ¾Ð¼Ð°Ð½Ð´Ðµ.
    if mode and text and text.startswith("/"):
        clear_admin_mode(uid)
        mode = None
    if mode == "set_aigate":
        # If user entered a command while waiting for a threshold value, cancel input mode and process the command normally.
        if text and text.startswith("/"):
            clear_admin_mode(uid)
            mode = None
        
        raw = (text or "").strip().replace(",", ".")
        try:
            v = float(raw)
        except Exception:
            send_message(chat_id, "â ÐÐµÐ²ÐµÑÐ½ÑÐ¹ ÑÐ¾ÑÐ¼Ð°Ñ. ÐÐ²ÐµÐ´Ð¸ ÑÐ¸ÑÐ»Ð¾ Ð¾Ñ 0 Ð´Ð¾ 100 (Ð½Ð°Ð¿ÑÐ¸Ð¼ÐµÑ: 5 Ð¸Ð»Ð¸ 12.5).")
            return
        if v < 0 or v > 100:
            send_message(chat_id, "â ÐÐ¸Ð°Ð¿Ð°Ð·Ð¾Ð½: Ð¾Ñ 0 Ð´Ð¾ 100.")
            return
        global MIN_AI_GATE
        MIN_AI_GATE = float(v)
        set_cfg_value("min_ai_gate", MIN_AI_GATE)
        clear_admin_mode(uid)
        send_message(chat_id, f"â ÐÐ¾ÑÐ¾Ð²Ð¾. ÐÐ¾Ð²ÑÐ¹ AIâgate Ð¿Ð¾ÑÐ¾Ð³: {MIN_AI_GATE:.1f}%")
        return

        if mode == "add_admin":
            _roles_add("admins", target_uid); send_message(chat_id, f"â ÐÐ¾Ð±Ð°Ð²Ð»ÐµÐ½ Ð°Ð´Ð¼Ð¸Ð½: {target_uid}")
        elif mode == "del_admin":
            _roles_del("admins", target_uid); send_message(chat_id, f"â Ð£Ð´Ð°Ð»ÑÐ½ Ð°Ð´Ð¼Ð¸Ð½: {target_uid}")
        elif mode == "add_mod":
            _roles_add("moderators", target_uid); send_message(chat_id, f"â ÐÐ¾Ð±Ð°Ð²Ð»ÐµÐ½ Ð¼Ð¾Ð´ÐµÑÐ°ÑÐ¾Ñ: {target_uid}")
        elif mode == "del_mod":
            _roles_del("moderators", target_uid); send_message(chat_id, f"â Ð£Ð´Ð°Ð»ÑÐ½ Ð¼Ð¾Ð´ÐµÑÐ°ÑÐ¾Ñ: {target_uid}")
        elif mode == "add_lead":
            _roles_add("leadership", target_uid); send_message(chat_id, f"â ÐÐ¾Ð±Ð°Ð²Ð»ÐµÐ½Ð¾ ÑÑÐºÐ¾Ð²Ð¾Ð´ÑÑÐ²Ð¾: {target_uid}")
        elif mode == "del_lead":
            _roles_del("leadership", target_uid); send_message(chat_id, f"â Ð£Ð´Ð°Ð»ÐµÐ½Ð¾ ÑÑÐºÐ¾Ð²Ð¾Ð´ÑÑÐ²Ð¾: {target_uid}")
        elif mode == "add_report_target":
            _roles_add("report_targets", target_uid); send_message(chat_id, f"â ÐÐ¾Ð±Ð°Ð²Ð»ÐµÐ½ Ð¿Ð¾Ð»ÑÑÐ°ÑÐµÐ»Ñ Ð¾ÑÑÑÑÐ¾Ð²: {target_uid}")
        elif mode == "del_report_target":
            _roles_del("report_targets", target_uid); send_message(chat_id, f"â Ð£Ð´Ð°Ð»ÑÐ½ Ð¿Ð¾Ð»ÑÑÐ°ÑÐµÐ»Ñ Ð¾ÑÑÑÑÐ¾Ð²: {target_uid}")
        pop_admin_mode(uid)
        return

    if text == "/admin":
        if not is_privileged(uid):
            send_message(chat_id, "â ÐÐµÑ Ð´Ð¾ÑÑÑÐ¿Ð°.")
            return
        send_message(chat_id, admin_menu_text(), reply_markup=admin_menu_kb())
        return
    chat_id = (msg.get("chat") or {}).get("id")
    from_user = (msg.get("from") or {}).get("id")
    if not chat_id or not from_user:
        return

    if text == "/admin":
        uid = get_sender_user_id(upd)
        if not (is_admin(from_user) or is_moderator(from_user) or is_lead(from_user)):
            send_message(chat_id, "â ÐÐµÑ Ð´Ð¾ÑÑÑÐ¿Ð°.")
            return

        onzs_cnt = len(ONZS_MAP) if isinstance(ONZS_MAP, dict) else 0
        yagpt_enabled = bool(YAGPT_API_KEY and YAGPT_FOLDER_ID)
        info = []
        info.append("ð  ÐÐ´Ð¼Ð¸Ð½-Ð¿Ð°Ð½ÐµÐ»Ñ")
        info.append(f"ID: {from_user}")
        info.append(f"YandexGPT: {'ON' if yagpt_enabled else 'OFF'} | model={YAGPT_MODEL}")
        info.append(f"AI-gate: {MIN_AI_GATE}% | HTTP_TIMEOUT={HTTP_TIMEOUT}s")
        info.append(f"ÐÐÐ·Ð¡ ÐºÐ°ÑÐ°Ð»Ð¾Ð³: {onzs_cnt} | ÑÐ°Ð¹Ð»: {ONZS_XLSX}")
        info.append(f"Admins: {len(ADMINS)} | Moderators: {len(MODERATORS)} | Leadership: {len(LEADERSHIP)}")
        send_message(chat_id, "\n".join(info), reply_markup=build_admin_keyboard())
        return

    if text == "/onzs_ai_stats":
        uid = get_sender_user_id(upd)
        if not (is_admin(from_user) or is_moderator(from_user) or is_lead(from_user)):
            send_message(chat_id, "â ÐÐµÑ Ð´Ð¾ÑÑÑÐ¿Ð°.")
            return
        send_message(chat_id, build_onzs_stats())
        return

    if text == "/start":
        send_message(chat_id, "ÐÐ¾Ñ Ð·Ð°Ð¿ÑÑÐµÐ½.")
        return

# ----------------------------- GETUPDATES LOOP -----------------------------def acquire_lock() -> bool:
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



def handle_callback_query(upd: Dict):
    # --- normalize callback update ---
    cq = upd.get('callback_query') or {}
    data = cq.get('data')
    cb_id = cq.get('id')
    msg = cq.get('message') or {}
    chat = msg.get('chat') or {}
    chat_id = int(chat.get('id', 0) or 0)
    msg_id = int(msg.get('message_id', 0) or 0)
    if not data or not chat_id or not msg_id:
        return
    # ---------------------------------
    # ---------------------------------

    cb = upd.get("callback_query") or {}
    # data already set above

    # --- ADMIN PANEL CALLBACKS ---
    uid = get_sender_user_id(upd)
    if data and data.startswith("admin:"):
        if not is_privileged(uid):
            answer_callback_query(cb_id, "ÐÐµÑ Ð´Ð¾ÑÑÑÐ¿Ð°")
            return
        action = data.split(":",1)[1]

        if action == "back":
            edit_message_text(chat_id, msg_id, admin_menu_text(), reply_markup=admin_menu_kb())
            answer_callback_query(cb_id, "OK")
            return
        if action == "roles":
            edit_message_text(chat_id, msg_id, build_roles_text(), reply_markup=admin_roles_kb())
            answer_callback_query(cb_id, "OK")
            return
        if action == "reports":
            edit_message_text(chat_id, msg_id, "ð§¾ ÐÑÑÑÑÑ", reply_markup=admin_reports_kb())
            answer_callback_query(cb_id, "OK")
            return
        if action == "settings":
            edit_message_text(chat_id, msg_id, "âï¸ ÐÐ°ÑÑÑÐ¾Ð¹ÐºÐ¸", reply_markup=admin_settings_kb())
            answer_callback_query(cb_id, "OK")
            return
        if action == "set_aigate":
            set_admin_mode(uid, "set_aigate")
            send_message(
                chat_id,
                f"ð AIâgate Ð¿Ð¾ÑÐ¾Ð³ (Ð² Ð¿ÑÐ¾ÑÐµÐ½ÑÐ°Ñ).\n\nÐ¢ÐµÐºÑÑÐ¸Ð¹: {MIN_AI_GATE:.1f}%\n\nÐÐ²ÐµÐ´Ð¸ ÑÐ¸ÑÐ»Ð¾ Ð¾Ñ 0 Ð´Ð¾ 100.",
            )
            answer_callback_query(cb_id, "ÐÐ²ÐµÐ´Ð¸ÑÐµ ÑÐ¸ÑÐ»Ð¾ 0â100")
            return

        if action == "stats":
            try:
                txt = build_onzs_stats()
            except Exception:
                txt = "ð Ð¡ÑÐ°ÑÐ¸ÑÑÐ¸ÐºÐ° Ð¿Ð¾ÐºÐ° Ð½ÐµÐ´Ð¾ÑÑÑÐ¿Ð½Ð°."
            edit_message_text(chat_id, msg_id, txt, reply_markup={"inline_keyboard":[[{"text":"â¬ï¸ ÐÐ°Ð·Ð°Ð´","callback_data":"admin:back"}]]})
            answer_callback_query(cb_id, "OK")
            return
        if action == "list_roles":
            answer_callback_query(cb_id, "OK")
            send_message(chat_id, build_roles_text())
            return
        if action in ("add_admin","del_admin","add_mod","del_mod","add_lead","del_lead","add_report_target","del_report_target"):
            set_admin_mode(uid, action)
            answer_callback_query(cb_id, "OK")
            send_message(chat_id, "âï¸ ÐÑÐ¸ÑÐ»Ð¸ÑÐµ Telegram ID Ð¿Ð¾Ð»ÑÐ·Ð¾Ð²Ð°ÑÐµÐ»Ñ (ÑÐ¸ÑÐ»Ð¾Ð¼).")
            return
        if action == "report_targets":
            answer_callback_query(cb_id, "OK")
            send_message(chat_id, build_roles_text())
            return
        if action == "report_day":
            answer_callback_query(cb_id, "OK")
            send_message(chat_id, build_daily_report_text())
            return
        if action == "reload_onzs":
            load_onzs_catalog()
            answer_callback_query(cb_id, "OK")
            send_message(chat_id, f"â ÐÐÐ·Ð¡ Ð¿ÐµÑÐµÐ·Ð°Ð³ÑÑÐ¶ÐµÐ½: {len(ONZS_MAP)}")
            return
        if action == "test_yagpt":
            answer_callback_query(cb_id, "OK")
            try:
                t = call_yandex_gpt_raw([{"role":"user","text":"Ð¢ÐµÑÑ. ÐÑÐ²ÐµÑÑ Ð¾Ð´Ð½Ð¸Ð¼ ÑÐ»Ð¾Ð²Ð¾Ð¼: ÐÐ"}])
                send_message(chat_id, f"ð§ª YandexGPT: {str(t)[:500]}")
            except Exception as e:
                send_message(chat_id, f"ð§ª YandexGPT Ð¾ÑÐ¸Ð±ÐºÐ°: {e}")
            return
    cb_id = cb.get("id") or ""
    msg = cb.get("message") or {}
    from_user = (cb.get("from") or {}).get("id")
    chat_id = (msg.get("chat") or {}).get("id")
    message_id = msg.get("message_id")

    if not from_user:
        answer_callback(cb_id, "ÐÑÐ¸Ð±ÐºÐ°", show_alert=True)
        return

    # -------------------- ADMIN ACTIONS --------------------
    if data.startswith("admin:"):
        # Ð´Ð¾ÑÑÑÐ¿: Ð°Ð´Ð¼Ð¸Ð½/Ð¼Ð¾Ð´ÐµÑÐ°ÑÐ¾Ñ/ÑÑÐºÐ¾Ð²Ð¾Ð´ÑÑÐ²Ð¾
        if not (is_admin(from_user) or is_moderator(from_user) or is_lead(from_user)):
            answer_callback(cb_id, "ÐÐµÑ Ð´Ð¾ÑÑÑÐ¿Ð°", show_alert=True)
            return

        op = data.split(":", 1)[1]

        if op == "onzs_stats":
            if chat_id:
                send_message(chat_id, build_onzs_stats())
            answer_callback(cb_id, "ÐÐ¾ÑÐ¾Ð²Ð¾")
            return

        if op == "reload_onzs":
            load_onzs_catalog()
            if chat_id:
                send_message(chat_id, f"ð ÐÐ°ÑÐ°Ð»Ð¾Ð³ ÐÐÐ·Ð¡ Ð¿ÐµÑÐµÐ·Ð°Ð³ÑÑÐ¶ÐµÐ½: {len(ONZS_MAP)} ÑÐ»ÐµÐ¼ÐµÐ½ÑÐ¾Ð²")
            answer_callback(cb_id, "ÐÐµÑÐµÐ·Ð°Ð³ÑÑÐ¶ÐµÐ½Ð¾")
            return

        if op == "test_yagpt":
            ok = False
            detail = ""
            try:
                out_text, meta = call_yandex_gpt_raw([
                    {"role": "system", "text": "ÐÑÐ²ÐµÑÐ°Ð¹ Ð¾Ð´Ð½Ð¾Ð¹ ÑÑÑÐ¾ÐºÐ¾Ð¹: OK."},
                    {"role": "user", "text": "ÐÑÐ²ÐµÑÑ Ð¾Ð´Ð½Ð¾Ð¹ ÑÑÑÐ¾ÐºÐ¾Ð¹: OK"},
                ])
                if isinstance(out_text, str) and "OK" in out_text.upper():
                    ok = True
                else:
                    detail = (out_text or "")[:200]
            except Exception as e:
                detail = str(e)[:200]

            if chat_id:
                send_message(chat_id, "â YandexGPT: OK" if ok else f"â ï¸ YandexGPT: Ð½ÐµÑ Ð¾ÑÐ²ÐµÑÐ°. {detail}")
            answer_callback(cb_id, "OK" if ok else "ÐÑÐ¾Ð±Ð»ÐµÐ¼Ð°")
            return

        answer_callback(cb_id, "ÐÐµÐ¸Ð·Ð²ÐµÑÑÐ½Ð°Ñ ÐºÐ¾Ð¼Ð°Ð½Ð´Ð°", show_alert=True)
        return

    # -------------------- ONZS ACTIONS --------------------
    if data.startswith("onzs:"):
        if not is_moderator(from_user):
            answer_callback(cb_id, "â ÐÐµÑ Ð´Ð¾ÑÑÑÐ¿Ð°.", show_alert=True)
            return

        parts = data.split(":")
        op = parts[1] if len(parts) > 1 else ""

        if op == "edit" and len(parts) == 3:
            card_id = parts[2]
            if chat_id and message_id:
                edit_reply_markup(chat_id, message_id, reply_markup=build_onzs_pick_keyboard(card_id))
            answer_callback(cb_id, "ÐÑÐ±ÐµÑÐ¸ ÐÐÐ·Ð¡ (1â12)")
            return

        if op == "set" and len(parts) == 4:
            card_id = parts[2]
            try:
                n = int(parts[3])
            except Exception:
                n = 0
            if n < 1 or n > 12:
                answer_callback(cb_id, "ÐÐÐ·Ð¡ Ð´Ð¾Ð»Ð¶ÐµÐ½ Ð±ÑÑÑ 1â12", show_alert=True)
                return

            card = load_card(card_id)
            if not card:
                answer_callback(cb_id, "ÐÐ°ÑÑÐ¾ÑÐºÐ° Ð½Ðµ Ð½Ð°Ð¹Ð´ÐµÐ½Ð°", show_alert=True)
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
            answer_callback(cb_id, f"ÐÐÐ·Ð¡ ÑÑÑÐ°Ð½Ð¾Ð²Ð»ÐµÐ½: {n}")
            return

        if op == "confirm" and len(parts) == 3:
            card_id = parts[2]
            card = load_card(card_id)
            if not card:
                answer_callback(cb_id, "ÐÐ°ÑÑÐ¾ÑÐºÐ° Ð½Ðµ Ð½Ð°Ð¹Ð´ÐµÐ½Ð°", show_alert=True)
                return

            oz = card.get("onzs") or {}
            val = oz.get("value") if oz.get("value") else oz.get("ai")
            if not val:
                answer_callback(cb_id, "ÐÐÐ·Ð¡ ÐµÑÑ Ð½Ðµ Ð¾Ð¿ÑÐµÐ´ÐµÐ»ÑÐ½", show_alert=True)
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
            answer_callback(cb_id, "ÐÐÐ·Ð¡ Ð¿Ð¾Ð´ÑÐ²ÐµÑÐ¶Ð´ÑÐ½")
            return

        if op == "back" and len(parts) == 3:
            card_id = parts[2]
            if chat_id and message_id:
                edit_reply_markup(chat_id, message_id, reply_markup=build_card_keyboard(card_id))
            answer_callback(cb_id, "ÐÐº")
            return

        answer_callback(cb_id, "ÐÐµÐ¸Ð·Ð²ÐµÑÑÐ½Ð°Ñ ÐºÐ¾Ð¼Ð°Ð½Ð´Ð° ÐÐÐ·Ð¡", show_alert=True)
        return

    # -------------------- CARD ACTIONS --------------------
    if data.startswith("card:"):
        parts = data.split(":")
        if len(parts) != 3:
            answer_callback(cb_id, "ÐÑÐ¸Ð±ÐºÐ°", show_alert=True)
            return
        card_id, action = parts[1], parts[2]
        if not is_moderator(from_user):
            answer_callback(cb_id, "â ÐÐµÑ Ð´Ð¾ÑÑÑÐ¿Ð°.", show_alert=True)
            return

        card = load_card(card_id)
        if not card:
            answer_callback(cb_id, "ÐÐ°ÑÑÐ¾ÑÐºÐ° Ð½Ðµ Ð½Ð°Ð¹Ð´ÐµÐ½Ð°", show_alert=True)
            return

        label = None
        if action == "work":
            label = "work"
            answer_callback(cb_id, "ÐÑÐ¸Ð½ÑÑÐ¾: Ð ÑÐ°Ð±Ð¾ÑÑ")
        elif action == "wrong":
            label = "wrong"
            answer_callback(cb_id, "ÐÑÐ¸Ð½ÑÑÐ¾: ÐÐµÐ²ÐµÑÐ½Ð¾")
        elif action == "attach":
            label = "attach"
            answer_callback(cb_id, "ÐÑÐ¸Ð½ÑÑÐ¾: ÐÑÐ¸Ð²ÑÐ·Ð°ÑÑ")
        else:
            answer_callback(cb_id, "ÐÐµÐ¸Ð·Ð²ÐµÑÑÐ½Ð¾Ðµ Ð´ÐµÐ¹ÑÑÐ²Ð¸Ðµ", show_alert=True)
            return

        append_history({"text": card.get("text", ""), "label": label, "channel": card.get("channel", ""), "reason": "user_action"})

        # Remove buttons after action
        if chat_id and message_id:
            edit_reply_markup(chat_id, message_id, reply_markup={"inline_keyboard": []})
        return

    answer_callback(cb_id, "OK")

# ----------------------------- COMMANDS -----------------------------def handle_message(upd: Dict):
    # --- ensure chat_id is always defined ---
    msg = upd.get('message') or {}
    chat = msg.get('chat') or {}
    chat_id = int(chat.get('id', 0) or 0)
    text = (msg.get('text') or '').strip()
    if not text: 
        return
    # ---------------------------------------

    msg = upd.get("message") or {}
    text = (msg.get("text") or "").strip()

    # --- ADMIN MODE INPUT (role management) ---
    uid = get_sender_user_id(upd)
    mode = get_admin_mode(uid)
    if mode and text and not text.startswith("/"):
        m_id = re.search(r"(\d+)", text)
        if not m_id:
            send_message(chat_id, "â ï¸ ÐÑÐ¸ÑÐ»Ð¸ÑÐµ ÑÐ¸ÑÐ»Ð¾Ð²Ð¾Ð¹ Telegram ID Ð¿Ð¾Ð»ÑÐ·Ð¾Ð²Ð°ÑÐµÐ»Ñ.")
            return
        target_uid = int(m_id.group(1))
        if mode == "add_admin":
            _roles_add("admins", target_uid); send_message(chat_id, f"â ÐÐ¾Ð±Ð°Ð²Ð»ÐµÐ½ Ð°Ð´Ð¼Ð¸Ð½: {target_uid}")
        elif mode == "del_admin":
            _roles_del("admins", target_uid); send_message(chat_id, f"â Ð£Ð´Ð°Ð»ÑÐ½ Ð°Ð´Ð¼Ð¸Ð½: {target_uid}")
        elif mode == "add_mod":
            _roles_add("moderators", target_uid); send_message(chat_id, f"â ÐÐ¾Ð±Ð°Ð²Ð»ÐµÐ½ Ð¼Ð¾Ð´ÐµÑÐ°ÑÐ¾Ñ: {target_uid}")
        elif mode == "del_mod":
            _roles_del("moderators", target_uid); send_message(chat_id, f"â Ð£Ð´Ð°Ð»ÑÐ½ Ð¼Ð¾Ð´ÐµÑÐ°ÑÐ¾Ñ: {target_uid}")
        elif mode == "add_lead":
            _roles_add("leadership", target_uid); send_message(chat_id, f"â ÐÐ¾Ð±Ð°Ð²Ð»ÐµÐ½Ð¾ ÑÑÐºÐ¾Ð²Ð¾Ð´ÑÑÐ²Ð¾: {target_uid}")
        elif mode == "del_lead":
            _roles_del("leadership", target_uid); send_message(chat_id, f"â Ð£Ð´Ð°Ð»ÐµÐ½Ð¾ ÑÑÐºÐ¾Ð²Ð¾Ð´ÑÑÐ²Ð¾: {target_uid}")
        elif mode == "add_report_target":
            _roles_add("report_targets", target_uid); send_message(chat_id, f"â ÐÐ¾Ð±Ð°Ð²Ð»ÐµÐ½ Ð¿Ð¾Ð»ÑÑÐ°ÑÐµÐ»Ñ Ð¾ÑÑÑÑÐ¾Ð²: {target_uid}")
        elif mode == "del_report_target":
            _roles_del("report_targets", target_uid); send_message(chat_id, f"â Ð£Ð´Ð°Ð»ÑÐ½ Ð¿Ð¾Ð»ÑÑÐ°ÑÐµÐ»Ñ Ð¾ÑÑÑÑÐ¾Ð²: {target_uid}")
        pop_admin_mode(uid)
        return

    if text == "/admin":
        if not is_privileged(uid):
            send_message(chat_id, "â ÐÐµÑ Ð´Ð¾ÑÑÑÐ¿Ð°.")
            return
        send_message(chat_id, admin_menu_text(), reply_markup=admin_menu_kb())
        return
    chat_id = (msg.get("chat") or {}).get("id")
    from_user = (msg.get("from") or {}).get("id")
    if not chat_id or not from_user:
        return

    if text == "/admin":
        uid = get_sender_user_id(upd)
        if not (is_admin(from_user) or is_moderator(from_user) or is_lead(from_user)):
            send_message(chat_id, "â ÐÐµÑ Ð´Ð¾ÑÑÑÐ¿Ð°.")
            return

        onzs_cnt = len(ONZS_MAP) if isinstance(ONZS_MAP, dict) else 0
        yagpt_enabled = bool(YAGPT_API_KEY and YAGPT_FOLDER_ID)
        info = []
        info.append("ð  ÐÐ´Ð¼Ð¸Ð½-Ð¿Ð°Ð½ÐµÐ»Ñ")
        info.append(f"ID: {from_user}")
        info.append(f"YandexGPT: {'ON' if yagpt_enabled else 'OFF'} | model={YAGPT_MODEL}")
        info.append(f"AI-gate: {MIN_AI_GATE}% | HTTP_TIMEOUT={HTTP_TIMEOUT}s")
        info.append(f"ÐÐÐ·Ð¡ ÐºÐ°ÑÐ°Ð»Ð¾Ð³: {onzs_cnt} | ÑÐ°Ð¹Ð»: {ONZS_XLSX}")
        info.append(f"Admins: {len(ADMINS)} | Moderators: {len(MODERATORS)} | Leadership: {len(LEADERSHIP)}")
        send_message(chat_id, "\n".join(info), reply_markup=build_admin_keyboard())
        return

    if text == "/onzs_ai_stats":
        uid = get_sender_user_id(upd)
        if not (is_admin(from_user) or is_moderator(from_user) or is_lead(from_user)):
            send_message(chat_id, "â ÐÐµÑ Ð´Ð¾ÑÑÑÐ¿Ð°.")
            return
        send_message(chat_id, build_onzs_stats())
        return

    if text == "/start":
        send_message(chat_id, "ÐÐ¾Ñ Ð·Ð°Ð¿ÑÑÐµÐ½.")
        return

# ----------------------------- GETUPDATES LOOP -----------------------------def acquire_lock() -> bool:
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
            elif ("message" in u) or ("edited_message" in u) or ("channel_post" in u) or ("edited_channel_post" in u):
                handle_message(u)

# ----------------------------- MAIN -----------------------------

def start_health_server():
    """Start a minimal HTTP server on PORT for Railway/Web-style deployments.

    Railway "Web" services expect a process to bind to $PORT. If you deploy this
    bot as a Web service, lack of an open port may cause the platform to stop
    the container as "unhealthy".

    If PORT is not set (or invalid), this function does nothing.
    """
    port_s = os.getenv("PORT")
    if not port_s:
        return
    try:
        port = int(port_s)
    except Exception:
        log.warning(f"[HEALTH] invalid PORT={port_s!r}; skipping health server")
        return

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"OK")

        def log_message(self, fmt, *args):
            # silence default http.server access logs
            return

    def _serve():
        try:
            httpd = HTTPServer(("0.0.0.0", port), _Handler)
            log.info(f"[HEALTH] listening on 0.0.0.0:{port}")
            httpd.serve_forever()
        except Exception as e:
            log.warning(f"[HEALTH] server failed: {e}")

    t = threading.Thread(target=_serve, name="health_server", daemon=True)
    t.start()

def main():
    init_db()
    load_onzs_catalog()
    log.info('=== VERSION: ONZS + AI-GATE + BUTTONS + STATS ===')

    # If deployed as a Web service, keep the platform health-checks satisfied.
    start_health_server()

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
    _acq = globals().get('acquire_lock')
    log.info(f"[LOCK] acquire_lock callable={callable(_acq)}")
    if callable(_acq) and (not _acq()):
        log.error("Lock exists: another poller is running. Exiting.")
        return

    log.info(f"Lock acquired: {LOCK_FILE}")
    log.info("Starting getUpdates poller...")
    try:
        run_poller()
    finally:
        release_lock()

if __name__ == "__main__":
    main()


# -*- coding: utf-8 -*-
"""
samastroi_extensions.py
Доп. блоки для прод-скрейпера: рецидивы, судебные ссылки, тепловая карта, web-дашборд.

Подключение:
- import samastroi_extensions as ext
- ext.ensure_extra_tables(conn) в init_db() перед commit()
- ext.enrich_recidiv_and_courts(db, now_ts, card, cid) в generate_card()
- ext.start_dashboard_thread(db, build_kpi_text, get_prob_threshold, compute_training_stats, REPORTS_DIR, now_ts) в main()
"""
from __future__ import annotations

import os
import json
import uuid
import urllib.parse
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Callable, Any

def ensure_extra_tables(conn) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS objects (
            object_key TEXT PRIMARY KEY,
            cadastral_number TEXT,
            address TEXT,
            municipality TEXT,
            lat REAL,
            lon REAL,
            first_seen_ts INTEGER NOT NULL,
            last_seen_ts INTEGER NOT NULL,
            occurrences INTEGER NOT NULL DEFAULT 1,
            last_card_id TEXT
        );
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS card_objects (
            card_id TEXT PRIMARY KEY,
            object_key TEXT NOT NULL
        );
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS court_links (
            card_id TEXT NOT NULL,
            url TEXT NOT NULL,
            title TEXT,
            source TEXT,
            created_ts INTEGER NOT NULL,
            PRIMARY KEY(card_id, url)
        );
    """)

def _norm_ws(s: str) -> str:
    return " ".join((s or "").replace("\t", " ").replace("\n", " ").split()).strip()

def normalize_address_ru(s: str) -> str:
    s = _norm_ws(s).lower()
    if not s:
        return ""
    s = s.replace("г.о.", "го ").replace("городской округ", "го ")
    s = s.replace("ул.", "улица ").replace("ул ", "улица ")
    s = s.replace("пр-т", "проспект ").replace("пр-т.", "проспект ").replace("пр.", "проспект ")
    s = s.replace("пер.", "переулок ").replace("пер ", "переулок ")
    s = s.replace("ш.", "шоссе ").replace("ш ", "шоссе ")
    s = s.replace("д.", "дом ").replace("д ", "дом ")
    s = s.replace(",", " ").replace(";", " ")
    while "  " in s:
        s = s.replace("  ", " ")
    # грубая эвристика падежа
    s = s.replace("ской ", "ская ")
    return s.strip()

def _coords_to_lat(coords: str) -> Tuple[Optional[float], Optional[float]]:
    if not coords:
        return None, None
    try:
        p = [x.strip() for x in coords.split(",")]
        return float(p[0]), float(p[1])
    except Exception:
        return None, None

def build_object_key(geo_info: Dict, municipality: str = "") -> Tuple[str, Dict]:
    cad = (geo_info or {}).get("cadastral_number") or ""
    addr = (geo_info or {}).get("address") or ""
    coords = (geo_info or {}).get("coordinates") or ""
    muni = (municipality or "").strip()

    meta = {"cadastral_number": cad, "address": addr, "coordinates": coords, "municipality": muni}

    if cad:
        return f"cad:{cad}", meta

    naddr = normalize_address_ru(addr)
    if naddr:
        key = f"addr:{naddr}"
        if muni:
            key += f"|{muni.lower()}"
        return key, meta

    if coords:
        lat, lon = _coords_to_lat(coords)
        if lat is not None and lon is not None:
            return f"ll:{round(lat,4)},{round(lon,4)}", meta

    return f"unknown:{uuid.uuid4().hex[:10]}", meta

def upsert_object_and_get_repeats(conn, now_ts: Callable[[], int], card_id: str, geo_info: Dict, municipality: str = "", ts: Optional[int] = None) -> Dict:
    ts = int(ts or now_ts())
    obj_key, meta = build_object_key(geo_info or {}, municipality=municipality)
    lat, lon = _coords_to_lat(meta.get("coordinates", ""))

    row = conn.execute("SELECT occurrences, first_seen_ts FROM objects WHERE object_key=?;", (obj_key,)).fetchone()
    if row:
        occ, first_ts = int(row[0]), int(row[1])
        occ2 = occ + 1
        conn.execute(
            """
            UPDATE objects
               SET last_seen_ts=?, occurrences=?, last_card_id=?,
                   cadastral_number=COALESCE(NULLIF(?,''), cadastral_number),
                   address=COALESCE(NULLIF(?,''), address),
                   municipality=COALESCE(NULLIF(?,''), municipality),
                   lat=COALESCE(?, lat),
                   lon=COALESCE(?, lon)
             WHERE object_key=?;
            """,
            (ts, occ2, card_id,
             meta.get("cadastral_number",""),
             meta.get("address",""),
             meta.get("municipality",""),
             lat, lon, obj_key)
        )
    else:
        conn.execute(
            """
            INSERT INTO objects(object_key,cadastral_number,address,municipality,lat,lon,first_seen_ts,last_seen_ts,occurrences,last_card_id)
            VALUES(?,?,?,?,?,?,?,?,?,?);
            """,
            (obj_key,
             meta.get("cadastral_number") or None,
             meta.get("address") or None,
             meta.get("municipality") or None,
             lat, lon,
             ts, ts, 1, card_id)
        )
        occ2 = 1
        first_ts = ts

    conn.execute("INSERT OR REPLACE INTO card_objects(card_id, object_key) VALUES(?,?);", (card_id, obj_key))
    return {"object_key": obj_key, "repeat_count": int(occ2), "first_seen_ts": int(first_ts), "is_repeat": int(occ2) >= 2}

def build_court_search_links(geo_info: Dict, municipality: str = "") -> List[Dict]:
    q_parts = []
    cad = (geo_info or {}).get("cadastral_number")
    addr = (geo_info or {}).get("address")
    if cad:
        q_parts.append(str(cad))
    if addr:
        q_parts.append(str(addr))
    if municipality:
        q_parts.append(str(municipality))
    q = " ".join([p for p in q_parts if p]).strip()
    if not q:
        return []
    qq = urllib.parse.quote(q, safe="")
    return [
        {"source":"kad_arbitr","title":"Поиск в КАД Арбитр","url":f"https://kad.arbitr.ru/#search?text={qq}"},
        {"source":"yandex_search","title":"Поиск по судебным упоминаниям (Яндекс)","url":f"https://yandex.ru/search/?text={qq}%20суд%20дело"},
    ]

def upsert_court_links(conn, now_ts: Callable[[], int], card_id: str, links: List[Dict]) -> None:
    for it in links or []:
        url = it.get("url")
        if not url:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO court_links(card_id,url,title,source,created_ts) VALUES(?,?,?,?,?);",
            (card_id, url, it.get("title"), it.get("source"), now_ts())
        )

def enrich_recidiv_and_courts(db_conn_factory: Callable[[], Any], now_ts: Callable[[], int], card: Dict, cid: str) -> None:
    muni = card.get("onzs_category_name") or ""
    ts = int(card.get("timestamp") or now_ts())
    conn = db_conn_factory()
    try:
        conn.execute("BEGIN IMMEDIATE;")
        card["recidiv"] = upsert_object_and_get_repeats(conn, now_ts, cid, card.get("geo_info") or {}, municipality=str(muni), ts=ts)
        links = build_court_search_links(card.get("geo_info") or {}, municipality=str(muni))
        if links:
            card["court_links"] = links
            upsert_court_links(conn, now_ts, cid, links)
        conn.execute("COMMIT;")
    except Exception as e:
        try:
            conn.execute("ROLLBACK;")
        except Exception:
            pass
        card["recidiv_error"] = str(e)
    finally:
        conn.close()

def fetch_top_recidiv(db_conn_factory: Callable[[], Any], limit: int = 25) -> List[Dict]:
    conn = db_conn_factory()
    try:
        rows = conn.execute(
            """
            SELECT object_key, cadastral_number, address, municipality, lat, lon, occurrences, last_seen_ts, last_card_id
              FROM objects
             WHERE occurrences >= 2
             ORDER BY occurrences DESC, last_seen_ts DESC
             LIMIT ?;
            """,
            (int(limit),)
        ).fetchall()
        return [
            {"object_key": r[0], "cadastral_number": r[1], "address": r[2], "municipality": r[3],
             "lat": r[4], "lon": r[5], "occurrences": int(r[6]), "last_seen_ts": int(r[7]), "last_card_id": r[8]}
            for r in rows
        ]
    finally:
        conn.close()

def build_heatmap_html(db_conn_factory: Callable[[], Any], reports_dir: str, now_ts: Callable[[], int], days: int = 180, min_occ: int = 1) -> str:
    os.makedirs(reports_dir, exist_ok=True)
    out_path = os.path.join(reports_dir, f"heatmap_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.html")
    since_ts = now_ts() - int(days) * 86400
    conn = db_conn_factory()
    try:
        rows = conn.execute(
            """
            SELECT lat, lon, occurrences, municipality, address, cadastral_number
              FROM objects
             WHERE lat IS NOT NULL AND lon IS NOT NULL
               AND last_seen_ts >= ?
               AND occurrences >= ?
             ORDER BY occurrences DESC, last_seen_ts DESC;
            """,
            (int(since_ts), int(min_occ))
        ).fetchall()
    finally:
        conn.close()

    pts = []
    for lat, lon, occ, mun, addr, cad in rows:
        try:
            pts.append({"lat": float(lat), "lon": float(lon), "w": float(max(1, int(occ))),
                        "municipality": mun or "", "address": addr or "", "cadastral": cad or ""})
        except Exception:
            pass

    center_lat, center_lon = 55.75, 37.62
    if pts:
        center_lat = sum(p["lat"] for p in pts) / len(pts)
        center_lon = sum(p["lon"] for p in pts) / len(pts)

    html = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Тепловая карта самостроя</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<style>html,body,#map{{height:100%;margin:0}} .panel{{position:absolute;top:10px;left:10px;z-index:999;background:rgba(255,255,255,.95);padding:10px 12px;border-radius:10px;box-shadow:0 2px 10px rgba(0,0,0,.12);font-family:Arial,sans-serif;font-size:12px;}}</style>
</head><body><div id="map"></div>
<div class="panel"><b>🧭 Тепловая карта самостроя</b><br/>Период: {days} дн. • Точек: {len(pts)}</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
<script>
  const map = L.map('map').setView([{center_lat:.6f},{center_lon:.6f}], 10);
  L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{maxZoom:19, attribution:'&copy; OpenStreetMap'}}).addTo(map);
  const pts = {json.dumps(pts, ensure_ascii=False)};
  L.heatLayer(pts.map(p=>[p.lat,p.lon,p.w]), {{radius:26, blur:18, maxZoom:17}}).addTo(map);
  pts.slice(0,60).forEach(p=>{{ const label=(p.municipality?p.municipality+' — ':'')+(p.address||p.cadastral||''); L.circleMarker([p.lat,p.lon],{{radius:4}}).addTo(map).bindPopup(label); }});
</script></body></html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path

def get_public_base_url() -> str:
    v = (os.getenv("PUBLIC_BASE_URL") or "").strip()
    if v:
        return v.rstrip("/")
    dom = (os.getenv("RAILWAY_PUBLIC_DOMAIN") or os.getenv("RAILWAY_STATIC_URL") or "").strip()
    if dom:
        if dom.startswith("http"):
            return dom.rstrip("/")
        return "https://" + dom.rstrip("/")
    return ""

def start_dashboard_thread(
    db_conn_factory: Callable[[], Any],
    build_kpi_text: Callable[[], str],
    get_prob_threshold: Callable[[], float],
    compute_training_stats: Callable[[], Dict],
    reports_dir: str,
    now_ts: Callable[[], int],
) -> None:
    enabled = str(os.getenv("DASHBOARD_ENABLED", "0")).strip().lower() in ("1","true","yes","on")
    if not enabled:
        return
    try:
        from flask import Flask, jsonify, request, send_file
    except Exception:
        return

    app = Flask(__name__)

    @app.get("/api/summary")
    def api_summary():
        return jsonify({"kpi_text": build_kpi_text(), "threshold": get_prob_threshold(), "training": compute_training_stats()})

    @app.get("/api/recidiv")
    def api_recidiv():
        limit = int(request.args.get("limit","50"))
        return jsonify(fetch_top_recidiv(db_conn_factory, limit=limit))

    @app.get("/map")
    def map_view():
        p = build_heatmap_html(db_conn_factory, reports_dir, now_ts, days=int(request.args.get("days","180")))
        return send_file(p, mimetype="text/html", as_attachment=False)

    @app.get("/")
    def root():
        base = get_public_base_url()
        return f"<pre>Самострой — web-дашборд\n\n{build_kpi_text()}\n\nBase: {base or 'PUBLIC_BASE_URL not set'}\n/map  /api/summary  /api/recidiv</pre>"

    port = int(os.getenv("PORT","8080"))
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=port, debug=False), daemon=True).start()

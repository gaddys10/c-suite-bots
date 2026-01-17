import sqlite3
import time

DB_PATH = "memory.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS signals (
        ts REAL,
        channel TEXT,
        role TEXT,
        kind TEXT,
        text TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS kv (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    """)

    conn.commit()
    conn.close()

def write_signal(channel, role, kind, text):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO signals VALUES (?, ?, ?, ?, ?)",
        (time.time(), channel, role, kind, text[:500])
    )
    conn.commit()
    conn.close()

def get_signals_since(ts):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT ts, channel, role, kind, text FROM signals WHERE ts > ? ORDER BY ts ASC",
        (ts,)
    )
    rows = cur.fetchall()
    conn.close()
    return rows

def get_last_brief_ts():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT value FROM kv WHERE key='last_brief_ts'")
    row = cur.fetchone()
    conn.close()
    return float(row[0]) if row else 0.0

def set_last_brief_ts(ts):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO kv VALUES ('last_brief_ts', ?)",
        (str(ts),)
    )
    conn.commit()
    conn.close()

def kv_get(key: str, default: str = "") -> str:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT value FROM kv WHERE key=?", (key,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else default

def kv_set(key: str, value: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO kv VALUES (?, ?)", (key, value))
    conn.commit()
    conn.close()

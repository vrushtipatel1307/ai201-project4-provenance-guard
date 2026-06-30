"""Storage layer for Provenance Guard.

Two SQLite tables, kept deliberately separate:

  * content   — the canonical, MUTABLE record for one submission. Appeals
                flip its `status` from "classified" to "under_review".
  * audit_log — an APPEND-ONLY event trail. Every submission and every appeal
                writes one immutable, fully self-contained JSON snapshot here.

Keeping them apart means an appeal can change the live status of a piece of
content without ever rewriting history in the audit log.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone

_DB_PATH = os.path.join(os.path.dirname(__file__), "provenance.db")
DB_PATH = _DB_PATH  # Export for analytics


def _connect():
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they do not exist. Safe to call on every startup."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS content (
                content_id        TEXT PRIMARY KEY,
                creator_id        TEXT NOT NULL,
                text              TEXT NOT NULL,
                attribution       TEXT,
                confidence        REAL,
                llm_score         REAL,
                stylometric_score REAL,
                linguistic_score  REAL,
                label             TEXT,
                status            TEXT NOT NULL,
                created_at        TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                content_id TEXT NOT NULL,
                event      TEXT NOT NULL,
                timestamp  TEXT NOT NULL,
                detail     TEXT NOT NULL
            )
            """
        )
        # Stretch Feature: Creator verification (Provenance Certificate)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS creator_verification (
                creator_id        TEXT PRIMARY KEY,
                verified          BOOLEAN NOT NULL,
                verification_date TEXT,
                verification_method TEXT
            )
            """
        )


def now_iso():
    """UTC timestamp with millisecond precision, e.g. 2026-06-30T14:32:10.123Z."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def save_content(record):
    """Insert (or replace) the canonical content record."""
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO content (
                content_id, creator_id, text, attribution, confidence,
                llm_score, stylometric_score, linguistic_score, label, status, created_at
            ) VALUES (
                :content_id, :creator_id, :text, :attribution, :confidence,
                :llm_score, :stylometric_score, :linguistic_score, :label, :status, :created_at
            )
            """,
            record,
        )


def get_content(content_id):
    """Return the content record as a dict, or None if it does not exist."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM content WHERE content_id = ?", (content_id,)
        ).fetchone()
    return dict(row) if row else None


def update_status(content_id, status):
    """Change a content record's status. Returns True if a row was updated."""
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE content SET status = ? WHERE content_id = ?",
            (status, content_id),
        )
        return cur.rowcount > 0


def add_audit_entry(content_id, event, detail):
    """Append an immutable audit event. `detail` is any JSON-serializable dict."""
    with _connect() as conn:
        conn.execute(
            "INSERT INTO audit_log (content_id, event, timestamp, detail) "
            "VALUES (?, ?, ?, ?)",
            (content_id, event, now_iso(), json.dumps(detail)),
        )


def get_log(limit=50):
    """Return the most recent audit entries, newest first, as flat dicts."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    entries = []
    for row in rows:
        entry = json.loads(row["detail"])
        # Stamp the canonical envelope fields on top of the snapshot.
        entry["content_id"] = row["content_id"]
        entry["event"] = row["event"]
        entry["timestamp"] = row["timestamp"]
        entries.append(entry)
    return entries


# Stretch Feature: Creator Verification (Provenance Certificate)

def verify_creator(creator_id, method="email"):
    """Mark a creator as verified (earned Provenance Certificate)."""
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO creator_verification 
            (creator_id, verified, verification_date, verification_method)
            VALUES (?, ?, ?, ?)
            """,
            (creator_id, True, now_iso(), method),
        )


def is_creator_verified(creator_id):
    """Check if a creator has earned a Provenance Certificate."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT verified FROM creator_verification WHERE creator_id = ?",
            (creator_id,),
        ).fetchone()
    return dict(row)["verified"] if row else False


def get_creator_verification_info(creator_id):
    """Return full verification record for a creator."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM creator_verification WHERE creator_id = ?",
            (creator_id,),
        ).fetchone()
    return dict(row) if row else None

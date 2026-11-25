"""Local SQLite persistence for accounts, patients, and image analyses."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from pneumonia.config import PROJECT_ROOT


def data_dir() -> Path:
    return Path(os.getenv("CLINICAL_DATA_DIR", PROJECT_ROOT / "clinical_data")).expanduser()


def database_path() -> Path:
    return data_dir() / "clinical.sqlite3"


@contextmanager
def connection() -> Iterator[sqlite3.Connection]:
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 15000")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def initialize_database() -> None:
    with connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                display_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                session_version INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS password_resets (
                token_hash TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                expires_at TEXT NOT NULL,
                used_at TEXT
            );
            CREATE TABLE IF NOT EXISTS patients (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                patient_code TEXT,
                name TEXT,
                age INTEGER,
                sex TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS patients_user_code
                ON patients(user_id, patient_code) WHERE patient_code IS NOT NULL;
            CREATE INDEX IF NOT EXISTS patients_by_user
                ON patients(user_id, updated_at DESC);
            CREATE TABLE IF NOT EXISTS analyses (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                patient_id TEXT REFERENCES patients(id) ON DELETE CASCADE,
                image_path TEXT NOT NULL,
                image_filename TEXT NOT NULL,
                image_width INTEGER NOT NULL,
                image_height INTEGER NOT NULL,
                symptoms TEXT NOT NULL DEFAULT '',
                clinical_history TEXT NOT NULL DEFAULT '',
                notes TEXT NOT NULL DEFAULT '',
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS analyses_by_user
                ON analyses(user_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS analyses_by_patient
                ON analyses(patient_id, created_at DESC);
            """
        )


def user_by_email(email: str) -> Optional[Dict[str, Any]]:
    with connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        return dict(row) if row else None


def user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    with connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None


def create_user(user_id: str, email: str, display_name: str, password_hash: str) -> Dict[str, Any]:
    timestamp = now_utc()
    with connection() as conn:
        conn.execute(
            """INSERT INTO users (id, email, display_name, password_hash, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (user_id, email, display_name, password_hash, timestamp, timestamp),
        )
        return dict(conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone())


def update_profile(user_id: str, display_name: str) -> None:
    with connection() as conn:
        conn.execute(
            "UPDATE users SET display_name = ?, updated_at = ? WHERE id = ?",
            (display_name, now_utc(), user_id),
        )


def update_password(user_id: str, password_hash: str) -> None:
    with connection() as conn:
        conn.execute(
            """UPDATE users
               SET password_hash = ?, session_version = session_version + 1, updated_at = ?
               WHERE id = ?""",
            (password_hash, now_utc(), user_id),
        )
        conn.execute("DELETE FROM password_resets WHERE user_id = ?", (user_id,))


def create_password_reset(user_id: str, token_hash: str, expires_at: str) -> None:
    with connection() as conn:
        conn.execute("DELETE FROM password_resets WHERE user_id = ?", (user_id,))
        conn.execute(
            "INSERT INTO password_resets (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
            (token_hash, user_id, expires_at),
        )


def consume_password_reset(token_hash: str, password_hash: str) -> bool:
    timestamp = now_utc()
    with connection() as conn:
        row = conn.execute(
            """SELECT user_id FROM password_resets
               WHERE token_hash = ? AND expires_at > ? AND used_at IS NULL""",
            (token_hash, timestamp),
        ).fetchone()
        if not row:
            return False
        conn.execute(
            """UPDATE users
               SET password_hash = ?, session_version = session_version + 1, updated_at = ?
               WHERE id = ?""",
            (password_hash, timestamp, row["user_id"]),
        )
        conn.execute("DELETE FROM password_resets WHERE user_id = ?", (row["user_id"],))
        return True


def create_analysis(
    *,
    analysis_id: str,
    user_id: str,
    image_path: Path,
    image_filename: str,
    image_width: int,
    image_height: int,
    result: Dict[str, Any],
    patient_code: Optional[str],
    patient_name: Optional[str],
    age: Optional[int],
    sex: Optional[str],
    symptoms: str,
    clinical_history: str,
    notes: str,
) -> None:
    timestamp = now_utc()
    patient_id: Optional[str] = None
    has_patient_details = any((patient_code, patient_name, age is not None, sex))
    with connection() as conn:
        if has_patient_details:
            if patient_code:
                patient = conn.execute(
                    "SELECT id FROM patients WHERE user_id = ? AND patient_code = ?",
                    (user_id, patient_code),
                ).fetchone()
                patient_id = patient["id"] if patient else None
            if patient_id:
                conn.execute(
                    """UPDATE patients SET name = COALESCE(?, name), age = COALESCE(?, age),
                       sex = COALESCE(?, sex), updated_at = ? WHERE id = ? AND user_id = ?""",
                    (patient_name, age, sex, timestamp, patient_id, user_id),
                )
            else:
                import uuid

                patient_id = uuid.uuid4().hex
                conn.execute(
                    """INSERT INTO patients
                       (id, user_id, patient_code, name, age, sex, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        patient_id,
                        user_id,
                        patient_code,
                        patient_name,
                        age,
                        sex,
                        timestamp,
                        timestamp,
                    ),
                )
        conn.execute(
            """INSERT INTO analyses
               (id, user_id, patient_id, image_path, image_filename, image_width,
                image_height, symptoms, clinical_history, notes, result_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                analysis_id,
                user_id,
                patient_id,
                str(image_path),
                image_filename,
                image_width,
                image_height,
                symptoms,
                clinical_history,
                notes,
                json.dumps(result),
                timestamp,
            ),
        )


def _analysis_dict(row: sqlite3.Row) -> Dict[str, Any]:
    result = dict(row)
    result["result"] = json.loads(result.pop("result_json"))
    return result


def get_analysis(user_id: str, analysis_id: str) -> Optional[Dict[str, Any]]:
    with connection() as conn:
        row = conn.execute(
            """SELECT a.*, p.patient_code, p.name AS patient_name, p.age, p.sex
               FROM analyses a LEFT JOIN patients p ON p.id = a.patient_id
               WHERE a.user_id = ? AND a.id = ?""",
            (user_id, analysis_id),
        ).fetchone()
        return _analysis_dict(row) if row else None


def list_analyses(
    user_id: str,
    *,
    query: str = "",
    prediction: str = "",
    order: str = "newest",
    limit: int = 20,
    offset: int = 0,
) -> Tuple[List[Dict[str, Any]], int]:
    order_by = {
        "newest": "a.created_at DESC",
        "oldest": "a.created_at ASC",
        "patient": "p.name COLLATE NOCASE ASC, a.created_at DESC",
        "prediction": "json_extract(a.result_json, '$.prediction') ASC, a.created_at DESC",
    }[order]
    clauses = ["a.user_id = ?"]
    values: List[Any] = [user_id]
    if prediction:
        clauses.append("json_extract(a.result_json, '$.prediction') = ?")
        values.append(prediction)
    if query:
        clauses.append(
            """(a.image_filename LIKE ? OR p.name LIKE ? OR p.patient_code LIKE ?
               OR json_extract(a.result_json, '$.prediction') LIKE ?)"""
        )
        term = f"%{query}%"
        values.extend([term, term, term, term])
    where = " AND ".join(clauses)
    with connection() as conn:
        total = conn.execute(
            f"SELECT COUNT(*) FROM analyses a LEFT JOIN patients p ON p.id = a.patient_id WHERE {where}",
            values,
        ).fetchone()[0]
        rows = conn.execute(
            f"""SELECT a.*, p.patient_code, p.name AS patient_name
                FROM analyses a LEFT JOIN patients p ON p.id = a.patient_id
                WHERE {where} ORDER BY {order_by} LIMIT ? OFFSET ?""",
            [*values, limit, offset],
        ).fetchall()
        return [_analysis_dict(row) for row in rows], int(total)


def list_patient_analyses(user_id: str, patient_id: str) -> List[Dict[str, Any]]:
    with connection() as conn:
        rows = conn.execute(
            """SELECT a.*, p.patient_code, p.name AS patient_name
               FROM analyses a JOIN patients p ON p.id = a.patient_id
               WHERE a.user_id = ? AND p.user_id = ? AND p.id = ?
               ORDER BY a.created_at DESC""",
            (user_id, user_id, patient_id),
        ).fetchall()
        return [_analysis_dict(row) for row in rows]


def delete_analysis(user_id: str, analysis_id: str) -> Optional[str]:
    with connection() as conn:
        row = conn.execute(
            "SELECT image_path FROM analyses WHERE user_id = ? AND id = ?",
            (user_id, analysis_id),
        ).fetchone()
        if not row:
            return None
        conn.execute("DELETE FROM analyses WHERE user_id = ? AND id = ?", (user_id, analysis_id))
        return row["image_path"]


def list_patients(user_id: str, query: str = "") -> List[Dict[str, Any]]:
    values: List[Any] = [user_id]
    clause = "p.user_id = ?"
    if query:
        clause += " AND (p.name LIKE ? OR p.patient_code LIKE ?)"
        term = f"%{query}%"
        values.extend([term, term])
    with connection() as conn:
        rows = conn.execute(
            f"""SELECT p.*, COUNT(a.id) AS analysis_count, MAX(a.created_at) AS last_seen
                FROM patients p LEFT JOIN analyses a ON a.patient_id = p.id
                WHERE {clause} GROUP BY p.id ORDER BY p.updated_at DESC""",
            values,
        ).fetchall()
        return [dict(row) for row in rows]


def get_patient(user_id: str, patient_id: str) -> Optional[Dict[str, Any]]:
    with connection() as conn:
        row = conn.execute(
            """SELECT p.*, COUNT(a.id) AS analysis_count, MAX(a.created_at) AS last_seen
               FROM patients p LEFT JOIN analyses a ON a.patient_id = p.id
               WHERE p.user_id = ? AND p.id = ? GROUP BY p.id""",
            (user_id, patient_id),
        ).fetchone()
        return dict(row) if row else None


def delete_patient(user_id: str, patient_id: str) -> Optional[List[str]]:
    with connection() as conn:
        exists = conn.execute(
            "SELECT 1 FROM patients WHERE user_id = ? AND id = ?",
            (user_id, patient_id),
        ).fetchone()
        if not exists:
            return None
        paths = conn.execute(
            "SELECT image_path FROM analyses WHERE user_id = ? AND patient_id = ?",
            (user_id, patient_id),
        ).fetchall()
        conn.execute("DELETE FROM patients WHERE user_id = ? AND id = ?", (user_id, patient_id))
        return [row["image_path"] for row in paths]


def dashboard_summary(user_id: str) -> Dict[str, Any]:
    with connection() as conn:
        analysis_count = conn.execute(
            "SELECT COUNT(*) FROM analyses WHERE user_id = ?", (user_id,)
        ).fetchone()[0]
        patient_count = conn.execute(
            "SELECT COUNT(*) FROM patients WHERE user_id = ?", (user_id,)
        ).fetchone()[0]
        class_rows = conn.execute(
            """SELECT json_extract(result_json, '$.prediction') AS prediction, COUNT(*) AS count
               FROM analyses WHERE user_id = ? GROUP BY prediction""",
            (user_id,),
        ).fetchall()
        recent = conn.execute(
            """SELECT a.*, p.patient_code, p.name AS patient_name
               FROM analyses a LEFT JOIN patients p ON p.id = a.patient_id
               WHERE a.user_id = ? ORDER BY a.created_at DESC LIMIT 6""",
            (user_id,),
        ).fetchall()
    counts = {row["prediction"]: int(row["count"]) for row in class_rows}
    return {
        "analysis_count": int(analysis_count),
        "patient_count": int(patient_count),
        "normal_count": counts.get("NORMAL", 0),
        "pneumonia_count": counts.get("PNEUMONIA", 0),
        "recent_analyses": [_analysis_dict(row) for row in recent],
    }

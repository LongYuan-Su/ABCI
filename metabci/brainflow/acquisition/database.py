# -*- coding: utf-8 -*-
"""Patient/session/event database manager using SQLite."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ..logger import get_logger

logger = get_logger("database")


def _default_db_path() -> Path:
    project_root = Path(__file__).resolve().parents[3]
    runtime_dir = project_root / ".runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return runtime_dir / "swallow_experiment.db"


@dataclass
class PatientRecord:
    """Patient data record aligned with the GUI database schema."""

    patient_id: str
    name: str = ""
    gender: str = ""
    age: int = 0
    height_cm: float = 0.0
    weight_kg: float = 0.0
    dysphagia_level: str = ""


class DBManager:
    """SQLite database manager for patients, sessions, and paradigm events."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(Path(db_path) if db_path else _default_db_path())
        self.conn: Optional[sqlite3.Connection] = None
        self.cursor: Optional[sqlite3.Cursor] = None

    def _connect(self) -> None:
        if self.conn is None:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(self.db_path)
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA foreign_keys=ON")
            self.conn.row_factory = sqlite3.Row
            self.cursor = self.conn.cursor()

    def init_db(self) -> None:
        """Create tables if they do not exist."""
        self._connect()
        assert self.cursor is not None
        self.cursor.executescript(
            """
            CREATE TABLE IF NOT EXISTS patients (
                patient_id TEXT PRIMARY KEY,
                name TEXT DEFAULT '',
                gender TEXT DEFAULT '',
                age INTEGER DEFAULT 0,
                height_cm REAL DEFAULT 0,
                weight_kg REAL DEFAULT 0,
                dysphagia_level TEXT DEFAULT '',
                created_at TEXT DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                experiment_date TEXT DEFAULT '',
                epoch_count INTEGER DEFAULT 1,
                paradigm_type TEXT DEFAULT 'swallow_assessment',
                notes TEXT DEFAULT '',
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
                    ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS swallow_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                event_name TEXT DEFAULT '',
                timestamp_sec REAL DEFAULT 0,
                marker_label TEXT DEFAULT '',
                created_at TEXT DEFAULT '',
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                    ON DELETE CASCADE
            );
            """
        )
        self.conn.commit()
        logger.info("Database initialized: %s", self.db_path)

    def upsert_patient(self, record: PatientRecord | dict) -> None:
        """Insert or update a patient record."""
        self._connect()
        assert self.cursor is not None and self.conn is not None
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(record, PatientRecord):
            data = {
                "patient_id": record.patient_id,
                "name": record.name,
                "gender": record.gender,
                "age": record.age,
                "height_cm": record.height_cm,
                "weight_kg": record.weight_kg,
                "dysphagia_level": record.dysphagia_level,
            }
        else:
            data = record

        existing = self.get_patient(data["patient_id"])
        if existing:
            self.cursor.execute(
                "UPDATE patients SET name=?, gender=?, age=?, "
                "height_cm=?, weight_kg=?, dysphagia_level=? "
                "WHERE patient_id=?",
                (
                    data.get("name", ""),
                    data.get("gender", ""),
                    data.get("age", 0),
                    data.get("height_cm", 0.0),
                    data.get("weight_kg", 0.0),
                    data.get("dysphagia_level", ""),
                    data["patient_id"],
                ),
            )
        else:
            self.cursor.execute(
                "INSERT INTO patients (patient_id, name, gender, age, "
                "height_cm, weight_kg, dysphagia_level, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    data["patient_id"],
                    data.get("name", ""),
                    data.get("gender", ""),
                    data.get("age", 0),
                    data.get("height_cm", 0.0),
                    data.get("weight_kg", 0.0),
                    data.get("dysphagia_level", ""),
                    now,
                ),
            )
        self.conn.commit()

    def get_patient(self, patient_id: str) -> Optional[dict]:
        """Get one patient by ID."""
        self._connect()
        assert self.cursor is not None
        self.cursor.execute("SELECT * FROM patients WHERE patient_id=?", (patient_id,))
        row = self.cursor.fetchone()
        return dict(row) if row else None

    def get_patients(self) -> List[dict]:
        """Get all patients."""
        self._connect()
        assert self.cursor is not None
        self.cursor.execute("SELECT * FROM patients ORDER BY patient_id")
        return [dict(row) for row in self.cursor.fetchall()]

    def delete_patient(self, patient_id: str) -> None:
        """Delete a patient and related sessions/events."""
        self._connect()
        assert self.cursor is not None and self.conn is not None
        self.cursor.execute("DELETE FROM patients WHERE patient_id=?", (patient_id,))
        self.conn.commit()
        logger.info("Deleted patient: %s", patient_id)

    def create_session(
        self,
        patient_id: str,
        experiment_date: str = "",
        epoch_count: int = 1,
        notes: str = "",
        paradigm_type: str = "swallow_assessment",
    ) -> int:
        """Create a session and return its ID."""
        self._connect()
        assert self.cursor is not None and self.conn is not None
        self.cursor.execute(
            "INSERT INTO sessions (patient_id, experiment_date, epoch_count, paradigm_type, notes) "
            "VALUES (?, ?, ?, ?, ?)",
            (patient_id, experiment_date, epoch_count, paradigm_type, notes),
        )
        self.conn.commit()
        session_id = int(self.cursor.lastrowid)
        logger.info("Session %d created for %s", session_id, patient_id)
        return session_id

    def get_sessions(self, patient_id: Optional[str] = None) -> List[dict]:
        """Get sessions, optionally filtered by patient."""
        self._connect()
        assert self.cursor is not None
        if patient_id:
            self.cursor.execute(
                "SELECT s.*, "
                "(SELECT COUNT(*) FROM swallow_events e WHERE e.session_id=s.session_id) AS event_count "
                "FROM sessions s WHERE s.patient_id=? ORDER BY s.experiment_date DESC",
                (patient_id,),
            )
        else:
            self.cursor.execute(
                "SELECT s.*, "
                "(SELECT COUNT(*) FROM swallow_events e WHERE e.session_id=s.session_id) AS event_count "
                "FROM sessions s ORDER BY s.experiment_date DESC"
            )
        return [dict(row) for row in self.cursor.fetchall()]

    def add_event(
        self,
        session_id: int,
        event_name: str,
        timestamp_sec: float = 0,
        marker_label: str = "",
    ) -> None:
        """Add a swallow event to a session."""
        self._connect()
        assert self.cursor is not None and self.conn is not None
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute(
            "INSERT INTO swallow_events (session_id, event_name, timestamp_sec, marker_label, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, event_name, timestamp_sec, marker_label, now),
        )
        self.conn.commit()

    def get_events(self, session_id: int) -> List[dict]:
        """Get all events for a session."""
        self._connect()
        assert self.cursor is not None
        self.cursor.execute(
            "SELECT * FROM swallow_events WHERE session_id=? ORDER BY timestamp_sec",
            (session_id,),
        )
        return [dict(row) for row in self.cursor.fetchall()]

    def get_patient_stats(self) -> dict:
        """Return per-patient session count and last test date."""
        self._connect()
        assert self.cursor is not None
        rows = self.cursor.execute(
            """
            SELECT p.patient_id,
                   COUNT(s.session_id) AS session_count,
                   MAX(s.experiment_date) AS last_date
            FROM patients p LEFT JOIN sessions s ON s.patient_id=p.patient_id
            GROUP BY p.patient_id ORDER BY p.patient_id
            """
        ).fetchall()
        stats = {}
        for row in rows:
            stats[row["patient_id"]] = {
                "session_count": row["session_count"] or 0,
                "last_date": row["last_date"] or "-",
            }
        return stats

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None
            logger.info("Database closed.")

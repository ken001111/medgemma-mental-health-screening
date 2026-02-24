"""
Database module for storing call records, scores, reports, and alerts.
"""
import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional
import os
from config import DB_PATH, DATA_DIR, DB_TABLE_CALLS, DB_TABLE_SCORES, DB_TABLE_REPORTS, DB_TABLE_ALERTS


class ScreeningDatabase:
    """
    Database handler for mental health screening application.
    """
    
    def __init__(self, db_path: str = DB_PATH):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Calls table
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {DB_TABLE_CALLS} (
                call_id TEXT PRIMARY KEY,
                soldier_id TEXT NOT NULL,
                call_timestamp TIMESTAMP NOT NULL,
                call_duration REAL,
                audio_path TEXT,
                transcript TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Scores table
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {DB_TABLE_SCORES} (
                score_id INTEGER PRIMARY KEY AUTOINCREMENT,
                call_id TEXT NOT NULL,
                phq9_score REAL,
                anxiety_risk REAL,
                ptsd_risk REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (call_id) REFERENCES {DB_TABLE_CALLS}(call_id)
            )
        """)
        
        # Reports table
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {DB_TABLE_REPORTS} (
                report_id INTEGER PRIMARY KEY AUTOINCREMENT,
                call_id TEXT NOT NULL,
                report_content TEXT NOT NULL,
                report_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (call_id) REFERENCES {DB_TABLE_CALLS}(call_id)
            )
        """)
        
        # Alerts table
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {DB_TABLE_ALERTS} (
                alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                call_id TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                sent_to TEXT,
                acknowledged BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (call_id) REFERENCES {DB_TABLE_CALLS}(call_id)
            )
        """)
        
        # Clinician sends (for GUI: track when report was sent to clinician)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS clinician_sends (
                call_id TEXT PRIMARY KEY,
                sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (call_id) REFERENCES calls(call_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def add_call(self, call_id: str, soldier_id: str, call_timestamp: datetime,
                 call_duration: float, audio_path: str, transcript: str = ""):
        """Add a new call record."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(f"""
            INSERT INTO {DB_TABLE_CALLS} 
            (call_id, soldier_id, call_timestamp, call_duration, audio_path, transcript)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (call_id, soldier_id, call_timestamp, call_duration, audio_path, transcript))
        
        conn.commit()
        conn.close()
    
    def add_scores(self, call_id: str, phq9_score: float = None,
                   anxiety_risk: float = None, ptsd_risk: float = None):
        """Add screening scores for a call."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(f"""
            INSERT INTO {DB_TABLE_SCORES} 
            (call_id, phq9_score, anxiety_risk, ptsd_risk)
            VALUES (?, ?, ?, ?)
        """, (call_id, phq9_score, anxiety_risk, ptsd_risk))
        
        conn.commit()
        conn.close()
    
    def add_report(self, call_id: str, report_content: str, report_path: str = None):
        """Add a medical report for a call."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(f"""
            INSERT INTO {DB_TABLE_REPORTS} 
            (call_id, report_content, report_path)
            VALUES (?, ?, ?)
        """, (call_id, report_content, report_path))
        
        conn.commit()
        conn.close()
    
    def add_alert(self, call_id: str, alert_type: str, severity: str,
                  message: str, sent_to: str = None):
        """Add an alert record."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(f"""
            INSERT INTO {DB_TABLE_ALERTS} 
            (call_id, alert_type, severity, message, sent_to)
            VALUES (?, ?, ?, ?, ?)
        """, (call_id, alert_type, severity, message, sent_to))
        
        conn.commit()
        conn.close()
    
    def get_call_history(self, soldier_id: str, limit: int = 10) -> List[Dict]:
        """Get call history for a soldier."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(f"""
            SELECT c.*, s.phq9_score, s.anxiety_risk, s.ptsd_risk
            FROM {DB_TABLE_CALLS} c
            LEFT JOIN {DB_TABLE_SCORES} s ON c.call_id = s.call_id
            WHERE c.soldier_id = ?
            ORDER BY c.call_timestamp DESC
            LIMIT ?
        """, (soldier_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_recordings_for_gui(self, soldier_id: str = None, limit: int = 50) -> List[Dict]:
        """Get all recordings with scores and severity, grouped for GUI. Optionally filter by soldier_id."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if soldier_id:
            cursor.execute(f"""
                SELECT c.call_id, c.soldier_id, c.call_timestamp, c.call_duration, c.transcript,
                       s.phq9_score, s.anxiety_risk, s.ptsd_risk
                FROM {DB_TABLE_CALLS} c
                LEFT JOIN {DB_TABLE_SCORES} s ON c.call_id = s.call_id
                WHERE c.soldier_id = ?
                ORDER BY c.call_timestamp DESC
                LIMIT ?
            """, (soldier_id, limit))
        else:
            cursor.execute(f"""
                SELECT c.call_id, c.soldier_id, c.call_timestamp, c.call_duration, c.transcript,
                       s.phq9_score, s.anxiety_risk, s.ptsd_risk
                FROM {DB_TABLE_CALLS} c
                LEFT JOIN {DB_TABLE_SCORES} s ON c.call_id = s.call_id
                ORDER BY c.call_timestamp DESC
                LIMIT ?
            """, (limit,))
        
        rows = cursor.fetchall()
        
        # Check which calls have been sent to clinician
        cursor.execute("SELECT call_id FROM clinician_sends")
        sent_ids = {r[0] for r in cursor.fetchall()}
        conn.close()
        
        result = []
        for row in rows:
            d = dict(row)
            phq9 = d.get("phq9_score")
            anxiety = d.get("anxiety_risk")
            ptsd = d.get("ptsd_risk")
            severity = "low"
            if phq9 is not None and phq9 >= 15:
                severity = "high"
            elif phq9 is not None and phq9 >= 10:
                severity = "moderate"
            if anxiety is not None and anxiety >= 0.8:
                severity = "high"
            elif anxiety is not None and anxiety >= 0.6 and severity == "low":
                severity = "moderate"
            if ptsd is not None and ptsd >= 0.8:
                severity = "high"
            elif ptsd is not None and ptsd >= 0.6 and severity == "low":
                severity = "moderate"
            d["severity"] = severity
            d["sent_to_clinician"] = d["call_id"] in sent_ids
            result.append(d)
        
        return result
    
    def get_call_context_for_chat(self, call_id: str) -> Optional[Dict]:
        """Get report summary and scores for a call (for chat context with Gemini)."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT c.call_id, c.transcript,
                   s.phq9_score, s.anxiety_risk, s.ptsd_risk
            FROM {DB_TABLE_CALLS} c
            LEFT JOIN {DB_TABLE_SCORES} s ON c.call_id = s.call_id
            WHERE c.call_id = ?
        """, (call_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return None
        d = dict(row)
        cursor.execute(f"""
            SELECT report_content FROM {DB_TABLE_REPORTS}
            WHERE call_id = ?
            ORDER BY created_at DESC LIMIT 1
        """, (call_id,))
        report_row = cursor.fetchone()
        conn.close()
        if report_row:
            d["report_content"] = report_row[0]
        else:
            d["report_content"] = None
        # Compute severity for context
        phq9 = d.get("phq9_score")
        anxiety = d.get("anxiety_risk")
        ptsd = d.get("ptsd_risk")
        severity = "low"
        if phq9 is not None and phq9 >= 15:
            severity = "high"
        elif phq9 is not None and phq9 >= 10:
            severity = "moderate"
        if anxiety is not None and anxiety >= 0.8:
            severity = "high"
        elif anxiety is not None and anxiety >= 0.6 and severity == "low":
            severity = "moderate"
        if ptsd is not None and ptsd >= 0.8:
            severity = "high"
        elif ptsd is not None and ptsd >= 0.6 and severity == "low":
            severity = "moderate"
        d["severity"] = severity
        return d

    def mark_sent_to_clinician(self, call_id: str):
        """Mark that this call's report was sent to clinician."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO clinician_sends (call_id) VALUES (?)",
            (call_id,)
        )
        conn.commit()
        conn.close()
    
    def get_pending_alerts(self) -> List[Dict]:
        """Get all unacknowledged alerts."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(f"""
            SELECT * FROM {DB_TABLE_ALERTS}
            WHERE acknowledged = 0
            ORDER BY created_at DESC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]

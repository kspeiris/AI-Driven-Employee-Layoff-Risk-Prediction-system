"""Database operations for storing predictions and logs."""

import sqlite3
from datetime import datetime
from contextlib import contextmanager
from config import DATABASE_PATH


@contextmanager
def get_connection():
    """Context manager for database connections."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_database():
    """Initialize database tables."""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                age INTEGER,
                years_experience INTEGER,
                industry TEXT,
                job_role TEXT,
                routine_task_percentage REAL,
                creativity_requirement REAL,
                ai_usage_hours REAL,
                tasks_automated REAL,
                ai_training_hours REAL,
                prediction TEXT,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER,
                was_correct BOOLEAN,
                actual_risk TEXT,
                comments TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prediction_id) REFERENCES predictions(id)
            )
        """)
        
        # Query logs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                endpoint TEXT,
                response_time_ms REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def save_prediction(data, prediction, confidence):
    """Save a prediction to the database."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (
                age, years_experience, industry, job_role,
                routine_task_percentage, creativity_requirement,
                ai_usage_hours, tasks_automated, ai_training_hours,
                prediction, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.get('Age'), data.get('Years_of_Experience'),
            data.get('Industry'), data.get('Job_Role'),
            data.get('Routine_Task_Percentage'),
            data.get('Creativity_Requirement'),
            data.get('AI_Usage_Hours_Per_Week'),
            data.get('Tasks_Automated_Percentage'),
            data.get('AI_Training_Hours'),
            prediction, confidence
        ))
        return cursor.lastrowid


def get_prediction_stats():
    """Get prediction statistics for analytics."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                prediction,
                COUNT(*) as count,
                AVG(confidence) as avg_confidence
            FROM predictions
            GROUP BY prediction
        """)
        return [dict(row) for row in cursor.fetchall()]


def log_query(endpoint, response_time_ms):
    """Log API query for monitoring."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO query_logs (endpoint, response_time_ms)
            VALUES (?, ?)
        """, (endpoint, response_time_ms))
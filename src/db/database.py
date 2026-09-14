import sqlite3
from pathlib import Path


DB_PATH = Path("data/safevision.db")


def get_connection():
    DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    connection = sqlite3.connect(
        DB_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


def initialize_database():
    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            timestamp TEXT NOT NULL,

            track_id INTEGER NOT NULL,

            helmet INTEGER NOT NULL,
            mask INTEGER NOT NULL,
            vest INTEGER NOT NULL,

            compliant INTEGER NOT NULL,

            missing_ppe TEXT,
            severity TEXT NOT NULL
        )
        """
    )

    connection.commit()
    connection.close()
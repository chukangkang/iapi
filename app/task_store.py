import json
import sqlite3
from typing import Any, Optional

import pymysql
from pymysql.connections import Connection as MySQLConnection
from pymysql.cursors import DictCursor

from app.config import Settings


class ImageTaskMetadataStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.backend = settings.task_db_backend
        self.db_path = settings.task_db_path
        if self.backend == "sqlite":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection | MySQLConnection:
        if self.backend == "mysql":
            return pymysql.connect(
                host=self.settings.mysql_host,
                port=self.settings.mysql_port,
                user=self.settings.mysql_user,
                password=self.settings.mysql_password_or_none,
                database=self.settings.mysql_database,
                charset=self.settings.mysql_charset,
                connect_timeout=self.settings.mysql_connect_timeout,
                cursorclass=DictCursor,
                autocommit=False,
            )
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            if self.backend == "mysql":
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        CREATE TABLE IF NOT EXISTS image_tasks (
                            id VARCHAR(80) PRIMARY KEY,
                            status VARCHAR(32) NOT NULL,
                            created BIGINT NOT NULL,
                            updated BIGINT NOT NULL,
                            started BIGINT NULL,
                            completed BIGINT NULL,
                            worker_id INT NULL,
                            payload_json JSON NULL,
                            reference_image LONGTEXT NULL,
                            result_json JSON NULL,
                            error TEXT NULL,
                            INDEX idx_image_tasks_status (status)
                        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                        """
                    )
                    self._ensure_mysql_column(cursor, "payload_json", "JSON NULL")
                    self._ensure_mysql_column(cursor, "reference_image", "LONGTEXT NULL")
                connection.commit()
                return

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS image_tasks (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created INTEGER NOT NULL,
                    updated INTEGER NOT NULL,
                    started INTEGER,
                    completed INTEGER,
                    worker_id INTEGER,
                    payload_json TEXT,
                    reference_image TEXT,
                    result_json TEXT,
                    error TEXT
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_image_tasks_status ON image_tasks(status)")
            self._ensure_sqlite_column(connection, "payload_json", "TEXT")
            self._ensure_sqlite_column(connection, "reference_image", "TEXT")

    def save(self, task: object) -> None:
        payload_json = self._payload_to_json(getattr(task, "payload", None))
        reference_image = getattr(task, "reference_image_data", None)
        result_json = self._result_to_json(getattr(task, "result", None))
        with self._connect() as connection:
            params = (
                getattr(task, "id"),
                getattr(task, "status"),
                getattr(task, "created"),
                getattr(task, "updated"),
                getattr(task, "started"),
                getattr(task, "completed"),
                getattr(task, "worker_id"),
                payload_json,
                reference_image,
                result_json,
                getattr(task, "error"),
            )
            if self.backend == "mysql":
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO image_tasks (
                            id, status, created, updated, started, completed, worker_id, payload_json, reference_image, result_json, error
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            status=VALUES(status),
                            updated=VALUES(updated),
                            started=VALUES(started),
                            completed=VALUES(completed),
                            worker_id=VALUES(worker_id),
                            payload_json=COALESCE(VALUES(payload_json), payload_json),
                            reference_image=COALESCE(VALUES(reference_image), reference_image),
                            result_json=VALUES(result_json),
                            error=VALUES(error)
                        """,
                        params,
                    )
                connection.commit()
                return

            connection.execute(
                """
                INSERT INTO image_tasks (
                    id, status, created, updated, started, completed, worker_id, payload_json, reference_image, result_json, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status=excluded.status,
                    updated=excluded.updated,
                    started=excluded.started,
                    completed=excluded.completed,
                    worker_id=excluded.worker_id,
                    payload_json=COALESCE(excluded.payload_json, payload_json),
                    reference_image=COALESCE(excluded.reference_image, reference_image),
                    result_json=excluded.result_json,
                    error=excluded.error
                """,
                params,
            )

    def get(self, task_id: str) -> Optional[dict]:
        with self._connect() as connection:
            if self.backend == "mysql":
                with connection.cursor() as cursor:
                    cursor.execute("SELECT * FROM image_tasks WHERE id = %s", (task_id,))
                    row = cursor.fetchone()
            else:
                row = connection.execute("SELECT * FROM image_tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["payload"] = json.loads(result.pop("payload_json")) if result.get("payload_json") else None
        result["result"] = json.loads(result.pop("result_json")) if result.get("result_json") else None
        return result

    def delete(self, task_id: str) -> None:
        with self._connect() as connection:
            if self.backend == "mysql":
                with connection.cursor() as cursor:
                    cursor.execute("DELETE FROM image_tasks WHERE id = %s", (task_id,))
                connection.commit()
                return

            connection.execute("DELETE FROM image_tasks WHERE id = ?", (task_id,))

    def _result_to_json(self, result: Any) -> Optional[str]:
        if result is None:
            return None
        if hasattr(result, "model_dump"):
            return json.dumps(result.model_dump(), ensure_ascii=False)
        return json.dumps(result, ensure_ascii=False)

    def _payload_to_json(self, payload: Any) -> Optional[str]:
        if payload is None:
            return None
        if hasattr(payload, "model_dump"):
            return json.dumps(payload.model_dump(), ensure_ascii=False)
        return json.dumps(payload, ensure_ascii=False)

    def _ensure_sqlite_column(self, connection: sqlite3.Connection, column: str, definition: str) -> None:
        rows = connection.execute("PRAGMA table_info(image_tasks)").fetchall()
        if column not in {row[1] for row in rows}:
            connection.execute(f"ALTER TABLE image_tasks ADD COLUMN {column} {definition}")

    def _ensure_mysql_column(self, cursor: Any, column: str, definition: str) -> None:
        cursor.execute(
            """
            SELECT COUNT(*) AS count
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
                AND TABLE_NAME = 'image_tasks'
                AND COLUMN_NAME = %s
            """,
            (column,),
        )
        row = cursor.fetchone()
        if not row or int(row["count"]) == 0:
            cursor.execute(f"ALTER TABLE image_tasks ADD COLUMN {column} {definition}")

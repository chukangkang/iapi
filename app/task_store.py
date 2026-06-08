import json
import sqlite3
from pathlib import Path
from typing import Optional

from app.config import Settings


class ImageTaskMetadataStore:
    def __init__(self, settings: Settings):
        self.db_path = settings.task_db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
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
                    result_json TEXT,
                    error TEXT
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_image_tasks_status ON image_tasks(status)")

    def save(self, task: object) -> None:
        result_json = self._result_to_json(getattr(task, "result", None))
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO image_tasks (
                    id, status, created, updated, started, completed, worker_id, result_json, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status=excluded.status,
                    updated=excluded.updated,
                    started=excluded.started,
                    completed=excluded.completed,
                    worker_id=excluded.worker_id,
                    result_json=excluded.result_json,
                    error=excluded.error
                """,
                (
                    getattr(task, "id"),
                    getattr(task, "status"),
                    getattr(task, "created"),
                    getattr(task, "updated"),
                    getattr(task, "started"),
                    getattr(task, "completed"),
                    getattr(task, "worker_id"),
                    result_json,
                    getattr(task, "error"),
                ),
            )

    def get(self, task_id: str) -> Optional[dict]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM image_tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["result"] = json.loads(result.pop("result_json")) if result.get("result_json") else None
        return result

    def delete(self, task_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM image_tasks WHERE id = ?", (task_id,))

    def _result_to_json(self, result: object) -> Optional[str]:
        if result is None:
            return None
        if hasattr(result, "model_dump"):
            return json.dumps(result.model_dump(), ensure_ascii=False)
        return json.dumps(result, ensure_ascii=False)

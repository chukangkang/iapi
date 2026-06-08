import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from PIL import Image

from app.config import Settings


TaskStatus = str


@dataclass
class ImageTask:
    id: str
    payload: object
    reference_image: Optional[Image.Image]
    status: TaskStatus = "queued"
    created: int = field(default_factory=lambda: int(time.time()))
    updated: int = field(default_factory=lambda: int(time.time()))
    started: Optional[int] = None
    completed: Optional[int] = None
    worker_id: Optional[int] = None
    result: Optional[object] = None
    error: Optional[str] = None


class ImageTaskManager:
    def __init__(
        self,
        settings: Settings,
        runner: Callable[[object, Optional[Image.Image]], Awaitable[object]],
    ):
        self.settings = settings
        self.runner = runner
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=settings.image_queue_maxsize)
        self.tasks: dict[str, ImageTask] = {}
        self._workers: list[asyncio.Task] = []
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if self._workers:
            return
        for worker_id in range(self.settings.image_worker_count):
            self._workers.append(asyncio.create_task(self._worker_loop(worker_id)))

    async def stop(self) -> None:
        for worker in self._workers:
            worker.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    async def submit(self, payload: object, reference_image: Optional[Image.Image]) -> ImageTask:
        task = ImageTask(id=f"img-{uuid.uuid4().hex}", payload=payload, reference_image=reference_image)
        async with self._lock:
            self.tasks[task.id] = task
        try:
            self.queue.put_nowait(task.id)
        except asyncio.QueueFull as exc:
            async with self._lock:
                self.tasks.pop(task.id, None)
            raise exc
        return task

    async def get(self, task_id: str) -> Optional[ImageTask]:
        async with self._lock:
            return self.tasks.get(task_id)

    async def _worker_loop(self, worker_id: int) -> None:
        while True:
            task_id = await self.queue.get()
            try:
                await self._run_task(worker_id, task_id)
            finally:
                self.queue.task_done()

    async def _run_task(self, worker_id: int, task_id: str) -> None:
        task = await self.get(task_id)
        if task is None:
            return

        task.status = "running"
        task.worker_id = worker_id
        task.started = int(time.time())
        task.updated = task.started
        try:
            task.result = await self.runner(task.payload, task.reference_image)
            task.status = "succeeded"
            task.completed = int(time.time())
            task.reference_image = None
        except Exception as exc:
            task.status = "failed"
            task.error = str(exc)
            task.completed = int(time.time())
            task.reference_image = None
        finally:
            task.updated = int(time.time())

import asyncio
import logging
import multiprocessing
import os
import signal
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from PIL import Image

from app.config import Settings
from app.image_utils import image_to_base64_png, string_to_image
from app.task_store import ImageTaskMetadataStore


TaskStatus = str
logger = logging.getLogger(__name__)


@contextmanager
def suppress_output():
    """临时抑制stdout/stderr,避免子进程输出干扰"""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def _task_execution_worker(payload: dict, reference_image_b64: Optional[str]) -> dict:
    """子进程执行函数,隔离OOM和崩溃"""
    import traceback
    
    # 在子进程中重新设置信号处理器
    def signal_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        sys.stderr.write(f"Child process received {sig_name}\n")
        sys.stderr.write(f"Stack: {''.join(traceback.format_stack(frame))}\n")
        sys.stderr.flush()
        os._exit(128 + signum)
    
    signal.signal(signal.SIGSEGV, signal_handler)
    signal.signal(signal.SIGBUS, signal_handler)
    signal.signal(signal.SIGFPE, signal_handler)
    
    try:
        from app.main import _run_task_payload
        from app.image_utils import string_to_image
        
        reference_image = string_to_image(reference_image_b64) if reference_image_b64 else None
        result = _run_task_payload(payload, reference_image)
        
        # 将结果转换为可序列化格式
        if isinstance(result, dict):
            return {"success": True, "data": result}
        else:
            return {"success": True, "data": {"result": result}}
    except MemoryError as e:
        return {"success": False, "error": "MemoryError", "message": str(e), "is_oom": True}
    except Exception as e:
        return {"success": False, "error": type(e).__name__, "message": str(e), "traceback": traceback.format_exc()}


async def _run_task_in_subprocess(self, payload: dict, reference_image: Optional[Image.Image]) -> dict:
    """在子进程中执行任务,隔离OOM和崩溃"""
    import base64
    from app.image_utils import image_to_base64_png
    
    reference_image_b64 = image_to_base64_png(reference_image) if reference_image else None
    
    loop = asyncio.get_event_loop()
    
    # 使用线程池运行子进程,避免阻塞事件循环
    result = await loop.run_in_executor(
        None,
        _task_execution_worker,
        payload,
        reference_image_b64
    )
    
    if not result["success"]:
        if result.get("is_oom"):
            raise MemoryError(result.get("message", "Out of memory"))
        else:
            error_msg = result.get("message", "Unknown error")
            raise RuntimeError(error_msg)
    
    return result["data"]


def check_gpu_memory() -> dict:
    """检查GPU显存使用情况"""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu,power.draw,power.limit', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(',')
            if len(parts) >= 2:
                used_mb = float(parts[0].strip())
                total_mb = float(parts[1].strip())
                return {
                    "used_mb": used_mb,
                    "total_mb": total_mb,
                    "used_percent": (used_mb / total_mb * 100) if total_mb > 0 else 0,
                    "utilization": parts[2].strip() if len(parts) > 2 else "N/A",
                    "power_draw": parts[3].strip() if len(parts) > 3 else "N/A",
                    "power_limit": parts[4].strip() if len(parts) > 4 else "N/A"
                }
    except Exception as e:
        logger.debug(f"Failed to check GPU memory: {e}")
    return None


def check_process_health() -> dict:
    """检查进程健康状态"""
    import psutil
    
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "status": process.status(),
            "memory_rss_mb": memory_info.rss / (1024 * 1024),
            "memory_vms_mb": memory_info.vms / (1024 * 1024),
            "cpu_percent": process.cpu_percent(),
            "num_threads": process.num_threads(),
            "is_running": process.is_running(),
            "is_zombie": process.status() == psutil.STATUS_ZOMBIE
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Unknown: {e}"}


@dataclass
class ImageTask:
    id: str
    payload: object
    reference_image: Optional[Image.Image]
    reference_image_data: Optional[str] = None
    status: TaskStatus = "queued"
    created: int = field(default_factory=lambda: int(time.time()))
    updated: int = field(default_factory=lambda: int(time.time()))
    started: Optional[int] = None
    completed: Optional[int] = None
    worker_id: Optional[int] = None
    worker_name: Optional[str] = None
    result: Optional[object] = None
    error: Optional[str] = None


class ImageTaskManager:
    def _log_task_health(self, worker_id: int, task_id: str, phase: str) -> None:
        """记录任务执行期间的健康状态"""
        try:
            gpu_info = check_gpu_memory()
            proc_info = check_process_health()
            
            if gpu_info:
                logger.info(
                    "Worker %s/%s task %s %s: GPU used=%.1f/%.1f MB (%.1f%%), util=%s",
                    self.settings.resolved_worker_name, worker_id, task_id, phase,
                    gpu_info["used_mb"], gpu_info["total_mb"], gpu_info["used_percent"],
                    gpu_info["utilization"]
                )
            
            if proc_info and "error" not in proc_info:
                logger.debug(
                    "Worker %s/%s task %s %s: PID=%d, RSS=%.1f MB, threads=%d, status=%s",
                    self.settings.resolved_worker_name, worker_id, task_id, phase,
                    proc_info["pid"], proc_info["memory_rss_mb"],
                    proc_info["num_threads"], proc_info["status"]
                )
        except Exception as e:
            logger.debug(f"Failed to log task health: {e}")

    def __init__(
        self,
        settings: Settings,
        runner: Callable[[object, Optional[Image.Image]], Awaitable[object]],
        payload_factory: Optional[Callable[[dict], object]] = None,
    ):
        self.settings = settings
        self.runner = runner
        self.payload_factory = payload_factory
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=settings.image_queue_maxsize)
        self.tasks: dict[str, ImageTask] = {}
        self.store = ImageTaskMetadataStore(settings)
        self.redis: Optional[Any] = None
        self._workers: list[asyncio.Task] = []
        self._maintenance_tasks: list[asyncio.Task] = []
        self._lock = asyncio.Lock()
        self._last_idle_log_at = 0.0

    async def start(self) -> None:
        if self.settings.service_role == "api":
            logger.info("Task workers disabled for SERVICE_ROLE=api")
            return
        if self._workers:
            return
        if self.settings.task_queue_backend == "redis":
            self.redis = self._redis_from_url()
            await self._recover_redis_tasks()
            if self.settings.redis_requeue_stale_enabled:
                self._maintenance_tasks.append(asyncio.create_task(self._requeue_stale_processing_loop()))
        for worker_id in range(self.settings.image_worker_count):
            self._workers.append(asyncio.create_task(self._worker_loop(worker_id)))
        logger.info(
            "Started %s image worker(s), queue_backend=%s",
            len(self._workers),
            self.settings.task_queue_backend,
        )

    async def stop(self) -> None:
        for worker in self._workers:
            worker.cancel()
        for task in self._maintenance_tasks:
            task.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        if self._maintenance_tasks:
            await asyncio.gather(*self._maintenance_tasks, return_exceptions=True)
        self._workers.clear()
        self._maintenance_tasks.clear()
        if self.redis is not None:
            await self.redis.aclose()
            self.redis = None

    async def submit(self, payload: object, reference_image: Optional[Image.Image]) -> ImageTask:
        reference_image_data = image_to_base64_png(reference_image) if reference_image is not None else None
        task = ImageTask(
            id=f"img-{uuid.uuid4().hex}",
            payload=payload,
            reference_image=reference_image,
            reference_image_data=reference_image_data,
        )
        if self.settings.task_queue_backend == "memory":
            async with self._lock:
                self.tasks[task.id] = task
        self.store.save(task)
        try:
            if self.settings.task_queue_backend == "redis":
                await self._enqueue_redis(task.id)
            else:
                self.queue.put_nowait(task.id)
        except Exception:
            async with self._lock:
                self.tasks.pop(task.id, None)
            self.store.delete(task.id)
            raise
        return task

    async def get(self, task_id: str) -> Optional[ImageTask]:
        async with self._lock:
            return self.tasks.get(task_id)

    async def queue_position(self, task_id: str) -> Optional[int]:
        if self.settings.task_queue_backend == "redis":
            redis = self.redis or self._redis_from_url()
            close_redis = self.redis is None
            try:
                position = await redis.lpos(self.settings.redis_queue_name, task_id)
                if position is None:
                    return None
                queue_size = await redis.llen(self.settings.redis_queue_name)
                return int(queue_size) - int(position)
            finally:
                if close_redis:
                    await redis.aclose()
        try:
            queued_task_ids = list(self.queue._queue)
        except AttributeError:
            return None
        try:
            return queued_task_ids.index(task_id) + 1
        except ValueError:
            return None

    async def _worker_loop(self, worker_id: int) -> None:
        logger.info("Worker %s is waiting for image tasks", worker_id)
        while True:
            task_id = None
            try:
                task_id = await self._next_task_id()
                if task_id is None:
                    await self._log_idle_redis_queue_state(worker_id)
                    continue
                logger.info("Worker %s picked task %s", worker_id, task_id)
                await self._run_task(worker_id, task_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Worker %s loop failed", worker_id)
                await self._reset_redis_after_error(exc)
            finally:
                if task_id is not None:
                    await self._ack_task_id(task_id)

    async def _run_task(self, worker_id: int, task_id: str) -> None:
        task = await self._load_task(task_id)
        if task is None:
            await self._fail_unloadable_task(worker_id, task_id)
            return

        task.status = "running"
        task.worker_id = worker_id
        task.worker_name = self.settings.resolved_worker_name
        task.started = int(time.time())
        task.updated = task.started
        self.store.save(task)
        logger.info("Worker %s/%s started task %s", self.settings.resolved_worker_name, worker_id, task_id)
        logger.info(f"Task details: id={task_id}, width={getattr(task.payload, 'width', 'N/A')}, height={getattr(task.payload, 'height', 'N/A')}, steps={getattr(task.payload, 'num_inference_steps', 'N/A')}")
        
        # 任务开始前检查
        self._log_task_health(worker_id, task_id, "start")
        
        heartbeat_task = asyncio.create_task(self._heartbeat_running_task(task))
        try:
            # 在子进程中执行任务,隔离OOM和崩溃
            task.result = await self._run_task_in_subprocess(task.payload, task.reference_image)
            task.status = "succeeded"
            task.completed = int(time.time())
            task.reference_image = None
            logger.info("Worker %s/%s completed task %s", self.settings.resolved_worker_name, worker_id, task_id)
            # 任务完成后检查
            self._log_task_health(worker_id, task_id, "end")
        except MemoryError as me:
            task.status = "failed"
            task.error = f"Out of memory (OOM): {me}"
            task.completed = int(time.time())
            task.reference_image = None
            logger.critical(f"Worker {worker_id}/{self.settings.resolved_worker_name} OOM on task {task_id}")
            logger.critical(f"Task payload: {task.payload}")
            logger.critical(f"Worker PID: {os.getpid()}")
            # 尝试获取GPU显存信息
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=5)
                logger.critical(f"GPU memory info: {result.stdout.strip()}")
            except Exception as gpu_error:
                logger.critical(f"Failed to get GPU info: {gpu_error}")
            raise  # 重新抛出以触发进程退出
        except Exception as exc:
            task.status = "failed"
            task.error = str(exc)
            task.completed = int(time.time())
            task.reference_image = None
            logger.exception("Worker %s/%s failed task %s", self.settings.resolved_worker_name, worker_id, task_id)
        finally:
            heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task, return_exceptions=True)
            task.updated = int(time.time())
            self.store.save(task)

    async def _heartbeat_running_task(self, task: ImageTask) -> None:
        while True:
            await asyncio.sleep(self.settings.task_running_heartbeat_interval)
            if task.status != "running":
                return
            task.updated = int(time.time())
            self.store.save(task)
            logger.debug("Heartbeat updated running task %s", task.id)

    async def _enqueue_redis(self, task_id: str) -> None:
        redis = self._redis_from_url()
        try:
            queue_size = await redis.llen(self.settings.redis_queue_name)
            if queue_size >= self.settings.image_queue_maxsize:
                raise asyncio.QueueFull
            await redis.lpush(self.settings.redis_queue_name, task_id)
            queue_size = await redis.llen(self.settings.redis_queue_name)
            processing_size = await redis.llen(self.settings.redis_processing_queue_name)
            logger.info(
                "Enqueued task %s to Redis queue=%s queue_size=%s processing_size=%s",
                task_id,
                self.settings.redis_queue_name,
                queue_size,
                processing_size,
            )
        finally:
            await redis.aclose()

    async def _next_task_id(self) -> Optional[str]:
        if self.settings.task_queue_backend == "redis":
            if self.redis is None:
                self.redis = self._redis_from_url()
            return await self.redis.brpoplpush(
                self.settings.redis_queue_name,
                self.settings.redis_processing_queue_name,
                timeout=self.settings.redis_block_timeout,
            )
        return await self.queue.get()

    async def _ack_task_id(self, task_id: str) -> None:
        if self.settings.task_queue_backend == "redis":
            if self.redis is not None:
                await self.redis.lrem(self.settings.redis_processing_queue_name, 1, task_id)
            return
        self.queue.task_done()

    async def _load_task(self, task_id: str) -> Optional[ImageTask]:
        task = await self.get(task_id)
        if task is not None:
            return task

        task_metadata = self.store.get(task_id)
        if task_metadata is None:
            logger.warning("Task %s was picked from queue but does not exist in metadata store", task_id)
            return None
        if task_metadata.get("payload") is None:
            logger.warning("Task %s has no payload_json and cannot be executed", task_id)
            return None

        payload = task_metadata["payload"]
        if self.payload_factory is not None:
            payload = self.payload_factory(payload)
        reference_image_data = task_metadata.get("reference_image")
        reference_image = string_to_image(reference_image_data)
        return ImageTask(
            id=task_metadata["id"],
            payload=payload,
            reference_image=reference_image,
            reference_image_data=reference_image_data,
            status=task_metadata["status"],
            created=task_metadata["created"],
            updated=task_metadata["updated"],
            started=task_metadata.get("started"),
            completed=task_metadata.get("completed"),
            worker_id=task_metadata.get("worker_id"),
            worker_name=task_metadata.get("worker_name"),
            result=task_metadata.get("result"),
            error=task_metadata.get("error"),
        )

    async def _fail_unloadable_task(self, worker_id: int, task_id: str) -> None:
        task_metadata = self.store.get(task_id)
        if task_metadata is None:
            return
        failed_task = ImageTask(
            id=task_metadata["id"],
            payload=task_metadata.get("payload"),
            reference_image=None,
            reference_image_data=task_metadata.get("reference_image"),
            status="failed",
            created=task_metadata["created"],
            updated=int(time.time()),
            started=task_metadata.get("started") or int(time.time()),
            completed=int(time.time()),
            worker_id=worker_id,
            worker_name=self.settings.resolved_worker_name,
            result=task_metadata.get("result"),
            error="Task payload is missing; submit the task again after Redis/MySQL configuration is fixed.",
        )
        self.store.save(failed_task)
        logger.error("Worker %s marked unloadable task %s as failed", worker_id, task_id)

    async def _recover_redis_tasks(self) -> None:
        if self.redis is None:
            return
        task_rows = self.store.list_by_status(["queued", "running"], limit=self.settings.image_queue_maxsize)
        recovered = 0
        for task_row in task_rows:
            task_id = task_row["id"]
            if task_row.get("payload") is None:
                logger.warning("Skipping task %s during recovery because payload_json is missing", task_id)
                continue
            if await self._redis_task_exists(task_id):
                continue
            await self.redis.lpush(self.settings.redis_queue_name, task_id)
            recovered += 1
        if recovered:
            logger.info("Recovered %s queued/running task(s) into Redis queue", recovered)
        await self._log_redis_queue_state("Redis queue state after recovery")

    async def _log_redis_queue_state(self, message: str) -> None:
        if self.redis is None:
            return
        queue_size = await self.redis.llen(self.settings.redis_queue_name)
        processing_size = await self.redis.llen(self.settings.redis_processing_queue_name)
        logger.info(
            "%s: queue=%s queue_size=%s processing_queue=%s processing_size=%s",
            message,
            self.settings.redis_queue_name,
            queue_size,
            self.settings.redis_processing_queue_name,
            processing_size,
        )

    async def _log_idle_redis_queue_state(self, worker_id: int) -> None:
        now = time.time()
        if now - self._last_idle_log_at < 60:
            return
        self._last_idle_log_at = now
        await self._log_redis_queue_state("Worker %s is still waiting for tasks" % worker_id)

    async def _reset_redis_after_error(self, exc: Exception) -> None:
        if self.settings.task_queue_backend != "redis":
            return
        module_name = exc.__class__.__module__
        if not module_name.startswith("redis"):
            return
        if self.redis is not None:
            await self.redis.aclose()
            self.redis = None
        logger.info("Redis connection reset after %s; worker will reconnect", exc.__class__.__name__)

    async def _redis_task_exists(self, task_id: str) -> bool:
        if self.redis is None:
            return False
        queued = await self.redis.lpos(self.settings.redis_queue_name, task_id)
        if queued is not None:
            return True
        processing = await self.redis.lpos(self.settings.redis_processing_queue_name, task_id)
        return processing is not None

    async def _requeue_stale_processing_loop(self) -> None:
        while True:
            try:
                await self._requeue_stale_processing_tasks()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Redis processing recovery loop failed")
                await self._reset_redis_after_error(exc)
            await asyncio.sleep(self.settings.redis_requeue_interval)

    async def _requeue_stale_processing_tasks(self) -> None:
        if self.redis is None:
            self.redis = self._redis_from_url()
        updated_before = int(time.time()) - self.settings.redis_processing_timeout
        stale_tasks = self.store.list_stale_running(updated_before, limit=self.settings.image_queue_maxsize)
        requeued = 0
        for task_row in stale_tasks:
            task_id = task_row["id"]
            processing_position = await self.redis.lpos(self.settings.redis_processing_queue_name, task_id)
            if processing_position is None:
                continue
            removed = await self.redis.lrem(self.settings.redis_processing_queue_name, 1, task_id)
            if not removed:
                continue
            if await self.redis.lpos(self.settings.redis_queue_name, task_id) is None:
                await self.redis.lpush(self.settings.redis_queue_name, task_id)
            requeued += 1
        if requeued:
            logger.warning(
                "Requeued %s stale processing task(s), timeout=%ss",
                requeued,
                self.settings.redis_processing_timeout,
            )

    def _redis_from_url(self) -> Any:
        from redis.asyncio import Redis

        return Redis.from_url(
            self.settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=self.settings.redis_socket_connect_timeout,
            socket_timeout=max(self.settings.redis_socket_timeout, self.settings.redis_block_timeout + 5),
            health_check_interval=30,
        )


ImageTaskManager._run_task_in_subprocess = _run_task_in_subprocess

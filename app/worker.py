import asyncio
import logging
import multiprocessing
import os
import signal
import sys
import traceback
from contextlib import asynccontextmanager

from app.config import get_settings
from app.main import ImageGenerationRequest, _prepare_task_payload, _run_task_payload, _task_affinity_key
from app.tasks import ImageTaskManager


logger = logging.getLogger(__name__)


# 全局信号处理,捕获致命信号并记录日志
def setup_signal_handlers():
    """设置信号处理器,捕获OOM和致命错误"""
    def signal_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.critical(f"Received {sig_name} (signal {signum}), worker is terminating")
        logger.critical(f"Stack trace:\n{''.join(traceback.format_stack(frame))}")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(128 + signum)
    
    # 捕获致命信号
    signal.signal(signal.SIGSEGV, signal_handler)  # 段错误(内存访问违规)
    signal.signal(signal.SIGBUS, signal_handler)   # 总线错误(内存对齐问题)
    signal.signal(signal.SIGFPE, signal_handler)   # 浮点异常(可能由OOM触发)
    signal.signal(signal.SIGABRT, signal_handler)  # abort()调用
    
    # Windows特有信号
    if sys.platform == 'win32':
        signal.signal(signal.SIGBREAK, signal_handler)


@asynccontextmanager
async def worker_lifespan(manager: ImageTaskManager):
    """Worker生命周期管理,确保异常时有完整日志"""
    try:
        yield
    except asyncio.CancelledError:
        logger.info("Worker cancellation requested")
        raise
    except MemoryError as e:
        logger.critical(f"MemoryError (OOM) detected: {e}")
        logger.critical(f"Worker process PID: {os.getpid()}")
        logger.critical(f"Worker process PPID: {os.getppid()}")
        # 尝试获取GPU显存信息
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            logger.critical(f"GPU memory info: {result.stdout.strip()}")
        except Exception as gpu_error:
            logger.critical(f"Failed to get GPU info: {gpu_error}")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    except Exception as e:
        logger.critical(f"Unexpected fatal error in worker: {e}")
        logger.critical(f"Exception type: {type(e).__module__}.{type(e).__name__}")
        logger.critical(f"Worker process PID: {os.getpid()}")
        logger.critical(f"Worker process PPID: {os.getppid()}")
        logger.critical(f"Stack trace:\n{traceback.format_exc()}")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


async def main() -> None:
    # 设置信号处理器(在主进程和子进程都会执行)
    setup_signal_handlers()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stdout,
    )
    settings = get_settings()
    logger.info(
        "Starting image worker service: service_role=%s queue_backend=%s db_backend=%s redis_url=%s queue=%s",
        settings.service_role,
        settings.task_queue_backend,
        settings.task_db_backend,
        settings.redis_url,
        settings.redis_queue_name,
    )
    logger.info(f"Worker process PID: {os.getpid()}, PPID: {os.getppid()}")
    logger.info(f"Worker name: {settings.resolved_worker_name}")
    logger.info(f"Worker count: {settings.image_worker_count}")
    
    if settings.service_role not in {"worker", "all"}:
        raise RuntimeError("Worker service requires SERVICE_ROLE=worker or SERVICE_ROLE=all.")
    if settings.task_queue_backend != "redis":
        raise RuntimeError("Worker service requires TASK_QUEUE_BACKEND=redis.")
    
    manager = ImageTaskManager(settings, _run_task_payload, ImageGenerationRequest.model_validate, _prepare_task_payload, _task_affinity_key)
    
    async with worker_lifespan(manager):
        await manager.start()
        logger.info("Image worker service is running. Press Ctrl+C to stop.")
        try:
            await asyncio.Event().wait()
        finally:
            logger.info("Stopping image worker service")
            await manager.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.critical(f"Failed to start worker: {e}")
        logger.critical(f"Stack trace:\n{traceback.format_exc()}")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
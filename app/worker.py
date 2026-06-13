import asyncio
import logging
import sys

from app.config import get_settings
from app.main import ImageGenerationRequest, _run_task_payload
from app.tasks import ImageTaskManager


logger = logging.getLogger(__name__)


async def main() -> None:
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
    if settings.service_role not in {"worker", "all"}:
        raise RuntimeError("Worker service requires SERVICE_ROLE=worker or SERVICE_ROLE=all.")
    if settings.task_queue_backend != "redis":
        raise RuntimeError("Worker service requires TASK_QUEUE_BACKEND=redis.")
    manager = ImageTaskManager(settings, _run_task_payload, ImageGenerationRequest.model_validate)
    await manager.start()
    logger.info("Image worker service is running. Press Ctrl+C to stop.")
    try:
        await asyncio.Event().wait()
    finally:
        logger.info("Stopping image worker service")
        await manager.stop()


if __name__ == "__main__":
    asyncio.run(main())
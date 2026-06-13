import asyncio

from app.config import get_settings
from app.main import ImageGenerationRequest, _run_task_payload
from app.tasks import ImageTaskManager


async def main() -> None:
    settings = get_settings()
    manager = ImageTaskManager(settings, _run_task_payload, ImageGenerationRequest.model_validate)
    await manager.start()
    try:
        await asyncio.Event().wait()
    finally:
        await manager.stop()


if __name__ == "__main__":
    asyncio.run(main())
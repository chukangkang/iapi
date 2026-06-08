import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image

from app.config import Settings
from app.image_utils import public_image_url, save_png


@dataclass
class StoredImage:
    url: str
    object_key: Optional[str] = None
    local_path: Optional[Path] = None


class ImageStorage:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._bucket = None

    def store_png(self, image: Image.Image, filename: str) -> StoredImage:
        if self._oss_is_configured():
            return self._upload_to_oss(image, filename)

        local_path = save_png(image, self.settings.output_dir, filename)
        return StoredImage(
            url=public_image_url(self.settings.normalized_public_base_url, filename),
            local_path=local_path,
        )

    def _upload_to_oss(self, image: Image.Image, filename: str) -> StoredImage:
        object_key = self._object_key(filename)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        headers = {
            "Content-Type": "image/png",
            "Cache-Control": "public, max-age=31536000, immutable",
        }
        expires_at = int(time.time()) + self.settings.oss_retention_days * 24 * 60 * 60
        headers["Expires"] = self._http_date(expires_at)

        bucket = self._get_bucket()
        bucket.put_object(object_key, buffer.getvalue(), headers=headers)

        return StoredImage(url=self._public_url(object_key), object_key=object_key)

    def _get_bucket(self):
        if self._bucket is not None:
            return self._bucket

        import oss2

        auth = oss2.Auth(self.settings.oss_access_key_id, self.settings.oss_access_key_secret)
        endpoint = self.settings.oss_endpoint
        if not endpoint.startswith(("http://", "https://")):
            endpoint = f"https://{endpoint}"
        self._bucket = oss2.Bucket(auth, endpoint, self.settings.oss_bucket)
        return self._bucket

    def _oss_is_configured(self) -> bool:
        return bool(self.settings.oss_enabled) and not self.settings.oss_access_key_id.startswith("your_")

    def _object_key(self, filename: str) -> str:
        prefix = self.settings.oss_object_prefix.strip("/")
        return f"{prefix}/{filename}" if prefix else filename

    def _public_url(self, object_key: str) -> str:
        if self.settings.normalized_oss_public_base_url:
            return f"{self.settings.normalized_oss_public_base_url}/{object_key}"

        endpoint = self.settings.oss_endpoint
        if endpoint.startswith("https://"):
            endpoint = endpoint.removeprefix("https://")
        elif endpoint.startswith("http://"):
            endpoint = endpoint.removeprefix("http://")
        return f"https://{self.settings.oss_bucket}.{endpoint}/{object_key}"

    @staticmethod
    def _http_date(timestamp: int) -> str:
        from email.utils import formatdate

        return formatdate(timestamp, usegmt=True)

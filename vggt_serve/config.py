from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VGGT_SERVE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    service_name: str = "vggt-serve"
    host: str = "0.0.0.0"
    port: int = 8000
    data_root: Path = Field(default_factory=lambda: Path("data/runs"))
    model_id: str = "facebook/VGGT-1B"
    checkpoint_path: Path | None = None
    square_image_size: int = 518
    default_depth_conf_threshold: float = 5.0
    max_images: int = 16
    max_upload_bytes_per_file: int = 25 * 1024 * 1024
    max_upload_bytes_total: int = 250 * 1024 * 1024
    max_point_cloud_points: int = 500_000

    def ensure_directories(self) -> None:
        self.data_root.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings


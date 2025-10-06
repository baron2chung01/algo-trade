"""Application configuration helpers."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataPaths(BaseModel):
    """Filesystem locations for cached datasets."""

    raw: Path = Field(default=Path("data/raw"))
    cache: Path = Field(default=Path("data/cache"))

    def ensure(self) -> None:
        """Create directories if they do not exist."""

        for path in (self.raw, self.cache):
            path.mkdir(parents=True, exist_ok=True)


class AppSettings(BaseSettings):
    """Project-wide settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        populate_by_name=True,
    )

    polygon_api_key: str | None = Field(default=None, alias="POLYGON_API_KEY")
    ibkr_host: str = Field(default="127.0.0.1", alias="IBKR_HOST")
    ibkr_port: int = Field(default=7497, alias="IBKR_PORT")
    ibkr_client_id: int = Field(default=101, alias="IBKR_CLIENT_ID")
    data_paths: DataPaths = Field(default_factory=DataPaths)

    def require_polygon_key(self) -> str:
        """Return the Polygon API key or raise a helpful error."""

        if not self.polygon_api_key:
            raise RuntimeError("Missing Polygon API key. Set POLYGON_API_KEY in your environment or .env file.")
        return self.polygon_api_key

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


class QuantConnectCredentials(BaseModel):
    """Credentials required for QuantConnect data API access."""

    user_id: str
    api_token: str
    organization_id: str | None = None


class AppSettings(BaseSettings):
    """Project-wide settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        populate_by_name=True,
    )

    quantconnect_user_id: str | None = Field(
        default=None, alias="QUANTCONNECT_USER_ID")
    quantconnect_api_token: str | None = Field(
        default=None, alias="QUANTCONNECT_API_TOKEN")
    quantconnect_organization_id: str | None = Field(
        default=None, alias="QUANTCONNECT_ORGANIZATION_ID")
    polygon_api_key: str | None = Field(default=None, alias="POLYGON_API_KEY")
    ibkr_host: str = Field(default="127.0.0.1", alias="IBKR_HOST")
    ibkr_port: int = Field(default=7497, alias="IBKR_PORT")
    ibkr_client_id: int = Field(default=101, alias="IBKR_CLIENT_ID")
    data_paths: DataPaths = Field(default_factory=DataPaths)

    def require_quantconnect_credentials(self) -> QuantConnectCredentials:
        """Return QuantConnect credentials or raise a helpful error."""

        if not self.quantconnect_user_id or not self.quantconnect_api_token:
            raise RuntimeError(
                "Missing QuantConnect credentials. Set QUANTCONNECT_USER_ID and QUANTCONNECT_API_TOKEN "
                "in your environment or .env file."
            )
        return QuantConnectCredentials(
            user_id=self.quantconnect_user_id,
            api_token=self.quantconnect_api_token,
            organization_id=self.quantconnect_organization_id,
        )

    def require_polygon_api_key(self) -> str:
        """Return Polygon API key or raise a helpful error."""

        if not self.polygon_api_key:
            raise RuntimeError(
                "Missing Polygon API key. Set POLYGON_API_KEY in your environment or .env file."
            )
        return self.polygon_api_key

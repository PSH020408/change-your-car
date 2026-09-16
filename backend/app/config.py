"""Central runtime configuration (12-factor, env-driven)."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Data layer
    data_root: Path = Path("../data")
    fastf1_cache_dir: Path = Path("../data/cache")
    model_registry_dir: Path = Path("../data/artifacts")

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_cors_origins: str = "http://localhost:3000"
    log_level: str = "INFO"
    # Built HUD (next build && output: "export"). Served at "/" when the directory exists,
    # so one container = one origin = no CORS in production. Absent in development.
    static_dir: Path = Path("../frontend/out")

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.api_cors_origins.split(",") if o.strip()]

    @property
    def bronze(self) -> Path:
        return self.data_root / "bronze"

    @property
    def silver(self) -> Path:
        return self.data_root / "silver"

    @property
    def gold(self) -> Path:
        return self.data_root / "gold"


@lru_cache
def get_settings() -> Settings:
    return Settings()

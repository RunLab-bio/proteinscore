"""
ProteinScore Configuration

Centralized configuration management with environment variable support.
Follows Python Backend Best Practices (PBP) Section 1.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from proteinscore.models import ScoringWeights


class Config(BaseSettings):
    """
    ProteinScore configuration with environment variable support.

    Environment variables are prefixed with PROTEINSCORE_ or RUNLAB_.
    Includes production validation for critical settings.
    """

    model_config = SettingsConfigDict(
        env_prefix="PROTEINSCORE_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="proteinscore", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Deployment environment",
    )
    debug: bool = Field(default=False, description="Enable debug mode")

    # API Configuration
    api_key: str | None = Field(
        default=None,
        description="RunLab API key (also accepts RUNLAB_API_KEY)",
    )
    api_base_url: str = Field(
        default="https://api.runlab.bio",
        description="Base URL for the RunLab API",
    )
    api_timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="API request timeout in seconds",
    )

    # Scoring Configuration
    default_weights: ScoringWeights = Field(
        default_factory=ScoringWeights,
        description="Default scoring weights",
    )

    # HLA Configuration
    hla_population: Literal["global", "european", "african", "asian", "hispanic"] = Field(
        default="global",
        description="Population for HLA allele selection",
    )
    hla_alleles: list[str] | None = Field(
        default=None,
        description="Specific HLA alleles to use (overrides population)",
    )

    # Cache Configuration
    cache_enabled: bool = Field(
        default=True,
        description="Enable result caching",
    )
    cache_dir: Path = Field(
        default_factory=lambda: Path.home() / ".proteinscore" / "cache",
        description="Directory for caching results",
    )
    cache_ttl: int = Field(
        default=86400,
        ge=0,
        description="Cache TTL in seconds (0 = never expire)",
    )

    # Execution Configuration
    local_only: bool = Field(
        default=False,
        description="Skip API calls, use local estimation for immunogenicity",
    )
    parallel: bool = Field(
        default=True,
        description="Enable parallel processing for batch operations",
    )
    max_workers: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Maximum worker threads for parallel processing",
    )

    # Logging Configuration
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )

    # Server Configuration (for API mode)
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port")

    def model_post_init(self, __context: object) -> None:
        """Post-initialization: check for RUNLAB_API_KEY if api_key not set."""
        if self.api_key is None:
            self.api_key = os.getenv("RUNLAB_API_KEY")

        # Ensure cache directory exists
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @model_validator(mode="after")
    def validate_production_settings(self) -> Config:
        """Ensure required settings are present in production."""
        if self.environment == "production":
            errors = []
            if self.debug:
                errors.append("DEBUG must be False in production")
            if self.log_level == "DEBUG":
                errors.append("LOG_LEVEL should not be DEBUG in production")
            if errors:
                raise ValueError(f"Production configuration errors: {', '.join(errors)}")
        return self

    @field_validator("hla_alleles", mode="before")
    @classmethod
    def parse_hla_alleles(cls, v: str | list[str] | None) -> list[str] | None:
        """Parse HLA alleles from string or list."""
        if v is None:
            return None
        if isinstance(v, str):
            return [a.strip() for a in v.split(",") if a.strip()]
        return v

    @property
    def is_authenticated(self) -> bool:
        """Check if an API key is configured."""
        return self.api_key is not None and self.api_key.startswith("rl_")

    @property
    def tier(self) -> str:
        """Infer the API tier from authentication status."""
        if not self.is_authenticated:
            return "anonymous"
        return "authenticated"

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == "development"


class PredictorConfig(BaseModel):
    """Configuration for individual predictors."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(default=True, description="Whether this predictor is enabled")
    timeout: int = Field(default=60, ge=1, description="Timeout for this predictor")
    fallback_on_error: bool = Field(
        default=True, description="Use fallback score on error"
    )
    fallback_score: float = Field(
        default=50.0, ge=0, le=100, description="Score to use on fallback"
    )


class StabilityConfig(PredictorConfig):
    """Configuration specific to stability prediction."""

    method: Literal["sequence_based", "foldx", "rosetta"] = Field(
        default="sequence_based",
        description="Stability prediction method",
    )
    foldx_path: Path | None = Field(
        default=None,
        description="Path to FoldX executable",
    )


class SolubilityConfig(PredictorConfig):
    """Configuration specific to solubility prediction."""

    method: Literal["sequence_based", "camsol", "protein_sol"] = Field(
        default="sequence_based",
        description="Solubility prediction method",
    )
    window_size: int = Field(
        default=7,
        ge=3,
        le=21,
        description="Window size for sequence-based prediction",
    )


class AggregationConfig(PredictorConfig):
    """Configuration specific to aggregation prediction."""

    method: Literal["sequence_based", "aggrescan3d", "tango"] = Field(
        default="sequence_based",
        description="Aggregation prediction method",
    )
    apr_threshold: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Threshold for identifying aggregation-prone regions",
    )


class ImmunogenicityConfig(PredictorConfig):
    """Configuration specific to immunogenicity prediction."""

    method: Literal["rip_api", "local_estimate"] = Field(
        default="rip_api",
        description="Immunogenicity prediction method",
    )
    peptide_lengths: list[int] = Field(
        default_factory=lambda: [9],
        description="Peptide lengths to scan",
    )
    threshold_percentile: float = Field(
        default=2.0,
        ge=0,
        le=100,
        description="Percentile threshold for epitope reporting",
    )


@lru_cache
def get_default_config() -> Config:
    """Get the default configuration (cached)."""
    return Config()


# Re-export ScoringWeights for convenience
__all__ = [
    "Config",
    "PredictorConfig",
    "StabilityConfig",
    "SolubilityConfig",
    "AggregationConfig",
    "ImmunogenicityConfig",
    "ScoringWeights",
    "get_default_config",
]

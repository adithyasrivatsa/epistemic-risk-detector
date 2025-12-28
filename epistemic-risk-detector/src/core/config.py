"""Configuration management with strict typing and validation."""

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: Literal["openai", "anthropic", "local", "ollama"] = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(4096, gt=0)
    api_key: Optional[str] = None  # Falls back to env var
    base_url: Optional[str] = None  # For local/ollama


class RetrievalConfig(BaseModel):
    """Evidence retrieval configuration."""

    chunk_size: int = Field(512, gt=0)
    chunk_overlap: int = Field(64, ge=0)
    top_k: int = Field(5, gt=0)
    similarity_threshold: float = Field(0.3, ge=0.0, le=1.0)
    embedding_model: str = "all-MiniLM-L6-v2"
    db_path: str = ".hallucination_debugger/evidence.db"


class CalibrationConfig(BaseModel):
    """Confidence calibration configuration."""

    no_evidence_penalty: float = Field(0.4, ge=0.0, le=1.0)
    contradiction_penalty: float = Field(0.6, ge=0.0, le=1.0)
    vague_language_penalty: float = Field(0.2, ge=0.0, le=1.0)
    weak_evidence_penalty: float = Field(0.15, ge=0.0, le=1.0)


class VerdictConfig(BaseModel):
    """Verdict computation configuration."""

    hallucination_threshold: float = Field(0.3, ge=0.0, le=1.0)
    grounded_threshold: float = Field(0.7, ge=0.0, le=1.0)
    confidence_weight: float = Field(0.4, ge=0.0, le=1.0)
    evidence_weight: float = Field(0.6, ge=0.0, le=1.0)


class ExtractionConfig(BaseModel):
    """Claim extraction configuration."""

    max_claims: int = Field(50, gt=0)
    min_claim_length: int = Field(10, gt=0)
    max_retries: int = Field(3, gt=0)
    include_opinions: bool = False


class Config(BaseModel):
    """Root configuration."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    verdict: VerdictConfig = Field(default_factory=VerdictConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls(**data)

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration with environment variable overrides."""
        import os

        config = cls()

        # Override LLM settings from env
        if api_key := os.getenv("OPENAI_API_KEY"):
            config.llm.api_key = api_key
        if api_key := os.getenv("ANTHROPIC_API_KEY"):
            config.llm.api_key = api_key
            config.llm.provider = "anthropic"

        return config

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration from file or defaults."""
    if path:
        return Config.from_yaml(path)

    # Check default locations
    default_paths = [
        Path("config.yaml"),
        Path("config.yml"),
        Path(".hallucination_debugger/config.yaml"),
    ]

    for p in default_paths:
        if p.exists():
            return Config.from_yaml(p)

    return Config.from_env()

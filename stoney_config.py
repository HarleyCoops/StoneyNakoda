"""Central configuration for Stoney Nakoda pipeline model choices."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from dotenv import load_dotenv


SUPPORTED_FINETUNE_MODELS = (
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-nano-2025-04-14",
)

DEFAULT_OPENAI_CHAT_MODEL = "gpt-4.1-mini-2025-04-14"
DEFAULT_OPENAI_FINETUNE_MODEL = "gpt-4.1-mini-2025-04-14"
DEFAULT_OPENAI_EXTRACTION_MODEL = "gpt-5"
DEFAULT_OPENAI_TASK_MODEL = "gpt-5"
DEFAULT_GEMINI_QA_MODEL = "gemini-2.5-pro"


class ConfigError(ValueError):
    """Raised when repository configuration is invalid."""


@dataclass(frozen=True)
class StoneyConfig:
    """Resolved model configuration for each pipeline purpose."""

    openai_chat_model: str
    openai_finetune_model: str
    openai_extraction_model: str
    openai_task_model: str
    gemini_qa_model: str
    legacy_openai_model_present: bool = False

    def diagnostics(self) -> str:
        """Return safe configuration diagnostics without exposing secrets."""

        lines = [
            "Stoney Nakoda configuration:",
            f"- OPENAI_CHAT_MODEL: {self.openai_chat_model}",
            f"- OPENAI_FINETUNE_MODEL: {self.openai_finetune_model}",
            f"- OPENAI_EXTRACTION_MODEL: {self.openai_extraction_model}",
            f"- OPENAI_TASK_MODEL: {self.openai_task_model}",
            f"- GEMINI_QA_MODEL: {self.gemini_qa_model}",
            "- Supported fine-tune models: " + ", ".join(SUPPORTED_FINETUNE_MODELS),
        ]
        if self.legacy_openai_model_present:
            lines.append("- OPENAI_MODEL: set but ignored; use purpose-specific variables instead")
        return "\n".join(lines)


def validate_finetune_model(model: str) -> None:
    """Raise when the fine-tune model is not in the supported allowlist.

    Args:
        model: OpenAI model name intended for supervised fine-tuning.

    Raises:
        ConfigError: When the model is not supported by the repository allowlist.
    """

    if model not in SUPPORTED_FINETUNE_MODELS:
        allowed = ", ".join(SUPPORTED_FINETUNE_MODELS)
        raise ConfigError(
            f"Unsupported OPENAI_FINETUNE_MODEL '{model}'. "
            f"Choose one of: {allowed}. Update stoney_config.py if OpenAI support changes."
        )


def load_stoney_config(
    env: Mapping[str, str] | None = None,
    *,
    load_dotenv_file: bool = True,
    validate_finetune: bool = True,
) -> StoneyConfig:
    """Resolve model configuration from environment variables.

    Args:
        env: Optional environment mapping. Defaults to `os.environ`.
        load_dotenv_file: Whether to load a local `.env` file first.
        validate_finetune: Whether to enforce the fine-tune model allowlist.

    Returns:
        Resolved, purpose-specific model configuration.

    Raises:
        ConfigError: When the fine-tune model is unsupported.
    """

    if load_dotenv_file:
        load_dotenv()
    values = os.environ if env is None else env
    config = StoneyConfig(
        openai_chat_model=values.get("OPENAI_CHAT_MODEL", DEFAULT_OPENAI_CHAT_MODEL),
        openai_finetune_model=values.get("OPENAI_FINETUNE_MODEL", DEFAULT_OPENAI_FINETUNE_MODEL),
        openai_extraction_model=values.get("OPENAI_EXTRACTION_MODEL", DEFAULT_OPENAI_EXTRACTION_MODEL),
        openai_task_model=values.get("OPENAI_TASK_MODEL", DEFAULT_OPENAI_TASK_MODEL),
        gemini_qa_model=values.get("GEMINI_QA_MODEL", DEFAULT_GEMINI_QA_MODEL),
        legacy_openai_model_present=bool(values.get("OPENAI_MODEL")),
    )
    if validate_finetune:
        validate_finetune_model(config.openai_finetune_model)
    return config

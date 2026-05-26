from __future__ import annotations

import pytest

from stoney_config import (
    DEFAULT_OPENAI_FINETUNE_MODEL,
    ConfigError,
    SUPPORTED_FINETUNE_MODELS,
    load_stoney_config,
)


def test_default_config_uses_supported_finetune_model() -> None:
    config = load_stoney_config({}, load_dotenv_file=False)

    assert config.openai_finetune_model == DEFAULT_OPENAI_FINETUNE_MODEL
    assert config.openai_finetune_model in SUPPORTED_FINETUNE_MODELS


def test_openai_model_is_not_finetune_fallback() -> None:
    config = load_stoney_config({"OPENAI_MODEL": "gpt-5"}, load_dotenv_file=False)

    assert config.openai_finetune_model == DEFAULT_OPENAI_FINETUNE_MODEL
    assert config.openai_finetune_model != "gpt-5"
    assert config.legacy_openai_model_present is True


def test_unsupported_finetune_model_fails_before_api_call() -> None:
    with pytest.raises(ConfigError, match="Unsupported OPENAI_FINETUNE_MODEL"):
        load_stoney_config(
            {"OPENAI_FINETUNE_MODEL": "gpt-5"},
            load_dotenv_file=False,
        )


def test_purpose_specific_models_are_resolved() -> None:
    config = load_stoney_config(
        {
            "OPENAI_CHAT_MODEL": "chat-model",
            "OPENAI_FINETUNE_MODEL": "gpt-4.1-nano-2025-04-14",
            "OPENAI_EXTRACTION_MODEL": "extract-model",
            "OPENAI_TASK_MODEL": "task-model",
            "GEMINI_QA_MODEL": "gemini-model",
        },
        load_dotenv_file=False,
    )

    assert config.openai_chat_model == "chat-model"
    assert config.openai_finetune_model == "gpt-4.1-nano-2025-04-14"
    assert config.openai_extraction_model == "extract-model"
    assert config.openai_task_model == "task-model"
    assert config.gemini_qa_model == "gemini-model"


def test_diagnostics_do_not_expose_secrets() -> None:
    config = load_stoney_config(
        {
            "OPENAI_API_KEY": "sk-secret-value",
            "OPENAI_FINETUNE_MODEL": "gpt-4.1-mini-2025-04-14",
        },
        load_dotenv_file=False,
    )

    assert "sk-secret-value" not in config.diagnostics()

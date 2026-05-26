"""Shared JSON-schema client wrapper for grammar pipeline LLM calls."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import jsonschema

logger = logging.getLogger(__name__)

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas"
TRUE_VALUES = {"1", "true", "yes", "y", "on"}


class LLMJsonError(ValueError):
    """Raised when an LLM JSON response is malformed or schema-invalid."""


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment flag."""

    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in TRUE_VALUES


def load_json_schema(name: str) -> dict[str, Any]:
    """Load a JSON schema from the repository schemas directory."""

    path = SCHEMA_DIR / name
    with path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    if not isinstance(schema, dict):
        raise LLMJsonError(f"Schema must be an object: {path}")
    jsonschema.Draft202012Validator.check_schema(schema)
    return schema


def _response_diagnostics(response: Any) -> str:
    """Return compact response diagnostics for logs/errors."""

    try:
        choice = response.choices[0]
    except (AttributeError, IndexError, TypeError):
        return "finish_reason=<unavailable>, refusal=<unavailable>"
    message = getattr(choice, "message", None)
    refusal = getattr(message, "refusal", None) if message is not None else None
    finish_reason = getattr(choice, "finish_reason", None)
    return f"finish_reason={finish_reason!r}, refusal={refusal!r}"


def _extract_message_content(response: Any) -> str:
    """Extract text content from a Chat Completions response."""

    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise LLMJsonError("Unable to read model response content") from exc
    if not isinstance(content, str):
        raise LLMJsonError(f"Expected string response content, got {type(content)}")
    if not content.strip():
        raise LLMJsonError(f"Model returned empty content ({_response_diagnostics(response)})")
    return content


def parse_json_object(content: str) -> dict[str, Any]:
    """Parse strict JSON object text."""

    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    if not cleaned.startswith("{"):
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            cleaned = match.group()

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMJsonError(f"Model returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise LLMJsonError("Model JSON response must be an object")
    return payload


def validate_json_payload(payload: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    """Validate a parsed JSON payload against a JSON schema."""

    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda err: list(err.path))
    if errors:
        details = "\n".join(f"- {'/'.join(map(str, err.path)) or '<root>'}: {err.message}" for err in errors)
        raise LLMJsonError(f"Model JSON response failed schema validation:\n{details}")


class LLMJsonClient:
    """Call an OpenAI chat model and validate the JSON response against a schema."""

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        allow_json_fallback: bool = False,
    ) -> None:
        self.client = client
        self.model = model
        self.allow_json_fallback = allow_json_fallback

    def create(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        schema: Mapping[str, Any],
        schema_name: str,
        max_output_tokens: int,
        response_format_mode: str = "json_schema",
    ) -> dict[str, Any]:
        """Create and validate a schema-constrained JSON response.

        Args:
            messages: Chat Completions messages.
            schema: JSON Schema that the model response must satisfy.
            schema_name: Name sent to the OpenAI structured-output API.
            max_output_tokens: Maximum completion tokens.
            response_format_mode: `json_schema`, `json_object`, or `json_prompt`.

        Returns:
            Parsed and schema-validated JSON object.

        Raises:
            LLMJsonError: When the model output is not valid schema-constrained JSON.
        """

        jsonschema.Draft202012Validator.check_schema(schema)
        request: dict[str, Any] = {
            "model": self.model,
            "max_completion_tokens": max_output_tokens,
            "messages": list(messages),
        }
        if response_format_mode == "json_schema":
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": dict(schema),
                },
            }
        elif response_format_mode == "json_object":
            request["response_format"] = {"type": "json_object"}
        elif response_format_mode != "json_prompt":
            raise LLMJsonError(
                "response_format_mode must be one of: json_schema, json_object, json_prompt"
            )

        try:
            response = self.client.chat.completions.create(**request)
        except Exception:
            if not self.allow_json_fallback:
                raise
            logger.warning(
                "Structured JSON schema call failed; retrying with strict JSON parsing "
                "because STONEY_ALLOW_JSON_FALLBACK is enabled.",
                exc_info=True,
            )
            response = self.client.chat.completions.create(
                model=self.model,
                max_completion_tokens=max_output_tokens,
                response_format={"type": "json_object"},
                messages=list(messages),
            )

        payload = parse_json_object(_extract_message_content(response))
        validate_json_payload(payload, schema)
        return payload

"""Shared JSON-schema client wrapper for grammar pipeline LLM calls."""

from __future__ import annotations

import json
import logging
import os
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


def _extract_message_content(response: Any) -> str:
    """Extract text content from a Chat Completions response."""

    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise LLMJsonError("Unable to read model response content") from exc
    if not isinstance(content, str):
        raise LLMJsonError(f"Expected string response content, got {type(content)}")
    return content


def parse_json_object(content: str) -> dict[str, Any]:
    """Parse strict JSON object text."""

    try:
        payload = json.loads(content)
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
    ) -> dict[str, Any]:
        """Create and validate a schema-constrained JSON response.

        Args:
            messages: Chat Completions messages.
            schema: JSON Schema that the model response must satisfy.
            schema_name: Name sent to the OpenAI structured-output API.
            max_output_tokens: Maximum completion tokens.

        Returns:
            Parsed and schema-validated JSON object.

        Raises:
            LLMJsonError: When the model output is not valid schema-constrained JSON.
        """

        jsonschema.Draft202012Validator.check_schema(schema)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_completion_tokens=max_output_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": True,
                        "schema": dict(schema),
                    },
                },
                messages=list(messages),
            )
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

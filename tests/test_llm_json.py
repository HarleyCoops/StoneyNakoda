from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from stoney_rl_grammar.llm_json import LLMJsonClient, LLMJsonError, load_json_schema


class FakeCompletions:
    def __init__(self, payloads: list[dict[str, Any]], fail_first: bool = False) -> None:
        self.payloads = payloads
        self.fail_first = fail_first
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.fail_first and len(self.calls) == 1:
            raise RuntimeError("structured output unsupported")
        payload = self.payloads[min(len(self.calls) - 1, len(self.payloads) - 1)]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(payload)),
                )
            ]
        )


class FakeClient:
    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


def test_valid_rule_extraction_payload_uses_json_schema_response_format() -> None:
    schema = load_json_schema("grammar_rule.schema.json")
    completions = FakeCompletions(
        [
            {
                "rules": [
                    {
                        "title": "Plural suffix",
                        "description": "A suffix marks plural nouns.",
                        "category": "morphology",
                        "stoney_examples": ["example"],
                        "english_examples": ["examples"],
                        "verification_hint": "check suffix",
                        "confidence": 0.8,
                        "page_number": 1,
                        "chunk_id": "page_001_chunk_00",
                    }
                ]
            }
        ]
    )
    client = LLMJsonClient(FakeClient(completions), "test-model")

    payload = client.create(
        messages=[{"role": "user", "content": "extract"}],
        schema=schema,
        schema_name="grammar_rule_extraction",
        max_output_tokens=2000,
    )

    assert payload["rules"][0]["title"] == "Plural suffix"
    assert completions.calls[0]["response_format"]["type"] == "json_schema"


def test_malformed_rule_extraction_payload_is_rejected() -> None:
    schema = load_json_schema("grammar_rule.schema.json")
    completions = FakeCompletions([{"rules": [{"title": "Missing required fields"}]}])
    client = LLMJsonClient(FakeClient(completions), "test-model")

    with pytest.raises(LLMJsonError, match="schema validation"):
        client.create(
            messages=[{"role": "user", "content": "extract"}],
            schema=schema,
            schema_name="grammar_rule_extraction",
            max_output_tokens=2000,
        )


def test_valid_rl_task_payload_validates() -> None:
    schema = load_json_schema("rl_task.schema.json")
    completions = FakeCompletions(
        [
            {
                "tasks": [
                    {
                        "task_type": "morphology",
                        "prompt": "Apply the rule.",
                        "ideal_answer": "Expected answer.",
                        "verification_pattern": "Expected",
                        "hints": ["Use the suffix."],
                        "difficulty": "easy",
                    }
                ]
            }
        ]
    )
    client = LLMJsonClient(FakeClient(completions), "test-model")

    payload = client.create(
        messages=[{"role": "user", "content": "generate"}],
        schema=schema,
        schema_name="rl_task_generation",
        max_output_tokens=2000,
    )

    assert payload["tasks"][0]["task_type"] == "morphology"


def test_json_fallback_is_only_used_when_enabled() -> None:
    schema = load_json_schema("rl_task.schema.json")
    payload = {
        "tasks": [
            {
                "task_type": "syntax_analysis",
                "prompt": "Analyze the sentence.",
                "ideal_answer": "The structure is valid.",
                "verification_pattern": "valid",
                "hints": [],
                "difficulty": "medium",
            }
        ]
    }

    disabled_completions = FakeCompletions([payload], fail_first=True)
    disabled_client = LLMJsonClient(FakeClient(disabled_completions), "test-model")
    with pytest.raises(RuntimeError, match="structured output unsupported"):
        disabled_client.create(
            messages=[{"role": "user", "content": "generate"}],
            schema=schema,
            schema_name="rl_task_generation",
            max_output_tokens=2000,
        )

    enabled_completions = FakeCompletions([payload], fail_first=True)
    enabled_client = LLMJsonClient(
        FakeClient(enabled_completions),
        "test-model",
        allow_json_fallback=True,
    )
    enabled_client.create(
        messages=[{"role": "user", "content": "generate"}],
        schema=schema,
        schema_name="rl_task_generation",
        max_output_tokens=2000,
    )

    assert enabled_completions.calls[0]["response_format"]["type"] == "json_schema"
    assert enabled_completions.calls[1]["response_format"]["type"] == "json_object"


def test_json_prompt_mode_omits_response_format_and_still_validates() -> None:
    schema = load_json_schema("grammar_rule.schema.json")
    completions = FakeCompletions(
        [
            {
                "rules": [
                    {
                        "title": "Word order",
                        "description": "A word-order rule.",
                        "category": "syntax",
                        "stoney_examples": [],
                        "english_examples": [],
                        "verification_hint": "",
                        "confidence": 0.7,
                        "page_number": 1,
                        "chunk_id": "page_001_chunk_00",
                    }
                ]
            }
        ]
    )
    client = LLMJsonClient(FakeClient(completions), "test-model")

    client.create(
        messages=[{"role": "user", "content": "extract"}],
        schema=schema,
        schema_name="grammar_rule_extraction",
        max_output_tokens=2000,
        response_format_mode="json_prompt",
    )

    assert "response_format" not in completions.calls[0]


def test_empty_model_content_includes_response_diagnostics() -> None:
    schema = load_json_schema("rl_task.schema.json")
    completions = FakeCompletions([{}])

    def create_empty(**kwargs: Any) -> Any:
        completions.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="content_filter",
                    message=SimpleNamespace(content="", refusal="cannot process page"),
                )
            ]
        )

    completions.create = create_empty  # type: ignore[method-assign]
    client = LLMJsonClient(FakeClient(completions), "test-model")

    with pytest.raises(LLMJsonError, match="finish_reason='content_filter'"):
        client.create(
            messages=[{"role": "user", "content": "generate"}],
            schema=schema,
            schema_name="rl_task_generation",
            max_output_tokens=2000,
        )

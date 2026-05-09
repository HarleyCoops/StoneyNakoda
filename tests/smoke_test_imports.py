from __future__ import annotations


def test_core_imports_do_not_require_api_keys() -> None:
    import stoney_config
    import stoney_rl_grammar
    from stoney_rl_grammar.llm_json import LLMJsonClient

    assert stoney_config.DEFAULT_OPENAI_FINETUNE_MODEL
    assert stoney_rl_grammar.DEFAULT_EXTRACTION_MODEL
    assert LLMJsonClient

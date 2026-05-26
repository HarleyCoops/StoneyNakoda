from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RL_ENV_PATH = REPO_ROOT / "environments" / "stoney_nakoda_translation"
if str(RL_ENV_PATH) not in sys.path:
    sys.path.insert(0, str(RL_ENV_PATH))


def test_core_imports_do_not_require_api_keys() -> None:
    import stoney_config
    import stoney_rl_grammar
    from stoney_nakoda_translation import load_environment
    from stoney_rl_grammar.llm_json import LLMJsonClient

    assert stoney_config.DEFAULT_OPENAI_FINETUNE_MODEL
    assert stoney_rl_grammar.DEFAULT_EXTRACTION_MODEL
    assert LLMJsonClient
    assert load_environment(max_examples=1)

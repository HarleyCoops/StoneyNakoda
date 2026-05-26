"""Configuration constants for the Stoney grammar RL pipeline."""

from __future__ import annotations

from pathlib import Path

from stoney_config import load_stoney_config

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Input assets
GRAMMAR_PDF_PATH = BASE_DIR / "Stoney; A Grammar of the Stony Language.pdf"

# Intermediate outputs
GRAMMAR_PAGES_DIR = DATA_DIR / "grammar_pages"
EXTRACTION_OUTPUT_DIR = DATA_DIR / "grammar_extracted_stoney"
RL_RULES_OUTPUT_PATH = DATA_DIR / "rl_training_rules_stoney.json"
TASK_DATASET_OUTPUT_PATH = DATA_DIR / "training_datasets_stoney.jsonl"

# Models and tuning defaults
_CONFIG = load_stoney_config(validate_finetune=False)
DEFAULT_EXTRACTION_MODEL = _CONFIG.openai_extraction_model
DEFAULT_TASK_MODEL = _CONFIG.openai_task_model

# Operational parameters
MAX_CHARS_PER_CHUNK = 2200
# Conservative limit to avoid API truncation.
MAX_OUTPUT_RULES_PER_CHUNK = 6
EXTRACTION_TEMPERATURE = 0.2
TASK_GENERATION_TEMPERATURE = 0.3


def ensure_directories() -> None:
    """Create any directories required for the pipeline outputs."""
    for path in (DATA_DIR, GRAMMAR_PAGES_DIR, EXTRACTION_OUTPUT_DIR):
        path.mkdir(parents=True, exist_ok=True)

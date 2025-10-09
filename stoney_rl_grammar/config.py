"""Configuration constants for the Stoney grammar RL pipeline."""

from __future__ import annotations

import os
from pathlib import Path

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
DEFAULT_RESPONSES_MODEL = os.getenv("OPENAI_RESPONSES_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-5"
DEFAULT_EXTRACTION_MODEL = os.getenv("STONEY_EXTRACTION_MODEL", DEFAULT_RESPONSES_MODEL)
DEFAULT_TASK_MODEL = os.getenv("STONEY_TASK_MODEL", DEFAULT_RESPONSES_MODEL)

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

# Stoney Nakoda Pipeline Operations Guide

## Repository Scope & Mission

This repository implements **two complementary pipelines** for building low-resource language AI models for Stoney Nakoda:

1. **Dictionary Fine-tuning Pipeline** - Supervised learning via OpenAI fine-tuning
   - Generates 150K Q&A pairs from bilingual dictionaries
   - Converts to OpenAI fine-tuning format (120K train / 30K validation)
   - Fine-tunes GPT models via OpenAI API

2. **Grammar RL Pipeline** - Reinforcement learning with grammar-based rewards
   - Extracts grammar rules from PDF using vision models
   - Organizes and curates rules by confidence and category
   - Generates RL training tasks for GRPO/prime-rl

**Operational Philosophy:** Run both pipelines end-to-end and **log all failures per stage** before attempting fixes. Document errors with context (command, timestamp, observed behavior) to build institutional knowledge.

---

## Environment & Secrets Checklist

### Required API Keys

Create a `.env` file (copy from `.env.example`) with the following **required** keys:

```bash
# REQUIRED: OpenAI API key for fine-tuning + grammar extraction
OPENAI_API_KEY=sk-your-openai-api-key-here

# REQUIRED: Google Gemini API key for Q&A generation
GOOGLE_API_KEY=your-google-api-key-here
```

### Optional Integration Keys

```bash
# OPTIONAL: Hugging Face dataset publishing
HUGGINGFACE_TOKEN=hf_your-hf-token-here
HUGGINGFACE_DATASET_REPO=username/stoney-nakoda-dataset
HUGGINGFACE_DATASET_PRIVATE=false

# OPTIONAL: Weights & Biases experiment tracking
WANDB_API_KEY=your-wandb-api-key-here
WANDB_PROJECT=stoney-nakoda-finetuning
WANDB_ENTITY=your-username-or-team
WANDB_RUN_NAME=stoney-run-001
```

### Model Override Configuration

**IMPORTANT:** These models have cost and capability implications. Update this file when changing models.

```bash
# Base model for fine-tuning (default: gpt-4o-mini)
# Options: gpt-4o-mini, gpt-3.5-turbo, gpt-4
OPENAI_MODEL=gpt-4o-mini
OPENAI_FINETUNE_MODEL=gpt-4o-mini

# Responses API model for grammar extraction/task generation (default: gpt-5)
# Options: gpt-5, gpt-4-turbo, gpt-4
OPENAI_RESPONSES_MODEL=gpt-5

# Gemini model for Q&A generation (default: gemini-2.5-pro)
# Options: gemini-2.5-pro, gemini-2.0-flash-exp, gemini-1.0-pro
GEMINI_MODEL=gemini-2.5-pro

# Grammar pipeline-specific overrides (default to OPENAI_RESPONSES_MODEL)
STONEY_EXTRACTION_MODEL=gpt-5
STONEY_TASK_MODEL=gpt-5
```

**Cost Estimates by Model:**
- `gpt-4o-mini`: $0.150/1M input tokens, $0.600/1M output tokens
- `gpt-3.5-turbo`: $0.500/1M input tokens, $1.500/1M output tokens
- `gpt-4`: $10.00/1M input tokens, $30.00/1M output tokens
- `gpt-5` (Responses API): Varies, typically $5-10/1M tokens
- `gemini-2.5-pro`: $1.25/1M input tokens, $5.00/1M output tokens
- `gemini-2.0-flash-exp`: $0.00/1M tokens (experimental, free tier)

---

## Data-Handling Policy

### DO NOT Commit These Directories

The following directories contain generated artifacts and should **NEVER** be committed to version control:

```
Dictionaries/bilingual_training_set*.jsonl
Dictionaries/checkpoints/
OpenAIFineTune/*.jsonl
data/grammar_pages/
data/grammar_extracted_stoney/
data/rl_training_rules_stoney.json
data/training_datasets_stoney.jsonl
```

**Why:** These files are pipeline outputs that can be regenerated. They are large (150K+ lines) and change frequently during development.

### External Artifact Recording

When documenting pipeline failures:
1. **DO NOT** commit the full output files
2. **DO** record file sizes, line counts, and sample content (first 10 lines)
3. **DO** capture error logs and command outputs in issue descriptions
4. **DO** note timestamps and environment variables used

Example failure documentation:
```
Stage: Q&A Generation
Command: python bilingual_qa_generator2.py
Timestamp: 2025-10-28 17:30:00 UTC
Error: google.api_core.exceptions.ResourceExhausted: 429 Quota exceeded
Output size: 45,234 lines generated before failure
Checkpoint: Dictionaries/checkpoints_v2/checkpoint_45.jsonl exists
Mitigation: Reduced context_size from 6 to 4, resumed from checkpoint
```

---

## Pipeline Execution Steps

### Setup (One-time)

```bash
# 1. Clone repository and navigate
cd StoneyNakoda

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
cp .env.example .env
# Edit .env and add your API keys

# CHECKPOINT: Verify environment
python -c "import openai, google.generativeai; print('Dependencies OK')"
```

**Failure Checkpoint:** If imports fail, check `pip list` and compare against `requirements.txt`. Document missing packages.

### Dictionary Fine-tuning Pipeline

#### Stage 1: Q&A Generation (2-4 hours, $5-15)

```bash
python bilingual_qa_generator2.py
```

**Expected Output:**
- `Dictionaries/bilingual_training_set_v2.jsonl` (10K Q&A pairs, 5K per language)
- `Dictionaries/checkpoints_v2/checkpoint_*.jsonl` (progress checkpoints every 1000 pairs)

**Failure Checkpoints:**
- After 1000 pairs: Verify checkpoint file exists and contains valid JSON
- If interrupted: Resume automatically from last checkpoint
- If API errors: Check `GOOGLE_API_KEY` validity and quota limits

**Common Failures:**
| Error | Cause | Mitigation |
|-------|-------|------------|
| `ResourceExhausted: 429` | Gemini API quota exceeded | Wait 60 seconds, script auto-retries |
| `FileNotFoundError` | Missing dictionary files | Verify `Dictionaries/*.jsonl` exist |
| `JSONDecodeError` | Malformed dictionary entry | Script logs and skips invalid lines |

#### Stage 2: Format Conversion (< 1 minute)

```bash
python finetunesetup.py
```

**Expected Output:**
- `OpenAIFineTune/stoney_train.jsonl` (120K examples)
- `OpenAIFineTune/stoney_valid.jsonl` (30K examples)

**Failure Checkpoint:**
- Validate JSONL structure: `head -n 1 OpenAIFineTune/stoney_train.jsonl | python -m json.tool`
- Check line counts: `wc -l OpenAIFineTune/*.jsonl`

**Common Failures:**
| Error | Cause | Mitigation |
|-------|-------|------------|
| Empty output files | Missing input file | Re-run `bilingual_qa_generator.py` |
| Invalid JSON | Encoding issues | Check UTF-8 encoding in input file |

#### Stage 3: Fine-tuning (1-3 hours, $20-50)

```bash
python openai_finetune.py
```

**Expected Output:**
- Fine-tuned model ID printed to console (e.g., `ft:gpt-4o-mini:org:model:abc123`)
- Optional: Dataset published to HuggingFace
- Optional: Metrics logged to Weights & Biases

**Failure Checkpoints:**
- After file upload: Verify file IDs printed (e.g., `file-abc123`)
- During training: Check status every 60 seconds (logged to console)
- After completion: Record fine-tuned model ID in issue/docs

**Common Failures:**
| Error | Cause | Mitigation |
|-------|-------|------------|
| `File validation failed` | Invalid JSONL format | Re-run `finetunesetup.py` |
| `Insufficient quota` | OpenAI API limits | Wait or upgrade plan |
| `Training failed` | Data quality issues | Review validation loss logs |

### Grammar RL Pipeline

#### Stage 4: PDF Ingest & Extraction (10-30 minutes, $10-30)

```bash
python run_stoney_grammar_pipeline.py
```

**Pipeline Stages (automated):**
1. **PDF Ingest** - Renders PDF pages to PNG images
2. **Rule Extraction** - Sends images to OpenAI vision model
3. **Rule Organization** - Filters by confidence, deduplicates
4. **Task Generation** - Creates RL training tasks

**Expected Output:**
- `data/grammar_pages/*.png` (rendered PDF pages)
- `data/grammar_extracted_stoney/*.json` (per-page raw extractions)
- `data/rl_training_rules_stoney.json` (curated rules)
- `data/training_datasets_stoney.jsonl` (RL tasks)

**Failure Checkpoints:**
- After PDF ingest: Verify PNG files exist: `ls -lh data/grammar_pages/`
- After extraction: Check rule count: `jq '.rules | length' data/grammar_extracted_stoney/page_001_chunk_00.json`
- After organization: Verify curated rules: `jq '.summary' data/rl_training_rules_stoney.json`
- After task generation: Check task count: `wc -l data/training_datasets_stoney.jsonl`

**Common Failures:**
| Error | Cause | Mitigation |
|-------|-------|------------|
| `FileNotFoundError: Stoney; A Grammar of the Stony Language.pdf` | PDF not in project root | Verify PDF location |
| `Empty extraction` | Vision model API error | Check `OPENAI_API_KEY` and vision model access |
| `Low confidence rules filtered` | Threshold too high | Lower `min_confidence` in `rule_organizer.py` (default 0.35) |
| `Few tasks generated` | Insufficient rules | Review extraction quality in `data/grammar_extracted_stoney/` |

**Retry/Backoff Behavior:**
- Extraction: 3 retries with exponential backoff (2s, 4s, 8s intervals)
- Task generation: 3 retries with exponential backoff
- Location: `stoney_rl_grammar/rule_extractor.py` line 33, `task_generator.py` line 30

#### Stage 5: RL Environment Installation

```bash
pip install -e environments/stoney_nakoda_translation
```

**Expected Output:**
- Package installed in editable mode
- Verify: `python -c "import stoney_nakoda_translation; print('OK')"`

**Failure Checkpoint:**
- If import fails: Check `environments/stoney_nakoda_translation/pyproject.toml` exists
- Verify dependencies: `pip show verifiers`

---

## Failure Logging Template

Use this template when documenting pipeline failures:

```markdown
### [Pipeline Stage] Failure Report

**Date/Time:** YYYY-MM-DD HH:MM:SS UTC
**Stage:** [QA Generation | Format Conversion | Fine-tuning | PDF Ingest | Extraction | Organization | Task Generation | RL Environment]
**Command:** `python script_name.py`

**Environment:**
- Python version: `python --version`
- OS: [Windows 11 | macOS | Linux]
- Virtual environment: [Yes | No]

**Observed Error:**
```
[Paste full error traceback here]
```

**Context:**
- Input file size: [X lines / Y MB]
- Output generated: [X lines / Y MB or "none"]
- Checkpoint available: [Yes at line X | No]
- API keys configured: [OPENAI_API_KEY: ✓ | GOOGLE_API_KEY: ✓ | ...]

**Mitigation Attempted:**
1. [Action 1]
2. [Action 2]
3. [Result]

**Resolution:**
[How was this resolved, or "Unresolved - needs investigation"]

**Follow-up Actions:**
- [ ] Update documentation
- [ ] Add error handling to code
- [ ] Create preventive validation
```

---

## Model & Cost Controls

### Updating Model References

**RULE:** When changing model names, retry policies, or cost estimates in this file, you MUST also update:

1. `.env.example` - Document new model options
2. `stoney_rl_grammar/config.py` - Update default model constants
3. `openai_finetune.py` - Update fine-tune model fallback
4. `bilingual_qa_generator2.py` - Update Gemini model reference

### Retry/Backoff Policies

**Grammar Extraction** (`stoney_rl_grammar/rule_extractor.py` line 33):
```python
@retry(wait=wait_exponential(multiplier=1, min=2, max=20), stop=stop_after_attempt(3))
```
- 3 attempts total
- Wait: 2s → 4s → 8s (doubles each time, max 20s)
- Configurable: Change `stop_after_attempt(X)` to adjust retries

**Task Generation** (`stoney_rl_grammar/task_generator.py` line 30):
```python
@retry(wait=wait_exponential(multiplier=1, min=2, max=20), stop=stop_after_attempt(3))
```
- Same policy as extraction
- Separate retry counter per task batch

**Q&A Generation** (`bilingual_qa_generator2.py`):
- No explicit retry decorator (relies on Gemini client defaults)
- Manual checkpoint system: resumes from last successful checkpoint
- Checkpoints saved every 1000 pairs in `Dictionaries/checkpoints_v2/`

### Cost Estimation Formula

**Dictionary Pipeline:**
```
Q&A Generation = (dict_entries / 5) * $0.0025 * 2  # Gemini, 5 entries per batch, both dicts
Fine-tuning = tokens_trained * epochs * $per_token  # OpenAI pricing
Total ≈ $25-65
```

**Grammar Pipeline:**
```
Extraction = pdf_pages * $0.01  # Vision model per page
Tasks = rules_count * 5 * $0.005  # 5 tasks per rule
Total ≈ $10-30
```

**To Monitor Costs:**
- OpenAI: https://platform.openai.com/usage
- Google Cloud: https://console.cloud.google.com/billing
- Weights & Biases: Check run summaries for token counts

---

## Prime Intellect RL Gym Integration

### Environment Installation

The Stoney Nakoda RL environment is a custom Verifiers/Prime Intellect gym environment:

```bash
# Install in editable mode
pip install -e environments/stoney_nakoda_translation

# Verify installation
python -c "from stoney_nakoda_translation.environment import StoneyNakodaEnv; print('OK')"
```

### Forwarding Datasets to RL Runs

After running the grammar pipeline, forward generated datasets to your RL training:

```python
# Example: GRPO training with Stoney environment
from stoney_nakoda_translation.environment import StoneyNakodaEnv
import json

# Load RL tasks
with open("data/training_datasets_stoney.jsonl", "r") as f:
    tasks = [json.loads(line) for line in f]

# Initialize environment
env = StoneyNakodaEnv(tasks=tasks)

# Run GRPO training (pseudocode)
# trainer = GRPOTrainer(env=env, model="ft:gpt-4o-mini:...")
# trainer.train()
```

**Dataset Path Arguments:**
- `--task-dataset`: Path to `data/training_datasets_stoney.jsonl`
- `--rules-json`: Path to `data/rl_training_rules_stoney.json`

**Environment Configuration:**
- Sample tasks: `environments/stoney_nakoda_translation/stoney_nakoda_translation/data/sample_tasks.jsonl`
- Environment code: `environments/stoney_nakoda_translation/stoney_nakoda_translation/environment.py`

---

## Contributing Conventions

### Coding Style

**Import Patterns:**
- **REQUIRED:** Use absolute imports from project root
- **FORBIDDEN:** Wrapping imports in `try/except` UNLESS they are optional integrations (e.g., `huggingface_hub`, `wandb`)
- **REQUIRED:** Group imports: stdlib → third-party → local modules
- **REQUIRED:** Use `from __future__ import annotations` for forward type references

Example (from `stoney_rl_grammar/pipeline.py`):
```python
from __future__ import annotations

import logging
from typing import List

from .config import GRAMMAR_PDF_PATH
from .models import GrammarRule
from .pdf_ingest import load_page_assets
```

### Dataclass Structures

**REQUIRED:** Respect dataclass field definitions in `stoney_rl_grammar/models.py`:

```python
@dataclass
class GrammarRule:
    rule_id: str
    title: str
    description: str
    category: str
    stoney_examples: List[str] = field(default_factory=list)
    # ... more fields
```

**DO NOT:**
- Add fields without updating the dataclass definition
- Change field types without updating type hints
- Remove required fields

**DO:**
- Use `to_dict()` methods for JSON serialization
- Validate field types in constructors if adding new models

### Error Handling

**REQUIRED:** Log errors at appropriate levels:
- `logger.info()`: Normal progress updates
- `logger.warning()`: Recoverable errors (skipped entries)
- `logger.error()`: Failures that prevent completion
- `logger.debug()`: Detailed debugging information (off by default)

**FORBIDDEN:** Silent failures (catching exceptions without logging)

Example (from `rule_extractor.py`):
```python
try:
    payload = self._call_model(prompt, chunk)
except RetryError as exc:
    logger.error("Unable to process %s: %s", chunk.chunk_id(), exc)
    continue  # Skip this chunk but continue processing
```

### File Naming Conventions

- Scripts: `snake_case.py` (e.g., `bilingual_qa_generator.py`)
- Modules: `snake_case.py` (e.g., `rule_extractor.py`)
- Classes: `PascalCase` (e.g., `StoneyGrammarExtractor`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_EXTRACTION_MODEL`)

### Documentation Requirements

**REQUIRED for all functions:**
```python
def function_name(param: str) -> ReturnType:
    """Brief one-line description.
    
    Detailed explanation if needed. Describe what the function does,
    not how it does it (that's what code comments are for).
    
    Args:
        param: Description of parameter
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When this specific error occurs
    """
```

### Pull Request Checklist

Before submitting changes:
- [ ] Updated `.env.example` if adding new environment variables
- [ ] Updated this AGENTS.md file if changing pipeline behavior
- [ ] Added failure checkpoints for new pipeline stages
- [ ] Tested locally with full pipeline run
- [ ] Documented any new model costs in "Model & Cost Controls" section
- [ ] Added retry logic for new API calls (use `tenacity` library)
- [ ] Verified no sensitive data in commits (API keys, tokens, etc.)

---

## Quick Reference

### Pipeline Scripts

| Script | Purpose | Duration | Cost |
|--------|---------|----------|------|
| `bilingual_qa_generator2.py` | Generate Q&A pairs from dictionaries (enriched prompts) | 2-4 hours | $5-15 |
| `finetunesetup.py` | Convert to OpenAI format | < 1 min | Free |
| `openai_finetune.py` | Fine-tune OpenAI model | 1-3 hours | $20-50 |
| `run_stoney_grammar_pipeline.py` | Extract grammar rules + generate tasks | 10-30 min | $10-30 |

### Required API Keys

- `OPENAI_API_KEY` - **ALWAYS REQUIRED**
- `GOOGLE_API_KEY` - Required for Q&A generation only
- `HUGGINGFACE_TOKEN` - Optional (dataset publishing)
- `WANDB_API_KEY` - Optional (experiment tracking)

### Output Directories (Do Not Commit)

- `Dictionaries/bilingual_training_set*.jsonl`
- `Dictionaries/checkpoints/`
- `OpenAIFineTune/*.jsonl`
- `data/grammar_pages/`
- `data/grammar_extracted_stoney/`
- `data/rl_training_rules_stoney.json`
- `data/training_datasets_stoney.jsonl`

### Key Configuration Files

- `.env` - API keys and model overrides (never commit)
- `.env.example` - Template for environment setup
- `requirements.txt` - Python dependencies
- `stoney_rl_grammar/config.py` - Pipeline constants
- `environments/stoney_nakoda_translation/pyproject.toml` - RL environment metadata

---

## Support & Troubleshooting

For issues not covered in this guide:
1. Check existing issues: [GitHub Issues](https://github.com/HarleyCoops/StoneyNakoda/issues)
2. Review pipeline logs in console output
3. Document failure using the template above
4. Open a new issue with failure report

**Remember:** The goal is to run pipelines end-to-end and log failures systematically, not to fix every error immediately. Building institutional knowledge through documentation is more valuable than perfect code.

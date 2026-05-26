# Stoney Nakoda Pipeline Test Report
**Date:** October 28, 2025
**Objective:** Run complete dual-pipeline from start to finish and document failure locations

## Test Environment Setup

### API Keys Configuration
- ✅ `.env` file created with test API keys
- ✅ OpenAI API Key configured
- ✅ Google API Key (Gemini) configured
- ⏳ Model overrides set (GPT-5, Gemini 2.5)

### Dependency Installation
- ⏳ **Status:** In progress (large packages downloading)
- **Packages being installed:**
  - TensorFlow 2.20.0 (620.6 MB) - ✅ Downloaded
  - PyTorch 2.9.0 (899.8 MB) - ✅ Downloaded
  - NVIDIA CUDA libraries (594+ MB) - ⏳ Downloading
  - OpenAI SDK, Google GenerativeAI, HuggingFace libraries
  - **Verifiers framework v0.1.6** (Prime Intellect RL)

- **Installation Command:** `pip install -r requirements.txt`
- **Estimated Time:** 10-15 minutes (depends on network speed)

---

## Prime Intellect RL Environments Hub Integration

### Research Summary

#### What is Prime Intellect Environments Hub?
A community platform for LLM RL environments that functions as both a discovery hub and Python package registry. Environments are distributed as wheels with dependencies declared in `pyproject.toml`.

#### Verifiers Framework
The native library for Prime Intellect's Environments Hub that provides:
- **Datasets**: Hugging Face datasets with `prompt` (chat messages) or `question` (string) columns
- **Rubrics**: Reward functions (sync or async) with configurable weights
- **Environment Types:**
  - `SingleTurnEnv`: Single model response per prompt
  - `ToolEnv`: Agentic loops with function calling
  - `MultiTurnEnv`: Custom multi-turn interactions
  - `StatefulToolEnv`: Tools with per-rollout state
  - `SandboxEnv`/`PythonEnv`: Long-running sandboxed execution

#### Integration Requirements for Stoney Nakoda RL Gym

**Current Status:**
- ✅ `verifiers>=0.1.5.dev1` specified in `requirements.txt` (installing v0.1.6.post0)
- ✅ Custom environment at `environments/stoney_nakoda_translation/`
- ⚠️ **Need to verify:** Environment implements required `load_environment` function
- ⚠️ **Need to add:** Prime CLI installation (`uv tool install prime`)
- ⚠️ **Need to create:** `pyproject.toml` for environment packaging

**Recommended Steps for Prime Intellect Integration:**
1. Install Prime CLI: `uv tool install prime`
2. Initialize environment module: `uv run vf-init stoney-nakoda-translation`
3. Ensure environment exposes `load_environment()` function
4. Create reward rubric matching grammar verification patterns
5. Test locally: `uv run vf-eval stoney-nakoda-translation -s`
6. Publish: `prime env push`

---

## Dictionary→Fine-tuning Pipeline (Supervised Learning)

### Pre-flight Checks
- ✅ `Dictionaries/english_dictionary.jsonl` exists (2.3 MB, ~8K entries)
- ✅ `Dictionaries/stoney_dictionary.jsonl` exists (955 KB, ~3K entries)
- ✅ `Dictionaries/bilingual_training_set.jsonl` exists (41 KB, partial data)
- ✅ `OpenAIFineTune/` directory exists
- ✅ `bilingual_qa_generator.py` script present
- ✅ `finetunesetup.py` script present
- ✅ `openai_finetune.py` script present

### Pipeline Execution

#### Step 1: bilingual_qa_generator.py
**Purpose:** Generate diverse Q&A pairs from dictionary entries using Google Gemini
**Status:** ⏳ Pending (awaiting dependency installation)

**Expected Behavior:**
- Process dictionaries in batches of 5 entries
- Generate 5 Q&A pairs per batch with cultural context
- Target: 75K pairs per language (150K total)
- Create checkpoints every 1000 pairs
- Output: `Dictionaries/bilingual_training_set.jsonl`

**Test Command:**
```bash
python bilingual_qa_generator.py
```

**Potential Failure Points:**
- [ ] Google Gemini API authentication (test keys may be invalid)
- [ ] API rate limits or quota exhaustion
- [ ] Memory constraints (batch processing)
- [ ] UTF-8 encoding issues with special Stoney characters

---

#### Step 2: finetunesetup.py
**Purpose:** Convert Q&A pairs to OpenAI format with 80/20 train/validation split
**Status:** ⏳ Pending

**Expected Behavior:**
- Read from `Dictionaries/bilingual_training_set.jsonl`
- Convert to OpenAI messages format
- 80/20 shuffle split
- Preserve UTF-8 encoding
- Output: `OpenAIFineTune/stoney_train.jsonl` + `stoney_valid.jsonl`

**Test Command:**
```bash
python finetunesetup.py
```

**Potential Failure Points:**
- [ ] Input file format validation
- [ ] JSON encoding errors
- [ ] Insufficient training pairs
- [ ] System prompt configuration

---

#### Step 3: openai_finetune.py
**Purpose:** Upload datasets and launch OpenAI fine-tuning job
**Status:** ⏳ Pending

**Expected Behavior:**
- Upload `stoney_train.jsonl` and `stoney_valid.jsonl`
- Create fine-tuning job on `gpt-4o-mini` (default)
- 3 epochs (configurable at line 237)
- Monitor job status every 60 seconds
- Optional: Publish to HuggingFace
- Optional: Track with Weights & Biases

**Test Command:**
```bash
python openai_finetune.py
```

**Potential Failure Points:**
- [ ] OpenAI API authentication (test keys may be invalid)
- [ ] File upload failures
- [ ] Job creation errors (quota/permissions)
- [ ] HuggingFace token issues (if configured)
- [ ] W&B authentication failures (if configured)
- [ ] Model availability (GPT-5 may not exist with test keys)

---

## Grammar→RL Pipeline (Reinforcement Learning)

### Pre-flight Checks
- ✅ `Stoney; A Grammar of the Stony Language.pdf` exists (20.6 MB)
- ✅ `stoney_rl_grammar/` module present
- ✅ `run_stoney_grammar_pipeline.py` entry point present
- ✅ `data/` directory exists
- ⚠️ `environments/stoney_nakoda_translation/` - need to verify structure

### Pipeline Execution

#### Complete Grammar Pipeline: run_stoney_grammar_pipeline.py
**Purpose:** Extract grammar rules from PDF and generate RL training tasks
**Status:** ⏳ Pending

**Expected Behavior:**
1. **PDF Ingestion** (`pdf_ingest.py`)
   - Render PDF pages to PNG images using PyMuPDF
   - Output: `data/grammar_pages/*.png`

2. **Rule Extraction** (`rule_extractor.py`)
   - Use OpenAI vision models (GPT-5 configured)
   - Extract structured grammar rules from images
   - Output: `data/grammar_extracted_stoney/*.json`

3. **Rule Organization** (`rule_organizer.py`)
   - Filter low-confidence rules
   - Remove duplicates
   - Organize by category
   - Output: `data/rl_training_rules_stoney.json`

4. **Task Generation** (`task_generator.py`)
   - Convert rules to 3-6 RL tasks each
   - Generate prompts, answers, hints, verification patterns
   - Output: `data/training_datasets_stoney.jsonl`

**Test Command:**
```bash
python run_stoney_grammar_pipeline.py
```

**Potential Failure Points:**
- [ ] PyMuPDF PDF rendering errors
- [ ] OpenAI vision API authentication (test keys)
- [ ] GPT-5 model availability (may not exist)
- [ ] Insufficient API quota for 100+ page PDF
- [ ] Image processing memory constraints
- [ ] JSON parsing errors in extraction
- [ ] Confidence threshold filtering (too aggressive/lenient)
- [ ] Task generation quality

---

#### Custom RL Environment: environments/stoney_nakoda_translation/
**Status:** ⏳ Need to verify compliance with Prime Intellect standards

**Required Components:**
- [ ] `load_environment()` function
- [ ] HuggingFace dataset with `prompt` or `question` column
- [ ] Rubric with reward functions
- [ ] Environment type selection (likely `SingleTurnEnv`)
- [ ] `pyproject.toml` for packaging

**Test Command:**
```bash
pip install -e environments/stoney_nakoda_translation
# Then test with verifiers
uv run vf-eval stoney-nakoda-translation -s
```

**Potential Failure Points:**
- [ ] Missing `load_environment()` implementation
- [ ] Dataset format mismatch
- [ ] Reward function errors
- [ ] Dependencies not declared
- [ ] Package installation failures

---

## Known Issues & Blockers

### Critical Blockers
1. **Test API Keys:** The provided keys appear to be hypothetical/test keys. They will likely fail authentication when used.
2. **GPT-5 Model:** The `.env` file specifies `gpt-5` which may not exist. Should fall back to `gpt-4o` or `gpt-4-turbo`.
3. **Gemini 2.5:** Model specified is `gemini-2.5-pro` and `gemini-2.5-flash` which may need version verification.

### Dependency Concerns
- Large package downloads (1.5+ GB) may fail on unstable connections
- CUDA libraries require compatible GPU (may run CPU-only mode)

### Data Volume Concerns
- Grammar PDF is 100+ pages: vision API costs could be significant
- Target 150K Q&A pairs: generation time could be 10+ hours
- Fine-tuning jobs can take several hours

---

## Next Steps (Post-Installation)

1. ✅ Complete dependency installation
2. ⏳ Run `bilingual_qa_generator.py` and document failures
3. ⏳ Run `finetunesetup.py` if Step 2 succeeds
4. ⏳ Run `openai_finetune.py` if Step 3 succeeds
5. ⏳ Run `run_stoney_grammar_pipeline.py` in parallel
6. ⏳ Verify `environments/stoney_nakoda_translation/` structure
7. ⏳ Create Prime Intellect environment package
8. ⏳ Test local evaluation with `vf-eval`
9. ⏳ Update this report with all failure locations
10. ⏳ Provide recommendations for fixes

---

## Timeline Estimate

- **Dependency Installation:** 10-15 minutes (in progress)
- **Dictionary Q&A Generation:** 2-8 hours (depends on batch size & rate limits)
- **OpenAI Fine-tuning:** 1-3 hours (after upload)
- **Grammar RL Pipeline:** 30-90 minutes (vision API calls)
- **Total End-to-End:** 4-12 hours (excluding waiting for fine-tuning completion)

---

**Report will be updated continuously as pipeline stages complete.**

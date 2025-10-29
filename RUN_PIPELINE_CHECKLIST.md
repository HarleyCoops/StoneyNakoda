# Stoney Nakoda Pipeline Run Checklist

This guide captures the exact commands to run each stage of the repository's two pipelines and highlights what to verify after every step. Use it to reproduce an end-to-end run on a fresh clone and to document any failures you encounter.

## 0. Environment preparation

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Populate environment variables**
   - Confirm `OPENAI_API_KEY` and `GOOGLE_API_KEY` are exported in the shell (or stored in `.env`).
   - Optionally set `OPENAI_MODEL`, `OPENAI_FINETUNE_MODEL`, `STONEY_EXTRACTION_MODEL`, and `STONEY_TASK_MODEL` to override defaults.
3. **Artifacts directory sanity check**
   - Ensure `Dictionaries/`, `OpenAIFineTune/`, and `data/` exist and are writable (the pipeline scripts will create their own sub-directories if needed).

## 1. Dictionary → Fine-tuning pipeline

### 1.1 Generate bilingual Q&A pairs

Run the Gemini-backed generator to build the 150k example corpus:
```bash
python bilingual_qa_generator.py
```
Expectation:
- `Dictionaries/bilingual_training_set.jsonl` grows toward 150,000 lines (75k English-perspective + 75k Stoney-perspective examples).
- Periodic checkpoints appear under `Dictionaries/checkpoints/` with progress metadata.
- The script logs API calls and warnings when malformed dictionary rows are skipped. 【F:bilingual_qa_generator.py†L23-L209】

Capture failures by noting the last log lines, checkpoint counts, and the batch size (`context_size=5`). Quota or timeout issues can be retried because the generator resumes from rolling buffers.

### 1.2 Convert to OpenAI chat format

Transform the raw corpus into train/validation JSONL files:
```bash
python finetunesetup.py
```
Expectation:
- `OpenAIFineTune/stoney_train.jsonl` and `OpenAIFineTune/stoney_valid.jsonl` are produced with an 80/20 split.
- Each entry is a chat-style `messages` array ready for OpenAI fine-tuning. 【F:finetunesetup.py†L10-L78】

If the input file is missing or malformed, the script logs the issue and aborts. Record the offending line number or JSON snippet when that happens.

### 1.3 Launch OpenAI fine-tune job

Submit the formatted datasets to OpenAI:
```bash
python openai_finetune.py
```
Expectation:
- The script validates the presence of the train/valid files and the `OPENAI_API_KEY`.
- A fine-tuning job is created for the base model defined by `OPENAI_FINETUNE_MODEL`/`OPENAI_MODEL`.
- Optional Hugging Face dataset publishing and Weights & Biases tracking kick in when the relevant environment variables are set. 【F:openai_finetune.py†L40-L199】

Document the returned job ID, status transitions, and any API errors (quota, billing, validation failures). If dataset uploads fail, capture the exception message and whether retries were attempted.

## 2. Grammar → RL pipeline

### 2.1 Run the orchestrator

Execute the single entry point that loads the grammar PDF, extracts rules, curates them, and generates RL tasks:
```bash
python run_stoney_grammar_pipeline.py
```
Pipeline flow:
1. `pdf_ingest.load_page_assets` renders the source PDF into 127 base64 PNG + text chunks. 【F:stoney_rl_grammar/pdf_ingest.py†L14-L35】
2. `StoneyGrammarExtractor.extract_rules` calls the Responses API to convert each chunk into structured grammar rules, persisting per-chunk JSON under `data/grammar_extracted_stoney/`. 【F:stoney_rl_grammar/rule_extractor.py†L62-L165】
3. `RuleOrganizer.organize` filters, deduplicates, and writes curated rules to `data/rl_training_rules_stoney.json`. 【F:stoney_rl_grammar/rule_organizer.py†L17-L81】
4. `StoneyTaskGenerator.generate_tasks` streams RL-ready tasks to `data/training_datasets_stoney.jsonl`. 【F:stoney_rl_grammar/task_generator.py†L63-L140】

### 2.2 Current blocking failure

During the extraction stage, the OpenAI Python SDK bundled with the repo rejects the `response_format` argument when calling `client.responses.create`, causing every chunk to fail and the retry loop to exhaust:
```
2025-10-29 00:05:48,141 - ERROR - Unable to process page_001_chunk_00: Responses.create() got an unexpected keyword argument 'response_format'
```
The error repeats for each page until the run is interrupted. 【d9a4f4†L1-L2】【8b4d30†L1-L16】

Action items while reproducing:
- Confirm the installed `openai` package version (`pip show openai`). Versions prior to 1.12.0 do not yet accept `response_format`.
- Option 1: upgrade the SDK (`pip install --upgrade openai`) so `response_format={"type": "json_object"}` is supported.
- Option 2: remove the `response_format` argument and parse raw JSON manually, but that reintroduces non-JSON responses.
- Document which option you choose and capture the resulting behavior (successful rule extraction counts or new errors).

## 3. Prime Intellect RL gym readiness

Once `data/training_datasets_stoney.jsonl` exists, install and smoke-test the bundled RL environment:
```bash
pip install -e environments/stoney_nakoda_translation
uv run vf-eval stoney-nakoda-translation -a '{"dataset_path": "data/training_datasets_stoney.jsonl", "max_examples": 50}'
```
The environment expects the task JSONL generated above and exposes configuration knobs documented in `environments/stoney_nakoda_translation/README.md`. 【F:environments/stoney_nakoda_translation/README.md†L1-L60】

Record evaluation metrics (`exact_match_reward`, `char_overlap_reward`, `pattern_reward`) and any loader issues (e.g., missing dataset path).

---

**Failure logging template**
```
Stage: <dictionary generation | format conversion | fine-tune | grammar extraction | task generation | RL eval>
Command: <exact command>
Timestamp (UTC): <YYYY-MM-DD HH:MM>
Observed behavior: <error message or anomaly>
Artifacts: <paths, line counts, checkpoint names>
Next action: <retry, upgrade dependency, open issue>
```
Keep this document updated as you resolve blockers so future operators have an authoritative runbook.

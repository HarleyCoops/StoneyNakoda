# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Community-In-The-Loop model distillation system for the Stoney Nakoda language. The project uses AI fine-tuning (primarily OpenAI's API) to create a bilingual translation model from a relatively small dictionary dataset (~8,000-10,000 words). The core innovation is iterative refinement through community feedback, where cultural context and narrative corrections improve the model over time.

**Active Model**: [Stoney Language Model App](https://huggingface.co/spaces/HarleyCooper/StoneyApp)
**Training Data**: [StoneyNakoda Training Dataset](https://huggingface.co/datasets/HarleyCooper/StoneyNakoda/blob/main/zSTONEY1_TRAINING_SET.jsonl)

## Dual Pipeline Architecture

The project contains **two parallel workflows**:

1. **Dictionary→Fine-tuning Pipeline** (Supervised Learning)
   - Primary path for building translation models
   - Uses dictionary data to generate Q&A pairs
   - Fine-tunes via OpenAI API
   - Files: `bilingual_qa_generator.py`, `finetunesetup.py`, `openai_finetune.py`

2. **Grammar→RL Pipeline** (Reinforcement Learning) **[October 2025 addition]**
   - Extracts grammar rules from PDF scans using vision models
   - Generates RL training tasks with verifiable rewards
   - Custom environment at `environments/stoney_nakoda_translation/`
   - Files: `run_stoney_grammar_pipeline.py`, `stoney_rl_grammar/` module

## Core Architecture

### Dictionary→Fine-tuning Pipeline (Supervised Learning)

Three-stage process:

1. **Data Generation** ([bilingual_qa_generator.py](bilingual_qa_generator.py)) - Uses Google Gemini to generate diverse Q&A pairs from dictionary entries
2. **Data Preparation** ([finetunesetup.py](finetunesetup.py)) - Converts Q&A pairs into OpenAI fine-tuning format with 80/20 train/validation split
3. **Model Training** ([openai_finetune.py](openai_finetune.py)) - Uploads data to OpenAI, creates fine-tuning jobs, monitors progress, optionally publishes to HuggingFace and tracks via Weights & Biases

**Data Flow:**
```
Dictionary Files (Dictionaries/)
    ↓
bilingual_qa_generator.py (Gemini generates Q&A pairs)
    ↓
Dictionaries/bilingual_training_set.jsonl
    ↓
finetunesetup.py (converts to OpenAI format, 80/20 split)
    ↓
OpenAIFineTune/stoney_train.jsonl + stoney_valid.jsonl
    ↓
openai_finetune.py (fine-tunes via OpenAI API)
    ↓
Fine-tuned Model (+ optional HF publish + W&B tracking)
```

### Grammar→RL Pipeline (Reinforcement Learning)

Four-stage process for extracting grammar rules from PDF and generating RL tasks:

1. **PDF Ingestion** ([stoney_rl_grammar/pdf_ingest.py](stoney_rl_grammar/pdf_ingest.py)) - Renders PDF pages to images
2. **Rule Extraction** ([stoney_rl_grammar/rule_extractor.py](stoney_rl_grammar/rule_extractor.py)) - Uses OpenAI vision models to extract structured grammar rules
3. **Rule Organization** ([stoney_rl_grammar/rule_organizer.py](stoney_rl_grammar/rule_organizer.py)) - Filters, deduplicates, and curates rules
4. **Task Generation** ([stoney_rl_grammar/task_generator.py](stoney_rl_grammar/task_generator.py)) - Converts rules into RL training tasks with hints and verification patterns

**Data Flow:**
```
Stoney; A Grammar of the Stony Language.pdf
    ↓
stoney_rl_grammar/pdf_ingest.py (renders to images)
    ↓
data/grammar_pages/*.png
    ↓
stoney_rl_grammar/rule_extractor.py (vision model extraction)
    ↓
data/grammar_extracted_stoney/*.json
    ↓
stoney_rl_grammar/rule_organizer.py (curate & filter)
    ↓
data/rl_training_rules_stoney.json
    ↓
stoney_rl_grammar/task_generator.py (create RL tasks)
    ↓
data/training_datasets_stoney.jsonl
    ↓
environments/stoney_nakoda_translation/ (custom RL env)
    ↓
GRPO/Prime-RL Training
```

## Key Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with OPENAI_API_KEY and GOOGLE_API_KEY
```

### Dictionary→Fine-tuning Pipeline (Complete)
```bash
# Step 1: Generate Q&A training data from dictionaries
python bilingual_qa_generator.py

# Step 2: Convert to OpenAI format and split train/validation
python finetunesetup.py

# Step 3: Upload and start fine-tuning job (with optional HF/W&B integration)
python openai_finetune.py
```

### Grammar→RL Pipeline (Complete)
```bash
# Run the complete grammar extraction and RL task generation pipeline
python run_stoney_grammar_pipeline.py

# After running, outputs will be in:
# - data/grammar_extracted_stoney/     (raw rule extractions)
# - data/rl_training_rules_stoney.json (curated rules)
# - data/training_datasets_stoney.jsonl (RL tasks)

# Then install the custom environment and run RL training:
pip install -e environments/stoney_nakoda_translation
# (Use with prime-rl or your RL framework)
```

### Utility Commands
```bash
# Convert legacy JSONL formats to OpenAI messages format
python convert_data_format.py

# Alternative Q&A generator (bilingual_qa_generator2.py)
python bilingual_qa_generator2.py
```

## File Structure

```
StoneyNakoda/
├── Dictionaries/                          # Source dictionary files
│   ├── english_dictionary.jsonl           # English→Stoney mappings
│   ├── stoney_dictionary.jsonl            # Stoney→English mappings
│   └── bilingual_training_set.jsonl       # Generated Q&A pairs
├── OpenAIFineTune/                        # Training files for OpenAI
│   ├── stoney_train.jsonl                 # 80% training split
│   └── stoney_valid.jsonl                 # 20% validation split
├── data/                                  # RL pipeline outputs
│   ├── grammar_pages/                     # Rendered PDF pages (PNG)
│   ├── grammar_extracted_stoney/          # Raw rule extractions (JSON)
│   ├── rl_training_rules_stoney.json      # Curated grammar rules
│   └── training_datasets_stoney.jsonl     # RL training tasks
├── environments/                          # Custom RL environments
│   └── stoney_nakoda_translation/         # Stoney RL environment package
├── stoney_rl_grammar/                     # Grammar RL pipeline module
│   ├── pipeline.py                        # High-level orchestration
│   ├── pdf_ingest.py                      # PDF→image conversion
│   ├── rule_extractor.py                  # Vision-based rule extraction
│   ├── rule_organizer.py                  # Rule curation and filtering
│   ├── task_generator.py                  # RL task generation
│   ├── config.py                          # Configuration and paths
│   └── models.py                          # Data models (GrammarRule, etc.)
├── Public/                                # Images and documentation assets
├── bilingual_qa_generator.py              # Q&A generation via Gemini
├── bilingual_qa_generator2.py             # Alternative Q&A generator
├── finetunesetup.py                       # Data formatting and splitting
├── openai_finetune.py                     # OpenAI fine-tuning + HF + W&B
├── convert_data_format.py                 # Legacy format converter
├── run_stoney_grammar_pipeline.py         # CLI entry for grammar RL pipeline
├── requirements.txt                       # Python dependencies
├── .env.example                           # API key template
├── RLHFrules.json                         # Language pattern rules for RL
├── Stoney; A Grammar of the Stony Language.pdf  # Source grammar PDF
└── README.md                              # Full project documentation
```

## Dictionary Format

**English Dictionary** (`english_dictionary.jsonl`):
```json
{
  "english_word": "example",
  "stoney_versions": [
    {
      "word": "stoney_word",
      "grammatical_classification": "noun/verb/etc",
      "meaning": "contextual explanation"
    }
  ]
}
```

**Stoney Dictionary** (`stoney_dictionary.jsonl`):
```json
{
  "stoney_word": "word",
  "english_translations": ["translation1", "translation2"],
  "grammatical_info": "classification",
  "cultural_context": "optional context"
}
```

**OpenAI Training Format** (`OpenAIFineTune/*.jsonl`):
```json
{
  "messages": [
    {"role": "system", "content": "You are a bilingual Stoney-English assistant..."},
    {"role": "user", "content": "question"},
    {"role": "assistant", "content": "answer"}
  ]
}
```

## Important Implementation Details

### Dictionary→Fine-tuning Pipeline Scripts

**bilingual_qa_generator.py**
- Processes dictionaries in batches (default: 5 entries at a time)
- Uses Google Gemini API (default: `gemini-1.0-pro`, configurable to `gemini-2.0-flash-exp` via `GEMINI_MODEL` env var)
- Generates 5 Q&A pairs per batch with cultural context
- Creates checkpoints every 1000 pairs for recovery
- Target: 75,000 pairs per language (150,000 total)
- Output includes metadata: `pair_id`, `source_language`, `generated_at`
- Reads from: `Dictionaries/english_dictionary.jsonl` and `Dictionaries/stoney_dictionary.jsonl`
- Outputs to: `Dictionaries/bilingual_training_set.jsonl`

**finetunesetup.py**
- Reads from: `Dictionaries/bilingual_training_set.jsonl`
- Converts to OpenAI messages format with specific system prompt
- Uses 80/20 shuffle split for train/validation
- Preserves UTF-8 encoding for special characters
- Outputs to: `OpenAIFineTune/stoney_train.jsonl` and `OpenAIFineTune/stoney_valid.jsonl`

**openai_finetune.py**
- Default model: `gpt-4o-mini` (configurable via `OPENAI_FINETUNE_MODEL` or `OPENAI_MODEL` env vars)
- Hyperparameters: 3 epochs (adjustable in code at line 237)
- Monitors job status every 60 seconds
- Logs trained tokens, accuracy, validation loss during training
- **Optional HuggingFace Integration**: Set `HUGGINGFACE_TOKEN`, `HUGGINGFACE_DATASET_REPO`, and optionally `HUGGINGFACE_DATASET_PRIVATE` to auto-publish training datasets
- **Optional Weights & Biases Integration**: Set `WANDB_API_KEY`, `WANDB_PROJECT`, and optionally `WANDB_ENTITY`/`WANDB_RUN_NAME` for experiment tracking
- Expects files at `OpenAIFineTune/stoney_train.jsonl` and `OpenAIFineTune/stoney_valid.jsonl`

**convert_data_format.py**
- Utility for converting legacy JSONL formats to OpenAI messages format
- Handles both old format and existing messages format
- Preserves special characters with `ensure_ascii=False`
- Currently configured for `data/raw/` and `data/processed/` directories

### Grammar→RL Pipeline Scripts

**run_stoney_grammar_pipeline.py**
- CLI entry point that orchestrates the complete grammar extraction pipeline
- Calls `stoney_rl_grammar.pipeline.run_pipeline()`
- No command-line arguments required

**stoney_rl_grammar/pdf_ingest.py**
- Renders PDF pages to PNG images using PyMuPDF (fitz)
- Saves to `data/grammar_pages/`
- Each page becomes a separate image file

**stoney_rl_grammar/rule_extractor.py**
- Uses OpenAI vision models (default: GPT-5 via `STONEY_EXTRACTION_MODEL` or `OPENAI_RESPONSES_MODEL` env vars)
- Sends page images to vision API to extract structured grammar rules
- Outputs raw extractions to `data/grammar_extracted_stoney/*.json`
- Each extraction includes: rule text, category, confidence score, page number, source metadata

**stoney_rl_grammar/rule_organizer.py**
- Filters low-confidence rules and removes duplicates
- Builds a compact catalogue organized by category (morphology, syntax, phonology, etc.)
- Outputs curated rules to `data/rl_training_rules_stoney.json`

**stoney_rl_grammar/task_generator.py**
- Converts each curated rule into 3-6 RL training tasks
- Uses OpenAI API (default: GPT-5 via `STONEY_TASK_MODEL` or `OPENAI_RESPONSES_MODEL` env vars)
- Task types: morphology exercises, translation prompts, pattern identification
- Outputs tasks to `data/training_datasets_stoney.jsonl`
- Each task includes: prompt, expected answer, hints, verification patterns, difficulty level

**environments/stoney_nakoda_translation/environment.py**
- Custom RL environment compatible with `verifiers` framework and GRPO training
- Exposes multi-signal rewards: exact match, character F1, pattern matching
- Integrates with grammar rules for verifiable grading
- Install with: `pip install -e environments/stoney_nakoda_translation`

## Community-In-The-Loop Methodology

The project implements an iterative distillation approach:

1. **Initial Model**: Fine-tune on dictionary + generated Q&A pairs
2. **Community Testing**: Stoney speakers interact with model, identify errors
3. **Narrative Corrections**: Instead of simple corrections, speakers provide cultural context, stories, and nuanced explanations
4. **Distillation Triplets**: Store (Prompt, Incorrect Response, Narrative Correction)
5. **LoRA Fine-tuning**: Lightweight re-training on correction triplets
6. **Adaptive Checkpoints**: Validate improvements; revert if quality degrades

This approach emphasizes **narrative feedback** over mere accuracy, embedding cultural authenticity into the model's responses.

## LoRA and Parameter Efficiency

The project documentation describes using Low-Rank Adaptation (LoRA) for efficient fine-tuning:
- LoRA parameterizes weight updates as ΔW = AB where A and B are low-rank matrices
- Dramatically reduces trainable parameters (99%+ reduction typical)
- Allows rapid iteration on community feedback without full model retraining
- Preserves base model knowledge while adding language-specific patterns

## Future Direction: RL-Based Training

The March 2025 update in the README outlines plans to move from OpenAI fine-tuning to open-source models with custom RL verifiers:

- Multi-faceted rewards: letter accuracy, word accuracy, semantic similarity (sentence-BERT), edit distance
- Custom `TranslationEnv`, `TranslationRubric`, and `TranslationParser` classes
- GRPO (Group Relative Policy Optimization) trainer
- Target models: Helsinki-NLP/opus-mt or similar open-source translation models
- See README section "Adapting Verifiers for Low-Resource Language Translation" for detailed implementation

## Language Pattern Rules

[RLHFrules.json](RLHFrules.json) contains automatically extracted linguistic patterns from the Stoney dictionary. These rules can be converted into fine-grained reward functions for RL-based training, enabling precise tuning around grammatical and phonological contours specific to Stoney Nakoda.

## Cultural Authenticity Score (CAS)

The loss function combines linguistic accuracy with community feedback:

```
L = α * L_CE + (1-α) * L_CAS
```

Where:
- L_CE: Cross-entropy loss (linguistic accuracy)
- L_CAS: Cultural Authenticity Score from community ratings (1-5 scale)
- α: Balance parameter (typically 0.7)

## Development Notes

- All scripts use UTF-8 encoding to preserve special characters in Stoney orthography
- Progress tracking via `tqdm` for long-running operations
- Checkpoint system for recovery during Q&A generation
- Logging configured with timestamps and severity levels
- Environment variables stored in `.env` (never committed to version control)

## API Requirements

**Required APIs:**

- **OpenAI API**: For fine-tuning and grammar rule extraction (vision models)
  - Get key from: https://platform.openai.com/api-keys
  - Set in `.env`: `OPENAI_API_KEY=sk-...`
  - Model overrides: `OPENAI_MODEL`, `OPENAI_FINETUNE_MODEL`, `OPENAI_RESPONSES_MODEL=gpt-5`, `STONEY_EXTRACTION_MODEL=gpt-5`, `STONEY_TASK_MODEL=gpt-5`
  - Current recommended: GPT-5 for vision/responses tasks

- **Google Gemini API**: For Q&A generation from dictionaries
  - Get key from: https://console.cloud.google.com/apis/credentials
  - Set in `.env`: `GOOGLE_API_KEY=...`
  - Model override: `GEMINI_MODEL=gemini-2.0-flash-exp` (latest recommended)

**Optional Integrations:**

- **HuggingFace**: Auto-publish training datasets
  - Set in `.env`: `HUGGINGFACE_TOKEN`, `HUGGINGFACE_DATASET_REPO`
  - Optional: `HUGGINGFACE_DATASET_PRIVATE=true`

- **Weights & Biases**: Experiment tracking for fine-tuning
  - Set in `.env`: `WANDB_API_KEY`, `WANDB_PROJECT`
  - Optional: `WANDB_ENTITY`, `WANDB_RUN_NAME`

## System Requirements

- Python 3.8+
- 8GB+ RAM (16GB recommended for Q&A generation)
- 10GB free disk space
- Stable internet connection for API calls

## Visualization

The project includes embeddings visualization via Nomic Atlas:
- [Original dictionary map](https://atlas.nomic.ai/data/harleycoops/stoney-1/map/8049523d-b620-4da3-962c-c509e08f586f)
- [42K expanded dictionary (April 2025)](https://atlas.nomic.ai/data/harleycoops/stoney-nakoda-language-synthetic/map/5c87caaf-6be0-4546-9e83-826569070b24)

These visualizations show semantic clustering using cosine similarity in embedding space, demonstrating how the model learns relationships between Stoney words and concepts.

## When to Use Each Pipeline

**Use Dictionary→Fine-tuning Pipeline when:**
- Building an initial translation model from scratch
- You have dictionary data (English↔Stoney word mappings)
- Goal is conversational fluency and general translation capability
- Leveraging pre-trained LLM knowledge (OpenAI models)
- Quick iteration and deployment is important

**Use Grammar→RL Pipeline when:**
- You have formal grammar documentation (PDF, scans, textbooks)
- Goal is precise morphological, syntactic, and phonological accuracy
- Need verifiable, rule-based training signals
- Working with cultural competence and nuanced language usage
- Building curriculum-based training with progressive difficulty
- Want to optimize for specific grammatical correctness (not just fluency)

**Recommended Approach:**
1. Start with Dictionary→Fine-tuning for base translation capability
2. Layer on Grammar→RL for grammatical refinement and cultural nuance
3. Iterate with community feedback (Community-In-The-Loop)

## Adapting for Other Languages

This approach can be applied to any low-resource language with:
1. A starting dictionary of ~8,000 words (for supervised pipeline)
2. Grammar documentation in PDF/scan form (for RL pipeline)
3. Community members who can provide narrative corrections
4. Cultural custodians to validate authenticity

The entire pipeline is language-agnostic and can be adapted by:
- Replacing dictionary files (`Dictionaries/`)
- Adjusting prompts in [bilingual_qa_generator.py](bilingual_qa_generator.py)
- Providing your language's grammar PDF
- Configuring `stoney_rl_grammar/config.py` for your language name

## Key Technical Notes

**Encoding & Special Characters:**
- All scripts use UTF-8 encoding with `ensure_ascii=False` to preserve Stoney orthography (š, ŋ, ć, doubled vowels, etc.)
- Critical for maintaining linguistic accuracy

**Checkpoint & Recovery:**
- `bilingual_qa_generator.py` creates checkpoints every 1000 Q&A pairs in `Dictionaries/checkpoints/`
- Resume from failures by checking checkpoint files

**Data Validation:**
- All JSONL files should have one JSON object per line
- OpenAI format requires `messages` array with `role` and `content` fields
- Dictionary format varies by file (see Dictionary Format section)

**Model Selection:**
- Dictionary pipeline: Uses Google Gemini (cheap, fast for Q&A generation) + OpenAI fine-tuning (quality)
- Grammar pipeline: Uses OpenAI vision models (GPT-4V or similar) for PDF rule extraction
- Override models via environment variables

**Progress Monitoring:**
- `tqdm` progress bars for all long-running operations
- `logging` module with timestamps for debugging
- W&B integration for real-time fine-tuning metrics (optional)

**RL Environment Integration:**
- The custom environment at `environments/stoney_nakoda_translation/` implements the `verifiers` framework interface
- Compatible with GRPO (Group Relative Policy Optimization) trainers
- Multi-signal reward functions: exact match, character-level F1, regex pattern matching
- Designed for qualitative (non-coding) language tasks

## Common Issues & Troubleshooting

**Issue: `FileNotFoundError` for dictionary files**
- Ensure `Dictionaries/english_dictionary.jsonl` and `Dictionaries/stoney_dictionary.jsonl` exist
- Check working directory is project root

**Issue: API rate limits or quota errors**
- Google Gemini: Batch size in `bilingual_qa_generator.py` can be reduced (line 80, default 5)
- OpenAI: Adjust sleep intervals if hitting rate limits
- Consider using checkpoints to resume after quota resets

**Issue: Fine-tuning job fails validation**
- Run `python finetunesetup.py` to ensure proper OpenAI format
- Check that all JSON in output files is valid (no trailing commas, proper quotes)
- Verify UTF-8 encoding is preserved

**Issue: Grammar RL pipeline produces no rules**
- Check that `Stoney; A Grammar of the Stony Language.pdf` exists in project root
- Ensure OpenAI API key has access to vision models (GPT-4V or similar)
- Review `data/grammar_extracted_stoney/*.json` for extraction errors
- Lower confidence threshold in `stoney_rl_grammar/rule_organizer.py` if too aggressive

**Issue: W&B or HuggingFace integration not working**
- Verify API tokens are correct in `.env`
- Check that `wandb` and `huggingface_hub` packages are installed (`pip install -r requirements.txt`)
- Review logs for specific authentication errors

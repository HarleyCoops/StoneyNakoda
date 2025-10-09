# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Community-In-The-Loop model distillation system for the Stoney Nakoda language. The project uses AI fine-tuning (primarily OpenAI's API) to create a bilingual translation model from a relatively small dictionary dataset (~8,000-10,000 words). The core innovation is iterative refinement through community feedback, where cultural context and narrative corrections improve the model over time.

**Active Model**: [Stoney Language Model App](https://huggingface.co/spaces/HarleyCooper/StoneyApp)
**Training Data**: [StoneyNakoda Training Dataset](https://huggingface.co/datasets/HarleyCooper/StoneyNakoda/blob/main/zSTONEY1_TRAINING_SET.jsonl)

## Core Architecture

The system operates as a three-stage pipeline:

1. **Data Generation** ([bilingual_qa_generator.py](bilingual_qa_generator.py)) - Uses Google Gemini to generate diverse Q&A pairs from dictionary entries
2. **Data Preparation** ([finetunesetup.py](finetunesetup.py)) - Converts Q&A pairs into OpenAI fine-tuning format with 80/20 train/validation split
3. **Model Training** ([openai_finetune.py](openai_finetune.py)) - Uploads data to OpenAI, creates fine-tuning jobs, and monitors progress

### Data Flow

```
Dictionary Files (Dictionaries/)
    ↓
bilingual_qa_generator.py (generates Q&A pairs using Gemini)
    ↓
bilingual_training_set.jsonl
    ↓
finetunesetup.py (converts to OpenAI format, splits data)
    ↓
OpenAIFineTune/stoney_train.jsonl + stoney_valid.jsonl
    ↓
openai_finetune.py (fine-tunes model via OpenAI API)
    ↓
Fine-tuned Model
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

### Complete Training Pipeline
```bash
# Step 1: Generate Q&A training data from dictionaries
python bilingual_qa_generator.py

# Step 2: Convert to OpenAI format and split train/validation
python finetunesetup.py

# Step 3: Upload and start fine-tuning job
python openai_finetune.py
```

### Data Format Conversion (if needed)
```bash
# Convert legacy formats to OpenAI messages format
python convert_data_format.py
```

## File Structure

```
StoneyNakoda/
├── Dictionaries/                    # Source dictionary files
│   ├── english_dictionary.jsonl     # English→Stoney mappings
│   ├── stoney_dictionary.jsonl      # Stoney→English mappings
│   └── bilingual_training_set.jsonl # Generated Q&A pairs
├── OpenAIFineTune/                  # Training files for OpenAI
│   ├── stoney_train.jsonl           # 80% training split
│   └── stoney_valid.jsonl           # 20% validation split
├── Public/                          # Images and documentation assets
├── bilingual_qa_generator.py        # Q&A generation via Gemini
├── finetunesetup.py                 # Data formatting and splitting
├── openai_finetune.py               # OpenAI fine-tuning orchestration
├── convert_data_format.py           # Legacy format converter
├── requirements.txt                 # Python dependencies
├── .env.example                     # API key template
├── RLHFrules.json                   # Language pattern rules for RL
└── README.md                        # Full project documentation
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

### bilingual_qa_generator.py
- Processes dictionaries in batches (default: 5 entries at a time)
- Uses Google Gemini API (`gemini-2.0-exp` model)
- Generates 5 Q&A pairs per batch with cultural context
- Creates checkpoints every 1000 pairs for recovery
- Target: 75,000 pairs per language (150,000 total)
- Output includes metadata: `pair_id`, `source_language`, `generated_at`

### finetunesetup.py
- Expects dictionary files in `English.Data/` directory
- Converts to OpenAI messages format with specific system prompt
- Uses 80/20 stratified split for train/validation
- Preserves UTF-8 encoding for special characters
- Output files: `stoney_dictionary_train.jsonl` and `stoney_dictionary_valid.jsonl`

### openai_finetune.py
- Default model: `gpt-3.5-turbo` (configurable to `gpt-4`)
- Hyperparameters: 3 epochs (adjustable in code)
- Monitors job status every 60 seconds
- Logs trained tokens, accuracy, validation loss during training
- Expects files at `OpenAIFineTune/stoney_train.jsonl` and `stoney_valid.jsonl`

### convert_data_format.py
- Utility for converting legacy JSONL formats
- Handles both old format and existing messages format
- Preserves special characters with `ensure_ascii=False`
- Currently configured for `data/raw/` and `data/processed/` directories

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

- **OpenAI API**: For fine-tuning (currently using GPT-3.5-turbo/GPT-4)
  - Get key from: https://platform.openai.com/api-keys
  - Set in `.env`: `OPENAI_API_KEY=sk-...`

- **Google Gemini API**: For Q&A generation
  - Get key from: https://console.cloud.google.com/apis/credentials
  - Set in `.env`: `GOOGLE_API_KEY=...`

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

## Adapting for Other Languages

This approach can be applied to any low-resource language with:
1. A starting dictionary of ~8,000 words
2. Community members who can provide narrative corrections
3. Cultural custodians to validate authenticity

The entire pipeline is language-agnostic and can be adapted by replacing dictionary files and adjusting prompts in [bilingual_qa_generator.py](bilingual_qa_generator.py).

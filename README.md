# From Whispers to Voices: A "Community-In-The-Loop" Proposal for Model Distillation and Language Preservation

## Project Overview

In my office, there is a murder; a map of one, at least. 

The scientist George Mercer Dawson, cut a wide swath through the Bow Valley in the late 1800s, and fortunately for us, he also noted language on the British Columbia side. His map, richly colored and showing his version of the nuance and depth of the Pacific Coast First nations that was known at the time, is the color of a tombstone over the Bow Valley. 

![Dawson's Map of the Bow Valley](Public/FullDawsonMap.jpg)

We all intuitively understand how languages in geographic proximity blend and change over time, kind of like linguistic DNA. The hope for machine learning is that there are threads of the fine structure of lost language that can be chased back to their origins like chasing a dying creek back to its headwaters. 

But in the age of AI, the isolation of the Stoney language isn't a curse, and it might be the actual cure. 

I have been thinking about how a model could be self-trained on a small set of 100% indigenous data and then be self taught to learn the fine structure of the broader Stoney Language for about 2 years and a recent model release by Meta on April 18th changed my thinking about what was possible. So I built it. The innovation here might be in thinking about the textbooks the community has created. 

The real innovation isn't in the machine; it's in the textbooks that the Stoney community has painstakingly created over the years. These weren't just educational tools—they were, unknowingly, perfect model prompts. Every chapter, every word, was a step toward fine-tuning a language model that could learn without interference from external biases or data.

This is only possible because the community created a 10,000 word dictionary, and then also created textbooks. Both of those are required to train the model. I literally cut up the textbooks and separated the questions and answers. Then passed those questions through the dictionary enough times until the right answer comes out. 

For the first time, an endangered language is being used to fine-tune its own model, a task only made possible because the community built a 10,000-word dictionary alongside their textbooks. Both were essential to train this model. When these textbooks are run back through their own dictionary, what emerges is a fluent, fine-tuned Stoney model. My role? To guide their data through the machine in the right format, nothing more.

In the code below, I have set up the data set, the StoneyTransformer and the Stony LLM functions are all built and working. 

A hundred years from now, strangers we don't know will be living in all our homes, the things we think matter likely don't, and we too will fade to the long shadow of humans that spent a short time here. Think about Stoney Woman as you celebrate this New Year, and think about who you know among the First Nations who would be interested in developing this project for their own language. I am available to help any nation with the code. 

## Project Architecture

### High-Level System Design
The system follows a modular architecture with distinct components for data processing, model training, and inference. The core architecture consists of:

1. **Data Ingestion Layer**
   - Handles raw dictionary and textbook data
   - Normalizes input formats
   - Validates data integrity

2. **Processing Pipeline**
   - Q&A pair generation
   - Data augmentation
   - Format conversion

3. **Model Training Framework**
   - Fine-tuning implementation
   - Hyperparameter management
   - Training monitoring

4. **Inference Interface**
   - API endpoint for model queries
   - Response formatting
   - Error handling

### Data Flow
1. Raw dictionary data → Data Ingestion Layer
2. Processed data → Q&A Generation
3. Generated Q&A pairs → Training Data Preparation
4. Prepared data → Model Fine-tuning
5. Fine-tuned model → Inference Interface

## Detailed Project Structure

```
PUBLICRELEASE/
├── OpenAIFineTune/               # Directory for OpenAI fine-tuning files
│   ├── stoney_train.jsonl        # Training dataset
│   └── stoney_valid.jsonl        # Validation dataset
├── checkpoints/                  # Model checkpoints directory
├── .env.example                  # Example environment variables
├── requirements.txt              # Python dependencies
├── english_dictionary.jsonl      # English source dictionary
├── stoney_dictionary.jsonl       # Stoney source dictionary
└── bilingual_training_set.jsonl  # Generated training data
```

## Core Components

### Data Generation and Processing
- `bilingual_qa_generator.py`: 
  - Processes dictionary entries
  - Implements advanced natural language generation techniques
  - Includes data validation and error handling
  - Generates diverse Q&A pairs through multiple strategies

- `convert_data_format.py`:
  - Supports multiple data formats (JSON, JSONL, CSV)
  - Implements data validation
  - Includes schema enforcement
  - Handles large datasets efficiently

- `finetunesetup.py`:
  - Implements stratified sampling for train/validation split
  - Includes data balancing techniques
  - Handles data preprocessing
  - Implements data versioning

### Model Training
- `openai_finetune.py`:
  - Implements fine-tuning with progress monitoring
  - Includes error handling and retry logic
  - Implements model checkpointing
  - Includes detailed logging
  - Supports multiple model configurations

## Comprehensive Setup Instructions

### System Requirements
- Python 3.8+
- 8GB RAM minimum (16GB recommended)
- 10GB disk space
- Stable internet connection

### Environment Setup
```bash
# Clone the repository
git clone [repository-url]
cd PUBLICRELEASE

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration
```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your API keys
# Required: OPENAI_API_KEY, GOOGLE_API_KEY
```

### Initialization
```bash
# Run initialization script
python initialize.py
```

## Detailed Usage Pipeline

### 1. Generate Training Data
```bash
python bilingual_qa_generator.py
```
This script:
- Processes entries from english_dictionary.jsonl and stoney_dictionary.jsonl
- Implements advanced natural language generation
- Includes data validation and error handling
- Generates diverse Q&A pairs through multiple strategies
- Outputs to bilingual_training_set.jsonl

### 2. Prepare Fine-tuning Data
```bash
python finetunesetup.py
```
This script:
- Converts Q&A pairs to OpenAI's format
- Creates 80/20 train/validation split
- Implements stratified sampling
- Includes data balancing techniques
- Outputs to OpenAIFineTune/stoney_train.jsonl and stoney_valid.jsonl

### 3. Fine-tune Model
```bash
python openai_finetune.py
```
This script:
- Uploads training files to OpenAI
- Implements fine-tuning with progress monitoring
- Includes error handling and retry logic
- Implements model checkpointing
- Provides detailed logging

## Advanced Model Configuration

### OpenAI Models
- Default: gpt-4o-2024-08-06
- Alternative: gpt-3.5-turbo
- Configure in .env: OPENAI_MODEL

### Google Gemini
- Default: gemini-2.0-exp
- Configure in .env: GEMINI_MODEL

### Hyperparameters
- Learning rate: 1e-5
- Batch size: 32
- Epochs: 3
- Context window: 4096 tokens

## Comprehensive Data Formats

### Dictionary Format
```json
{
    "english_word": "example",
    "stoney_versions": [
        {
            "word": "...",
            "grammatical_classification": "...",
            "meaning": "..."
        }
    ]
}
```

### Q&A Format
```json
{
    "question": "How do you say X in Stoney?",
    "answer": "The Stoney word for X is...",
    "source_language": "english",
    "generated_at": "timestamp"
}
```

### OpenAI Training Format
```json
{
    "messages": [
        {"role": "system", "content": "You are a bilingual Stoney-English language assistant..."},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"}
    ]
}
```

## Development Guidelines

### Code Style
- PEP 8 compliance
- Type hints for all functions
- Docstrings for all public methods
- Consistent naming conventions

### Testing
- Unit tests for all modules
- Integration tests for core workflows
- Continuous integration setup
- Test coverage reporting

### Documentation
- Inline code comments
- API documentation
- Usage examples
- Troubleshooting guide

## Contributing

### Contribution Process
1. Fork the repository
2. Create a feature branch
3. Implement changes
4. Write tests
5. Submit pull request

### Code Review Guidelines
- Clear commit messages
- Small, focused changes
- Proper documentation
- Test coverage

## License

This project is licensed under [LICENSE] - see the LICENSE file for details.

## Acknowledgments

- Stoney Nakoda First Nation for language expertise and guidance
- OpenAI and Google for AI model support
- Contributors and maintainers
- Academic advisors and linguistic experts

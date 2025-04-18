# From Whispers to Voices: A "Community-In-The-Loop" Proposal for Model Distillation and Language Preservation

A working model of the Stoney Nakoda language has been developed and is now available for community-in-the-loop testing in 2025:

- **Model App**: [Stoney Language Model App](https://huggingface.co/spaces/HarleyCooper/StoneyApp)  
- **Training Data**: [StoneyNakoda Training Dataset](https://huggingface.co/datasets/HarleyCooper/StoneyNakoda/blob/main/zSTONEY1_TRAINING_SET.jsonl)


Any First Nations community seeking to apply this approach to their own language is warmly invited to reach out. 

By following this code, you can build a model for any low-resource language. The starting dictionary size should be ~8,000 words.

---

<div align="center">
  <a href="https://huggingface.co/spaces/HarleyCooper/AskAboutCIL" target="_blank">
    <img src="https://img.shields.io/badge/Ask%20Google%20What%20This%20is%20About-FF0000?style=for-the-badge" alt="Ask Google What This is About">
  </a>
</div>

---

## Table of Contents

1. [New Years Day, Canadian Rockies, 2025](#introduction)  
2. [Understanding How AI Learns Stoney Words Using Cosine Similarity](#understanding-how-ai-learns-stoney-words-using-cosine-similarity)
3. [Project Architecture](#project-architecture)  
   - [High-Level System Design](#high-level-system-design)  
   - [Data Flow](#data-flow)  
4. [Detailed Project Structure](#detailed-project-structure)  
5. [Core Components](#core-components)  
   - [Data Generation & Processing](#data-generation--processing)
   - [Model Training](#model-training)
6. [Comprehensive Setup Instructions](#comprehensive-setup-instructions)  
   - [System Requirements](#system-requirements)
   - [Environment Setup](#environment-setup)
   - [Configuration](#configuration)
   - [Initialization](#initialization)
7. [Detailed Usage Pipeline](#detailed-usage-pipeline)  
   1. [Generate Training Data](#1-generate-training-data)  
   2. [Prepare Fine-tuning Data](#2-prepare-fine-tuning-data)  
   3. [Fine-tune Model](#3-fine-tune-model)  
8. [Advanced Model Configuration](#advanced-model-configuration)  
   - [OpenAI Models](#openai-models)  
   - [Google Gemini](#google-gemini)  
   - [Hyperparameters](#hyperparameters)  
9. [Comprehensive Data Formats](#comprehensive-data-formats)  
   - [Dictionary Format](#dictionary-format)  
   - [Q&A Format](#qa-format)  
   - [OpenAI Training Format](#openai-training-format)  
10. [Development Guidelines](#development-guidelines)  
11. [Contributing](#contributing)  
12. [License](#license)  
13. [Acknowledgments](#acknowledgments)  
14. [The Community-in-the-Loop Revolution](#the-community-in-the-loop-revolution)  
    - [Introduction](#introduction-1)  
    - [Conceptual Overview](#conceptual-overview)  
    - [Heart of the Approach](#heart-of-the-approach)  
    - [LoRA Fine-Tuning](#lora-fine-tuning)  
    - [Mathematical Foundations](#mathematical-foundations)  
    - [Mermaid Diagram](#mermaid-diagram)  
    - [Cultural Integrity](#cultural-integrity)  
    - [Data Sources](#data-sources)  
    - [Expanding the Concept](#expanding-the-concept)  
    - [Adaptive Checkpoints](#adaptive-checkpoints)  
    - [Example Workflow](#example-workflow)  
    - [Monitoring & QA](#monitoring--qa)  
    - [Future Directions](#future-directions)  
    - [Glossary](#glossary)  
15. [March 2025 RL Update](#march-2025-rl-update)
    - [Adapting Verifiers for Low-Resource Language Translation](#adapting-verifiers-for-low-resource-language-translation)
    - [Overview](#overview)
    - [Key Concepts](#key-concepts)
    - [Code Structure and Components](#code-structure-and-components)
    - [Running the Example (Not Built Yet)](#running-the-example-not-built-yet)
    - [Adapting to Your Specific Low-Resource Task](#adapting-to-your-specific-low-resource-task)
    - [Generating Reward Functions from Language Patterns](#generating-reward-functions-from-language-patterns)
16. [StoneyNakoda 42K Dictionary - April 2025](#stoney-nakoda-42k-dictionary---april-2025)

---

## Introduction

New Years Day, 2025


In my office, there is a murder; a map of one, at least.  

![Dawson's Map of the Bow Valley](Public/FullDawsonMap.jpg)

George Mercer Dawson explored the Bow Valley in the late 1800s as a geologist but also as a linguist, noting language on the British Columbia side. His map, though richly colored, stands like a tombstone over the Bow Valley where the Stoney people lived because he made no notes on their language and simply noted the people as "recent immigrants". Much of this work was based on the notes of W. Faser Tolmie and was published after his death. 

![Detail of Dawson Map](Public/dawsondetail.jpg)

What is very obvious from the linguistic patterns among the Haida, Tshimsia, Thlinkit, Kwakiool and Kawitshin dialects nearby is that languages blend like "linguistic DNA," and machine learning could help trace faint threads of lost speech to their roots. Where some see isolation as a curse, in the age of AI, Stoney's isolation turns out to be its strength.

For about two years, I thought about the size of the vector space that would be needed to get a model to self-train on a set of 100% indigenous data, and how that model could refine its grasp of the broader Stoney Language. This is now publicly and freely available. 


Two key releases influenced my thinking of what was possible:

1. [Meta's Llama-3 Model (April 18th, 2024)](https://www.reuters.com/technology/meta-releases-early-versions-its-llama-3-ai-model-2024-04-18/)  
2. [OpenAI Fine-Tuning API (October 2024)](https://openai.com/index/api-model-distillation/)

Both gave me the motivation to build what's presented here. The true innovation here lies in how communities can narratively correct the initially flawed response (about 10% of the time, the model works every time.) then that feeback be passed seamleslly back into the fine-tuning process. The [textbooks](https://globalnews.ca/news/9430501/stoney-nakota-language-textbook/) that the Stoney community created—intended as educational tools—became perfect concept of a model prompts, each chapter or word offering pure indigenous data devoid of external weights or biases to the fine-tuning process.


Early in 2023, I found an original, unpublished sketch by James Hector likely drawn in the summer of 1858 or 1859 along the Bow River in Southern Alberta:

![Sketch by James Hector of a Stoney Woman](Public/StoneyWoman.jpg)

Finding this, and already aware of George Mercer Dawson's work on First Nation's language on the British Columbia side, I was inspired to put the effort in and build a working model of the language and implement the Community-In-The-Loop distillation method.

This sketch shifted my thinking from considering the "Stoney People" to this "Stoney Woman" who saw these same mountains and rivers I see everyday, yet who had a very different way to think about and communicate to the world around her.  The Community-in-the-Loop model distillation will quickly converge this initial model toward fluencey. I suspect this will require the community to correct about 80,000 question and answer pairs and would cost less than $800 in OpenAI computing power. Recent releases by Google and the Chinese Lab DeepSeek, could effectively reduce the cost to zero.  

I think what this project has left me considering most is that a century from now, strangers will live in all our homes and most of what we worry about today will not matter. But we can honor "Stoney Woman" by making sure her language endures, forging a living record in an age of AI. Incredibly, this tool will work with any first nations language, as long as there is a starting dictionary of about 8,000 words. 

**I am freely available to help any First Nation in Canada.**

## Understanding How AI Learns Stoney Words Using Cosine Similarity

Word Embeddings: Mapping Words in Space
Word embeddings are like placing words in a high-dimensional map, where similar words are positioned closer together. For example, "strawberry," "orange," and "cherry" might form a cluster because they are fruits, while "laptop," "Microsoft," and "Android" might cluster elsewhere as tech-related terms. Each axis in this space represents a characteristic of the words, such as their context or meaning.

Context Shapes Meaning
A word's position in this space isn't fixed—it shifts based on context. For instance, the word "apple" could mean a fruit or the tech brand, depending on its surrounding words, like "buy" (tech) or "tree" (fruit). This dynamic placement captures the nuances of meaning.

Cosine Similarity: Measuring Relationships
Cosine similarity quantifies how similar two words are by measuring the angle between their vectors in the embedding space:

- Similar words have vectors pointing in nearly the same direction (cosine similarity close to 1)
- Unrelated words have vectors at a right angle (cosine similarity near 0)
- Opposite meanings have vectors pointing in opposite directions (cosine similarity close to -1)
- For example, "cherry" and "orange" might have a similarity of 0.97, while "cherry" and "laptop" might score 0.24

How AI Learns Stoney Words

- **Stoney Dictionary as a Starting Point:**
  The AI begins with a structured dictionary of Stoney words, including translations, categories, pronunciations, and cultural context.

- **Community Feedback for Learning:**
  The AI makes initial translations, which are often incorrect. Stoney speakers provide corrections, enriched with cultural context, stories, and humor. This feedback helps refine the AI's understanding.

The Role of Cosine Similarity in AI Learning

- The AI uses word embeddings to group Stoney words based on their meaning. For example, it determines whether a word belongs to a category like "fruit," "animal," or "spiritual."
- Community corrections and cosine similarity guide the AI in repositioning words closer to their accurate groupings in the embedding space.

Iterative Refinement
Through repeated feedback and fine-tuning, the AI improves its ability to place Stoney words correctly, not just individually but in the context of sentences and paragraphs. Over time, it develops a detailed, dynamic map of the Stoney language, with words clustered according to their community-informed meanings and uses.

Although this is not cosine similarity, you can see the relationships among words can concepts in Stoney as I have mapped them here:

[![Stoney Nakoda Language Map in Nomic Atlas](Public/nomic_atlas_preview.jpg)](https://atlas.nomic.ai/data/harleycoops/stoney-1/map/8049523d-b620-4da3-962c-c509e08f586f#iE2b)

### StoneyNakoda 42K Dictionary - April 2025
[![StoneyNakoda 42K Dictionary - April 2025 Nomic Atlas Map](Public/StoneyNakoda42k.jpg)](https://atlas.nomic.ai/data/harleycoops/stoney-nakoda-language-synthetic/map/5c87caaf-6be0-4546-9e83-826569070b24#JI7c)

---

## Project Architecture

This code forms a complete pipeline for training and deploying a Stoney model. It is fully functional—but not correct 100% of the time—and is designed to improve through Community-In-The-Loop feedback. Access the model here:  
[Stoney Language Model App](https://huggingface.co/spaces/HarleyCooper/StoneyApp)

### High-Level System Design

1. **Data Ingestion Layer**  
2. **Processing Pipeline** (Q&A generation, augmentation, conversion)  
3. **Model Training Framework** (fine-tuning, hyperparameters, monitoring)  
4. **Inference Interface** (API endpoint, response formatting, error handling)

### Data Flow

1. Raw dictionary data → Data Ingestion  
2. Processed data → Q&A Generation  
3. Generated Q&A pairs → Training Data Preparation  
4. Prepared data → Model Fine-tuning  
5. Fine-tuned model → Inference Interface  

---

## Detailed Project Structure

```
PUBLICRELEASE/
├── OpenAIFineTune/           # OpenAI fine-tuning files
│   ├── stoney_train.jsonl    # Training dataset
│   └── stoney_valid.jsonl    # Validation dataset
├── checkpoints/              # Model checkpoints
├── .env.example             # Env variables example
├── requirements.txt         # Python dependencies
├── english_dictionary.jsonl
├── stoney_dictionary.jsonl
└── bilingual_training_set.jsonl
```

---

## Core Components

### Data Generation & Processing

- **`bilingual_qa_generator.py`**  
  Generates Q&A pairs from dictionaries, using advanced language generation.

- **`convert_data_format.py`**  
  Supports multiple data formats; validates and enforces schemas.

- **`finetunesetup.py`**  
  Splits data (80/20) with stratified sampling and prepares files.

### Model Training

- **`openai_finetune.py`**  
  Handles fine-tuning, error handling, checkpointing, and logging.

---

## Comprehensive Setup Instructions

### System Requirements

- Python 3.8+  
- 8GB+ RAM (16GB recommended)  
- 10GB free disk space  
- Stable internet connection  

### Environment Setup

```bash
# Clone the repository
git clone [repository-url]
cd PUBLICRELEASE

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

### Configuration

```bash
# Copy example environment file
cp .env.example .env
# Provide OPENAI_API_KEY, GOOGLE_API_KEY in .env

```

### Initialization

```bash
python initialize.py

```

----------

## Detailed Usage Pipeline

### 1. Generate Training Data

```bash
python bilingual_qa_generator.py

```

-   Processes `english_dictionary.jsonl` & `stoney_dictionary.jsonl`
-   Produces `bilingual_training_set.jsonl`

### 2. Prepare Fine-tuning Data

```bash
python finetunesetup.py

```

-   Converts Q&A to OpenAI format
-   Outputs `OpenAIFineTune/stoney_train.jsonl` & `stoney_valid.jsonl`

### 3. Fine-tune Model

```bash
python openai_finetune.py

```

-   Uploads files to OpenAI
-   Monitors fine-tuning progress
-   Implements checkpointing & logs

----------

## Advanced Model Configuration

### OpenAI Models

-   Default: `gpt-4o-2024-08-06`
-   Alternative: `gpt-3.5-turbo`
-   `.env`: `OPENAI_MODEL`

### Google Gemini

-   Default: `gemini-2.0-exp`
-   `.env`: `GEMINI_MODEL`

### Hyperparameters

-   LR: `1e-5`
-   Batch size: `32`
-   Epochs: `3`
-   Context window: `4096`

----------

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
    {"role": "system", "content": "You are a bilingual Stoney-English assistant..."},
    {"role": "user", "content": "question"},
    {"role": "assistant", "content": "answer"}
  ]
}

```

----------

## Development Guidelines

-   **Style**: PEP 8, type hints, docstrings, consistent naming
-   **Testing**: Unit tests, integration tests, CI, coverage
-   **Documentation**: Inline comments, usage examples, troubleshooting

----------

## Contributing

1.  Fork, branch, implement changes, test
2.  Submit a pull request

**Code Review**

-   Clear commits, small changes, documentation, test coverage

----------

## The Community-in-the-Loop Revolution

### Introduction

This project aims to preserve, refine, and resurrect endangered languages via AI fine-tuning and model distillation. Minimal lexical data can evolve into a culturally rich digital speaker of Stoney Nakoda. This subverts assumptions that massive datasets are necessary, instead emphasizing:

-   Iterative improvement with community feedback
-   Narrative corrections (cultural context over simple dictionary entries)
-   Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning

### Conceptual Overview

**Community-in-the-Loop Model Distillation**:

1.  Start with a small dictionary/text set.
2.  Prompt an initial model.
3.  Let the community correct errors with storytelling and context, not just words.
4.  LoRA-based fine-tuning absorbs these narrative corrections.
5.  The model evolves iteratively, guided by cultural custodians.

### Heart of the Approach

-   **Intentional Errors**: Poke the model with tough or context-specific queries.
-   **Narrative Corrections**: Rich cultural commentary instead of bare "right vs. wrong."
-   **Distillation Triplets**: (Prompt, Disallowed Reply, Narrative Reply).
-   **Iterative Improvement**: If the model stumbles, revert and add more context.

### LoRA Fine-Tuning

LoRA attaches small, low-rank matrices to the base model. This dramatically reduces compute and speeds up retraining:

-   **Efficiency**: Fraction of resources required vs. full retraining
-   **Focused Updates**: Capturing the "essence" of new knowledge
-   **Rapid Iterations**: Frequent refinement without heavy overhead

### Mathematical Foundations

<div align="center">
  <img src="https://latex.codecogs.com/png.latex?\dpi{120}&space;\large&space;W&space;=&space;W_0&space;&plus;&space;\Delta&space;W&space;=&space;W_0&space;&plus;&space;AB" title="LoRA Weight Decomposition" />
</div>

LoRA (Low-Rank Adaptation) works by decomposing weight updates into smaller matrices, dramatically reducing parameters while preserving learning capacity.

#### Core Principles

- **Weight Decomposition**: If $W_0 \in \mathbb{R}^{d \times k}$ is the pre-trained weight matrix, LoRA parameterizes the update as:
  
  $$\Delta W = AB$$
  
  Where $A \in \mathbb{R}^{d \times r}$ and $B \in \mathbb{R}^{r \times k}$, with rank $r \ll \min(d,k)$

- **Parameter Efficiency**: For a typical transformer layer where $d=k=4096$, using $r=16$ reduces parameters by 99.2%

- **Forward Pass Computation**:
  
  $$h = W_0x + \Delta Wx = W_0x + ABx$$

#### Cultural-Linguistic Loss Function

Our approach combines standard cross-entropy loss with a novel Cultural Authenticity Score (CAS):

$$\mathcal{L} = \alpha\mathcal{L}_{CE} + (1-\alpha)\mathcal{L}_{CAS}$$

Where:
- $\mathcal{L}_{CE}$ is cross-entropy loss measuring linguistic accuracy
- $\mathcal{L}_{CAS}$ is derived from community feedback ratings (1-5 scale)
- $\alpha$ balances linguistic precision with cultural authenticity (typically 0.7)

#### Adaptation Dynamics

The adaptation process creates a specialized subspace that captures Stoney-specific linguistic patterns while preserving the base model's general capabilities. This enables:

1. **Targeted Learning**: Focus on Stoney-specific knowledge without catastrophic forgetting
2. **Efficient Transfer**: Leverage pre-trained knowledge while adding cultural nuance
3. **Iterative Refinement**: Quick adaptation cycles based on community feedback

### Mermaid Diagram

```mermaid
graph TD
    A[Initial Model] --> B[Generate Response]
    B --> C{Correct?}
    C -->|No| D[Community Correction]
    D --> E[Create Distillation Triplet]
    E --> F[LoRA Fine-Tuning]
    F --> A
    C -->|Yes| G[Validation]

```

### Cultural Integrity

Every correction preserves cultural norms—idioms, humor, oral traditions—and ensures the community wields control over the AI's "mindset."

### Data Sources

A 10,000-word Stoney Nakoda dictionary and community textbooks serve as seeds. Community feedback enriches this data over time, weaving historical memory into the model.

### Expanding the Concept

From a tiny dictionary to an AI that:

-   **Understands context** (formal/informal usage)
-   **Integrates cultural references** (stories, metaphors)
-   **Remembers history** (ancestors, ceremonies, seasonal events)

### Adaptive Checkpoints

-   **Forward Progress**: Keep the new checkpoint if improved.
-   **Reversion**: If degraded, roll back and increase context in corrections.
-   **Convergence**: Repeat until stable authenticity and fluency metrics are met.

### Example Workflow

1.  **Prompt**: "How to say 'taste slightly with the tip of your tongue' in Stoney?"
2.  **Model's Flawed Reply**: "`supthîyach`" (incorrect).
3.  **Community Correction**: Shares the correct phrase plus a story from childhood.
4.  **Distillation Triplet**: (Prompt, Disallowed, Narrative).
5.  **LoRA Fine-Tuning**: Model adjusts swiftly.
6.  **Re-Evaluation**: Answers improve in subsequent queries.

### Monitoring & QA

-   **Cultural Authenticity Score (CAS)**
-   **Linguistic Fluency** (perplexity, cross-entropy)
-   **Validation Loops** (watch for regressions, revert if needed)

### Future Directions

-   **Oral Histories**: Model retells century-old stories.
-   **Seasonal Knowledge**: Terms tied to ceremonies and ecological cycles.
-   **Dialects/Accents**: Respecting sub-regional differences.
-   **Educational Tools**: Interactive AI for language learning.
-   **Ethical AI**: Centered on consent, community governance, cultural integrity.

### Glossary

-   **CAS**: Cultural Authenticity Score
-   **Distillation Triplet**: (Prompt, Flawed Reply, Narrative Reply)
-   **LoRA**: Low-Rank Adaptation
-   **Community-in-the-Loop**: Paradigm of continuous human-guided refinement

---

## March 2025 RL Update

# Adapting Verifiers for Low-Resource Language Translation

This document details how to adapt the `verifiers` framework, originally designed for verifiable environments like math and coding, to the task of low-resource language translation.  This approach focuses on providing nuanced, multi-faceted rewards, going beyond simple correct/incorrect evaluations.

## Overview

The core idea is to treat translation as a multi-step process (even if it's just a single-turn translation) where the model receives rewards for various aspects of translation quality.  This allows for partial credit and provides more informative training signals, particularly beneficial in low-resource settings where data scarcity is a major challenge.

We will be customizing the following components of the `verifiers` library:

*   **Environment:** A custom `TranslationEnv` to handle the interaction with the translation model (LLM).
*   **Parser:** A simplified `TranslationParser` to extract the translated text from the LLM's output.  We won't require strict XML formatting for this task.
*   **Rubric:**  A `TranslationRubric` containing several reward functions that evaluate different quality dimensions (letter accuracy, word accuracy, semantic similarity, and edit distance).
*   **Training:** Using the `GRPOEnvTrainer` with our custom components and a small, low-resource translation dataset.

## Key Concepts

*   **Ground Truth:** A parallel corpus of source and target language sentences.  Essential for calculating rewards.  In low-resource scenarios, this might be a small, curated dataset.
*   **Multi-faceted Reward:**  Instead of a single reward, we provide separate rewards for:
    *   **Letter Accuracy:**  Proportion of correctly translated letters.
    *   **Word Accuracy:** Proportion of correctly translated words (space-separated).
    *   **Semantic Similarity:**  Uses pre-trained sentence embeddings (Sentence-BERT) to measure how close the *meaning* of the translation is to the ground truth, even if the exact words differ.
    *   **Edit Distance Similarity.** Levenshtein distances.
*   **Iterative Refinement (Optional):**  The environment can be designed to support multiple turns, allowing the LLM to refine its translation based on feedback (hints).  This example shows a rudimentary character by character suggestion technique, although a better version might provide hints more sparingly based on confidence scores.
*   **Low-Resource Focus:**  The techniques are tailored for scenarios with limited training data. This involves using smaller, specialized translation models (rather than massive general-purpose LLMs) and careful hyperparameter tuning (particularly `beta` in GRPO).

## Code Structure and Components

The code consists of the following main parts, each described in detail below:

1.  **`TranslationParser`:** A class to extract the translation from the LLM's output string.
2.  **`TranslationEnv`:**  A class inheriting from `MultiStepEnv` (or a simplified version) that defines the interaction loop between the trainer and the LLM.
3.  **`TranslationRubric`:**  A class inheriting from `Rubric` that defines the reward functions.
4.  **Dataset Creation (`create_dummy_dataset`):**  A function to load or create your low-resource translation dataset.  *You will replace this with your own dataset loading logic.*
5.  **Model Loading (`get_model_and_tokenizer`):** Uses functions from `verifiers` to load a suitable pre-trained translation model.
6.  **Training Setup (`GRPOEnvTrainer`):**  Sets up and runs the training process.

### 1. `TranslationParser`

```python
from types import SimpleNamespace

class TranslationParser:
    def parse(self, text: str, strip: bool = True) -> Any:
        translation = text.strip()
        return SimpleNamespace(translation=translation)
```

This simplified parser extracts the raw translated text from the LLM's output. We are not requiring or enforcing XML formatting, keeping the interaction straightforward.

### 2. TranslationEnv

```python
import verifiers as vf
from verifiers.envs import MultiStepEnv
from verifiers.rubrics import Rubric  # Will be used later.
from datasets import Dataset
from typing import List, Dict, Any

def check_prefix(text: str, suggested: str):
    if len(suggested) < 1:
        return False
    return text.startswith(suggested[:len(text)])
    
class TranslationEnv(MultiStepEnv):
    def __init__(self, dataset, system_prompt, max_steps=3):
        super().__init__(system_prompt=system_prompt, max_steps=max_steps, mask_env_response=False)
        self.dataset = dataset
        self.rubric = None # Set during get_rubric

    def get_dataset(self, **kwargs):
        return self.dataset
    def get_eval_dataset(self, **kwargs: Any):
      return self.dataset  # You might want separate eval set.

    def get_rubric(self):
        if self.rubric is None:
          self.rubric = TranslationRubric() # instantiate later.
        return self.rubric

    def is_completed(self, messages, **kwargs):
       assistant_text = self.rubric.parser.parse(messages[-1]['content']).translation
        user_query = self.get_last_user_prompt(messages)
        ground_truth = self.dataset.filter(lambda x: x["prompt"][0]['content'] == user_query)
        for element in ground_truth:
            target = element['answer']

        return check_prefix(target, assistant_text)

    def get_last_user_prompt(self, messages):
        i = len(messages) -1
        while i > -1:
           if messages[i]['role'] == 'user':
               return messages[i]['content']
           i-= 1
        return None
     # Suggest letters sequentially
    def env_response(self, messages, **kwargs):
        assistant_text = self.rubric.parser.parse(messages[-1]['content']).translation
        user_query = self.get_last_user_prompt(messages)
        ground_truth = self.dataset.filter(lambda x: x["prompt"][0]['content'] == user_query)

        response = "Check your word beginnings:"
        for element in ground_truth:
          target = element['answer']
          for i in range(0, min(len(target), len(assistant_text))):
              if target[i] != assistant_text[i]:
                   response += f" Your next correct letter choice starts with {target[i]}"
        return {"role": "user", "content": response}
```

Key Functions:

__init__: Initializes the environment with the dataset and system prompt. mask_env_response is set to False so suggestions/hints appear.

get_dataset: Returns the training dataset.

get_eval_dataset: Gets eval dataset

get_rubric: Returns an instance of the TranslationRubric.

is_completed: Checks if translation matches target, to terminate an interaction. We use custom checking logic by suggesting prefix matching, enabling hints, and then do similarity comparisons.

env_response Uses basic sequential suggestion algorithm. It will guide completion letter-by-letter if LLM fails.

### 3. TranslationRubric

```python
from verifiers.rubrics import Rubric
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict

class TranslationRubric(Rubric):
    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2'):
        super().__init__()
        self.parser = TranslationParser()
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.reward_funcs = [
            self.letter_accuracy_reward_func,
            self.word_accuracy_reward_func,
            self.semantic_similarity_reward_func,
            self.levenshtein_distance_reward_func,
        ]

    def letter_accuracy_reward_func(self, completions, answer, **kwargs) -> List[float]:
        rewards = []
        for completion, target in zip(completions, answer):
            completion_text = self.parser.parse(completion[0]["content"]).translation
            target_text = target.strip()

            min_len = min(len(completion_text), len(target_text))
            correct_letters = sum(1 for c1, c2 in zip(completion_text, target_text) if c1 == c2)
            reward = correct_letters / max(len(target_text), 1)  # Avoid division by zero

            rewards.append(reward)
        return rewards

    def word_accuracy_reward_func(self, completions, answer, **kwargs) -> List[float]:
        rewards = []
        for completion, target in zip(completions, answer):
            completion_text = self.parser.parse(completion[0]["content"]).translation
            target_words = target.strip().split()
            completion_words = completion_text.split()

            correct_words = sum(1 for cw in completion_words if cw in target_words)
            reward = correct_words / max(len(target_words), 1)
            rewards.append(reward)
        return rewards

    def semantic_similarity_reward_func(self, completions, answer, **kwargs) -> List[float]:
      rewards = []
      for completion, target in zip(completions, answer):
          completion_text = self.parser.parse(completion[0]["content"]).translation
          target_text = target.strip()

          try:
              completion_embedding = self.embedding_model.encode(completion_text, convert_to_numpy=True)
              target_embedding = self.embedding_model.encode(target_text, convert_to_numpy=True)
              # Cosine similarity
              similarity = np.dot(completion_embedding, target_embedding) / (np.linalg.norm(completion_embedding) * np.linalg.norm(target_embedding))
              rewards.append(max(0, similarity))  # Clip to be >= 0
          except Exception as e:
            print("Error during semantic similarity", e)
            rewards.append(0.0)
      return rewards

    def levenshtein_distance_reward_func(self, completions, answer, **kwargs) -> List[float]:
        def levenshtein_distance(s1, s2):
          if len(s1) > len(s2):
            s1, s2 = s2, s1
          distances = range(len(s1) + 1)
          for i2, c2 in enumerate(s2):
            distances_ = [i2+1]
            for i1, c1 in enumerate(s1):
              if c1 == c2:
                  distances_.append(distances[i1])
              else:
                  distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
          return distances[-1]

        rewards = []
        for completion, target in zip(completions, answer):
          completion_text = self.parser.parse(completion[0]["content"]).translation
          target_text = target.strip()
          distance = levenshtein_distance(completion_text, target_text)
          normalized_distance =  distance / max(len(completion_text), len(target_text), 1) # Avoid division by zero
          rewards.append(1.0 - normalized_distance)
        return rewards
```

Key Components:

__init__: Initializes the rubric with a TranslationParser and a Sentence-BERT model for semantic similarity calculations. You can change the embedding_model_name to use different pre-trained embeddings.

letter_accuracy_reward_func: Calculates the proportion of correct letters.

word_accuracy_reward_func: Calculates the proportion of correct words.

semantic_similarity_reward_func: Calculates the cosine similarity between the sentence embeddings of the generated translation and the ground truth.

levenshtein_distance_reward_func: Provides similarity based on edit distances

### 4. Dataset Creation (create_dummy_dataset)

```python
from datasets import Dataset
import verifiers as vf

def create_dummy_dataset():
  data = {
      'prompt': [
        vf.format_prompt("Translate to French: 'The cat is on the mat.'", "You are a translation expert."),
        vf.format_prompt("Translate to French: good morning", "You are a translation expert.")
      ],
      'answer': ["Le chat est sur le tapis.", "Bonjour"]
  }
  return Dataset.from_dict(data)
```

Important: This is a placeholder. You'll need to replace this with code that loads your low-resource parallel text dataset and creates a Hugging Face Dataset object with 'prompt' and 'answer' columns. The 'prompt' should contain the source sentence and any system prompt, and the 'answer' should contain the target translation.

### 5. Model Loading (get_model_and_tokenizer)

```python
import verifiers as vf

model_name = "Helsinki-NLP/opus-mt-en-fr"  # Example: English to French
model, tokenizer = vf.get_model_and_tokenizer(model_name)
```

This uses the verifiers utility functions to load a pre-trained translation model and its corresponding tokenizer. Choose a model appropriate for your language pair. Start with smaller models for efficiency, especially in a low-resource setting.

### 6. Training Setup (GRPOEnvTrainer)

```python
from verifiers.trainers.grpo_env_trainer import GRPOEnvTrainer

# Create dataset instances.  YOU WILL REPLACE create_dummy_dataset!
train_dataset = create_dummy_dataset()
eval_dataset = create_dummy_dataset()

# Set up environment and rubric.
vf_env = TranslationEnv(dataset=train_dataset, system_prompt="You are a translation expert.")
rubric = vf_env.get_rubric()  # Get the rubric *from* the environment

run_name = "translation_example"
# set training to be short
training_args = vf.get_default_grpo_config(run_name=run_name, num_gpus=8)
training_args.num_generations = 1 # reduce data
training_args.max_steps = 3       # Short training for illustration

trainer = GRPOEnvTrainer(
    model=model,
    tokenizer=tokenizer,
    env=vf_env,
    reward_funcs=rubric.reward_funcs,
    args=training_args,
    train_dataset=train_dataset,
   # eval_dataset=eval_dataset
)

trainer.train()
```

This part sets up the GRPOEnvTrainer with the custom environment, rubric, dataset, model, and tokenizer. Key parameters to consider tuning, especially in low-resource settings, are in training_args.

## Running the Example (Not Built Yet)
- The idea here is to get completely away from the OpenAI fine tuning I use now to any open source model. The idea I'm going to build here is to give any community the tool to input their language as they understand it, train that model on any opensource model, likey with LoRA, and achieve better and better output. 

Install Dependencies: Make sure you have the required packages installed (see your original pyproject.toml). Notably: sentence-transformers torch transformers. Use uv or other packaging method.

Run the Code: Combine the code snippets above into a single Python file (e.g., translation_trainer.py). Execute the script:

```bash
python translation_trainer.py
```

This will run a very short training demonstration on the dummy dataset. You should see output from the trainer and (if you enable logging) see the prompts, completions, and the calculated rewards.

## Adapting to Your Specific Low-Resource Task

Dataset: Replace create_dummy_dataset() with your data loading.

Model: Choose a suitable pre-trained translation model for your languages.

is_completed and Hints. Change these parts to improve hints.

Hyperparameters: Experiment with the GRPOConfig parameters. Start with a low learning rate and consider increasing beta (the KL divergence penalty) to prevent overfitting on a small dataset. A larger beta keeps the model's weights closer to the pre-trained values.

## Generating Reward Functions from Language Patterns

One idea I have for generating a series of reward functions on a low resource language is to simply pass the JSON dictionary of the Stoney Nakoda, and ask for the rules or patterns the LLM notices. 

It will give you a full set of rules that you can then use to define a very large number of very small reward functions that can be used to very precisely fine tune even low resource languages around contours. 

Here is the actual LLM output using this simple idea: [RLHFrules.json](RLHFrules.json)

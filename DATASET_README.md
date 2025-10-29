---
language:
- en
- srs
license: cc-by-4.0
task_categories:
- translation
- question-answering
pretty_name: Stoney Nakoda Bilingual Training Set
size_categories:
- 10K<n<100K
---

# Stoney Nakoda Bilingual Training Dataset (10K)

## Dataset Description

This dataset contains 10,000 bilingual question-answer pairs designed for training language models on English-Stoney Nakoda translation tasks. The Stoney Nakoda language (also known as Nakoda or Assiniboine) is an Indigenous language spoken by the Stoney Nakoda peoples in Alberta, Canada.

### Dataset Summary

- **Total Entries:** 10,000 question-answer pairs
- **Languages:** English (source) and Stoney Nakoda (target)
- **Format:** JSONL (JSON Lines)
- **Task:** Translation and language learning

### Data Structure

Each entry in the dataset contains the following fields:

```json
{
  "question": "How would you express 'to be ablaze' in Stoney Nakoda?",
  "answer": "\"tâga înech\". This is the direct verb translation for 'to be ablaze' from the linguistic material provided.",
  "source_language": "english",
  "generated_at": "2025-10-28T20:03:38.435460",
  "pair_id": 1
}
```

**Field Descriptions:**
- `question`: The English prompt asking for translation
- `answer`: The Stoney Nakoda translation with contextual explanation
- `source_language`: The source language of the question (always "english")
- `generated_at`: Timestamp of when the entry was generated
- `pair_id`: Unique identifier for each question-answer pair

## Intended Use

### Primary Use Cases

1. **Machine Translation:** Training models to translate between English and Stoney Nakoda
2. **Language Model Fine-tuning:** Enhancing LLMs with Stoney Nakoda language understanding
3. **Educational Resources:** Supporting language learning and preservation efforts
4. **Linguistic Research:** Analyzing Stoney Nakoda language patterns and structure

### Out-of-Scope Use

This dataset should not be used for:
- Commercial applications without proper consultation with Stoney Nakoda communities
- Any purpose that could harm or misrepresent Indigenous peoples or their languages

## Dataset Creation

### Source Data

The dataset was generated from linguistic materials related to the Stoney Nakoda language, focusing on vocabulary, phrases, and grammatical structures documented in linguistic research.

### Data Collection Process

The question-answer pairs were systematically generated to cover a wide range of vocabulary and linguistic concepts, ensuring comprehensive coverage of the language's core elements.

## Considerations for Using the Data

### Social Impact

This dataset represents an important effort in Indigenous language preservation and revitalization. Users should:
- Respect the cultural significance of the Stoney Nakoda language
- Consult with Stoney Nakoda communities when developing applications
- Credit Indigenous knowledge holders and language keepers
- Support ongoing language revitalization efforts

### Limitations

- The dataset may not capture all dialectal variations of Stoney Nakoda
- Contemporary usage and evolving language patterns may not be fully represented
- Context-specific or ceremonial language uses require community guidance

## Licensing and Attribution

**License:** CC-BY-4.0

When using this dataset, please:
1. Acknowledge the Stoney Nakoda people as the source of this linguistic knowledge
2. Support Indigenous language revitalization initiatives
3. Cite this dataset in academic or research publications

## Citation

```
@dataset{stoney_nakoda_10k_2025,
  title={Stoney Nakoda Bilingual Training Dataset},
  author={HarleyCooper},
  year={2025},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/datasets/HarleyCooper/Stoney10kRL}}
}
```

## Contact

For questions, corrections, or collaboration opportunities related to this dataset, please open an issue in the repository.

## Acknowledgments

This dataset is part of ongoing efforts to preserve and promote the Stoney Nakoda language. We acknowledge the Stoney Nakoda peoples and their ancestral territories in what is now Alberta, Canada.

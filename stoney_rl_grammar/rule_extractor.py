"""LLM-driven grammar rule extraction from PDF chunks."""

from __future__ import annotations

import json
import logging
from typing import Dict, Iterable, List, Sequence

from dotenv import load_dotenv
from openai import OpenAI
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential

from .config import (
    DEFAULT_EXTRACTION_MODEL,
    EXTRACTION_OUTPUT_DIR,
    MAX_OUTPUT_RULES_PER_CHUNK,
    ensure_directories,
)
from .models import GrammarRule, PageChunk

logger = logging.getLogger(__name__)


def _build_extraction_prompt(chunk: PageChunk) -> str:
    instructions = f"""
You are an expert linguist specializing in Stoney Nakoda grammar.
You are given a scanned grammar page (page {chunk.page_number}). Inspect the image carefully.
Extract up to {MAX_OUTPUT_RULES_PER_CHUNK} distinct grammar rules from the page.
Focus on morphological, phonological, syntactic, and translation notes.

For each rule provide:
- title: short human-readable heading
- description: concise explanation in English
- category: choose from ["morphology","syntax","phonology","translation","semantics","phonotactics","misc"]
- stoney_examples: list of Stoney examples with translations inline if given
- english_examples: list of English examples if present
- verification_hint: regex-style or checklist hint for evaluation
- confidence: value between 0 and 1 based on clarity of source evidence
- page_number: {chunk.page_number}
- chunk_id: "{chunk.chunk_id()}"

Return JSON with schema:
{{
  "rules": [
    {{
      "title": "...",
      "description": "...",
      "category": "...",
      "stoney_examples": ["..."],
      "english_examples": ["..."],
      "verification_hint": "...",
      "confidence": 0.0,
      "page_number": {chunk.page_number},
      "chunk_id": "{chunk.chunk_id()}"
    }}
  ]
}}
"""
    return instructions.strip()


class StoneyGrammarExtractor:
    """Orchestrates LLM calls to convert text chunks into structured rules."""

    def __init__(
        self,
        model: str = DEFAULT_EXTRACTION_MODEL,
        temperature: float = 0.2,
        max_output_tokens: int = 2000,
    ) -> None:
        ensure_directories()
        load_dotenv()
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    @retry(wait=wait_exponential(multiplier=1, min=2, max=20), stop=stop_after_attempt(3))
    def _call_model(self, prompt: str, chunk: PageChunk) -> Dict:
        content = [{"type": "text", "text": prompt}]
        if chunk.text:
            content.append({"type": "text", "text": chunk.text})
        if chunk.image_b64:
            content.append({"type": "input_image", "image_base64": chunk.image_b64})

        response = self.client.responses.create(
            model=self.model,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            response_format={"type": "json_object"},
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You extract Stoney Nakoda grammar rules from scholarly text.",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": content,
                },
            ],
        )
        content = response.output[0].content[0].text  # type: ignore[index]
        return json.loads(content)

    def extract_rules(self, chunks: Sequence[PageChunk]) -> List[GrammarRule]:
        extracted_rules: List[GrammarRule] = []
        for chunk in chunks:
            if not chunk.image_b64 and not chunk.text:
                logger.warning("Skipping %s because no source content was provided", chunk.chunk_id())
                continue
            prompt = _build_extraction_prompt(chunk)
            try:
                payload = self._call_model(prompt, chunk)
            except RetryError as exc:
                if hasattr(exc, "last_attempt") and exc.last_attempt:
                    last_exc = exc.last_attempt.exception()
                else:
                    last_exc = exc
                logger.error("Unable to process %s: %s", chunk.chunk_id(), last_exc)
                continue

            rules_data = payload.get("rules", [])
            if not isinstance(rules_data, list):
                logger.warning("Unexpected payload structure for %s", chunk.chunk_id())
                continue

            chunk_rules: List[Dict] = []
            for idx, raw in enumerate(rules_data, start=1):
                rule_id = f"{chunk.chunk_id()}_rule_{idx:02d}"
                try:
                    rule = GrammarRule(
                        rule_id=rule_id,
                        title=raw.get("title", "").strip(),
                        description=raw.get("description", "").strip(),
                        category=raw.get("category", "misc").strip() or "misc",
                        stoney_examples=[s.strip() for s in raw.get("stoney_examples", [])],
                        english_examples=[s.strip() for s in raw.get("english_examples", [])],
                        verification_hint=raw.get("verification_hint", "").strip(),
                        confidence=raw.get("confidence"),
                        source_page=raw.get("page_number", chunk.page_number),
                        source_chunk=raw.get("chunk_id", chunk.chunk_id()),
                    )
                except Exception as err:  # defensive parsing
                    logger.error("Failed to build rule for %s: %s", rule_id, err)
                    continue

                extracted_rules.append(rule)
                chunk_rules.append(rule.to_dict())

            if chunk_rules:
                output_path = EXTRACTION_OUTPUT_DIR / f"{chunk.chunk_id()}.json"
                output_path.write_text(
                    json.dumps(
                        {"chunk_id": chunk.chunk_id(), "rules": chunk_rules},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
        return extracted_rules

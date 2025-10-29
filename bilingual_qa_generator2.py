"""Standalone bilingual QA generator with richer prompting.

This script mirrors the original generator but provides a more
deliberate prompt design aimed at producing deeper, scenario-rich
question / answer pairs while keeping the JSON structure identical
(`{"question": "...", "answer": "..."}`).
"""

import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import Dict, Generator, List

import google.generativeai as genai
from dotenv import load_dotenv
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class BilingualQAGeneratorV2:
    """Generates bilingual QA pairs using an enriched prompting strategy."""

    def __init__(self, english_dict_file: str, stoney_dict_file: str):
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        genai.configure(api_key=api_key)
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        self.model = genai.GenerativeModel(
            self.model_name,
            generation_config={"response_mime_type": "application/json"},
        )
        self.english_dict_file = Path(english_dict_file)
        self.stoney_dict_file = Path(stoney_dict_file)

        if not self.english_dict_file.exists():
            raise FileNotFoundError(
                f"English dictionary file not found: {english_dict_file}"
            )
        if not self.stoney_dict_file.exists():
            raise FileNotFoundError(
                f"Stoney dictionary file not found: {stoney_dict_file}"
            )

    def create_english_context_prompt(self, entries: List[Dict]) -> str:
        """Builds a context block for English-to-Stoney entries."""
        context = dedent(
            """
            ROLE
            - You are a Stoney Nakoda linguist, curriculum designer, and cultural consultant.

            OBJECTIVE
            - Using the English -> Stoney entries below, craft high-quality question / answer pairs
              that stretch learners beyond direct translations while preserving accuracy.

            DESIGN PRINCIPLES
            - Weave scenarios that force learners to reason about nuance, part-of-speech,
              morphology, and cultural appropriateness.
            - Reference multiple dictionary entries in at least two questions to encourage
              comparative thinking.
            - Highlight when multiple Stoney options exist for one English concept and explain
              the selection criteria in the answer.
            - Blend question styles: direct translation, situational usage, grammatical analysis,
              pragmatic / cultural insight, and pattern discovery.
            - Keep question text concise but meaningful (1-2 sentences). Answers can be up to two
              sentences that provide the translation plus a short explanation.
            - If you introduce contextual details (e.g., who is speaking, location, social setting),
              ensure they align with the cultural norms implied by the dictionary entries.

            LINGUISTIC MATERIAL
            """
        )

        for entry in entries:
            context += f"\n{json.dumps(entry, ensure_ascii=False)}"

        return context

    def create_stoney_context_prompt(self, entries: List[Dict]) -> str:
        """Builds a context block for Stoney-to-English entries."""
        context = dedent(
            """
            ROLE
            - You are a fluent Stoney Nakoda language bearer collaborating with pedagogy experts.

            OBJECTIVE
            - Using the Stoney -> English entries below, produce question / answer pairs that demand
              careful interpretation of meaning, usage constraints, and grammatical behaviour.

            DESIGN PRINCIPLES
            - Encourage learners to justify why particular English renderings best fit the Stoney
              terms, referencing affixes, classifiers, aspect, or animacy where relevant.
            - Include questions that show how context shifts the preferred English translation.
            - Blend question styles: translation back to English, fill-in-the-blank with
              explanations, morphological breakdowns, communicative scenarios, and cultural notes.
            - Draw explicit attention to semantic contrasts or complementary pairs across entries.
            - Keep questions to 1-2 sentences, and answers up to two sentences with both the
              translation and a brief rationale.
            - Maintain cultural respect; if referencing ceremonies, kinship, or land, ensure the
              tone is appropriate and aligns with the provided entries.

            LINGUISTIC MATERIAL
            """
        )

        for entry in entries:
            context += f"\n{json.dumps(entry, ensure_ascii=False)}"

        return context

    def _build_generation_prompt(self, perspective: str) -> str:
        """Creates the instruction block presented alongside the dictionary context."""
        coverage = dedent(
            f"""
            TASK
            - Produce exactly five complementary question / answer pairs that will be added to a
              bilingual training set. Work from the {perspective} viewpoint implied by the context.

            COVERAGE REQUIREMENTS
            1. Direct translation check grounded in a single entry.
            2. Scenario-based usage question that embeds cultural or situational nuance.
            3. Morphology or grammar-focused question (e.g., classifiers, aspect, word class).
            4. Contrastive reasoning question weaving at least two provided entries.
            5. Pattern or extension question that helps learners generalise responsibly.

            AUTHORING GUIDELINES
            - Use varied question openings (e.g., "How would...", "Which term...", "In this situation...").
            - When giving the answer, cite the relevant Stoney or English terms verbatim so the data
              stays searchable.
            - Answers must stay within two sentences and include the direct response plus a concise
              justification grounded in the entries.
            - Do not fabricate vocabulary not present in the supplied entries; if additional context
              is needed, build it using culturally respectful, high-level descriptions.
            - Avoid duplicate questions, and ensure each pair targets a distinct learning objective.

            OUTPUT FORMAT (STRICT)
            [
              {{
                "question": "<string>",
                "answer": "<string>"
              }}
            ]
            - Return a valid JSON array with exactly five objects matching the schema above.
            - Do not include comments, trailing commas, or additional fields.
            """
        )
        return coverage

    def generate_qa_pairs(
        self, dictionary_file: Path, is_english: bool, context_size: int = 6
    ) -> Generator[Dict, None, None]:
        """Streams QA pairs, yielding one at a time."""
        entries_buffer: List[Dict] = []

        with open(dictionary_file, "r", encoding="utf-8") as f:
            for line in tqdm(
                f, desc=f"Processing {'English' if is_english else 'Stoney'} entries"
            ):
                try:
                    entry = json.loads(line.strip())
                    entries_buffer.append(entry)

                    if len(entries_buffer) >= context_size:
                        if is_english:
                            context = self.create_english_context_prompt(entries_buffer)
                            perspective = "English -> Stoney"
                        else:
                            context = self.create_stoney_context_prompt(entries_buffer)
                            perspective = "Stoney -> English"

                        prompt = context + "\n\n" + self._build_generation_prompt(
                            perspective
                        )

                        response = self.model.generate_content(prompt)
                        if self._is_blocked(response):
                            entries_buffer = []
                            continue
                        if not response or not response.candidates:
                            logger.warning("No response from model, skipping batch")
                            entries_buffer = []
                            continue

                        text = self._collect_response_text(response)
                        try:
                            qa_list = self._parse_qa_list(text)
                            for qa in qa_list:
                                yield {
                                    "question": qa["question"],
                                    "answer": qa["answer"],
                                    "source_language": "english"
                                    if is_english
                                    else "stoney",
                                }
                        except ValueError as err:
                            logger.warning(
                                "Invalid JSON payload returned, skipping batch: %s", err
                            )

                        entries_buffer = []

                except json.JSONDecodeError:
                    logger.warning("Skipping invalid JSON line")
                    continue

    def generate_training_set(self, output_file: str, pairs_per_language: int = 5000):
        """Generates a bilingual training set with enriched prompts."""
        output_path = Path(output_file)
        checkpoint_dir = output_path.parent / "checkpoints_v2"
        checkpoint_dir.mkdir(exist_ok=True)

        total_pairs = pairs_per_language * 2
        pair_count = 0
        checkpoint_count = 0
        start_time = time.time()

        logger.info(
            "Starting enriched generation of %s total Q&A pairs (%s per language)...",
            total_pairs,
            pairs_per_language,
        )

        with open(output_path, "w", encoding="utf-8") as f:
            logger.info("Generating English-perspective Q&A pairs...")
            english_count = 0
            for qa_pair in self.generate_qa_pairs(self.english_dict_file, True):
                if english_count >= pairs_per_language:
                    break
                qa_pair["generated_at"] = datetime.now().isoformat()
                qa_pair["pair_id"] = pair_count + 1
                f.write(json.dumps(qa_pair, ensure_ascii=False) + "\n")
                pair_count += 1
                english_count += 1

                if pair_count % 1000 == 0:
                    self._create_checkpoint(
                        checkpoint_dir, checkpoint_count, pair_count, total_pairs
                    )
                    checkpoint_count += 1

            logger.info("Generating Stoney-perspective Q&A pairs...")
            stoney_count = 0
            for qa_pair in self.generate_qa_pairs(self.stoney_dict_file, False):
                if stoney_count >= pairs_per_language:
                    break
                qa_pair["generated_at"] = datetime.now().isoformat()
                qa_pair["pair_id"] = pair_count + 1
                f.write(json.dumps(qa_pair, ensure_ascii=False) + "\n")
                pair_count += 1
                stoney_count += 1

                if pair_count % 1000 == 0:
                    self._create_checkpoint(
                        checkpoint_dir, checkpoint_count, pair_count, total_pairs
                    )
                    checkpoint_count += 1

        elapsed = time.time() - start_time
        logger.info("Generation completed in %.2f seconds", elapsed)

    def _create_checkpoint(
        self, checkpoint_dir: Path, checkpoint_count: int, pair_count: int, total_pairs: int
    ) -> None:
        """Writes lightweight progress checkpoints to disk."""
        checkpoint_file = checkpoint_dir / f"checkpoint_{checkpoint_count}.jsonl"
        with open(checkpoint_file, "w", encoding="utf-8") as cf:
            cf.write(
                json.dumps(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "pairs_generated": pair_count,
                        "target_pairs": total_pairs,
                        "percent_complete": (pair_count / total_pairs) * 100,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


    def _collect_response_text(self, response) -> str:
        """Concatenates text parts from a Gemini response into a single string."""
        text_chunks: List[str] = []
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []) or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    text_chunks.append(part_text)
        return "\n".join(text_chunks).strip()


    def _parse_qa_list(self, raw_text: str) -> List[Dict[str, str]]:
        """Parses and validates the JSON array of QA objects returned by Gemini."""
        cleaned = raw_text.strip()
        if not cleaned:
            raise ValueError("response text was empty")

        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        match = re.search(r"\[[\s\S]*\]", cleaned)
        if not match:
            raise ValueError("no JSON array found in response")

        try:
            qa_list = json.loads(match.group())
        except json.JSONDecodeError as exc:
            raise ValueError(f"unable to decode JSON: {exc}") from exc

        if not isinstance(qa_list, list):
            raise ValueError("decoded payload was not a list")

        normalised: List[Dict[str, str]] = []
        for index, item in enumerate(qa_list):
            if not isinstance(item, dict):
                raise ValueError(f"item {index} was not an object")
            question = item.get("question")
            answer = item.get("answer")
            if not isinstance(question, str) or not isinstance(answer, str):
                raise ValueError(f"item {index} missing string question/answer fields")
            normalised.append({"question": question.strip(), "answer": answer.strip()})

        if len(normalised) != 5:
            raise ValueError(f"expected 5 QA pairs, received {len(normalised)}")

        return normalised


def main():
    """Convenience entrypoint for manual runs."""
    try:
        english_dict_path = "Dictionaries/english_dictionary.jsonl"
        stoney_dict_path = "Dictionaries/stoney_dictionary.jsonl"
        output_path = "Dictionaries/bilingual_training_set_v2.jsonl"

        generator = BilingualQAGeneratorV2(english_dict_path, stoney_dict_path)

        logger.info("Starting standalone enriched training set generation...")
        generator.generate_training_set(output_path, pairs_per_language=5000)
        logger.info("Enriched training set generation completed successfully")
    except Exception as exc:
        logger.error("Error during generation: %s", exc)


if __name__ == "__main__":
    main()

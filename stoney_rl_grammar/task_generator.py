"""Generate RL training tasks from curated grammar rules."""

from __future__ import annotations

import json
import logging
from typing import Dict, Iterable, List, Sequence

from dotenv import load_dotenv
from openai import OpenAI
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential

from .config import (
    DEFAULT_TASK_MODEL,
    TASK_DATASET_OUTPUT_PATH,
    TASK_GENERATION_TEMPERATURE,
    ensure_directories,
)
from .models import GrammarRule, RLTrainingTask

logger = logging.getLogger(__name__)


def _build_task_prompt(rule: GrammarRule, max_tasks: int) -> str:
    return f"""
You are building reinforcement-learning tasks to teach the Stoney Nakoda grammar rule below.

Rule:
Title: {rule.title}
Category: {rule.category}
Description: {rule.description}
Stoney examples: {rule.stoney_examples or "None"}
English examples: {rule.english_examples or "None"}
Verification hint: {rule.verification_hint or "None"}

Create between 3 and {max_tasks} concise tasks that help a model practice this rule.
Allowed task_type values: ["morphology","forward_translation","reverse_translation","pattern_identification","syntax_analysis"].

For each task provide:
- task_type
- prompt (should include instructions for the model)
- ideal_answer
- verification_pattern (regex or checklist for auto-verification)
- hints (list of brief hints for multi-turn RL environments)
- difficulty (easy/medium/hard)

Return JSON with schema:
{{
  "tasks": [
    {{
      "task_type": "...",
      "prompt": "...",
      "ideal_answer": "...",
      "verification_pattern": "...",
      "hints": ["..."],
      "difficulty": "easy"
    }}
  ]
}}
"""


class StoneyTaskGenerator:
    """Transforms grammar rules into RL-friendly training tasks."""

    def __init__(
        self,
        model: str = DEFAULT_TASK_MODEL,
        temperature: float = TASK_GENERATION_TEMPERATURE,
        max_output_tokens: int = 2000,
        max_tasks_per_rule: int = 6,
    ) -> None:
        ensure_directories()
        load_dotenv()
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_tasks_per_rule = max_tasks_per_rule

    @retry(wait=wait_exponential(multiplier=1, min=2, max=20), stop=stop_after_attempt(3))
    def _call_model(self, prompt: str) -> Dict:
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
                            "text": "You design short RL tasks that evaluate specific Stoney Nakoda grammar rules.",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                },
            ],
        )
        content = response.output[0].content[0].text  # type: ignore[index]
        return json.loads(content)

    def generate_tasks(self, rules: Sequence[GrammarRule]) -> List[RLTrainingTask]:
        """Generate and persist RL tasks for each grammar rule."""
        tasks: List[RLTrainingTask] = []
        with TASK_DATASET_OUTPUT_PATH.open("w", encoding="utf-8") as jsonl_file:
            for rule in rules:
                prompt = _build_task_prompt(rule, self.max_tasks_per_rule)
                try:
                    payload = self._call_model(prompt)
                except RetryError as exc:
                    logger.error("Unable to generate tasks for %s: %s", rule.rule_id, exc)
                    continue

                raw_tasks = payload.get("tasks", [])
                if not isinstance(raw_tasks, list):
                    logger.warning("Unexpected task payload for %s", rule.rule_id)
                    continue

                for idx, raw in enumerate(raw_tasks, start=1):
                    task_id = f"{rule.rule_id}_task_{idx:02d}"
                    task = RLTrainingTask(
                        task_id=task_id,
                        rule_id=rule.rule_id,
                        task_type=raw.get("task_type", "misc"),
                        prompt=raw.get("prompt", "").strip(),
                        ideal_answer=raw.get("ideal_answer", "").strip(),
                        verification_pattern=raw.get("verification_pattern", "").strip(),
                        hints=[hint.strip() for hint in raw.get("hints", [])],
                        difficulty=raw.get("difficulty", "medium"),
                    )
                    tasks.append(task)
                    jsonl_file.write(json.dumps(task.to_dict(), ensure_ascii=False) + "\n")

        logger.info("Generated %d tasks across %d rules", len(tasks), len(rules))
        return tasks

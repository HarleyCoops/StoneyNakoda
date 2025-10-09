"""Prime Intellect environment wrapper for Stoney Nakoda translation tasks."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import verifiers as vf
from datasets import Dataset
from verifiers.envs.singleturn_env import SingleTurnEnv
from verifiers.parsers.parser import Parser
from verifiers.rubrics.rubric import Rubric
from verifiers.types import Messages

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a Stoney Nakoda language expert. Translate or explain each prompt "
    "concisely while preserving the cultural and grammatical nuance present in the "
    "reference answer."
)

PACKAGE_ROOT = Path(__file__).resolve().parent
FALLBACK_DATASET = PACKAGE_ROOT / "data" / "sample_tasks.jsonl"
REPO_DATASET = Path(__file__).resolve().parents[3] / "data" / "training_datasets_stoney.jsonl"


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _char_f1(prediction: str, target: str) -> float:
    pred_chars = Counter(_normalize(prediction).replace(" ", ""))
    target_chars = Counter(_normalize(target).replace(" ", ""))
    if not target_chars:
        return 0.0
    overlap = sum(min(pred_chars[ch], target_chars[ch]) for ch in target_chars)
    precision = overlap / max(sum(pred_chars.values()), 1)
    recall = overlap / max(sum(target_chars.values()), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        logger.info("Dataset file %s missing or empty.", path)
        return []

    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON on line %d of %s (%s).", line_no, path, exc)
                continue
            if "prompt" not in payload or "ideal_answer" not in payload:
                logger.debug(
                    "Skipping entry without prompt or ideal_answer on line %d of %s.",
                    line_no,
                    path,
                )
                continue
            entries.append(payload)
    return entries


def _prepare_records(
    entries: Sequence[dict[str, Any]],
    difficulties: Optional[Sequence[str]] = None,
    task_types: Optional[Sequence[str]] = None,
    include_hints: bool = True,
) -> list[dict[str, Any]]:
    allowed_difficulties = {d.lower() for d in difficulties or []}
    allowed_tasks = {t.lower() for t in task_types or []}
    records: list[dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        difficulty = str(entry.get("difficulty", "medium"))
        task_type = str(entry.get("task_type", "default"))
        if allowed_difficulties and difficulty.lower() not in allowed_difficulties:
            continue
        if allowed_tasks and task_type.lower() not in allowed_tasks:
            continue
        prompt = str(entry.get("prompt", "")).strip()
        answer = str(entry.get("ideal_answer", "")).strip()
        if not prompt or not answer:
            continue
        info = {
            "rule_id": entry.get("rule_id"),
            "verification_pattern": entry.get("verification_pattern"),
            "difficulty": difficulty,
            "task_type": task_type,
        }
        if include_hints:
            info["hints"] = entry.get("hints", [])
        records.append(
            {
                "id": entry.get("task_id") or f"stoney_task_{idx:05d}",
                "question": prompt,
                "answer": answer,
                "task": task_type,
                "info": info,
            }
        )
    return records


def _ensure_dataset(records: list[dict[str, Any]]) -> Dataset:
    if not records:
        raise ValueError(
            "No usable Stoney Nakoda tasks were found. "
            "Ensure the RL dataset has been generated or provide `dataset_path`."
        )
    return Dataset.from_list(records)


@dataclass
class DatasetBundle:
    train: Dataset
    eval: Dataset | None = None


def _build_dataset_bundle(
    dataset_path: Optional[Path],
    eval_path: Optional[Path],
    max_examples: int,
    eval_examples: int,
    eval_fraction: float,
    seed: Optional[int],
    difficulty_filter: Optional[Sequence[str]],
    task_filter: Optional[Sequence[str]],
    include_hints: bool,
) -> DatasetBundle:
    candidates: list[dict[str, Any]] = []
    explicit_path = dataset_path if dataset_path else REPO_DATASET
    candidates.extend(_prepare_records(_load_jsonl(explicit_path), difficulty_filter, task_filter, include_hints))

    if not candidates:
        fallback_entries = _load_jsonl(FALLBACK_DATASET)
        if fallback_entries:
            logger.warning(
                "Using bundled fallback tasks because no generated dataset was found at %s.",
                explicit_path,
            )
            candidates = _prepare_records(
                fallback_entries, difficulty_filter, task_filter, include_hints
            )
    if not candidates:
        raise ValueError(
            f"Unable to locate any Stoney Nakoda tasks. Checked: {explicit_path} and fallback sample."
        )

    if max_examples > 0:
        candidates = candidates[: max_examples]

    train_dataset = _ensure_dataset(candidates)

    if eval_path:
        eval_candidates = _prepare_records(
            _load_jsonl(eval_path), difficulty_filter, task_filter, include_hints
        )
        if eval_examples > 0:
            eval_candidates = eval_candidates[:eval_examples]
        eval_dataset = _ensure_dataset(eval_candidates) if eval_candidates else None
        return DatasetBundle(train=train_dataset, eval=eval_dataset)

    if eval_fraction > 0 and len(train_dataset) > 1:
        split = train_dataset.train_test_split(test_size=eval_fraction, seed=seed or 42)
        eval_dataset = split["test"]
        if eval_examples > 0:
            eval_dataset = eval_dataset.select(range(min(eval_examples, len(eval_dataset))))
        train_dataset = split["train"]
    else:
        eval_dataset = None
    return DatasetBundle(train=train_dataset, eval=eval_dataset)


class StoneyTranslationParser(Parser):
    """Parser that strips whitespace and normalizes assistant output."""

    def parse(self, text: str) -> str:
        return text.strip()

    def parse_answer(self, completion: Messages) -> str:
        parsed = super().parse_answer(completion) or ""
        return parsed.strip()


class StoneyTranslationRubric(Rubric):
    """Reward rubric capturing literal and pattern-based correctness."""

    def __init__(self, parser: Optional[Parser] = None):
        parser = parser or StoneyTranslationParser()
        funcs = [
            self.exact_match_reward,
            self.char_overlap_reward,
            self.pattern_reward,
        ]
        weights = [0.6, 0.3, 0.1]
        super().__init__(funcs=funcs, weights=weights, parser=parser, parallelize_scoring=False)

    def _prediction(self, completion: vf.types.Messages, parser: Parser) -> str:
        response = parser.parse_answer(completion) or ""
        return response.strip()

    def exact_match_reward(
        self,
        completion: Messages,
        answer: str,
        parser: Parser,
        **_: Any,
    ) -> float:
        prediction = self._prediction(completion, parser)
        return float(_normalize(prediction) == _normalize(answer))

    def char_overlap_reward(
        self,
        completion: Messages,
        answer: str,
        parser: Parser,
        **_: Any,
    ) -> float:
        prediction = self._prediction(completion, parser)
        return float(_char_f1(prediction, answer))

    def pattern_reward(
        self,
        completion: Messages,
        answer: str,
        parser: Parser,
        info: Dict[str, Any],
        **_: Any,
    ) -> float:
        prediction = self._prediction(completion, parser)
        score = 0.0
        pattern = (info or {}).get("verification_pattern")
        if pattern:
            try:
                if re.search(pattern, prediction, flags=re.IGNORECASE):
                    score = 1.0
            except re.error:
                if pattern.lower() in _normalize(prediction):
                    score = 1.0
        if score < 1.0:
            hints = (info or {}).get("hints", []) or []
            if hints:
                covered = sum(1 for hint in hints if hint.lower() in prediction.lower())
                score = max(score, covered / len(hints))
        return float(score)


class StoneyTranslationEnv(SingleTurnEnv):
    """Single-turn chat environment tailored to Stoney Nakoda translation tasks."""

    def __init__(
        self,
        dataset: Dataset,
        eval_dataset: Optional[Dataset],
        system_prompt: str,
        rubric: StoneyTranslationRubric,
        sampling_args: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=system_prompt,
            parser=rubric.parser,
            rubric=rubric,
            sampling_args=sampling_args,
            message_type="chat",
            **kwargs,
        )


def load_environment(
    dataset_path: str | Path | None = None,
    eval_path: str | Path | None = None,
    max_examples: int = -1,
    eval_examples: int = -1,
    eval_fraction: float = 0.1,
    difficulty_filter: Optional[Sequence[str]] = None,
    task_filter: Optional[Sequence[str]] = None,
    system_prompt: Optional[str] = None,
    sampling_args: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    include_hints: bool = True,
) -> vf.Environment:
    """
    Load the Stoney Nakoda translation environment.

    Args:
        dataset_path: Path to a JSONL file produced by the RL grammar pipeline.
        eval_path: Optional path to a JSONL evaluation split.
        max_examples: Cap on training examples (-1 uses full dataset).
        eval_examples: Cap on evaluation examples (-1 uses full split).
        eval_fraction: Fraction of training split reserved for evaluation if `eval_path` is not provided.
        difficulty_filter: Optional iterable of difficulty labels to keep.
        task_filter: Optional iterable of task types to keep.
        system_prompt: Override the default system prompt.
        sampling_args: Optional sampling overrides passed to verifiers.
        seed: RNG seed used when performing internal splits.
        include_hints: Keep or drop hint metadata in the info payload.

    Returns:
        A configured verifiers `Environment` instance.
    """

    resolved_dataset = Path(dataset_path).expanduser().resolve() if dataset_path else None
    resolved_eval = Path(eval_path).expanduser().resolve() if eval_path else None

    bundle = _build_dataset_bundle(
        dataset_path=resolved_dataset,
        eval_path=resolved_eval,
        max_examples=max_examples,
        eval_examples=eval_examples,
        eval_fraction=eval_fraction,
        seed=seed,
        difficulty_filter=difficulty_filter,
        task_filter=task_filter,
        include_hints=include_hints,
    )

    rubric = StoneyTranslationRubric()
    env = StoneyTranslationEnv(
        dataset=bundle.train,
        eval_dataset=bundle.eval,
        system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
        rubric=rubric,
        sampling_args=sampling_args,
    )

    logger.info(
        "Loaded Stoney Nakoda environment with %d train and %s eval examples.",
        len(env.dataset) if env.dataset else 0,
        len(env.eval_dataset) if env.eval_dataset else 0,
    )
    return env

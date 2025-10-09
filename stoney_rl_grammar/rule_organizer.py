"""Normalize extracted grammar rules into an RL-friendly catalogue."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import asdict
from typing import Iterable, List, Sequence

from .config import RL_RULES_OUTPUT_PATH, ensure_directories
from .models import GrammarRule

logger = logging.getLogger(__name__)


class RuleOrganizer:
    """Validate and persist extracted grammar rules for downstream RL usage."""

    def __init__(
        self,
        rules_output_path=RL_RULES_OUTPUT_PATH,
        min_confidence: float = 0.35,
    ) -> None:
        ensure_directories()
        self.rules_output_path = rules_output_path
        self.min_confidence = min_confidence

    def filter_rules(self, rules: Sequence[GrammarRule]) -> List[GrammarRule]:
        """Remove low-confidence or incomplete rules."""
        filtered: List[GrammarRule] = []
        for rule in rules:
            if not rule.title or not rule.description:
                logger.debug("Skipping rule %s due to missing fields", rule.rule_id)
                continue

            if rule.confidence is not None and rule.confidence < self.min_confidence:
                logger.debug("Skipping rule %s due to confidence %.2f", rule.rule_id, rule.confidence)
                continue

            filtered.append(rule)
        return filtered

    def deduplicate(self, rules: Sequence[GrammarRule]) -> List[GrammarRule]:
        """Collapse duplicate titles by keeping the highest-confidence instance."""
        best_by_title: dict[str, GrammarRule] = {}
        for rule in rules:
            key = rule.title.lower()
            current = best_by_title.get(key)
            if current is None:
                best_by_title[key] = rule
                continue
            current_conf = current.confidence or 0.0
            new_conf = rule.confidence or 0.0
            if new_conf > current_conf:
                best_by_title[key] = rule
        return list(best_by_title.values())

    def organize(self, rules: Sequence[GrammarRule]) -> List[GrammarRule]:
        """Return curated rule list ready for RL generation and persist as JSON."""
        curated = self.filter_rules(rules)
        curated = self.deduplicate(curated)
        curated.sort(key=lambda r: (r.category, r.rule_id))

        payload = {
            "summary": self._build_summary(curated),
            "rules": [rule.to_dict() for rule in curated],
        }
        self.rules_output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Persisted %d curated rules to %s", len(curated), self.rules_output_path)
        return curated

    def _build_summary(self, rules: Sequence[GrammarRule]) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for rule in rules:
            counts[rule.category] += 1
        counts["total"] = len(rules)
        return dict(counts)

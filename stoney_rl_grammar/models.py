"""Dataclasses for grammar extraction and RL task generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PageChunk:
    """Represents a single extraction asset (page + optional chunk metadata)."""

    page_number: int
    chunk_index: int
    text: str = ""
    image_b64: Optional[str] = None

    def chunk_id(self) -> str:
        return f"page_{self.page_number:03d}_chunk_{self.chunk_index:02d}"


@dataclass
class GrammarRule:
    """Structured grammar rule extracted from reference text."""

    rule_id: str
    title: str
    description: str
    category: str
    stoney_examples: List[str] = field(default_factory=list)
    english_examples: List[str] = field(default_factory=list)
    verification_hint: str = ""
    source_page: int = 0
    source_chunk: str = ""
    confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "stoney_examples": self.stoney_examples,
            "english_examples": self.english_examples,
            "verification_hint": self.verification_hint,
            "source_page": self.source_page,
            "source_chunk": self.source_chunk,
            "confidence": self.confidence,
        }


@dataclass
class RLTrainingTask:
    """Representation of an RL task derived from a grammar rule."""

    task_id: str
    rule_id: str
    task_type: str
    prompt: str
    ideal_answer: str
    verification_pattern: str
    hints: List[str] = field(default_factory=list)
    difficulty: str = "medium"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "rule_id": self.rule_id,
            "task_type": self.task_type,
            "prompt": self.prompt,
            "ideal_answer": self.ideal_answer,
            "verification_pattern": self.verification_pattern,
            "hints": self.hints,
            "difficulty": self.difficulty,
        }

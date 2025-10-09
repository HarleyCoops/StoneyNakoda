"""High-level orchestration for the Stoney grammar RL pipeline."""

from __future__ import annotations

import logging
from typing import List

from .config import (
    GRAMMAR_PDF_PATH,
    DEFAULT_EXTRACTION_MODEL,
    DEFAULT_TASK_MODEL,
    ensure_directories,
)
from .models import GrammarRule
from .pdf_ingest import load_page_assets
from .rule_extractor import StoneyGrammarExtractor
from .rule_organizer import RuleOrganizer
from .task_generator import StoneyTaskGenerator

logger = logging.getLogger(__name__)


class StoneyGrammarPipeline:
    """Runs the extraction -> organization -> task generation flow."""

    def __init__(
        self,
        pdf_path=GRAMMAR_PDF_PATH,
        extraction_model: str | None = None,
        task_model: str | None = None,
    ) -> None:
        ensure_directories()
        self.pdf_path = pdf_path
        self.extractor = StoneyGrammarExtractor(
            model=extraction_model or DEFAULT_EXTRACTION_MODEL
        )
        self.organizer = RuleOrganizer()
        self.task_generator = StoneyTaskGenerator(
            model=task_model or DEFAULT_TASK_MODEL
        )

    def run(self) -> List[GrammarRule]:
        """Execute the full pipeline."""
        logger.info("Loading PDF pages from %s", self.pdf_path)
        page_assets = load_page_assets(self.pdf_path)
        logger.info("Prepared %d page assets for extraction", len(page_assets))

        rules = self.extractor.extract_rules(page_assets)
        logger.info("Extracted %d rules", len(rules))

        curated_rules = self.organizer.organize(rules)
        logger.info("Curated down to %d rules", len(curated_rules))

        self.task_generator.generate_tasks(curated_rules)
        logger.info("Pipeline complete")
        return curated_rules


def run_pipeline() -> None:
    """Convenience wrapper for CLI execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    pipeline = StoneyGrammarPipeline()
    pipeline.run()


if __name__ == "__main__":
    run_pipeline()

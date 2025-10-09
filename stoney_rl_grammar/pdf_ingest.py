"""Utilities for preparing PDF pages as image assets for VLM extraction."""

from __future__ import annotations

import base64
from typing import List

import fitz  # PyMuPDF

from .config import GRAMMAR_PDF_PATH
from .models import PageChunk


def load_page_assets(pdf_path=GRAMMAR_PDF_PATH, scale: float = 2.0) -> List[PageChunk]:
    """Render each PDF page to PNG and pair with extracted text for the VLM."""
    doc = fitz.open(str(pdf_path))
    matrix = fitz.Matrix(scale, scale)
    assets: List[PageChunk] = []

    for page_index, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=matrix)
        image_bytes = pix.tobytes("png")
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        page_text = (page.get_text("text") or "").strip()

        assets.append(
            PageChunk(
                page_number=page_index,
                chunk_index=0,
                text=page_text,
                image_b64=image_b64,
            )
        )
    return assets

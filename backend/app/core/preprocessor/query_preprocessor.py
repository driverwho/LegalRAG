"""Query preprocessor — orchestrates classification, correction, and metadata extraction.

Module layout
-------------
classifier.py         — QueryClassifier  (LLM-primary, rule fallback)
spell_checker.py      — LegalSpellChecker (rules + optional LLM correction)
metadata_extractor.py — MetadataExtractor (structured filter extraction)
query_preprocessor.py — QueryPreprocessor (this file, the public façade)

The pipeline owns a ``process_async`` method that is the only thing the
``AsyncRAGPipeline`` calls.  All three sub-components are optional — if
not provided, reasonable defaults are used.
"""

from __future__ import annotations

import logging
from typing import Optional

from backend.app.core.retriever.models import PreprocessResult
from backend.app.core.preprocessor.classifier import QueryClassifier
from backend.app.core.preprocessor.spell_checker import LegalSpellChecker
from backend.app.core.preprocessor.metadata_extractor import MetadataExtractor

logger = logging.getLogger(__name__)


class QueryPreprocessor:
    """Façade that chains classifier → spell-checker → metadata-extractor.

    Parameters
    ----------
    llm_client : optional
        An ``AsyncOpenAI``-compatible client instance.  Shared by both the
        classifier (intent detection) and the spell-checker (correction).
    model : str
        Model name forwarded to LLM components.
    classifier : QueryClassifier | None
        Custom classifier; a default one is created if omitted.
    spell_checker : LegalSpellChecker | None
        Custom spell-checker; a default one is created if omitted.
    metadata_extractor : MetadataExtractor | None
        Custom metadata extractor; a default one is created if omitted.
    """

    def __init__(
        self,
        llm_client=None,
        model: str = "qwen-plus",
        *,
        classifier: Optional[QueryClassifier] = None,
        spell_checker: Optional[LegalSpellChecker] = None,
        metadata_extractor: Optional[MetadataExtractor] = None,
    ) -> None:
        self.llm_client = llm_client
        self.model = model

        # Classifier receives the LLM client so it can run the LLM path
        self.classifier = classifier or QueryClassifier(
            llm_client=llm_client, model=model,
        )
        self.spell_checker = spell_checker or LegalSpellChecker(
            llm_client=llm_client, model=model,
        )
        self.metadata_extractor = metadata_extractor or MetadataExtractor()

    # ── Public entry-point ────────────────────────────────────────────────────

    async def process_async(
        self,
        query: str,
        enable_correction: bool = False,
    ) -> PreprocessResult:
        """Full preprocessing pipeline (async).

        Steps
        -----
        1. **Spell correction** — rule-based always; LLM only when
           ``enable_correction=True`` and the heuristic fires.
        2. **Classification** — LLM-primary with rule fallback; returns
           fine-grained type + coarse retrieval type + complexity + strategy.
        3. **Metadata extraction** — structured filters (region, year, court,
           article numbers, …) and ranking preferences.
        4. **Assemble** — pack everything into PreprocessResult.

        Returns
        -------
        PreprocessResult
        """
        # ── Step 1: Correction ────────────────────────────────────────
        corrected = await self.spell_checker.correct_async(
            query, enable_llm=enable_correction,
        )

        # ── Step 2: Classification (async — may call LLM) ─────────────
        cls = await self.classifier.classify_async(corrected)

        # ── Step 3: Metadata extraction ───────────────────────────────
        meta = self.metadata_extractor.extract(corrected, cls.primary_type)

        # ── Step 4: Assemble result ───────────────────────────────────
        result = PreprocessResult(
            original_query=query,
            corrected_query=corrected,
            query_type=cls.primary_type,
            retrieval_type=cls.retrieval_type,
            extracted_metadata=meta,
            confidence=cls.confidence,
            complexity=cls.complexity,
            processing_strategy=cls.processing_strategy,
        )

        logger.info(
            "Preprocessing complete: type=%s retrieval=%s complexity=%s "
            "conf=%.2f source=%s corrected=%r",
            cls.primary_type, cls.retrieval_type, cls.complexity,
            cls.confidence, cls.source, corrected != query,
        )

        return result

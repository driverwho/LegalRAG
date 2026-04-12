"""Rule-based document classifier for Chinese legal document types.

Classifies documents into two categories:

- **law** (法律条文): Legal statutes, regulations, codes — structured documents
  with 章/节/条 hierarchy, high density of article numbering patterns.
- **case** (司法判例): Judicial decisions, court rulings — narrative documents
  with fixed section headings, case numbers, and argumentative prose.

Classification is based on a weighted scoring system that analyses both
document content (regex pattern density, keyword matching) and metadata
(filename, source field).  Each document in the input list is assumed to
be a *page* of the same file, so scores are aggregated across all pages.

This module is used by the chunking step of the document processing pipeline
to route documents to the appropriate splitting strategy:

- law  → ``LegalParentChildSplitter`` (parent-child by 章/节/条)
- case → ``DocumentSplitter`` (recursive character splitting)
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classification result
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    """Result of document type classification."""

    doc_type: str           # "law" or "case"
    confidence: float       # 0.0 – 1.0 (winning score / total score)
    law_score: float
    case_score: float
    matched_features: Dict[str, List[str]] = field(default_factory=lambda: {
        "law": [], "case": [],
    })


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class DocumentClassifier:
    """Rule-based classifier that distinguishes legal statutes from court cases.

    Uses a weighted scoring system based on four categories of evidence:

    1. **Content patterns** — regex matches counted across full text
       (e.g. density of ``第X条`` patterns).
    2. **Structural patterns** — chapter / section headings (``第X章``, ``第X节``).
    3. **Metadata keywords** — filename or source field contains type-specific
       words (e.g. ``法``, ``判决书``).
    4. **Narrative keywords** — phrases strongly associated with one category
       (e.g. ``本院认为``, ``总则``).

    Each match adds a weighted score to the corresponding category.  The
    category with the higher total score wins.  When no features match at all
    the document defaults to ``"case"`` (generic / unknown documents are safer
    to chunk with the recursive splitter).

    Usage::

        classifier = DocumentClassifier()
        result = classifier.classify(documents)

        if result.doc_type == "law":
            ...  # use LegalParentChildSplitter
        else:
            ...  # use DocumentSplitter
    """

    # ------------------------------------------------------------------ #
    # Law indicators                                                      #
    # ------------------------------------------------------------------ #

    # "第X条", "第X款", "第X项" — article-level numbering
    _ARTICLE_RE = re.compile(r"第[一二三四五六七八九十百千万零\d]+[条款项]")

    # "第X章", "第X节" — structural headings
    _STRUCTURE_RE = re.compile(r"第[一二三四五六七八九十百千万零\d]+[章节]")

    # Filename / metadata keywords for statutes
    _LAW_META_KEYWORDS: set = {
        "法", "条例", "规定", "办法", "规则", "细则",
        "准则", "章程", "法典", "法规",
    }

    # Content keywords strongly associated with statutes
    _LAW_CONTENT_KEYWORDS: set = {
        "总则", "分则", "附则", "施行日期",
        "本法自", "公布之日起施行", "法律责任",
    }

    # ------------------------------------------------------------------ #
    # Case indicators                                                     #
    # ------------------------------------------------------------------ #

    # Fixed section headings found in court rulings
    _CASE_SECTION_KEYWORDS: set = {
        "原告诉称", "被告辩称", "本院查明", "本院认为",
        "判决如下", "裁定如下", "审判长", "审判员",
        "书记员", "合议庭", "经审理查明", "裁判要旨",
        "案件基本", "基本案情", "案件焦点",
    }

    # Case number: (2023)京01民终1234号
    _CASE_NUMBER_RE = re.compile(r"[（(]\d{4}[)）][^\s]{2,20}?号")

    # Filename keywords for judicial decisions
    _CASE_META_KEYWORDS: set = {
        "判决书", "裁定书", "通知书", "调解书",
        "决定书", "起诉书", "案例",
    }

    # Narrative phrases typical in case documents
    _CASE_NARRATIVE_KEYWORDS: set = {
        "诉至本院", "提起诉讼", "不服判决", "上诉",
        "赔偿损失", "承担责任", "事故发生",
        "原告", "被告", "当事人",
    }

    # ------------------------------------------------------------------ #
    # Scoring weights                                                     #
    # ------------------------------------------------------------------ #

    _WEIGHTS: Dict[str, float] = {
        # --- law ---
        "article_pattern":       2.0,   # per match (capped at 50)
        "structure_pattern":     5.0,   # per match
        "law_meta_keyword":     15.0,   # per keyword hit in filename
        "law_content_keyword":   5.0,   # per keyword hit in text
        # --- case ---
        "case_section_keyword":  8.0,   # per keyword hit
        "case_number":          12.0,   # per case-number match
        "case_meta_keyword":    15.0,   # per keyword hit in filename
        "case_narrative_keyword": 3.0,  # per keyword hit
    }

    # When article pattern density (matches / char_count) exceeds this
    # threshold, a large bonus is added to the law score.
    _ARTICLE_DENSITY_THRESHOLD: float = 0.005
    _ARTICLE_DENSITY_BONUS: float = 20.0

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def classify(self, documents: List[Document]) -> ClassificationResult:
        """Classify a list of documents (pages of the same file).

        All pages are assumed to belong to a single logical document (one
        file upload).  Content from every page is concatenated for analysis,
        and metadata from all pages is inspected for filename keywords.

        Args:
            documents: LangChain Document objects from the same file.

        Returns:
            A :class:`ClassificationResult` with ``doc_type``, confidence
            score, and a breakdown of matched features.
        """
        law_score: float = 0.0
        case_score: float = 0.0
        matched_features: Dict[str, List[str]] = {"law": [], "case": []}

        # ---- Aggregate text and filenames ----
        full_text = "\n".join(doc.page_content for doc in documents)
        text_len = max(len(full_text), 1)

        filenames: set = set()
        for doc in documents:
            for key in ("file_name", "original_filename", "source"):
                val = doc.metadata.get(key)
                if val:
                    filenames.add(str(val))

        # ============================================================== #
        # Score LAW features                                              #
        # ============================================================== #

        # 1. Article numbering patterns (第X条/款/项)
        article_matches = self._ARTICLE_RE.findall(full_text)
        if article_matches:
            capped = min(len(article_matches), 50)
            law_score += capped * self._WEIGHTS["article_pattern"]
            matched_features["law"].append(
                f"article_patterns: {len(article_matches)} matches"
            )

            # Density bonus
            density = len(article_matches) / text_len
            if density >= self._ARTICLE_DENSITY_THRESHOLD:
                law_score += self._ARTICLE_DENSITY_BONUS
                matched_features["law"].append(
                    f"article_density: {density:.4f} "
                    f"(>={self._ARTICLE_DENSITY_THRESHOLD})"
                )

        # 2. Structural patterns (第X章/节)
        structure_matches = self._STRUCTURE_RE.findall(full_text)
        if structure_matches:
            law_score += len(structure_matches) * self._WEIGHTS["structure_pattern"]
            matched_features["law"].append(
                f"structure_patterns: {len(structure_matches)} matches"
            )

        # 3. Metadata keywords (filename / source)
        for fname in filenames:
            for kw in self._LAW_META_KEYWORDS:
                if kw in fname:
                    law_score += self._WEIGHTS["law_meta_keyword"]
                    matched_features["law"].append(
                        f"meta_keyword: '{kw}' in '{fname}'"
                    )

        # 4. Content keywords
        for kw in self._LAW_CONTENT_KEYWORDS:
            if kw in full_text:
                law_score += self._WEIGHTS["law_content_keyword"]
                matched_features["law"].append(f"content_keyword: '{kw}'")

        # ============================================================== #
        # Score CASE features                                             #
        # ============================================================== #

        # 1. Case section keywords
        for kw in self._CASE_SECTION_KEYWORDS:
            if kw in full_text:
                case_score += self._WEIGHTS["case_section_keyword"]
                matched_features["case"].append(f"section_keyword: '{kw}'")

        # 2. Case number patterns
        case_numbers = self._CASE_NUMBER_RE.findall(full_text)
        if case_numbers:
            case_score += len(case_numbers) * self._WEIGHTS["case_number"]
            matched_features["case"].append(
                f"case_numbers: {len(case_numbers)} matches"
            )

        # 3. Metadata keywords (filename / source)
        for fname in filenames:
            for kw in self._CASE_META_KEYWORDS:
                if kw in fname:
                    case_score += self._WEIGHTS["case_meta_keyword"]
                    matched_features["case"].append(
                        f"meta_keyword: '{kw}' in '{fname}'"
                    )

        # 4. Narrative keywords
        for kw in self._CASE_NARRATIVE_KEYWORDS:
            if kw in full_text:
                case_score += self._WEIGHTS["case_narrative_keyword"]
                matched_features["case"].append(f"narrative_keyword: '{kw}'")

        # ============================================================== #
        # Determine final classification                                  #
        # ============================================================== #

        total_score = law_score + case_score
        if total_score == 0:
            # No features matched — default to case (generic document
            # is safer to chunk with the recursive splitter).
            doc_type = "case"
            confidence = 0.0
        else:
            doc_type = "law" if law_score > case_score else "case"
            confidence = max(law_score, case_score) / total_score

        result = ClassificationResult(
            doc_type=doc_type,
            confidence=confidence,
            law_score=law_score,
            case_score=case_score,
            matched_features=matched_features,
        )

        logger.info(
            "Document classified as '%s' (confidence=%.2f, law_score=%.1f, "
            "case_score=%.1f)",
            result.doc_type, result.confidence, law_score, case_score,
        )
        if matched_features["law"]:
            logger.debug("Law features: %s", "; ".join(matched_features["law"]))
        if matched_features["case"]:
            logger.debug("Case features: %s", "; ".join(matched_features["case"]))

        return result

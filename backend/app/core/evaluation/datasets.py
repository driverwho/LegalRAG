"""Dataset loaders for V2 retrieval evaluation.

Supports two legal benchmark formats:

  1. **QA format** — question + reference (list of ground-truth law articles)
     + gold answer.
  2. **MCQ format** — multiple-choice question + analysis/answer in ``output``.

Both are normalised into ``EvalSample`` so the evaluator has a single interface.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# ── Unified sample representation ─────────────────────────────────────────────

@dataclass
class EvalSample:
    """One evaluation item, regardless of original dataset format."""

    question: str
    ground_truth_docs: List[str]
    reference_answer: Optional[str] = None
    source_type: str = "qa"          # "qa" | "mcq"
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_data: Dict[str, Any] = field(default_factory=dict)


# ── Internal helpers ──────────────────────────────────────────────────────────

_MCQ_QUESTION_RE = re.compile(
    r"Question:\s*(.+?)(?:\n[A-D]\.|$)", re.DOTALL,
)
_MCQ_ANSWER_RE = re.compile(
    r"(?:正确答案[是为：:]+\s*|因此[，,]\s*正确答案是\s*|答案是\s*)([A-D])",
)
_LAW_REF_RE = re.compile(
    r"《(.+?)》[^》]*?(第[零一二三四五六七八九十百千万\d]+条[之的]?[^\n。，,；;]*)",
)


def _extract_mcq_question(text: str) -> str:
    """Extract the question portion from an MCQ ``input`` field.

    Returns everything after ``Question:`` up to the first option marker
    (``\\nA.``).  Falls back to the full text if parsing fails.
    """
    m = _MCQ_QUESTION_RE.search(text)
    if m:
        return m.group(1).strip()
    # Fallback: split on first option
    for marker in ("\nA.", "\nA、", "\nA "):
        idx = text.find(marker)
        if idx != -1:
            q = text[:idx].strip()
            # strip "Question:" prefix if present
            if q.lower().startswith("question:"):
                q = q[len("question:"):].strip()
            return q
    return text.strip()


def _extract_mcq_correct_answer(output: str) -> Optional[str]:
    """Extract the correct answer letter from an MCQ ``output`` field."""
    m = _MCQ_ANSWER_RE.search(output)
    if m:
        return m.group(1)
    # Fallback: look for last standalone letter at end
    last = re.findall(r"[是为]\s*([A-D])\s*[。.]?\s*$", output)
    return last[-1] if last else None


def _extract_law_refs_from_text(text: str) -> List[str]:
    """Extract law article references (``《法名》第X条...``) from free text.

    Returns a list of short reference strings that can be used for
    substring matching against retrieved documents.
    """
    refs = []
    for m in _LAW_REF_RE.finditer(text):
        law_name = m.group(1)
        article = m.group(2).strip().rstrip("。，,；;、")
        refs.append(f"{law_name}{article}")
    return refs


def _parse_qa_reference(ref: str) -> str:
    """Normalise a QA-format reference string.

    Input example::

        '诉讼与非诉讼程序法-刑事诉讼法2018-10-26:    "第二百九十四条 ..."'

    Returns the content after the colon (with quotes stripped and trimmed).
    """
    # Split on first colon to separate category label from content
    parts = ref.split(":", 1)
    content = parts[-1].strip()
    # Strip surrounding quotes and whitespace
    content = content.strip().strip('"').strip("',").strip()
    return content


# ── Public loaders ────────────────────────────────────────────────────────────

class EvalDatasetLoader:
    """Load evaluation datasets from JSON files."""

    @staticmethod
    def load_qa_dataset(path: Union[str, Path]) -> List[EvalSample]:
        """Load a QA-format dataset.

        Expected JSON structure: a list of objects with keys
        ``question``, ``reference`` (list of strings), ``answer``.
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            # Some datasets wrap the list under a key
            for key in ("data", "samples", "items", "questions"):
                if key in data:
                    data = data[key]
                    break

        samples: List[EvalSample] = []
        for item in data:
            question = item.get("question", "").strip()
            if not question:
                continue

            # Parse ground-truth documents from reference list
            raw_refs = item.get("reference", [])
            gt_docs = [_parse_qa_reference(r) for r in raw_refs if r.strip()]

            samples.append(
                EvalSample(
                    question=question,
                    ground_truth_docs=gt_docs,
                    reference_answer=item.get("answer", "").strip() or None,
                    source_type="qa",
                    raw_data=item,
                )
            )

        logger.info("Loaded %d QA samples from %s", len(samples), path)
        return samples

    @staticmethod
    def load_mcq_dataset(path: Union[str, Path]) -> List[EvalSample]:
        """Load an MCQ-format dataset.

        Expected JSON structure: a list of objects with keys
        ``input`` (question + options), ``output`` (analysis + answer),
        ``type`` (optional).
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            for key in ("data", "samples", "items", "questions"):
                if key in data:
                    data = data[key]
                    break

        samples: List[EvalSample] = []
        for item in data:
            raw_input = item.get("input", "").strip()
            raw_output = item.get("output", "").strip()
            if not raw_input:
                continue

            question = _extract_mcq_question(raw_input)
            correct_answer = _extract_mcq_correct_answer(raw_output)

            # Extract law references from the analysis as ground truth
            gt_refs = _extract_law_refs_from_text(raw_output)

            samples.append(
                EvalSample(
                    question=question,
                    ground_truth_docs=gt_refs,
                    reference_answer=correct_answer,
                    source_type="mcq",
                    metadata={
                        "type": item.get("type", ""),
                        "full_input": raw_input,
                        "correct_option": correct_answer,
                    },
                    raw_data=item,
                )
            )

        logger.info("Loaded %d MCQ samples from %s", len(samples), path)
        return samples

    @classmethod
    def load_auto(cls, path: Union[str, Path]) -> List[EvalSample]:
        """Auto-detect dataset format and load.

        Detection heuristic: peek at the first item —
          - has ``question`` + ``reference`` keys → QA format
          - has ``input`` + ``output`` keys → MCQ format
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Unwrap wrapper objects
        raw = data
        if isinstance(raw, dict):
            for key in ("data", "samples", "items", "questions"):
                if key in raw and isinstance(raw[key], list):
                    raw = raw[key]
                    break

        if not isinstance(raw, list) or len(raw) == 0:
            logger.error("Cannot auto-detect format for %s — empty or invalid", path)
            return []

        first = raw[0]
        if "question" in first and "reference" in first:
            logger.info("Auto-detected QA format for %s", path)
            return cls.load_qa_dataset(path)
        if "input" in first and "output" in first:
            logger.info("Auto-detected MCQ format for %s", path)
            return cls.load_mcq_dataset(path)

        logger.error(
            "Cannot auto-detect format for %s — first item keys: %s",
            path, list(first.keys()),
        )
        return []

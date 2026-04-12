"""Parent-child document splitter for Chinese legal documents in Markdown format.

Splits legal texts along their natural 章 → 节 → 条 hierarchy, producing:

- **Parent chunks** (chapter / section level): store the complete text of the
  structural unit so that downstream retrieval can supply rich context to the LLM.
- **Child chunks** (article level): individual 条 for vector-similarity search,
  each carrying metadata that references its parent chunk, positional info, and
  legal attributes (chapter, section, article number, law name …).

Hierarchy model::

    章 (Chapter)
    ├── 条 (Article)             ← chapter is parent when no sections exist
    └── 节 (Section)
        └── 条 (Article)         ← section is parent when sections exist

This module is **self-contained** and is NOT wired into the main preprocessing
pipeline.  Import and call ``LegalParentChildSplitter`` directly when processing
legal Markdown files.
"""

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chinese numeral conversion
# ---------------------------------------------------------------------------

_CN_DIGIT: Dict[str, int] = {
    "零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_CN_UNIT: Dict[str, int] = {"十": 10, "百": 100, "千": 1000, "万": 10000}

# Regex fragment that matches one or more Chinese numeral characters or digits.
_CN_NUM = r"[一二三四五六七八九十百千万零\d]+"


def cn_to_int(text: str) -> Optional[int]:
    """Convert a Chinese numeral string (e.g. ``'二十三'``) or Arabic digit
    string (e.g. ``'23'``) to :pyclass:`int`.

    Returns ``None`` when *text* cannot be parsed.

    Examples::

        >>> cn_to_int("十二")
        12
        >>> cn_to_int("一百零三")
        103
        >>> cn_to_int("42")
        42
    """
    text = text.strip()
    if text.isdigit():
        return int(text)

    result, current = 0, 0
    for ch in text:
        if ch in _CN_DIGIT:
            current = _CN_DIGIT[ch]
        elif ch in _CN_UNIT:
            result += (current or 1) * _CN_UNIT[ch]
            current = 0
        else:
            return None
    return (result + current) or None


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class _Marker:
    """A structural element (章 / 节 / 条) detected in the source text."""

    level: str            # "chapter" | "section" | "article"
    number: Optional[int] # parsed numeric value, e.g. 12
    number_raw: str       # raw numeral text, e.g. "十二"
    title: str            # full heading text, e.g. "第一章 总则"
    start: int            # char offset — start of the heading line
    end: int              # char offset — end of the heading line


@dataclass
class _ParentGroup:
    """A group of articles that share the same parent (chapter or section)."""

    chapter: Optional[_Marker] = None
    section: Optional[_Marker] = None
    parent_marker_idx: Optional[int] = None   # index into the markers list
    article_indices: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------

class LegalParentChildSplitter:
    """Split Chinese legal Markdown documents into parent and child chunks.

    Usage::

        splitter = LegalParentChildSplitter(law_name="中华人民共和国民法典")
        parents, children = splitter.split(documents)

        # parents  → store for context retrieval (keyed by parent_chunk_id)
        # children → store in vector DB for semantic search

    Parent chunk metadata::

        chunk_type       = "parent"
        parent_chunk_id  = <uuid>         # unique id for this parent
        law_name         = "..."          # from constructor
        chapter          = "第一章 总则"   # chapter title
        chapter_number   = 1
        section          = "第一节 ..."   # section title (None if N/A)
        section_number   = 1              # (None if N/A)
        child_count      = 5              # number of child articles
        char_count       = 1234

    Child chunk metadata::

        chunk_type       = "child"
        chunk_id         = <uuid>
        parent_chunk_id  = <uuid>         # FK → parent
        law_name         = "..."
        chapter          = "第一章 总则"
        chapter_number   = 1
        section          = "第一节 ..."   # (None if N/A)
        section_number   = 1              # (None if N/A)
        article_number   = 3
        article_title    = "第三条"
        char_count       = 256
    """

    # Compiled regex patterns ------------------------------------------------
    # Chapters:  "## 第一章 总则" or plain "第一章 总则"
    _CHAPTER_RE = re.compile(
        rf"^(?:#{{1,6}}\s*)?第({_CN_NUM})章\s*(.*?)$", re.MULTILINE,
    )
    # Sections:  "### 第一节 一般规定" or plain "第一节 一般规定"
    _SECTION_RE = re.compile(
        rf"^(?:#{{1,6}}\s*)?第({_CN_NUM})节\s*(.*?)$", re.MULTILINE,
    )
    # Articles:  "第一条 ..." or bold "**第一条** ..."
    _ARTICLE_RE = re.compile(
        rf"^(?:\*{{0,2}})第({_CN_NUM})条(?:\*{{0,2}})\s*", re.MULTILINE,
    )

    def __init__(self, law_name: Optional[str] = None):
        """
        Args:
            law_name: Name of the law / regulation.  Attached to every
                chunk's metadata for downstream filtering and display.
        """
        self.law_name = law_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(
        self,
        documents: List[Document],
    ) -> Tuple[List[Document], List[Document]]:
        """Split *documents* into parent and child chunks.

        Args:
            documents: LangChain Document objects whose ``page_content``
                contains legal text in Markdown format.

        Returns:
            ``(parent_chunks, child_chunks)`` tuple.
        """
        all_parents: List[Document] = []
        all_children: List[Document] = []

        for doc in documents:
            parents, children = self._split_single(doc)
            all_parents.extend(parents)
            all_children.extend(children)

        logger.info(
            "Legal split complete: %d parent chunks, %d child chunks",
            len(all_parents),
            len(all_children),
        )
        return all_parents, all_children

    # ------------------------------------------------------------------
    # Marker detection
    # ------------------------------------------------------------------

    def _find_markers(self, text: str) -> List[_Marker]:
        """Scan *text* and return all structural markers sorted by position."""
        markers: List[_Marker] = []

        for m in self._CHAPTER_RE.finditer(text):
            markers.append(_Marker(
                level="chapter",
                number=cn_to_int(m.group(1)),
                number_raw=m.group(1),
                title=re.sub(r"^#+\s*", "", m.group(0)).strip(),
                start=m.start(),
                end=m.end(),
            ))

        for m in self._SECTION_RE.finditer(text):
            markers.append(_Marker(
                level="section",
                number=cn_to_int(m.group(1)),
                number_raw=m.group(1),
                title=re.sub(r"^#+\s*", "", m.group(0)).strip(),
                start=m.start(),
                end=m.end(),
            ))

        for m in self._ARTICLE_RE.finditer(text):
            markers.append(_Marker(
                level="article",
                number=cn_to_int(m.group(1)),
                number_raw=m.group(1),
                title=f"第{m.group(1)}条",
                start=m.start(),
                end=m.end(),
            ))

        markers.sort(key=lambda mk: mk.start)
        return markers

    # ------------------------------------------------------------------
    # Content extraction
    # ------------------------------------------------------------------

    def _slice_content(
        self, markers: List[_Marker], text: str, idx: int,
    ) -> str:
        """Return the text belonging to the marker at *idx*.

        The slice runs from the marker's own start position to the start of
        the next marker (or end-of-text).
        """
        start = markers[idx].start
        end = markers[idx + 1].start if idx + 1 < len(markers) else len(text)
        return text[start:end].strip()

    # ------------------------------------------------------------------
    # Core splitting logic
    # ------------------------------------------------------------------

    def _split_single(
        self,
        doc: Document,
    ) -> Tuple[List[Document], List[Document]]:
        """Process one Document into parent + child chunks.

        Algorithm
        ---------
        1. Detect all 章 / 节 / 条 markers and sort by position.
        2. Walk through markers sequentially, tracking the current chapter
           and section context.  Each time a new chapter or section is
           encountered, a new :class:`_ParentGroup` is opened; articles
           are appended to the currently active group.
        3. For every non-empty group, emit one parent Document (full text
           of the structural unit) and one child Document per article.
        """
        text = doc.page_content
        base_meta = dict(doc.metadata)

        markers = self._find_markers(text)
        if not markers:
            logger.warning(
                "No legal structure (章/节/条) found — returning document as-is"
            )
            return [doc], []

        # -- Phase 1: group articles under their parent ------------------
        groups: List[_ParentGroup] = []
        cur_group: Optional[_ParentGroup] = None
        cur_chapter: Optional[_Marker] = None

        for i, mk in enumerate(markers):
            if mk.level == "chapter":
                cur_chapter = mk
                # Open a chapter-level group (may remain empty if sections
                # follow immediately — that is fine, empty groups are skipped).
                cur_group = _ParentGroup(
                    chapter=mk, parent_marker_idx=i,
                )
                groups.append(cur_group)

            elif mk.level == "section":
                # Sections are more granular parents; open a new group.
                cur_group = _ParentGroup(
                    chapter=cur_chapter, section=mk, parent_marker_idx=i,
                )
                groups.append(cur_group)

            elif mk.level == "article":
                if cur_group is None:
                    # Articles appearing before any chapter / section
                    # (preamble — rare but possible).
                    cur_group = _ParentGroup(parent_marker_idx=i)
                    groups.append(cur_group)
                cur_group.article_indices.append(i)

        # -- Phase 2: emit Documents ------------------------------------
        parents: List[Document] = []
        children: List[Document] = []

        for grp in groups:
            if not grp.article_indices:
                continue  # chapter header with no direct articles (sections follow)

            parent_chunk_id = str(uuid.uuid4())

            # ---- parent text range ----
            # Start: the group's own structural marker (chapter / section header).
            # End:   start of the next marker after the last article in this group,
            #        or end-of-text.
            parent_start_idx = (
                grp.parent_marker_idx
                if grp.parent_marker_idx is not None
                else grp.article_indices[0]
            )
            parent_start = markers[parent_start_idx].start

            last_article_idx = grp.article_indices[-1]
            parent_end = (
                markers[last_article_idx + 1].start
                if last_article_idx + 1 < len(markers)
                else len(text)
            )
            parent_text = text[parent_start:parent_end].strip()

            # ---- metadata fields ----
            ch_title = grp.chapter.title if grp.chapter else None
            ch_num = grp.chapter.number if grp.chapter else None
            sec_title = grp.section.title if grp.section else None
            sec_num = grp.section.number if grp.section else None

            parent_meta = {
                **base_meta,
                "chunk_type": "parent",
                "parent_chunk_id": parent_chunk_id,
                "law_name": self.law_name,
                "chapter": ch_title,
                "chapter_number": ch_num,
                "section": sec_title,
                "section_number": sec_num,
                "child_count": len(grp.article_indices),
                "char_count": len(parent_text),
            }
            parents.append(
                Document(page_content=parent_text, metadata=parent_meta)
            )

            # ---- child documents (one per article) ----
            for art_idx in grp.article_indices:
                art_mk = markers[art_idx]
                art_text = self._slice_content(markers, text, art_idx)

                child_meta = {
                    **base_meta,
                    "chunk_type": "child",
                    "chunk_id": str(uuid.uuid4()),
                    "parent_chunk_id": parent_chunk_id,
                    "law_name": self.law_name,
                    "chapter": ch_title,
                    "chapter_number": ch_num,
                    "section": sec_title,
                    "section_number": sec_num,
                    "article_number": art_mk.number,
                    "article_title": art_mk.title,
                    "char_count": len(art_text),
                }
                children.append(
                    Document(page_content=art_text, metadata=child_meta)
                )

        return parents, children

"""Metadata extractor — pulls structured filter conditions from free-text queries.

Extracts:
  - region         (province / court jurisdiction)
  - year_range     (explicit year or "最近N年")
  - court_level    (基层 / 中级 / 高级 / 最高)
  - article_number (法条引用)
  - case_number    (案号)
  - law_name       (referenced statute)
  - doc_type       (law / case — derived from query_type)

All methods are pure (no I/O, no LLM) so they can be called synchronously.
"""

from __future__ import annotations

import re
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Province keywords ─────────────────────────────────────────────────────────
_PROVINCES = [
    "北京", "天津", "上海", "重庆",
    "河北", "山西", "辽宁", "吉林", "黑龙江",
    "江苏", "浙江", "安徽", "福建", "江西", "山东",
    "河南", "湖北", "湖南", "广东", "海南",
    "四川", "贵州", "云南", "陕西", "甘肃",
    "青海", "内蒙古", "广西", "西藏", "宁夏", "新疆",
]

# ── Court level keywords ─────────────────────────────────────────────────────
_COURT_LEVELS = {
    "最高人民法院":   "最高",
    "高级人民法院":   "高级",
    "中级人民法院":   "中级",
    "基层人民法院":   "基层",
    "人民法庭":       "基层",
    # abbreviations (already expanded by spell_checker, but be safe)
    "最高法":         "最高",
    "高院":           "高级",
    "中院":           "中级",
}

# ── Law name keywords ─────────────────────────────────────────────────────────
_LAW_NAMES = [
    "民法典", "刑法", "宪法", "合同法", "劳动法", "劳动合同法",
    "公司法", "婚姻法", "继承法", "行政诉讼法", "民事诉讼法",
    "刑事诉讼法", "行政处罚法", "治安管理处罚法", "道路交通安全法",
    "消费者权益保护法", "侵权责任法", "物权法", "担保法",
    "商标法", "专利法", "著作权法", "反不正当竞争法",
]


class MetadataExtractor:
    """Extract structured filter metadata from a legal query."""

    def extract(self, query: str, query_type: str) -> Dict[str, Any]:
        """Build a metadata dict from *query* given the classified *query_type*.

        Only filters that are **explicitly mentioned** in the query are added.
        Defaults (e.g. "last 5 years", "region = 全国") are intentionally
        omitted so that broad queries like "介绍反分裂国家法" retrieve documents
        regardless of their year or region metadata.

        Returns
        -------
        dict
            ``filters``  — hard filter conditions for the vector store.
            ``ranking_preferences`` — soft hints for result ordering.
        """
        filters: Dict[str, Any] = {}

        # 1. Region — only when explicitly mentioned in the query
        region = self._extract_region(query)
        if region:
            filters["region"] = region

        # 2. Time range — only when explicitly mentioned in the query
        year_range = self._extract_time_range(query)
        if year_range is not None:
            filters["year_range"] = year_range

        # 3. Court level
        court = self._extract_court_level(query)
        if court:
            filters["court_level"] = court

        # 4. Article number(s)
        articles = self._extract_article_numbers(query)
        if articles:
            filters["article_numbers"] = articles

        # 5. Case number
        case = self._extract_case_number(query)
        if case:
            filters["case_number"] = case

        # 6. Referenced law names
        laws = self._extract_law_names(query)
        if laws:
            filters["law_names"] = laws

        # 7. Type-specific doc_type constraint
        # chunk_type is intentionally omitted: it is not guaranteed to be
        # stamped on documents ingested before the parent-child splitter
        # was introduced, which would cause those chunks to be silently
        # filtered out.
        if query_type in ("simple_law_query", "法条"):
            filters["doc_type"] = ["law"]
        elif query_type in ("case_retrieval", "case_analysis", "案例"):
            filters["doc_type"] = ["case"]

        # Ranking preferences
        ranking = self._build_ranking_preferences(query_type, query)

        logger.debug("Extracted metadata: filters=%s, ranking=%s", filters, ranking)

        return {
            "filters": filters,
            "ranking_preferences": ranking,
        }

    # ── Region ────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_region(query: str) -> List[str]:
        """Extract province / jurisdiction explicitly mentioned in the query.

        Returns an empty list when no region is mentioned — callers must skip
        the filter entirely in that case so that documents without a ``region``
        metadata field are not accidentally excluded.
        """
        regions: List[str] = []
        for prov in _PROVINCES:
            if prov in query:
                regions.append(prov)

        # Nationwide signals — only when explicitly stated
        if any(kw in query for kw in ("最高人民法院", "最高法", "全国")):
            if "全国" not in regions:
                regions.append("全国")

        return regions  # empty = no region filter

    # ── Time range ────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_time_range(query: str) -> Optional[Dict[str, int]]:
        """Extract year or year range explicitly mentioned in the query.

        Returns ``None`` when no time signal is found — callers must skip the
        filter entirely so that documents from any year remain retrievable.
        The old default of "last 5 years" silently excluded older statutes
        (e.g. 反分裂国家法 enacted 2005) from all results.
        """
        current_year = datetime.now().year

        # "2020年至2023年" / "2020-2023"
        range_match = re.search(r"(20\d{2})\s*[年至\-~—]+\s*(20\d{2})", query)
        if range_match:
            return {
                "start": int(range_match.group(1)),
                "end": int(range_match.group(2)),
            }

        # Single year: "2023年"
        year_match = re.search(r"(20\d{2})\s*年", query)
        if year_match:
            year = int(year_match.group(1))
            return {"start": year, "end": year}

        # "最近N年"
        recent_match = re.search(r"最近\s*(\d+)\s*年", query)
        if recent_match:
            n = int(recent_match.group(1))
            return {"start": current_year - n, "end": current_year}

        # "最新" / "现行" / "当前"
        if re.search(r"(?:最新|现行|当前)", query):
            return {"start": current_year - 2, "end": current_year}

        # No explicit time signal → do not filter by year
        return None

    # ── Court level ───────────────────────────────────────────────────────────

    @staticmethod
    def _extract_court_level(query: str) -> List[str]:
        """Extract court level(s) mentioned."""
        levels: List[str] = []
        for keyword, level in _COURT_LEVELS.items():
            if keyword in query and level not in levels:
                levels.append(level)
        return levels

    # ── Article numbers ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_article_numbers(query: str) -> List[Dict[str, Any]]:
        """Extract referenced article numbers (Arabic and Chinese)."""
        articles: List[Dict[str, Any]] = []

        # Arabic: "第1165条"
        for m in re.finditer(r"第\s*(\d+)\s*条", query):
            articles.append({"number": int(m.group(1)), "format": "arabic"})

        # Chinese: "第一百六十五条"
        for m in re.finditer(r"第\s*([一二三四五六七八九十百千]+)\s*条", query):
            articles.append({"number_chinese": m.group(1), "format": "chinese"})

        return articles

    # ── Case number ───────────────────────────────────────────────────────────

    @staticmethod
    def _extract_case_number(query: str) -> Optional[Dict[str, Any]]:
        """Extract case number (案号) e.g. "(2023)京01民初123号"."""
        # Full format
        full = re.search(
            r"[（(]\s*(20\d{2})\s*[）)]\s*(\S+?)\s*(\d+)\s*号", query
        )
        if full:
            return {
                "year": int(full.group(1)),
                "court_code": full.group(2),
                "number": int(full.group(3)),
                "raw": full.group(0),
            }

        # Loose format: "2023...123号"
        loose = re.search(r"(20\d{2}).*?(\d+)\s*号", query)
        if loose:
            return {
                "year": int(loose.group(1)),
                "number": int(loose.group(2)),
                "raw": loose.group(0),
            }

        return None

    # ── Law names ─────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_law_names(query: str) -> List[str]:
        """Find law / statute names referenced in query."""
        found: List[str] = []
        for name in _LAW_NAMES:
            if name in query and name not in found:
                found.append(name)
        return found

    # ── Ranking preferences ───────────────────────────────────────────────────

    @staticmethod
    def _build_ranking_preferences(
        query_type: str, query: str
    ) -> Dict[str, Any]:
        """Generate soft ranking hints based on query type."""
        prefs: Dict[str, Any] = {
            "sort_by": "relevance",
            "time_weight": 0.2,
            "authority_weight": 0.3,
        }

        if query_type in ("case_retrieval", "case_analysis", "案例"):
            prefs["time_weight"] = 0.3
            prefs["authority_weight"] = 0.4

        if query_type in ("simple_law_query", "法条"):
            prefs["time_weight"] = 0.1        # statutes are less time-sensitive
            prefs["authority_weight"] = 0.2

        # If user asks for "最新", boost time weight
        if re.search(r"(?:最新|最近|近期)", query):
            prefs["sort_by"] = "date"
            prefs["time_weight"] = 0.5

        return prefs

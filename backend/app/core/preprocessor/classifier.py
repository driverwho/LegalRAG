"""Query classifier — LLM-primary, rule-based fallback.

Architecture
------------
                       ┌─────────────────────┐
    query ──────────▶  │  LLM structured JSON │ ──▶ ClassifyResult
                       └──────────┬──────────┘
                          fail /  │ timeout
                                  ▼
                       ┌─────────────────────┐
                       │  Rule-based fallback │ ──▶ ClassifyResult
                       └─────────────────────┘

LLM path
  - Sends a compact prompt listing all 9 query types.
  - Expects a single JSON object (no markdown, no prose).
  - Model: configurable, defaults to project's LLM_MODEL.
  - Temperature: 0 (deterministic).
  - Timeout: 3 s — tight enough to be invisible to the user.

Fallback path
  - Preserves the original regex logic exactly.
  - Activated on any exception: network error, JSON parse failure,
    type-key not in QUERY_TYPES, timeout, etc.

Agent upgrade path
  The public interface (classify_async → ClassifyResult) is
  intentionally thin.  To turn this into a Router Agent later:
    1. Replace the LLM prompt with a tool-definition list.
    2. Let the model emit tool_calls instead of raw JSON.
    3. ClassifyResult stays the same — callers see no change.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


# ── Type registry ─────────────────────────────────────────────────────────────

QUERY_TYPES: Dict[str, str] = {
    "simple_law_query":       "简单法条查询 — 用户想知道某条法律的具体规定内容（如'民法典第1165条规定什么'）",
    "case_retrieval":         "案例检索 — 用户想找相关判例、裁定、案件（如'交通事故赔偿的典型案例'）",
    "legal_consultation":     "法律咨询 — 用户遇到具体法律问题求助（如'被公司辞退了怎么办'）",
    "document_generation":    "文书生成 — 用户需要生成或参考法律文书模板（如'帮我写一份劳动仲裁申请书'）",
    "case_analysis":          "案例分析 — 用户需要分析某个案件的法律要点（如'分析这个合同纠纷案的争议焦点'）",
    "comparative_analysis":   "对比分析 — 用户需要比较两种情况或概念的异同（如'对比有限责任和无限责任'）",
    "procedure_consultation": "程序咨询 — 用户询问法律程序/流程/材料（如'起诉需要哪些材料'）",
    "concept_explanation":    "概念解释 — 用户需要解释法律术语或概念（如'什么是善意取得'）",
    "mixed_law_case":         "法条+案例混合 — 查询同时涉及法条引用和案例检索（如'民法典第XXX条的相关判决'）",
}

# Coarse retrieval bucket for each fine-grained type
_RETRIEVAL_TYPE: Dict[str, str] = {
    "simple_law_query":       "法条",
    "case_retrieval":         "案例",
    "case_analysis":          "案例",
    "mixed_law_case":         "general",
    "legal_consultation":     "general",
    "document_generation":    "general",
    "comparative_analysis":   "general",
    "procedure_consultation": "general",
    "concept_explanation":    "general",
}


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ClassifyResult:
    """Output of the classifier — consumed by QueryPreprocessor."""

    primary_type: str
    retrieval_type: str
    confidence: float
    complexity: str                                         # simple | medium | complex
    processing_strategy: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""                                        # LLM explanation (debug)
    source: str = "llm"                                     # "llm" | "rule"


# ── LLM prompt ────────────────────────────────────────────────────────────────

def _build_prompt(query: str) -> str:
    type_lines = "\n".join(
        f'  "{k}": {v}' for k, v in QUERY_TYPES.items()
    )
    return f"""\
你是一个法律查询意图分类系统，请分析用户查询并输出结构化 JSON。

## 可选查询类型
{type_lines}

## 复杂度定义
- simple：单一明确意图，短查询
- medium：有一定背景或包含一个子问题
- complex：多个子问题、假设条件、或需要综合多方面知识

## 输出要求
仅输出合法 JSON，不要添加任何解释、markdown、代码块。
必须包含以下字段：
{{
  "primary_type": "<上方类型 key 之一>",
  "confidence": <0.0-1.0 的浮点数>,
  "complexity": "<simple|medium|complex>",
  "reason": "<一句话解释分类理由>"
}}

## 用户查询
{query}"""


# ── Main classifier ───────────────────────────────────────────────────────────

class QueryClassifier:
    """LLM-primary query classifier with rule-based fallback.

    Parameters
    ----------
    llm_client : AsyncOpenAI-compatible client, optional
        When provided, the LLM path is active.  When absent (or on any
        error), the rule-based fallback is used transparently.
    model : str
        Model name for the classification call.
    timeout : float
        Max seconds to wait for the LLM.  Keep tight (≤ 5 s) since
        classification is on the hot path.
    """

    def __init__(
        self,
        llm_client=None,
        model: str = "qwen-plus",
        timeout: float = 5.0,
    ) -> None:
        self._llm = llm_client
        self._model = model
        self._timeout = timeout
        self._fallback = _RuleClassifier()

    # ── Public API ────────────────────────────────────────────────────────────

    async def classify_async(self, query: str) -> ClassifyResult:
        """Classify *query* — LLM path with automatic fallback."""
        if self._llm is not None:
            try:
                result = await asyncio.wait_for(
                    self._llm_classify(query),
                    timeout=self._timeout,
                )
                logger.info(
                    "LLM classify: type=%s retrieval=%s conf=%.2f complexity=%s | %s",
                    result.primary_type, result.retrieval_type,
                    result.confidence, result.complexity, result.reason,
                )
                return result
            except asyncio.TimeoutError:
                logger.warning("LLM classifier timed out (%.1fs) — falling back", self._timeout)
            except Exception as exc:
                logger.warning("LLM classifier error (%s) — falling back", exc)

        # ── Rule-based fallback ───────────────────────────────────────
        result = self._fallback.classify(query)
        logger.info(
            "Rule classify: type=%s retrieval=%s conf=%.2f complexity=%s",
            result.primary_type, result.retrieval_type,
            result.confidence, result.complexity,
        )
        return result

    # ── LLM path ─────────────────────────────────────────────────────────────

    async def _llm_classify(self, query: str) -> ClassifyResult:
        prompt = _build_prompt(query)
        response = await self._llm.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()
        return self._parse_llm_response(raw)

    def _parse_llm_response(self, raw: str) -> ClassifyResult:
        """Parse and validate the LLM JSON output.

        Raises on any structural problem so the caller falls back to rules.
        """
        # Strip accidental markdown fences
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
            raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)

        data = json.loads(raw)                                # raises on invalid JSON

        primary_type: str = data["primary_type"]
        if primary_type not in QUERY_TYPES:
            raise ValueError(f"Unknown primary_type from LLM: {primary_type!r}")

        confidence = float(data.get("confidence", 0.8))
        complexity  = str(data.get("complexity", "simple"))
        reason      = str(data.get("reason", ""))

        if complexity not in ("simple", "medium", "complex"):
            complexity = "simple"

        retrieval_type = _RETRIEVAL_TYPE[primary_type]
        strategy = _determine_strategy(primary_type, complexity)

        return ClassifyResult(
            primary_type=primary_type,
            retrieval_type=retrieval_type,
            confidence=min(1.0, max(0.0, confidence)),
            complexity=complexity,
            processing_strategy=strategy,
            reason=reason,
            source="llm",
        )


# ── Rule-based fallback (preserved from original, zero dependencies) ──────────

@dataclass
class _QueryPattern:
    name: str
    patterns: List[str]
    priority: int = 0


class _RuleClassifier:
    """Pure-regex classifier — used as fallback when LLM is unavailable."""

    _PATTERNS: List[_QueryPattern] = [
        _QueryPattern("simple_law_query", [
            r"第\s*\d+\s*条",
            r"第\s*[一二三四五六七八九十百千]+\s*条",
            r"《[^》]+》\s*第",
            r"(?:民法典|刑法|宪法|合同法|劳动法|公司法|婚姻法|继承法|行政诉讼法|民事诉讼法|刑事诉讼法)",
            r"法律\s*规定", r"法条", r"条款\s*内容",
        ], priority=100),
        _QueryPattern("case_retrieval", [
            r"案例", r"判决", r"裁定", r"判例",
            r"(?:案件|纠纷)", r"法院\s*(?:认为|判|裁)",
            r"\d{4}.*?号", r"(?:赔偿|损害|侵权)\s*案",
        ], priority=90),
        _QueryPattern("case_analysis", [
            r"分析.*(?:案例|案件|争议)",
            r"(?:争议焦点|裁判要旨|判决理由)",
            r"(?:这个|该)\s*案",
        ], priority=85),
        _QueryPattern("document_generation", [
            r"(?:写|起草|拟|生成|模板).*(?:起诉状|答辩状|合同|协议|申请书|委托书|遗嘱)",
            r"(?:起诉状|答辩状|合同|协议|申请书|委托书|遗嘱).*(?:模板|范本|格式|怎么写)",
        ], priority=80),
        _QueryPattern("procedure_consultation", [
            r"(?:起诉|上诉|申诉|仲裁|立案)\s*(?:需要|要|得)\s*(?:什么|哪些|准备)",
            r"(?:流程|程序|步骤|手续|材料)",
            r"怎么\s*(?:起诉|立案|上诉|申请)",
        ], priority=75),
        _QueryPattern("concept_explanation", [
            r"什么\s*是\s*",
            r"(?:解释|含义|定义|概念)\s*(?:是|指)",
            r"(?:善意取得|不当得利|缔约过失|诉讼时效|公序良俗|显失公平|情势变更)",
        ], priority=70),
        _QueryPattern("comparative_analysis", [
            r"(?:对比|比较|区别|异同|不同)",
            r"(?:A|甲)\s*(?:和|与|跟)\s*(?:B|乙)",
        ], priority=65),
        _QueryPattern("legal_consultation", [
            r"怎么\s*办",
            r"(?:应该|可以|能不能|是否|能否)",
            r"(?:被.*?了|遇到|碰到)",
            r"(?:维权|赔偿|责任|义务|权利)",
        ], priority=50),
    ]

    def classify(self, query: str) -> ClassifyResult:
        primary_type, confidence = self._match(query)
        primary_type = self._detect_mixed(query, primary_type)
        complexity = self._complexity(query, primary_type)
        return ClassifyResult(
            primary_type=primary_type,
            retrieval_type=_RETRIEVAL_TYPE.get(primary_type, "general"),
            confidence=confidence,
            complexity=complexity,
            processing_strategy=_determine_strategy(primary_type, complexity),
            source="rule",
        )

    def _match(self, query: str) -> tuple[str, float]:
        for pat in sorted(self._PATTERNS, key=lambda p: p.priority, reverse=True):
            for regex in pat.patterns:
                if re.search(regex, query, re.IGNORECASE):
                    return pat.name, min(1.0, 0.7 + pat.priority / 500)
        return "legal_consultation", 0.5

    @staticmethod
    def _detect_mixed(query: str, current: str) -> str:
        has_law = any(re.search(p, query) for p in [r"第\s*\d+\s*条", r"法条", r"《[^》]+》"])
        has_case = any(re.search(p, query) for p in [r"案例", r"判决", r"案件", r"\d{4}.*?号"])
        return "mixed_law_case" if (has_law and has_case) else current

    @staticmethod
    def _complexity(query: str, qtype: str) -> str:
        signals = sum([
            bool(re.search(r"(?:并且|而且|同时|以及|另外)", query)),
            bool(re.search(r"(?:如果|假设|假如|倘若)", query)),
            bool(re.search(r"(?:对比|比较|区别)", query)),
            query.count("，") >= 3,
            query.count("？") >= 2,
        ])
        if signals >= 2 or len(query) > 100:
            return "complex"
        if signals >= 1 or len(query) > 50 or qtype in (
            "case_analysis", "comparative_analysis", "mixed_law_case",
        ):
            return "medium"
        return "simple"


# ── Shared strategy builder ───────────────────────────────────────────────────

def _determine_strategy(query_type: str, complexity: str) -> Dict[str, Any]:
    """Pipeline scheduling hints — shared by both LLM and rule paths."""
    base: Dict[str, Any] = {
        "search_mode": "hybrid",
        "k_multiplier": 1.0,
        "enable_rerank": False,
        "enable_parent_fetch": False,
        "enable_sliding_window": False,
    }
    overrides: Dict[str, Dict[str, Any]] = {
        "simple_law_query":  {"enable_parent_fetch": True},
        "case_retrieval":    {"enable_sliding_window": True, "k_multiplier": 1.5},
        "case_analysis":     {"enable_sliding_window": True, "enable_rerank": True, "k_multiplier": 2.0},
        "mixed_law_case":    {"enable_parent_fetch": True, "enable_sliding_window": True,
                              "enable_rerank": True, "k_multiplier": 2.0},
        "document_generation": {"k_multiplier": 0.5},
    }
    base.update(overrides.get(query_type, {}))
    if complexity == "complex":
        base["k_multiplier"] = max(base["k_multiplier"], 2.0)
        base["enable_rerank"] = True
    return base

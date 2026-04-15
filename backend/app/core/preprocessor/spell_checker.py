"""Legal spell-checker — rule-based fast path + optional LLM correction.

Design
------
1. **Fast path** (no LLM, no I/O):
   - Common legal-term typo table (static dict swap).
   - Citation format normalisation (add missing 书名号, etc.).

2. **LLM path** (async, opt-in):
   - Called only when ``enable_llm=True`` AND a heuristic detects likely errors.
   - Uses a domain-specific prompt tuned for legal Chinese.
   - Gracefully degrades: on timeout / error, returns the original query.

The class is intentionally stateless (aside from the injected client ref).
"""

from __future__ import annotations

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ── Static typo table ─────────────────────────────────────────────────────────
# key = common mis-spelling, value = correct term
_LEGAL_TYPOS: dict[str, str] = {
    "诉讼实效": "诉讼时效",
    "起诉书":   "起诉状",
    "辨护":     "辩护",
    "辨护人":   "辩护人",
    "辩护律":   "辩护律师",
    "答辨状":   "答辩状",
    "上述状":   "上诉状",
    "申述":     "申诉",
    "坦白从宽":  "坦白从宽",   # already correct — keep for completeness
    "自首从宽":  "自首从宽",
    "公正处":   "公证处",
    "公正书":   "公证书",
    "公正员":   "公证员",
    "仲才":     "仲裁",
    "管辖全":   "管辖权",
    "法定带理人": "法定代理人",
    "法定带表人": "法定代表人",
    "执行意":   "执行异议",
    "财产保权":  "财产保全",
    "证据保权":  "证据保全",
    "强制直行":  "强制执行",
    "取保后审":  "取保候审",
    "拒留":     "拘留",
    "逮扑":     "逮捕",
    "附带民事":  "附带民事",   # already correct
    "行事责任":  "刑事责任",
    "明事责任":  "民事责任",
    "行事案件":  "刑事案件",
    "明事案件":  "民事案件",
    "知识产全":  "知识产权",
    "商标全":   "商标权",
    "专利全":   "专利权",
    "著作全":   "著作权",
    "不正当竞正": "不正当竞争",
    "善意曲得":  "善意取得",
    "不当德利":  "不当得利",
    "缔约过实":  "缔约过失",
    "显失公评":  "显失公平",
}

# ── Court name shortcuts ──────────────────────────────────────────────────────
_COURT_SHORTCUTS: dict[str, str] = {
    "最高法":   "最高人民法院",
    "最高检":   "最高人民检察院",
    "高法":     "高级人民法院",
    "中法":     "中级人民法院",
    "基层法院":  "基层人民法院",
}

# ── Province abbreviations ────────────────────────────────────────────────────
_PROVINCE_COURT: dict[str, str] = {
    "北京高院": "北京市高级人民法院",
    "上海高院": "上海市高级人民法院",
    "广东高院": "广东省高级人民法院",
    "江苏高院": "江苏省高级人民法院",
    "浙江高院": "浙江省高级人民法院",
    "山东高院": "山东省高级人民法院",
    "四川高院": "四川省高级人民法院",
    "湖北高院": "湖北省高级人民法院",
    "河南高院": "河南省高级人民法院",
    "福建高院": "福建省高级人民法院",
}

# ── LLM prompt ────────────────────────────────────────────────────────────────
_CORRECTION_PROMPT = """\
你是一个资深法律专家，负责纠正用户查询中的错误。

请特别注意以下法律领域的特殊性：
1. **法律术语准确性**：
   - "诉讼时效"而非"诉讼实效"
   - "起诉状"而非"起诉书"
   - "借条"和"欠条"的法律区别
   - "法人"（公司）和"法定代表人"（负责人）的区别

2. **法条引用格式**：
   - 正确："《民法典》第1165条"
   - 错误："民法典1165条"或"民法典第1165条"（缺少书名号）

3. **口语与法言法语转换**：
   - "欠钱不还" → "借款合同纠纷"
   - "被开除了" → "用人单位单方解除劳动合同"
   - "打架伤了人" → "故意伤害他人身体"

4. **地域性表达**：
   - "上海高院" → "上海市高级人民法院"
   - "最高法" → "最高人民法院"

用户输入：{query}

请直接输出纠正后的版本，不要添加任何解释。如果没有错误，原样输出。"""


class LegalSpellChecker:
    """Legal-domain spell checker with rule-based + LLM correction."""

    def __init__(
        self,
        llm_client=None,
        model: str = "qwen-plus",
    ) -> None:
        self.llm_client = llm_client
        self.model = model

    # ── Public API ────────────────────────────────────────────────────────────

    def correct_sync(self, query: str) -> str:
        """Fast rule-based correction only (no LLM, no async)."""
        return self._apply_rules(query)

    async def correct_async(
        self,
        query: str,
        enable_llm: bool = False,
    ) -> str:
        """Full correction: rules first, then optional LLM pass.

        Parameters
        ----------
        query : str
            Raw user query.
        enable_llm : bool
            If True and an ``llm_client`` is available, run the LLM pass
            after rule-based correction.
        """
        corrected = self._apply_rules(query)

        if enable_llm and self.llm_client and self._should_use_llm(corrected):
            corrected = await self._llm_correct(corrected)

        if corrected != query:
            logger.info("Spell correction: '%s' → '%s'", query, corrected)

        return corrected

    # ── Rule-based transforms ─────────────────────────────────────────────────

    def _apply_rules(self, query: str) -> str:
        """Apply all deterministic correction rules."""
        text = query

        # 1. Static typo table swap
        for wrong, right in _LEGAL_TYPOS.items():
            if wrong in text:
                text = text.replace(wrong, right)

        # 2. Court name shortcut expansion
        for short, full in _COURT_SHORTCUTS.items():
            if short in text and full not in text:
                text = text.replace(short, full)

        # 3. Province court abbreviations
        for short, full in _PROVINCE_COURT.items():
            if short in text and full not in text:
                text = text.replace(short, full)

        # 4. Add missing 书名号 for known law names
        text = self._fix_law_citations(text)

        return text

    @staticmethod
    def _fix_law_citations(text: str) -> str:
        """Add 书名号 around bare law names when missing.

        e.g. "民法典第1165条" → "《民法典》第1165条"
        """
        law_names = [
            "民法典", "刑法", "宪法", "合同法", "劳动法", "劳动合同法",
            "公司法", "婚姻法", "继承法", "行政诉讼法", "民事诉讼法",
            "刑事诉讼法", "行政处罚法", "治安管理处罚法", "道路交通安全法",
            "消费者权益保护法", "侵权责任法", "物权法", "担保法",
            "商标法", "专利法", "著作权法", "反不正当竞争法",
        ]
        for name in law_names:
            # Match bare name NOT already inside 书名号
            pattern = rf"(?<!《){re.escape(name)}(?!》)"
            if re.search(pattern, text):
                text = re.sub(pattern, f"《{name}》", text)
        return text

    # ── LLM-based correction ──────────────────────────────────────────────────

    def _should_use_llm(self, query: str) -> bool:
        """Heuristic: should we spend an LLM call on this query?"""
        # 1. Repeated characters (e.g. "法法律")
        if re.search(r"(.)\1{2,}", query):
            return True

        # 2. Very short digit-only query
        if len(query) < 3 and query.isdigit():
            return True

        # 3. Colloquial / informal language (high value for LLM normalisation)
        colloquial = [
            r"怎么办", r"咋办", r"咋整", r"咋弄",
            r"欠钱不还", r"被开除", r"被辞退", r"被炒了",
            r"打架", r"打人", r"被打了",
            r"离婚.*?孩子", r"房子.*?归谁",
        ]
        if any(re.search(p, query) for p in colloquial):
            return True

        # 4. Suspiciously long query with no legal terms
        if len(query) > 30 and not re.search(
            r"(?:法|诉|判|裁|条|款|权|院|合同|纠纷|赔偿|责任)", query
        ):
            return True

        return False

    async def _llm_correct(self, query: str) -> str:
        """Send query through LLM for advanced correction."""
        try:
            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": _CORRECTION_PROMPT.format(query=query)}
                ],
                temperature=0.1,
                max_tokens=200,
            )
            result = response.choices[0].message.content.strip()

            # Sanity: LLM should not drastically alter length
            if result and 0.3 < len(result) / max(len(query), 1) < 3.0:
                return result

            logger.warning(
                "LLM correction suspicious (len %d→%d), keeping original",
                len(query), len(result),
            )
        except Exception as exc:
            logger.error("LLM spell correction failed: %s", exc)

        return query

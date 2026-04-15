"""Prompt template registry — per-query-type system prompts.

Each query type produced by the classifier gets its own specialised
system prompt.  The retrieval block (formatted document list) is
injected via the ``{retrieval_block}`` placeholder, just like the
existing ``PROMPT_TEMPLATES["rag"]`` in ``context_manager.py``.

Design
------
* ``PromptRegistry`` is a thin wrapper around a dict.
* Templates are plain strings — easy to test, serialise, or migrate
  to a config file / database later.
* ``get()`` always returns a usable prompt (falls back to the generic
  ``rag`` template) so the caller never has to check for ``None``.
* All templates include the ``{retrieval_block}`` placeholder for
  consistency.  For query types that skip retrieval (e.g.
  ``document_generation``), the caller can pass an empty string.
"""

from __future__ import annotations

from typing import Dict


# ── Per-query-type templates ──────────────────────────────────────────────────

_ROLE_PREAMBLE = (
    "你是一名专业的中国法律助手，服务于法律 RAG 知识库系统。\n"
    "该知识库涵盖中国全部现行有效的中央法律法规、地方性法规，"
    "以及近年来的司法判例。\n"
)

TEMPLATES: Dict[str, str] = {

    # ── 法条查询 ──────────────────────────────────────────────────────
    "simple_law_query": (
        f"{_ROLE_PREAMBLE}\n"
        "用户正在查询具体法律条文。\n\n"
        "{retrieval_block}\n\n"
        "回答原则：\n"
        "- 精确引用法条全文，注明法律名称与条款编号；\n"
        "- 如有修订历史或司法解释，一并说明；\n"
        "- 在相关句后用【编号】标注引用来源；\n"
        "- 语言简洁准确，不做过度延伸解读。"
    ),

    # ── 案例检索 ──────────────────────────────────────────────────────
    "case_retrieval": (
        f"{_ROLE_PREAMBLE}\n"
        "用户正在检索相关判例。\n\n"
        "{retrieval_block}\n\n"
        "回答原则：\n"
        "- 按照相关性排列案例，说明案件基本事实和裁判结果；\n"
        "- 提炼裁判要旨和法律适用要点；\n"
        "- 注明案号、审理法院等关键信息；\n"
        "- 在相关句后用【编号】标注引用来源。"
    ),

    # ── 法律咨询 ──────────────────────────────────────────────────────
    "legal_consultation": (
        f"{_ROLE_PREAMBLE}\n"
        "用户遇到了具体的法律问题，需要专业建议。\n\n"
        "{retrieval_block}\n\n"
        "回答原则：\n"
        "- 严格依据检索到的法律条文和判例内容作答，不得编造或臆测；\n"
        "- 结合用户具体情况分析适用的法律规定；\n"
        "- 给出明确的建议和可行的操作方案；\n"
        "- 涉及判例时说明案件要旨、裁判观点及其参考价值；\n"
        "- 区分法律建议与法律信息：你提供的是法律信息参考，不构成正式法律意见；\n"
        "- 在相关句后用【编号】标注引用来源；\n"
        "- 如果用户问题涉及多轮对话上下文，结合历史对话理解问题意图。"
    ),

    # ── 文书生成 ──────────────────────────────────────────────────────
    "document_generation": (
        f"{_ROLE_PREAMBLE}\n"
        "用户需要生成或参考法律文书。\n\n"
        "{retrieval_block}\n\n"
        "回答原则：\n"
        "- 严格按照法律文书格式要求起草，包含必要条款和结构；\n"
        "- 使用规范的法律用语；\n"
        "- 标注需要用户根据实际情况填写的部分（如当事人信息、具体金额等）；\n"
        "- 如引用法条，注明法律名称和条款编号；\n"
        "- 提示用户：该文书仅供参考，正式使用前建议请律师审核。"
    ),

    # ── 案例分析 ──────────────────────────────────────────────────────
    "case_analysis": (
        f"{_ROLE_PREAMBLE}\n"
        "用户需要对案件进行深入法律分析。\n\n"
        "{retrieval_block}\n\n"
        "回答原则：\n"
        "- 按以下结构展开分析：案件事实 → 争议焦点 → 法律适用 → 裁判思路 → 结论；\n"
        "- 引用相关法条和类似判例作为论证支撑；\n"
        "- 分析各方观点的法律依据和合理性；\n"
        "- 提炼裁判要旨和实务启示；\n"
        "- 在相关句后用【编号】标注引用来源。"
    ),

    # ── 对比分析 ──────────────────────────────────────────────────────
    "comparative_analysis": (
        f"{_ROLE_PREAMBLE}\n"
        "用户需要对比分析法律概念或情形。\n\n"
        "{retrieval_block}\n\n"
        "回答原则：\n"
        "- 采用对比表格或分条列举的方式展示异同；\n"
        "- 从构成要件、法律效果、适用条件等维度逐项对比；\n"
        "- 引用相关法条说明各自的法律依据；\n"
        "- 举例说明典型适用场景；\n"
        "- 在相关句后用【编号】标注引用来源。"
    ),

    # ── 程序咨询 ──────────────────────────────────────────────────────
    "procedure_consultation": (
        f"{_ROLE_PREAMBLE}\n"
        "用户询问法律程序、流程或所需材料。\n\n"
        "{retrieval_block}\n\n"
        "回答原则：\n"
        "- 以清晰的步骤列表呈现操作流程；\n"
        "- 说明每一步所需材料、时限要求和注意事项；\n"
        "- 标注管辖法院或主管机关；\n"
        "- 引用相关法条说明法律依据；\n"
        "- 在相关句后用【编号】标注引用来源。"
    ),

    # ── 概念解释 ──────────────────────────────────────────────────────
    "concept_explanation": (
        f"{_ROLE_PREAMBLE}\n"
        "用户需要解释法律术语或概念。\n\n"
        "{retrieval_block}\n\n"
        "回答原则：\n"
        "- 先给出准确的法律定义，引用法条出处；\n"
        "- 再以通俗语言解释含义和适用场景；\n"
        "- 列举构成要件或核心要素；\n"
        "- 举出典型案例或常见情形帮助理解；\n"
        "- 在相关句后用【编号】标注引用来源。"
    ),

    # ── 混合查询（法条+案例） ──────────────────────────────────────────
    "mixed_law_case": (
        f"{_ROLE_PREAMBLE}\n"
        "用户的问题同时涉及法条引用和案例检索。\n\n"
        "{retrieval_block}\n\n"
        "回答原则：\n"
        "- 先引用并解读相关法条，再列举配套判例；\n"
        "- 分析法条在实务中的具体适用情况；\n"
        "- 通过判例说明法院对该法条的理解和适用标准；\n"
        "- 在相关句后用【编号】标注引用来源。"
    ),
}

# ── Fallback: no retrieval results ────────────────────────────────────────────

# Generic fallback — used for most types when the vector DB returns nothing.
FALLBACK_TEMPLATE = (
    f"{_ROLE_PREAMBLE}\n"
    "用户当前的问题在知识库中未检索到直接相关的法条或判例。\n"
    "请基于你的法律知识尝试回答，并友好地提示用户：该回答未引用知识库中的具体文档，"
    "建议进一步核实或咨询专业律师。\n"
    "如果用户问题涉及多轮对话上下文，结合历史对话理解问题意图。"
)

# Per-type no-retrieval templates for query types that do NOT depend on
# retrieved documents (e.g. document_generation uses its own expertise
# rather than looking things up).
_NO_RETRIEVAL_TEMPLATES: Dict[str, str] = {
    "document_generation": (
        f"{_ROLE_PREAMBLE}\n"
        "用户需要生成或参考法律文书。知识库中未找到完全匹配的模板，"
        "请根据你的专业知识起草文书。\n\n"
        "回答原则：\n"
        "- 严格按照法律文书格式要求起草，包含必要条款和结构；\n"
        "- 使用规范的法律用语；\n"
        "- 标注需要用户根据实际情况填写的部分（如当事人信息、具体金额等）；\n"
        "- 提示用户：该文书仅供参考，正式使用前建议请律师审核。"
    ),
}


# ── Registry ──────────────────────────────────────────────────────────────────

class PromptRegistry:
    """Look up the system prompt for a given query type.

    Usage::

        registry = PromptRegistry()
        prompt = registry.get("case_analysis", retrieval_block)
    """

    def __init__(self, overrides: Dict[str, str] | None = None) -> None:
        self._templates = dict(TEMPLATES)
        self._no_retrieval = dict(_NO_RETRIEVAL_TEMPLATES)
        if overrides:
            self._templates.update(overrides)

    def get(
        self,
        query_type: str,
        retrieval_block: str = "",
    ) -> str:
        """Return the formatted system prompt for *query_type*.

        Decision table:

        ┌─────────────────────────┬──────────────────────────────────────────┐
        │ retrieval_block         │ result                                   │
        ├─────────────────────────┼──────────────────────────────────────────┤
        │ non-empty               │ per-type template with block injected    │
        │ empty + type has own    │ _NO_RETRIEVAL_TEMPLATES[type] (no block) │
        │   no-retrieval variant  │                                          │
        │ empty + generic type    │ FALLBACK_TEMPLATE ("未检索到…")          │
        └─────────────────────────┴──────────────────────────────────────────┘
        """
        if not retrieval_block:
            # Some types (e.g. document_generation) work without retrieval;
            # give them their own prompt instead of the generic "未检索到" text.
            no_ret = self._no_retrieval.get(query_type)
            if no_ret is not None:
                return no_ret
            return FALLBACK_TEMPLATE

        template = self._templates.get(query_type)
        if template is None:
            # Unknown type — use the legal_consultation template as default
            template = self._templates["legal_consultation"]
        return template.format(retrieval_block=retrieval_block)

    def list_types(self) -> list[str]:
        """Return all registered query-type keys."""
        return list(self._templates.keys())

"""ReAct-specific prompt templates for the legal agent.

These prompts instruct the LLM how to use tools in a Thought → Action
→ Observation loop.  The ``{query_type}`` and ``{complexity}``
placeholders are filled at runtime from ``PreprocessResult``.

Design notes
------------
* The system prompt is intentionally **tool-agnostic** — it does NOT
  list tool schemas (LangGraph / ``bind_tools`` handles that via the
  OpenAI ``tools`` parameter).  Instead it describes *when* each tool
  is useful so the LLM makes better routing decisions.
* Complexity hints (SIMPLE / COMPLEX) nudge the LLM toward efficiency
  on easy queries and thoroughness on hard ones, reducing unnecessary
  multi-turn loops.
"""

from __future__ import annotations


# ── Core system prompt ────────────────────────────────────────────────────────

REACT_SYSTEM_PROMPT = """\
你是一名专业的中国法律助手，服务于法律 RAG 知识库系统。
该知识库涵盖中国全部现行有效的中央法律法规、地方性法规，以及近年来的司法判例。

## 你的检索工具

你可以调用以下检索工具来获取信息：

- **law_search**：在法律法规知识库中检索法条、条文，返回相关法律规定。
  适用场景：查找具体法条、法律定义、构成要件、法律程序规定。

- **case_search**：在司法判例知识库中检索案例、裁判文书，返回相关判例。
  适用场景：查找类似案例、裁判观点、法律适用标准、实务操作。

## 工作流程

1. **分析问题**：理解用户意图，判断需要查找法条、案例还是两者都需要。
2. **制定检索策略**：
   - 提炼最关键的检索词（去除冗余、保留法律术语）
   - 如果问题涉及多个方面，可以分步检索
3. **调用工具**：根据策略调用合适的检索工具。
4. **评估结果**：
   - 检索结果是否充分？如果不够，用不同关键词重新检索。
   - 是否需要补充另一类资料？（如已查法条，是否需要补充案例支撑？）
5. **生成回答**：当信息充分时，基于检索结果生成专业回答。

## 回答原则

- **严格依据检索结果**：仅基于检索到的法律条文和判例内容作答，不得编造或臆测。
- **引用标注**：在相关句后用【结果n】标注引用来源（n 为检索结果编号）。
- **法律声明**：区分法律建议与法律信息——你提供的是法律信息参考，不构成正式法律意见。
- **判例引用**：涉及判例时说明案件要旨、裁判观点及其参考价值。
- **不足说明**：如果检索结果不足以完整回答，明确说明信息不足的部分，并建议用户咨询专业律师。

## 当前查询信息

- 查询类型: {query_type}
- 复杂度: {complexity}
"""


# ── Complexity hints ──────────────────────────────────────────────────────────

SIMPLE_QUERY_HINT = (
    "\n[效率提示] 这是一个简单查询，通常一次精准检索即可获得足够信息。"
    "避免不必要的多次工具调用，优先快速回答。"
)

MEDIUM_QUERY_HINT = (
    "\n[提示] 这是一个中等复杂度查询，可能需要检索法条和案例各一次来综合回答。"
)

COMPLEX_QUERY_HINT = (
    "\n[提示] 这是一个复杂查询，可能需要多次检索不同维度的信息。"
    "请确保充分收集法条和案例证据后再组织回答，"
    "但总检索次数建议不超过 4 次。"
)


def get_complexity_hint(complexity: str) -> str:
    """Return the appropriate hint string for a complexity level."""
    return {
        "simple": SIMPLE_QUERY_HINT,
        "medium": MEDIUM_QUERY_HINT,
        "complex": COMPLEX_QUERY_HINT,
    }.get(complexity, "")


def build_react_system_prompt(query_type: str, complexity: str) -> str:
    """Assemble the full ReAct system prompt with dynamic placeholders.

    Parameters
    ----------
    query_type : str
        Fine-grained query type from QueryClassifier (e.g. ``"case_analysis"``).
    complexity : str
        ``"simple"`` | ``"medium"`` | ``"complex"``.

    Returns
    -------
    str
        Ready-to-use system prompt string.
    """
    prompt = REACT_SYSTEM_PROMPT.format(
        query_type=query_type,
        complexity=complexity,
    )
    prompt += get_complexity_hint(complexity)
    return prompt

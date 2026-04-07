"""Document quality checker: rule-based + LLM hybrid analysis."""

import logging
import re
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from openai import OpenAI
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class ErrorDetail:
    """Detailed information about a document error."""

    error_type: str
    description: str
    position: Optional[int]
    severity: str


@dataclass
class CheckResult:
    """Result of document quality check."""

    total_errors: int
    rule_based_errors: int
    llm_detected_errors: int
    error_details: List[ErrorDetail]
    error_type_distribution: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        """Convert CheckResult to dictionary for serialization."""
        return {
            "total_errors": self.total_errors,
            "rule_based_errors": self.rule_based_errors,
            "llm_detected_errors": self.llm_detected_errors,
            "error_details": [
                {
                    "error_type": e.error_type,
                    "description": e.description,
                    "position": e.position,
                    "severity": e.severity,
                }
                for e in self.error_details
            ],
            "error_type_distribution": self.error_type_distribution,
        }


class DocumentChecker:
    """Document quality checker using rule-based + LLM hybrid approach.

    Performs rule-based checks for common formatting issues and
    uses Qwen LLM for deep linguistic analysis.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        enable_llm_check: bool = True,
    ):
        """Initialize the document checker.

        Args:
            api_key: API key for the LLM service.
            base_url: Base URL for the LLM API.
            model: Model name for the LLM.
            enable_llm_check: Whether to enable LLM-based checking.
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.enable_llm_check = enable_llm_check

    def _rule_based_check(self, text: str) -> List[ErrorDetail]:
        """Perform rule-based checks on the text.

        Args:
            text: The text to check.

        Returns:
            List of ErrorDetail objects found by rules.
        """
        errors: List[ErrorDetail] = []

        # 1. Excessive whitespace: 3+ consecutive spaces or 3+ consecutive newlines
        excessive_space_pattern = r" {3,}"
        excessive_newline_pattern = r"\n{3,}"

        for match in re.finditer(excessive_space_pattern, text):
            errors.append(
                ErrorDetail(
                    error_type="format",
                    description=f"Excessive whitespace: {len(match.group())} consecutive spaces",
                    position=match.start(),
                    severity="low",
                )
            )

        for match in re.finditer(excessive_newline_pattern, text):
            errors.append(
                ErrorDetail(
                    error_type="format",
                    description=f"Excessive newlines: {len(match.group())} consecutive line breaks",
                    position=match.start(),
                    severity="low",
                )
            )

        # 2. Inconsistent case: mixed UPPER and lower in same word
        # Pattern: words with both uppercase and lowercase, excluding camelCase at start
        # Looks for words like "hElLo" or "HeLLo" that are likely typos
        inconsistent_case_pattern = (
            r"\b[a-z]*[A-Z][a-z]*[A-Z][a-zA-Z]*\b|\b[A-Z]*[a-z][A-Z]*[a-z][a-zA-Z]*\b"
        )

        for match in re.finditer(inconsistent_case_pattern, text):
            word = match.group()
            # Exclude valid camelCase and PascalCase patterns
            if not self._is_valid_mixed_case(word):
                errors.append(
                    ErrorDetail(
                        error_type="case",
                        description=f"Inconsistent case in word: '{word}'",
                        position=match.start(),
                        severity="medium",
                    )
                )

        # 3. Duplicate punctuation: repeated punctuation marks
        duplicate_punct_pattern = r"([!\.，。？！,;:\\-])\1+"

        for match in re.finditer(duplicate_punct_pattern, text):
            errors.append(
                ErrorDetail(
                    error_type="punctuation",
                    description=f"Duplicate punctuation: '{match.group()}'",
                    position=match.start(),
                    severity="low",
                )
            )

        # 4. Unstandard symbols: & @ # when not in URLs or emails
        # Look for these symbols not preceded/followed by URL/email patterns
        lines = text.split("\n")
        pos = 0
        for line in lines:
            # Skip if line looks like URL or email
            if re.match(r"^https?://|^[\w.-]+@[\w.-]+\.\w+$", line.strip()):
                pos += len(line) + 1
                continue

            # Find standalone symbols
            for symbol in ["&", "@", "#"]:
                for match in re.finditer(re.escape(symbol), line):
                    # Check context to avoid URLs
                    start = max(0, match.start() - 10)
                    end = min(len(line), match.end() + 10)
                    context = line[start:end]

                    # Skip if in URL pattern
                    if re.search(r"https?://[^\s]*" + re.escape(symbol), context):
                        continue
                    # Skip if in email pattern
                    if re.match(r"[\w.-]*@[\w.-]*", context):
                        continue

                    errors.append(
                        ErrorDetail(
                            error_type="symbol",
                            description=f"Unstandard symbol '{symbol}' should be converted to text equivalent",
                            position=pos + match.start(),
                            severity="medium",
                        )
                    )
            pos += len(line) + 1

        # 5. Control characters: non-printable except \n, \t, \r
        control_char_pattern = r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"

        for match in re.finditer(control_char_pattern, text):
            errors.append(
                ErrorDetail(
                    error_type="format",
                    description=f"Control character found: \\x{ord(match.group()):02x}",
                    position=match.start(),
                    severity="high",
                )
            )

        # 6. Unmatched brackets/parentheses
        bracket_pairs = [
            ("(", ")"),
            ("[", "]"),
            ("{", "}"),
            ("【", "】"),
            ("（", "）"),
        ]

        for open_br, close_br in bracket_pairs:
            stack = []
            for idx, char in enumerate(text):
                if char == open_br:
                    stack.append((idx, char))
                elif char == close_br:
                    if not stack:
                        errors.append(
                            ErrorDetail(
                                error_type="format",
                                description=f"Unmatched closing bracket: '{char}'",
                                position=idx,
                                severity="medium",
                            )
                        )
                    else:
                        stack.pop()

            # Remaining unclosed brackets
            for idx, char in stack:
                errors.append(
                    ErrorDetail(
                        error_type="format",
                        description=f"Unmatched opening bracket: '{char}'",
                        position=idx,
                        severity="medium",
                    )
                )

        return errors

    def _is_valid_mixed_case(self, word: str) -> bool:
        """Check if a mixed-case word is valid (camelCase/PascalCase).

        Args:
            word: The word to check.

        Returns:
            True if the mixed case pattern is valid.
        """
        # Valid patterns:
        # - PascalCase: starts with uppercase, rest camel case (HttpRequest)
        # - camelCase: starts with lowercase (httpRequest)
        # - All uppercase abbreviations followed by word (URLPattern)

        if word.isupper() or word.islower():
            return True

        # Check for alternating case like hElLo (invalid)
        upper_count = sum(1 for c in word if c.isupper())
        lower_count = len(word) - upper_count

        # If mixed but looks like camelCase/PascalCase
        if upper_count > 0 and lower_count > 0:
            # Check for alternating pattern (invalid)
            transitions = sum(
                1
                for i in range(1, len(word))
                if word[i].isupper() != word[i - 1].isupper()
            )

            # More than 4 transitions suggests random casing like hElLo
            if transitions > 4:
                return False

            # Check if it follows camelCase pattern
            # Allow: startLower then Upper at word boundaries
            if word[0].islower():
                # camelCase: check for uppercase only at word boundaries
                for i in range(1, len(word)):
                    if word[i].isupper():
                        # Should be start of a new word part
                        if i > 0 and not word[i - 1].islower():
                            return False
                return True

            # PascalCase: similar check
            if word[0].isupper():
                for i in range(1, len(word)):
                    if word[i].isupper():
                        # Should follow lowercase or be start
                        if i > 0 and word[i - 1].isupper():
                            # Check for common abbreviations like URL, HTTP
                            if upper_count <= 4:
                                return True
                            return False
                return True

        return True

    def _llm_check(self, text: str) -> List[ErrorDetail]:
        """Perform LLM-based quality check on the text.

        Args:
            text: The text to check.

        Returns:
            List of ErrorDetail objects found by LLM.
        """
        if not self.enable_llm_check:
            return []

        # Text length limit
        original_length = len(text)
        if original_length > 4000:
            text = text[:4000]
            logger.info(
                "Text truncated from %d to 4000 chars for LLM check", original_length
            )

        system_prompt = """你是一个文档质量检查助手。请分析以下文本中的错误，并以JSON格式返回结果。
检查项目：
1. 错别字（中文和英文拼写错误）
2. 语法错误
3. 标点符号使用不当
4. 格式问题（如多余空格、不一致的格式）

请严格按以下JSON格式返回，不要添加其他内容：
{"errors": [{"type": "错误类型", "description": "错误描述", "severity": "low/medium/high"}]}
如果没有错误，返回：{"errors": []}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]

        try:
            completion = self.client.chat.completions.create(
                model=self.model, messages=messages
            )
            response_text = completion.choices[0].message.content

            # Parse JSON response
            try:
                result = json.loads(response_text)
                errors_data = result.get("errors", [])

                errors: List[ErrorDetail] = []
                for error_data in errors_data:
                    errors.append(
                        ErrorDetail(
                            error_type=error_data.get("type", "unknown"),
                            description=error_data.get("description", ""),
                            position=None,  # LLM doesn't provide positions
                            severity=error_data.get("severity", "medium"),
                        )
                    )

                logger.info("LLM check completed: %d errors found", len(errors))
                return errors

            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse LLM JSON response: %s", exc)
                return []

        except Exception as exc:
            logger.error("LLM check failed: %s", exc)
            return []

    def _is_similar_error(self, err1: ErrorDetail, err2: ErrorDetail) -> bool:
        """Check if two errors are similar enough to be considered duplicates.

        Args:
            err1: First error detail.
            err2: Second error detail.

        Returns:
            True if errors are similar.
        """
        # Same type and high description similarity
        if err1.error_type != err2.error_type:
            return False

        similarity = SequenceMatcher(None, err1.description, err2.description).ratio()
        return similarity > 0.8

    def _deduplicate_errors(self, errors: List[ErrorDetail]) -> List[ErrorDetail]:
        """Remove duplicate errors based on similarity.

        Args:
            errors: List of errors to deduplicate.

        Returns:
            Deduplicated list with rule-based errors prioritized.
        """
        result: List[ErrorDetail] = []

        for error in errors:
            is_duplicate = False
            for existing in result:
                if self._is_similar_error(error, existing):
                    is_duplicate = True
                    break
            if not is_duplicate:
                result.append(error)

        return result

    def check(self, text: str) -> CheckResult:
        """Check a single text for quality issues.

        Args:
            text: The text to check.

        Returns:
            CheckResult with all findings.
        """
        # Run rule-based check
        rule_errors = self._rule_based_check(text)

        # Run LLM check if enabled
        llm_errors = []
        if self.enable_llm_check:
            llm_errors = self._llm_check(text)

        # Merge and deduplicate
        all_errors = rule_errors + llm_errors
        deduplicated_errors = self._deduplicate_errors(all_errors)

        # Calculate distribution
        distribution: Dict[str, int] = {}
        for error in deduplicated_errors:
            distribution[error.error_type] = distribution.get(error.error_type, 0) + 1

        # Count LLM-only errors (those without position are from LLM)
        llm_count = sum(1 for e in deduplicated_errors if e.position is None)
        rule_count = len(deduplicated_errors) - llm_count

        result = CheckResult(
            total_errors=len(deduplicated_errors),
            rule_based_errors=rule_count,
            llm_detected_errors=llm_count,
            error_details=deduplicated_errors,
            error_type_distribution=distribution,
        )

        logger.info(
            "Document check complete: %d errors found (%d rule-based, %d LLM-detected)",
            result.total_errors,
            result.rule_based_errors,
            result.llm_detected_errors,
        )

        return result

    def check_documents(self, documents: List[Document]) -> List[CheckResult]:
        """Check multiple documents for quality issues.

        Args:
            documents: List of Document objects to check.

        Returns:
            List of CheckResult objects (one per document).
        """
        results: List[CheckResult] = []

        for idx, doc in enumerate(documents):
            logger.debug("Checking document %d/%d", idx + 1, len(documents))
            result = self.check(doc.page_content)
            results.append(result)

        return results

    def compare_before_after(
        self,
        before_docs: List[Document],
        after_docs: List[Document],
    ) -> Dict[str, Any]:
        """Compare quality metrics before and after processing.

        Args:
            before_docs: Documents before processing.
            after_docs: Documents after processing.

        Returns:
            Comparison report dictionary.
        """
        before_results = self.check_documents(before_docs)
        after_results = self.check_documents(after_docs)

        # Aggregate metrics
        before_total = sum(r.total_errors for r in before_results)
        after_total = sum(r.total_errors for r in after_results)

        # Aggregate distributions
        before_dist: Dict[str, int] = {}
        after_dist: Dict[str, int] = {}

        for result in before_results:
            for error_type, count in result.error_type_distribution.items():
                before_dist[error_type] = before_dist.get(error_type, 0) + count

        for result in after_results:
            for error_type, count in result.error_type_distribution.items():
                after_dist[error_type] = after_dist.get(error_type, 0) + count

        errors_reduced = before_total - after_total
        reduction_rate = (
            (errors_reduced / before_total * 100) if before_total > 0 else 0.0
        )

        report = {
            "before": {"total_errors": before_total, "distribution": before_dist},
            "after": {"total_errors": after_total, "distribution": after_dist},
            "improvement": {
                "errors_reduced": errors_reduced,
                "reduction_rate": round(reduction_rate, 2),
            },
        }

        logger.info(
            "Comparison complete: %d errors reduced (%.1f%% improvement)",
            errors_reduced,
            reduction_rate,
        )

        return report

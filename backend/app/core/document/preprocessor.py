"""Document preprocessing pipeline: regex-based cleaning + multi-level LLM fallback processing."""

import logging
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import List, Optional

from langchain_core.documents import Document
from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMAllProvidersFailedError(Exception):
    """Raised when all LLM providers in the fallback chain have failed."""


@dataclass
class LLMProvider:
    """Configuration for a single LLM provider in the fallback chain."""

    name: str
    client: OpenAI
    model: str


class DocumentPreprocessor:
    """Preprocesses documents through regex cleaning and multi-level LLM correction.

    Two-stage pipeline:
    1. Regex-based text cleaning (HTML removal, normalization, desensitization)
    2. LLM processing with fallback chain: DashScope → Kimi k2.5 → mark pending

    When the primary LLM (DashScope/Qwen) fails after retries, the preprocessor
    degrades to the next provider (Kimi). If all providers fail, the document is
    marked as pending for later reprocessing.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        enable_llm_preprocessing: bool = True,
        fallback_providers: Optional[List[LLMProvider]] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        # Build the ordered provider chain: primary + fallbacks
        primary = LLMProvider(
            name="dashscope",
            client=OpenAI(api_key=api_key, base_url=base_url),
            model=model,
        )
        self._provider_chain: List[LLMProvider] = [primary]
        if fallback_providers:
            self._provider_chain.extend(fallback_providers)

        self.model = model
        self.enable_llm_preprocessing = enable_llm_preprocessing
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _regex_preprocess(self, text: str) -> str:
        """Regex-based text cleaning and normalization.

        Steps:
        1. Strip leading/trailing whitespace
        2. Normalize unicode (NFC)
        3. Remove HTML/XML tags
        4. Remove control characters and normalize special chars
        5. Normalize date formats to YYYY-MM-DD
        6. Normalize punctuation (half-width to full-width)
        7. Desensitize sensitive info (ID card, phone numbers)
        8. Remove excessive whitespace
        """
        # 1. Strip leading/trailing whitespace
        text = text.strip()

        # 2. Normalize unicode (NFC normalization)
        text = unicodedata.normalize("NFC", text)

        # 3. Remove HTML/XML tags
        text = re.sub(r"<[^>]*>", "", text)

        # 4. Remove control characters (0x00-0x1F, 0x7F)
        # Keep newlines and tabs for structure
        text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", text)

        # Remove other special characters, keep: word chars, whitespace, punctuation, CJK
        # This pattern keeps: \w (word chars), \s (whitespace), specified punctuation, and CJK characters
        text = re.sub(r"[^\w\s.,!?;:，。！？；：\u4e00-\u9fff]", "", text)

        # 5. Normalize date formats to YYYY-MM-DD
        # Pattern: YYYY年MM月DD日 → YYYY-MM-DD
        text = re.sub(
            r"(\d{4})年(\d{1,2})月(\d{1,2})日",
            lambda m: f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}",
            text,
        )

        # Pattern: DD/MM/YYYY or MM/DD/YYYY (assume DD/MM/YYYY for non-US context)
        # If day > 12, it's definitely DD/MM/YYYY
        # If both <= 12, we assume DD/MM/YYYY (common in CN context)
        def normalize_slash_date(match: re.Match) -> str:
            part1 = int(match.group(1))
            part2 = int(match.group(2))
            year = int(match.group(3))
            # Normalize 2-digit year to 4-digit
            if year < 50:
                year += 2000
            elif year < 100:
                year += 1900
            # Assume DD/MM/YYYY format
            return f"{year}-{part2:02d}-{part1:02d}"

        text = re.sub(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", normalize_slash_date, text)

        # 6. Normalize punctuation: half-width to full-width
        punctuation_map = {
            ",": "，",
            ".": "。",
            "!": "！",
            "?": "？",
            ";": "；",
            ":": "：",
        }
        for half, full in punctuation_map.items():
            text = text.replace(half, full)

        # 7. Desensitize sensitive information
        # Chinese ID card: 18 digits (or 17 digits + X), or 15 digits (old format)
        text = re.sub(r"\b\d{17}[\dXx]\b", "***", text)
        text = re.sub(r"\b\d{15}\b", "***", text)
        # Phone numbers: 11 digits starting with 1, or formatted like 138-1234-5678
        text = re.sub(r"\b1[3-9]\d{9}\b", "***", text)
        text = re.sub(r"\b1[3-9]\d-\d{4}-\d{4}\b", "***", text)
        text = re.sub(r"\b1[3-9]\d{2}-\d{4}-\d{4}\b", "***", text)

        # 8. Remove excessive whitespace (collapse multiple spaces/newlines)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n+", "\n", text)
        text = re.sub(r"\s+", " ", text)

        return text

    def _llm_call_single(self, provider: LLMProvider, messages: list) -> str:
        """Call a single LLM provider with retries and exponential backoff.

        Args:
            provider: The LLM provider to call.
            messages: Chat messages to send.

        Returns:
            The LLM response text.

        Raises:
            Exception: If all retries for this provider are exhausted.
        """
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                completion = provider.client.chat.completions.create(
                    model=provider.model, messages=messages
                )
                return completion.choices[0].message.content
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "[%s] LLM call attempt %d/%d failed: %s — retrying in %.1fs",
                        provider.name,
                        attempt,
                        self.max_retries,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "[%s] LLM call failed after %d retries: %s",
                        provider.name,
                        self.max_retries,
                        exc,
                    )

        raise last_exc  # type: ignore[misc]

    def _llm_process(self, text: str) -> str:
        """Process text through the LLM fallback chain.

        Tries each provider in order (DashScope → Kimi → ...).
        If all providers fail, raises LLMAllProvidersFailedError.

        Args:
            text: Input text to process.

        Returns:
            Processed text from the first successful provider.

        Raises:
            LLMAllProvidersFailedError: When every provider in the chain exhausted retries.
        """
        # Text length limit check: chunk if > 6000 chars
        if len(text) > 6000:
            logger.info("Text exceeds 6000 chars — processing in chunks")
            return self._llm_process_chunked(text)

        system_prompt = (
            "你是一名资深法律文书校对专家。请严格按以下规则处理文本：\n"
            "1. **纠正错别字和语法错误**，但绝对保留所有专业法律术语（如'善意取得'、'无因管理'）的原貌。\n"
            "2. **统一法律引用格式**：将文中对法律法规的引用，统一为《中华人民共和国XXXX法》第X条的格式。\n"
            "3. **补全常见法律简称**：如将《刑法》补全为《中华人民共和国刑法》，但需根据上下文判断，避免错误补全。\n"
            "4. 保持原文段落结构，除上述修改外，不做任何删减或转写。\n"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]

        provider_errors: List[str] = []

        for provider in self._provider_chain:
            try:
                result = self._llm_call_single(provider, messages)
                if provider is not self._provider_chain[0]:
                    logger.warning(
                        "LLM preprocessing degraded to fallback provider [%s]",
                        provider.name,
                    )
                else:
                    logger.debug("LLM processing completed via [%s]", provider.name)
                return result
            except Exception as exc:
                provider_errors.append(f"{provider.name}: {exc}")
                logger.warning(
                    "Provider [%s] exhausted — degrading to next provider",
                    provider.name,
                )

        # All providers failed
        error_summary = "; ".join(provider_errors)
        raise LLMAllProvidersFailedError(f"All LLM providers failed: {error_summary}")

    def _llm_process_chunked(self, text: str) -> str:
        """Process long text in chunks and rejoin results.

        Splits on paragraph boundaries when possible to maintain coherence.
        Each chunk goes through the full fallback chain independently.

        Raises:
            LLMAllProvidersFailedError: If any chunk fails all providers.
        """
        chunk_size = 5000  # Conservative chunk size to stay under limit
        chunks = []

        # Try to split on paragraph boundaries
        paragraphs = text.split("\n\n")
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= chunk_size:
                if current_chunk:
                    current_chunk += "\n\n"
                current_chunk += para
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                # Handle paragraph that itself exceeds chunk size
                if len(para) > chunk_size:
                    # Hard split on sentence boundaries using regex
                    # Split on punctuation followed by space or end of string
                    sentences = re.split(r"([。！？\.\!\?]\s*)", para)
                    current_chunk = ""
                    for i in range(0, len(sentences), 2):
                        sent = sentences[i]
                        if i + 1 < len(sentences):
                            sent += sentences[i + 1]  # Add punctuation back
                        if len(current_chunk) + len(sent) + 1 <= chunk_size:
                            if current_chunk:
                                current_chunk += " "
                            current_chunk += sent
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = sent
                else:
                    current_chunk = para

        if current_chunk:
            chunks.append(current_chunk)

        logger.info("Split text into %d chunks for processing", len(chunks))

        # Process each chunk
        processed_chunks = []
        for idx, chunk in enumerate(chunks, 1):
            logger.debug(
                "Processing chunk %d/%d (%d chars)", idx, len(chunks), len(chunk)
            )
            processed = self._llm_process(chunk)
            processed_chunks.append(processed)

        # Rejoin with appropriate separators
        return "\n\n".join(processed_chunks)

    def preprocess(self, documents: List[Document]) -> List[Document]:
        """Process documents through the preprocessing pipeline.

        Args:
            documents: List of LangChain Document objects to process.

        Returns:
            List of processed Document objects with cleaned page_content
            and updated metadata:
            - preprocessed: True if LLM succeeded
            - preprocessed_by: provider name that handled the text (e.g. "dashscope", "kimi")
            - pending_preprocessing: True if all LLM providers failed (text only regex-cleaned)
        """
        if not documents:
            logger.info("No documents to preprocess")
            return []

        logger.info("Preprocessing %d documents...", len(documents))
        processed_docs = []

        for idx, doc in enumerate(documents, 1):
            try:
                # Stage 1: Regex preprocessing
                content = self._regex_preprocess(doc.page_content)

                # Stage 2: LLM processing with fallback chain (if enabled)
                metadata = dict(doc.metadata)
                if self.enable_llm_preprocessing:
                    try:
                        content = self._llm_process(content)
                        metadata["preprocessed"] = True
                    except LLMAllProvidersFailedError as llm_exc:
                        logger.warning(
                            "Document %d/%d: all LLM providers failed — marking pending. %s",
                            idx,
                            len(documents),
                            llm_exc,
                        )
                        metadata["preprocessed"] = False
                        metadata["pending_preprocessing"] = True
                        metadata["preprocessing_error"] = str(llm_exc)
                        # content keeps regex-cleaned version
                else:
                    metadata["preprocessed"] = False

                processed_doc = Document(page_content=content, metadata=metadata)
                processed_docs.append(processed_doc)

                logger.debug(
                    "Document %d/%d preprocessed successfully", idx, len(documents)
                )

            except Exception as exc:
                logger.error(
                    "Failed to preprocess document %d/%d: %s — keeping original",
                    idx,
                    len(documents),
                    exc,
                )
                # Keep original document on error
                metadata = dict(doc.metadata)
                metadata["preprocessed"] = False
                metadata["preprocessing_error"] = str(exc)
                processed_docs.append(
                    Document(page_content=doc.page_content, metadata=metadata)
                )

        # Summary logging
        pending_count = sum(
            1 for d in processed_docs if d.metadata.get("pending_preprocessing")
        )
        if pending_count:
            logger.warning(
                "Preprocessing complete: %d documents processed, %d marked pending",
                len(processed_docs),
                pending_count,
            )
        else:
            logger.info(
                "Preprocessing complete: %d documents processed", len(processed_docs)
            )

        return processed_docs

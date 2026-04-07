"""Document preprocessing pipeline: NLTK + Qwen LLM processing."""

import logging
import re
import unicodedata
from typing import List

from langchain_core.documents import Document
from openai import OpenAI

logger = logging.getLogger(__name__)

try:
    import nltk
    from nltk.tokenize import sent_tokenize
except ImportError:
    nltk = None
    sent_tokenize = None
    logger.warning(
        "NLTK not installed — NLTK preprocessing stage will be disabled. "
        "Install with: pip install nltk"
    )


class DocumentPreprocessor:
    """Preprocesses documents through NLTK cleaning and Qwen LLM correction.

    Two-stage pipeline:
    1. NLTK preprocessing: basic text cleaning (whitespace, unicode, tokenization)
    2. Qwen LLM processing: intelligent text correction via DashScope API
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        enable_llm_preprocessing: bool = True,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.enable_llm_preprocessing = enable_llm_preprocessing

    def _nltk_preprocess(self, text: str) -> str:
        """Basic text cleaning using NLTK.

        Steps:
        1. Strip leading/trailing whitespace
        2. Normalize unicode (NFC)
        3. Remove excessive whitespace
        4. Remove control characters (except newlines/tabs)
        5. Sentence tokenization and rejoin
        """
        if nltk is None:
            logger.debug("NLTK not available — skipping NLTK preprocessing")
            return text

        # 1. Strip leading/trailing whitespace
        text = text.strip()

        # 2. Normalize unicode (NFC normalization)
        text = unicodedata.normalize("NFC", text)

        # 3. Remove excessive whitespace (collapse multiple spaces/newlines)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n+", "\n", text)

        # 4. Remove control characters (except newlines and tabs)
        text = "".join(
            char
            for char in text
            if not unicodedata.category(char).startswith("C") or char in ("\n", "\t")
        )

        # 5. Basic sentence tokenization
        """
        try:
            sentences = sent_tokenize(text)
        except LookupError:
            # Missing punkt data, download and retry
            logger.info("Downloading NLTK punkt_tab data...")
            nltk.download("punkt_tab", quiet=True)
            try:
                sentences = sent_tokenize(text)
            except Exception as exc:
                logger.warning(
                    "NLTK sentence tokenization failed: %s — returning as-is", exc
                )
                return text
        except Exception as exc:
            logger.warning(
                "NLTK sentence tokenization failed: %s — returning as-is", exc
            )
            return text
        """
        # Rejoin sentences with single space
        #return " ".join(sentences)
        return text

    def _qwen_process(self, text: str) -> str:
        """Intelligent text correction using Qwen LLM via DashScope API.

        Handles text length limits by chunking if necessary.
        Returns original text on API errors (resilient pipeline).
        """
        # Text length limit check: chunk if > 6000 chars
        if len(text) > 6000:
            logger.info("Text exceeds 6000 chars — processing in chunks")
            return self._qwen_process_chunked(text)

        system_prompt = (
            "你是一个文档预处理助手。请对输入文本进行以下处理，直接返回处理后的文本，不要添加任何解释或标记：\n"
            "1. 修正错别字（保持原意）\n"
            "2. 将所有英文字母转换为小写\n"
            "3. 移除多余和错误的符号（保留必要的标点）\n"
            "4. 将特殊符号转为统一格式（例如：& 转为 and，@ 转为 at，# 转为 number）"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]

        try:
            completion = self.client.chat.completions.create(
                model=self.model, messages=messages
            )
            processed = completion.choices[0].message.content
            logger.debug("Qwen processing completed for text chunk")
            return processed
        except Exception as exc:
            logger.error(
                "Qwen LLM processing failed: %s — returning original text", exc
            )
            return text

    def _qwen_process_chunked(self, text: str) -> str:
        """Process long text in chunks and rejoin results.

        Splits on paragraph boundaries when possible to maintain coherence.
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
                    # Split on sentence boundaries
                    if nltk is not None:
                        try:
                            sentences = sent_tokenize(para)
                            current_chunk = ""
                            for sent in sentences:
                                if len(current_chunk) + len(sent) + 1 <= chunk_size:
                                    if current_chunk:
                                        current_chunk += " "
                                    current_chunk += sent
                                else:
                                    if current_chunk:
                                        chunks.append(current_chunk)
                                    current_chunk = sent
                        except Exception:
                            # Fallback: hard split
                            for i in range(0, len(para), chunk_size):
                                chunks.append(para[i : i + chunk_size])
                            current_chunk = ""
                    else:
                        # Hard split if NLTK unavailable
                        for i in range(0, len(para), chunk_size):
                            chunks.append(para[i : i + chunk_size])
                        current_chunk = ""
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
            processed = self._qwen_process(chunk)
            processed_chunks.append(processed)

        # Rejoin with appropriate separators
        return "\n\n".join(processed_chunks)

    def preprocess(self, documents: List[Document]) -> List[Document]:
        """Process documents through the preprocessing pipeline.

        Args:
            documents: List of LangChain Document objects to process.

        Returns:
            List of processed Document objects with cleaned page_content
            and updated metadata (preprocessed: True).
        """
        if not documents:
            logger.info("No documents to preprocess")
            return []

        logger.info("Preprocessing %d documents...", len(documents))
        processed_docs = []

        for idx, doc in enumerate(documents, 1):
            try:
                # Stage 1: NLTK preprocessing
                content = self._nltk_preprocess(doc.page_content)

                # Stage 2: Qwen LLM processing (if enabled)
                if self.enable_llm_preprocessing:
                    content = self._qwen_process(content)

                # Create new Document with processed content
                metadata = dict(doc.metadata)
                metadata["preprocessed"] = True

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

        logger.info(
            "Preprocessing complete: %d documents processed", len(processed_docs)
        )
        return processed_docs

"""Evaluate Qwen-Plus, Qwen3-Max and Kimi-K2.5 preprocessing independently.

Reads a txt file, splits into chunks, applies each model's preprocessing
**in parallel** (async), and uses DocumentChecker to measure how many errors
each reduces.  Processing speed for every model is recorded and reported.

Usage:
    python -m backend.app.core.quality.evaluate --file path/to/file.txt
    python -m backend.app.core.quality.evaluate --file path/to/file.txt --chunk-size 1000
"""

import argparse
import asyncio
import logging
import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

# Ensure project root is on sys.path
project_root = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from langchain_core.documents import Document

from backend.app.config.settings import get_settings
from backend.app.core.document.preprocessor import DocumentPreprocessor
from backend.app.core.document.splitter import DocumentSplitter
from backend.app.core.quality.checker import DocumentChecker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ModelResult(NamedTuple):
    """Container for one model's preprocessing + check outcome."""

    name: str
    model: str
    chunks: list
    check_result: object  # CheckResult
    elapsed_preprocess: float  # seconds spent in preprocessing
    elapsed_check: float  # seconds spent in quality checking


def _process_chunks_sync(
    preprocessor: DocumentPreprocessor,
    chunks: list[Document],
) -> list[Document]:
    """Apply LLM processing to each chunk (synchronous)."""
    return [
        Document(
            page_content=preprocessor._llm_process(chunk.page_content),
            metadata=chunk.metadata,
        )
        for chunk in chunks
    ]


def _reduction(before: int, after: int) -> tuple[int, float]:
    """Calculate error reduction count and rate."""
    reduced = before - after
    rate = (reduced / before * 100) if before > 0 else 0.0
    return reduced, round(rate, 2)


# ------------------------------------------------------------------
# Async helpers
# ------------------------------------------------------------------


async def _process_model_async(
    name: str,
    model_name: str,
    preprocessor: DocumentPreprocessor,
    chunks: list[Document],
    checker: DocumentChecker,
    executor: ThreadPoolExecutor,
) -> ModelResult:
    """Run preprocessing + quality check for one model in a thread.

    The synchronous OpenAI API calls are offloaded to *executor* so that
    multiple models can proceed concurrently.
    """
    loop = asyncio.get_running_loop()

    logger.info("[%s] (%s) 开始预处理 (%d 块) ...", name, model_name, len(chunks))

    # --- Preprocessing (in thread) ---
    t0 = time.perf_counter()
    processed_chunks = await loop.run_in_executor(
        executor,
        _process_chunks_sync,
        preprocessor,
        chunks,
    )
    elapsed_preprocess = time.perf_counter() - t0
    logger.info(
        "[%s] 预处理完成, 耗时 %.2f 秒 (%.2f 秒/块)",
        name,
        elapsed_preprocess,
        elapsed_preprocess / max(len(chunks), 1),
    )

    # --- Quality check (in thread) ---
    t1 = time.perf_counter()
    check_result = await loop.run_in_executor(
        executor,
        checker.check_documents,
        processed_chunks,
    )
    elapsed_check = time.perf_counter() - t1
    logger.info("[%s] 质量检查完成, 耗时 %.2f 秒", name, elapsed_check)

    return ModelResult(
        name=name,
        model=model_name,
        chunks=processed_chunks,
        check_result=check_result,
        elapsed_preprocess=elapsed_preprocess,
        elapsed_check=elapsed_check,
    )


async def _evaluate_async(
    file_path: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> dict:
    """Core async evaluation logic."""

    # --- Read file ---
    if not os.path.exists(file_path):
        logger.error("文件不存在: %s", file_path)
        sys.exit(1)

    with open(file_path, "r", encoding="utf-8") as f:
        original_text = f.read()

    logger.info("已读取文件: %s (%d 字符)", file_path, len(original_text))

    # --- Init components ---
    settings = get_settings()

    splitter = DocumentSplitter(
        chunk_size=chunk_size or settings.CHUNK_SIZE,
        chunk_overlap=chunk_overlap or settings.CHUNK_OVERLAP,
    )

    preprocessor_plus = DocumentPreprocessor(
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.LLM_BASE_URL,
        model=settings.LLM_MODEL,
        enable_llm_preprocessing=True,
    )

    preprocessor_max = DocumentPreprocessor(
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.LLM_BASE_URL,
        model=settings.LLM_MODEL_MAX,
        enable_llm_preprocessing=True,
    )

    preprocessor_kimi = DocumentPreprocessor(
        api_key=settings.MOONSHOT_API_KEY,
        base_url=settings.MOONSHOT_BASE_URL,
        model=settings.MOONSHOT_MODEL,
        enable_llm_preprocessing=True,
    )

    checker = DocumentChecker(
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.LLM_BASE_URL,
        model=settings.LLM_MODEL,
        enable_llm_check=settings.ENABLE_LLM_QUALITY_CHECK,
    )

    # --- Split into chunks ---
    raw_doc = [Document(page_content=original_text)]
    chunks = splitter.split(raw_doc)
    logger.info(
        "文档已分块: 共 %d 个块 (chunk_size=%d, chunk_overlap=%d)",
        len(chunks),
        splitter.chunk_size,
        splitter.chunk_overlap,
    )

    # --- 1. Check original chunks (before any preprocessing) ---
    logger.info("=" * 60)
    logger.info("阶段 1/2: 检查原始文档 (%d 块)", len(chunks))
    t_orig = time.perf_counter()
    original_result = checker.check_documents(chunks)
    elapsed_orig = time.perf_counter() - t_orig
    logger.info("原始文档检查完成, 耗时 %.2f 秒", elapsed_orig)

    # --- 2. Run all three models in parallel ---
    logger.info("=" * 60)
    logger.info("阶段 2/2: 三个模型并行预处理 + 检查")

    # Each model gets its own thread so blocking I/O can overlap
    executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="model")

    total_start = time.perf_counter()

    plus_task = _process_model_async(
        "Qwen-Plus",
        settings.LLM_MODEL,
        preprocessor_plus,
        chunks,
        checker,
        executor,
    )
    max_task = _process_model_async(
        "Qwen3-Max",
        settings.LLM_MODEL_MAX,
        preprocessor_max,
        chunks,
        checker,
        executor,
    )
    kimi_task = _process_model_async(
        "Kimi-K2.5",
        settings.MOONSHOT_MODEL,
        preprocessor_kimi,
        chunks,
        checker,
        executor,
    )

    plus_res, max_res, kimi_res = await asyncio.gather(
        plus_task,
        max_task,
        kimi_task,
    )

    total_elapsed = time.perf_counter() - total_start
    executor.shutdown(wait=False)

    # --- Build report ---
    plus_reduced, plus_rate = _reduction(
        original_result.total_errors,
        plus_res.check_result.total_errors,
    )
    max_reduced, max_rate = _reduction(
        original_result.total_errors,
        max_res.check_result.total_errors,
    )
    kimi_reduced, kimi_rate = _reduction(
        original_result.total_errors,
        kimi_res.check_result.total_errors,
    )

    report = {
        "file": file_path,
        "original": {
            **original_result.to_dict(),
            "check_time": round(elapsed_orig, 2),
        },
        "qwen_plus": {
            "model": settings.LLM_MODEL,
            **plus_res.check_result.to_dict(),
            "errors_reduced": plus_reduced,
            "reduction_rate": plus_rate,
            "preprocess_time": round(plus_res.elapsed_preprocess, 2),
            "check_time": round(plus_res.elapsed_check, 2),
        },
        "qwen3_max": {
            "model": settings.LLM_MODEL_MAX,
            **max_res.check_result.to_dict(),
            "errors_reduced": max_reduced,
            "reduction_rate": max_rate,
            "preprocess_time": round(max_res.elapsed_preprocess, 2),
            "check_time": round(max_res.elapsed_check, 2),
        },
        "kimi_k2_5": {
            "model": settings.MOONSHOT_MODEL,
            **kimi_res.check_result.to_dict(),
            "errors_reduced": kimi_reduced,
            "reduction_rate": kimi_rate,
            "preprocess_time": round(kimi_res.elapsed_preprocess, 2),
            "check_time": round(kimi_res.elapsed_check, 2),
        },
        "parallel_total_time": round(total_elapsed, 2),
    }

    # --- Print summary ---
    logger.info("=" * 60)
    logger.info("评估结果汇总: %s", os.path.basename(file_path))
    logger.info("-" * 60)
    logger.info(
        "原始文档          : %d 个错误  %s  (检查耗时 %.2f 秒)",
        original_result.total_errors,
        dict(original_result.error_type_distribution),
        elapsed_orig,
    )
    logger.info(
        "Qwen-Plus  处理后 : %d 个错误  减少 %d 个 (%.1f%%)  "
        "预处理 %.2f 秒  检查 %.2f 秒",
        plus_res.check_result.total_errors,
        plus_reduced,
        plus_rate,
        plus_res.elapsed_preprocess,
        plus_res.elapsed_check,
    )
    logger.info(
        "Qwen3-Max  处理后 : %d 个错误  减少 %d 个 (%.1f%%)  "
        "预处理 %.2f 秒  检查 %.2f 秒",
        max_res.check_result.total_errors,
        max_reduced,
        max_rate,
        max_res.elapsed_preprocess,
        max_res.elapsed_check,
    )
    logger.info(
        "Kimi-K2.5  处理后 : %d 个错误  减少 %d 个 (%.1f%%)  "
        "预处理 %.2f 秒  检查 %.2f 秒",
        kimi_res.check_result.total_errors,
        kimi_reduced,
        kimi_rate,
        kimi_res.elapsed_preprocess,
        kimi_res.elapsed_check,
    )
    logger.info("-" * 60)
    logger.info(
        "三模型并行总耗时: %.2f 秒  (串行预估: %.2f 秒, 加速比: %.1fx)",
        total_elapsed,
        (
            plus_res.elapsed_preprocess
            + plus_res.elapsed_check
            + max_res.elapsed_preprocess
            + max_res.elapsed_check
            + kimi_res.elapsed_preprocess
            + kimi_res.elapsed_check
        ),
        (
            plus_res.elapsed_preprocess
            + plus_res.elapsed_check
            + max_res.elapsed_preprocess
            + max_res.elapsed_check
            + kimi_res.elapsed_preprocess
            + kimi_res.elapsed_check
        )
        / max(total_elapsed, 0.01),
    )
    logger.info("=" * 60)

    return report


# ------------------------------------------------------------------
# Public synchronous wrapper
# ------------------------------------------------------------------


def evaluate(file_path: str, chunk_size: int = None, chunk_overlap: int = None) -> dict:
    """Run evaluation pipeline on a single txt file.

    Reads the file, splits into chunks via DocumentSplitter,
    then compares Qwen-Plus vs Qwen3-Max vs Kimi-K2.5 preprocessing
    results **in parallel**.  Processing speed for each model is recorded.

    Args:
        file_path: Path to the txt file to evaluate.
        chunk_size: Override chunk size (default: from settings).
        chunk_overlap: Override chunk overlap (default: from settings).

    Returns a report dict with error counts and timing for each model.
    """
    return asyncio.run(_evaluate_async(file_path, chunk_size, chunk_overlap))


def main():
    parser = argparse.ArgumentParser(
        description="评估 Qwen-Plus、Qwen3-Max 和 Kimi-K2.5 预处理各自的错误消除效果 (并行)"
    )
    parser.add_argument("--file", required=True, help="待评估的 txt 文件路径")
    parser.add_argument(
        "--chunk-size", type=int, default=None, help="分块大小 (默认使用 settings 配置)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="分块重叠 (默认使用 settings 配置)",
    )
    args = parser.parse_args()
    evaluate(args.file, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)


if __name__ == "__main__":
    main()

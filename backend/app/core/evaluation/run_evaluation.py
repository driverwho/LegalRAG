"""CLI entry-point for V2 retrieval evaluation.

Usage examples::

    # Evaluate a single QA dataset
    python -m backend.app.core.evaluation.run_evaluation \\
        --dataset path/to/qa_dataset.json \\
        --collection-name law_collection

    # Evaluate multiple datasets with generation metrics
    python -m backend.app.core.evaluation.run_evaluation \\
        --dataset path/to/qa.json \\
        --dataset path/to/mcq.json \\
        --output-dir ./evaluation_results \\
        --enable-generation \\
        --k-values 1,3,5,10

    # Quick evaluation (skip generation, limit samples)
    python -m backend.app.core.evaluation.run_evaluation \\
        --dataset path/to/qa.json \\
        --max-samples 50 \\
        --no-generation
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

# Ensure project root is on sys.path
_project_root = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    )
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from backend.app.core.evaluation.datasets import EvalDatasetLoader
from backend.app.core.evaluation.pipeline_evaluator import PipelineEvaluator
from backend.app.core.evaluation.report import save_report_json, save_report_markdown

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _build_pipeline():
    """Wire the same components as async_deps.py without lru_cache."""
    from openai import AsyncOpenAI

    from backend.app.config.settings import get_settings
    from backend.app.core.llm.embedding import EmbeddingManager
    from backend.app.core.vector_store.chroma import ChromaVectorStore
    from backend.app.core.retriever.async_rag import AsyncRAGPipeline
    from backend.app.core.retriever.bm25 import HybridBM25Retriever
    from backend.app.core.preprocessor.query_preprocessor import QueryPreprocessor

    settings = get_settings()

    em = EmbeddingManager(
        embedding_model=settings.EMBEDDING_MODEL,
        dashscope_api_key=settings.DASHSCOPE_API_KEY,
    )
    vector_store = ChromaVectorStore(
        collection_name=settings.COLLECTION_NAME,
        embeddings=em.embeddings,
        persist_directory=settings.CHROMADB_PERSIST_DIR,
    )

    bm25 = HybridBM25Retriever(vector_store=vector_store)

    llm_client = AsyncOpenAI(
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.LLM_BASE_URL,
    )
    preprocessor = QueryPreprocessor(llm_client=llm_client, model=settings.LLM_MODEL)

    pipeline = AsyncRAGPipeline(
        vector_store=vector_store,
        bm25_retriever=bm25,
        reranker=None,
        preprocessor=preprocessor,
        similarity_threshold=settings.SIMILARITY_THRESHOLD,
        max_results=settings.MAX_RESULTS,
        vector_weight=settings.VECTOR_WEIGHT,
        bm25_weight=settings.BM25_WEIGHT,
    )

    return pipeline, settings


def _build_chat_manager(settings):
    """Optionally build a chat manager for generation evaluation."""
    try:
        from backend.app.core.llm.async_chat import AsyncContextualChatManager
        from backend.app.core.context import ContextManager, ContextConfig
        from backend.app.core.database.session_service import SessionService

        session_service = SessionService()
        ctx_config = ContextConfig.from_settings()
        ctx_manager = ContextManager(session_service, ctx_config)

        chat_manager = AsyncContextualChatManager(
            api_key=settings.DASHSCOPE_API_KEY,
            base_url=settings.LLM_BASE_URL,
            model=settings.LLM_MODEL,
            context_manager=ctx_manager,
        )
        return chat_manager
    except Exception as exc:
        logger.warning("Could not build chat manager: %s — generation metrics disabled", exc)
        return None


async def _run_evaluation(args) -> None:
    """Core async evaluation logic."""
    # ── Load datasets ─────────────────────────────────────────────────
    all_samples = []
    loader = EvalDatasetLoader()

    for dataset_path in args.dataset:
        if not os.path.exists(dataset_path):
            logger.error("Dataset file not found: %s", dataset_path)
            continue
        samples = loader.load_auto(dataset_path)
        if args.max_samples and len(samples) > args.max_samples:
            logger.info(
                "Limiting %s to %d samples (from %d)",
                dataset_path, args.max_samples, len(samples),
            )
            samples = samples[:args.max_samples]
        all_samples.extend(samples)

    if not all_samples:
        logger.error("No evaluation samples loaded — exiting")
        return

    logger.info("Total evaluation samples: %d", len(all_samples))

    # ── Build pipeline ────────────────────────────────────────────────
    pipeline, settings = _build_pipeline()

    chat_manager = None
    if args.enable_generation:
        chat_manager = _build_chat_manager(settings)

    # ── Parse k values ────────────────────────────────────────────────
    k_values = [int(x.strip()) for x in args.k_values.split(",")]

    # ── Run evaluation ────────────────────────────────────────────────
    evaluator = PipelineEvaluator(
        pipeline=pipeline,
        chat_manager=chat_manager,
        k_values=k_values,
        max_concurrency=args.concurrency,
        fetch_k=args.fetch_k,
    )

    # Checkpoint callback: save intermediate JSON
    def checkpoint_cb(completed, total, partial_results):
        logger.info("Checkpoint: %d/%d samples complete", completed, total)

    dataset_name = ", ".join(
        os.path.basename(p) for p in args.dataset
    )

    report = await evaluator.evaluate_dataset(
        samples=all_samples,
        dataset_name=dataset_name,
        collection_name=args.collection_name,
        enable_generation=args.enable_generation and chat_manager is not None,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_callback=checkpoint_cb,
    )

    # ── Save outputs ──────────────────────────────────────────────────
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "eval_report.json")
    save_report_json(report, json_path)

    md_path = os.path.join(output_dir, "eval_report.md")
    save_report_markdown(report, md_path, k_main=5 if 5 in k_values else k_values[-1])

    logger.info("Evaluation complete. Reports saved to %s", output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="V2 Retrieval Pipeline Evaluation — evaluate each pipeline "
        "stage independently on legal benchmark datasets.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Path to a dataset JSON file (can be specified multiple times). "
        "Format is auto-detected (QA or MCQ).",
    )
    parser.add_argument(
        "--collection-name",
        default=None,
        help="Restrict retrieval to this vector store collection. "
        "If omitted, searches all collections.",
    )
    parser.add_argument(
        "--output-dir",
        default="./evaluation_results",
        help="Directory to save evaluation reports (default: ./evaluation_results)",
    )
    parser.add_argument(
        "--k-values",
        default="1,3,5,10",
        help="Comma-separated K values for multi-cutoff metrics (default: 1,3,5,10)",
    )
    parser.add_argument(
        "--fetch-k",
        type=int,
        default=20,
        help="Number of candidates to fetch per retriever before fusion (default: 20)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit the number of samples evaluated per dataset (for quick tests)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Maximum number of samples evaluated concurrently (default: 5)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=20,
        help="Save intermediate results every N samples (default: 20)",
    )

    gen_group = parser.add_mutually_exclusive_group()
    gen_group.add_argument(
        "--enable-generation",
        action="store_true",
        default=False,
        help="Enable end-to-end LLM generation evaluation (requires API access)",
    )
    gen_group.add_argument(
        "--no-generation",
        action="store_true",
        default=False,
        help="Explicitly disable generation evaluation",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.no_generation:
        args.enable_generation = False

    asyncio.run(_run_evaluation(args))


if __name__ == "__main__":
    main()

"""V2 pipeline evaluator — runs each retrieval stage independently and
collects per-stage metrics so bottlenecks are immediately visible.

Usage (programmatic)::

    evaluator = PipelineEvaluator(pipeline, chat_manager)
    report = await evaluator.evaluate_dataset(samples, collection_name="law")

The evaluator reuses the **same** ``AsyncRAGPipeline`` instance that serves
production traffic, calling its internal stage methods directly (they are all
public on ``PipelineStagesMixin``).  Pipeline caching is disabled during
evaluation so measurements are realistic.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.app.core.evaluation.datasets import EvalSample
from backend.app.core.evaluation.metrics import (
    DEFAULT_K_VALUES,
    StageMetrics,
    GenerationMetrics,
    compute_stage_metrics,
    compute_generation_metrics,
)
from backend.app.core.retriever.async_rag import AsyncRAGPipeline
from backend.app.core.retriever.rag import RetrievalResult
from backend.app.core.retriever.fusion import rrf_fusion

logger = logging.getLogger(__name__)


# ── Per-sample result container ───────────────────────────────────────────────

@dataclass
class SampleResult:
    """Detailed evaluation result for a single sample."""

    question: str
    source_type: str
    ground_truth_docs: List[str]

    # Per-stage retrieved doc contents (for debugging / drill-down)
    stage_results: Dict[str, List[str]] = field(default_factory=dict)
    # Per-stage latencies in ms
    stage_latencies: Dict[str, float] = field(default_factory=dict)

    # Generation
    generated_answer: Optional[str] = None
    reference_answer: Optional[str] = None

    # Preprocessing info
    preprocess_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "question": self.question,
            "source_type": self.source_type,
            "ground_truth_count": len(self.ground_truth_docs),
            "stage_result_counts": {
                k: len(v) for k, v in self.stage_results.items()
            },
            "stage_latencies_ms": self.stage_latencies,
            "preprocess_info": self.preprocess_info,
            "generated_answer": self.generated_answer,
        }


# ── Evaluation report ─────────────────────────────────────────────────────────

@dataclass
class EvalReport:
    """Complete evaluation report for one dataset run."""

    dataset_name: str
    dataset_size: int
    timestamp: str

    stage_metrics: Dict[str, StageMetrics] = field(default_factory=dict)
    generation_metrics: Optional[GenerationMetrics] = None
    per_sample_results: List[SampleResult] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "dataset_name": self.dataset_name,
            "dataset_size": self.dataset_size,
            "timestamp": self.timestamp,
            "stage_metrics": {
                k: v.to_dict() for k, v in self.stage_metrics.items()
            },
            "generation_metrics": (
                self.generation_metrics.to_dict()
                if self.generation_metrics
                else None
            ),
            "per_sample_results": [s.to_dict() for s in self.per_sample_results],
            "config": self.config,
        }


# ── Helper: extract contents from RetrievalResult list ────────────────────────

def _contents(results: List[RetrievalResult]) -> List[str]:
    return [r.content for r in results]


# ── Core evaluator ────────────────────────────────────────────────────────────

class PipelineEvaluator:
    """Evaluate retrieval pipeline performance at each stage.

    Parameters
    ----------
    pipeline : AsyncRAGPipeline
        The pipeline instance to evaluate.  Must already be wired with
        vector_store, bm25_retriever, preprocessor, etc.
    chat_manager : optional
        An ``AsyncContextualChatManager`` for end-to-end generation evaluation.
        Pass ``None`` to skip generation metrics.
    k_values : list of int
        Cutoff values for Recall@K, Precision@K, etc.
    max_concurrency : int
        Maximum number of samples evaluated concurrently.
    fetch_k : int
        Number of candidates to fetch per retriever (before fusion/rerank).
    """

    # Stage name constants
    STAGE_VECTOR = "vector_only"
    STAGE_BM25 = "bm25_only"
    STAGE_FUSION = "rrf_fusion"
    STAGE_CONTEXT = "context_assembly"
    STAGE_RERANK = "rerank"
    STAGE_DEDUP = "deduplicate"

    STAGE_ORDER = [
        STAGE_VECTOR,
        STAGE_BM25,
        STAGE_FUSION,
        STAGE_CONTEXT,
        STAGE_RERANK,
        STAGE_DEDUP,
    ]

    def __init__(
        self,
        pipeline: AsyncRAGPipeline,
        chat_manager=None,
        k_values: List[int] = None,
        max_concurrency: int = 5,
        fetch_k: int = 20,
    ):
        self.pipeline = pipeline
        self.chat_manager = chat_manager
        self.k_values = k_values or DEFAULT_K_VALUES
        self.max_concurrency = max_concurrency
        self.fetch_k = fetch_k
        self._semaphore = asyncio.Semaphore(max_concurrency)

    # ── Evaluate a single sample through all stages ───────────────────────────

    async def _evaluate_sample(
        self,
        sample: EvalSample,
        collection_name: Optional[str] = None,
        enable_generation: bool = True,
    ) -> SampleResult:
        """Run one sample through the entire pipeline, recording per-stage results."""

        result = SampleResult(
            question=sample.question,
            source_type=sample.source_type,
            ground_truth_docs=sample.ground_truth_docs,
            reference_answer=sample.reference_answer,
        )

        async with self._semaphore:
            try:
                await self._run_stages(sample, result, collection_name)

                if enable_generation and self.chat_manager and sample.reference_answer:
                    await self._run_generation(sample, result)

            except Exception as exc:
                logger.error(
                    "Evaluation failed for question '%.50s': %s",
                    sample.question, exc,
                )

        return result

    async def _run_stages(
        self,
        sample: EvalSample,
        result: SampleResult,
        collection_name: Optional[str],
    ) -> None:
        """Execute pipeline stages 1-5 and record outputs."""

        query = sample.question

        # ── Stage 1: Preprocessing ────────────────────────────────────
        t0 = time.perf_counter()
        pre = await self.pipeline.preprocess_query(query, enable_correction=False)
        preprocess_ms = (time.perf_counter() - t0) * 1000

        result.preprocess_info = {
            "query_type": pre.query_type,
            "retrieval_type": pre.retrieval_type,
            "complexity": pre.complexity,
            "confidence": pre.confidence,
            "corrected_query": pre.corrected_query,
            "latency_ms": round(preprocess_ms, 1),
        }
        search_query = pre.corrected_query
        query_type = pre.retrieval_type

        # ── Stage 2a: Vector search only ──────────────────────────────
        t0 = time.perf_counter()
        vector_results = await self.pipeline._vector_search_async(
            search_query, self.fetch_k, collection_name, None,
        )
        vector_ms = (time.perf_counter() - t0) * 1000
        result.stage_results[self.STAGE_VECTOR] = _contents(vector_results)
        result.stage_latencies[self.STAGE_VECTOR] = round(vector_ms, 1)

        # ── Stage 2b: BM25 search only ───────────────────────────────
        bm25_results: List[RetrievalResult] = []
        bm25_ms = 0.0
        if self.pipeline.bm25_retriever is not None:
            t0 = time.perf_counter()
            bm25_results = await self.pipeline._bm25_search_async(
                search_query, self.fetch_k, collection_name, None,
            )
            bm25_ms = (time.perf_counter() - t0) * 1000
        result.stage_results[self.STAGE_BM25] = _contents(bm25_results)
        result.stage_latencies[self.STAGE_BM25] = round(bm25_ms, 1)

        # ── Stage 2c: RRF Fusion ──────────────────────────────────────
        t0 = time.perf_counter()
        fused = rrf_fusion(
            vector_results,
            bm25_results,
            self.fetch_k,
            vector_weight=self.pipeline.vector_weight,
            bm25_weight=self.pipeline.bm25_weight,
        )
        fusion_ms = (time.perf_counter() - t0) * 1000 + vector_ms + bm25_ms
        result.stage_results[self.STAGE_FUSION] = _contents(fused)
        result.stage_latencies[self.STAGE_FUSION] = round(fusion_ms, 1)

        # ── Stage 3: Context assembly ─────────────────────────────────
        t0 = time.perf_counter()
        assembled = await self.pipeline.assemble_context_async(fused, query_type)
        assembly_ms = (time.perf_counter() - t0) * 1000 + fusion_ms
        result.stage_results[self.STAGE_CONTEXT] = _contents(assembled)
        result.stage_latencies[self.STAGE_CONTEXT] = round(assembly_ms, 1)

        # ── Stage 4: Rerank ───────────────────────────────────────────
        if self.pipeline.reranker:
            t0 = time.perf_counter()
            reranked = await self.pipeline.rerank_async(
                assembled, search_query, top_k=max(self.k_values),
            )
            rerank_ms = (time.perf_counter() - t0) * 1000 + assembly_ms
            result.stage_results[self.STAGE_RERANK] = _contents(reranked)
            result.stage_latencies[self.STAGE_RERANK] = round(rerank_ms, 1)
            assembled = reranked
        else:
            # No reranker — mark as N/A
            result.stage_results[self.STAGE_RERANK] = []
            result.stage_latencies[self.STAGE_RERANK] = 0.0

        # ── Stage 5: Deduplication ────────────────────────────────────
        t0 = time.perf_counter()
        final = await self.pipeline.deduplicate_async(assembled)
        dedup_ms = (time.perf_counter() - t0) * 1000 + result.stage_latencies.get(
            self.STAGE_RERANK, 0.0
        ) or assembly_ms
        result.stage_results[self.STAGE_DEDUP] = _contents(final)
        result.stage_latencies[self.STAGE_DEDUP] = round(dedup_ms, 1)

    async def _run_generation(
        self,
        sample: EvalSample,
        result: SampleResult,
    ) -> None:
        """Generate an answer using the final retrieved documents."""
        final_docs = result.stage_results.get(self.STAGE_DEDUP, [])
        if not final_docs:
            final_docs = result.stage_results.get(self.STAGE_FUSION, [])

        contexts = [
            {"source": "evaluation", "text": doc}
            for doc in final_docs[:max(self.k_values)]
        ]

        try:
            answer = await self.chat_manager.generate_rag_response(
                question=sample.question,
                contexts=contexts,
            )
            result.generated_answer = answer
        except Exception as exc:
            logger.error("Generation failed for '%.50s': %s", sample.question, exc)
            result.generated_answer = f"[ERROR] {exc}"

    # ── Evaluate full dataset ─────────────────────────────────────────────────

    async def evaluate_dataset(
        self,
        samples: List[EvalSample],
        dataset_name: str = "unnamed",
        collection_name: Optional[str] = None,
        enable_generation: bool = True,
        checkpoint_interval: int = 20,
        checkpoint_callback=None,
    ) -> EvalReport:
        """Evaluate a full dataset and produce a report.

        Parameters
        ----------
        samples : list of EvalSample
            The evaluation samples to run.
        dataset_name : str
            Human-readable name for the report.
        collection_name : str, optional
            Restrict retrieval to this collection.
        enable_generation : bool
            Whether to run LLM generation for end-to-end metrics.
        checkpoint_interval : int
            Save intermediate results every N samples.
        checkpoint_callback : callable, optional
            ``callback(completed, total, partial_results)`` called at each
            checkpoint for progress reporting / persistence.
        """
        import datetime

        logger.info(
            "Starting V2 evaluation: dataset=%s, samples=%d, k_values=%s",
            dataset_name, len(samples), self.k_values,
        )

        all_results: List[SampleResult] = []

        for i, sample in enumerate(samples):
            if not sample.ground_truth_docs:
                logger.warning(
                    "Skipping sample %d (no ground truth): '%.50s'",
                    i, sample.question,
                )
                continue

            sr = await self._evaluate_sample(
                sample,
                collection_name=collection_name,
                enable_generation=enable_generation,
            )
            all_results.append(sr)

            # Progress logging
            if (i + 1) % 10 == 0 or i == len(samples) - 1:
                logger.info(
                    "Progress: %d/%d samples evaluated", i + 1, len(samples),
                )

            # Checkpoint
            if checkpoint_callback and (i + 1) % checkpoint_interval == 0:
                checkpoint_callback(i + 1, len(samples), all_results)

        # ── Aggregate stage metrics ───────────────────────────────────
        stage_metrics: Dict[str, StageMetrics] = {}

        for stage in self.STAGE_ORDER:
            all_retrieved = [sr.stage_results.get(stage, []) for sr in all_results]
            all_gt = [sr.ground_truth_docs for sr in all_results]
            all_latencies = [sr.stage_latencies.get(stage, 0.0) for sr in all_results]

            # Skip stages with no results (e.g. rerank when no reranker)
            if stage == self.STAGE_RERANK and not self.pipeline.reranker:
                stage_metrics[stage] = StageMetrics(
                    stage_name=self._stage_display_name(stage),
                    num_samples=0,
                )
                continue

            metrics = compute_stage_metrics(
                all_retrieved=all_retrieved,
                all_ground_truth=all_gt,
                all_latencies_ms=all_latencies,
                stage_name=self._stage_display_name(stage),
                k_values=self.k_values,
            )
            stage_metrics[stage] = metrics

        # ── Generation metrics ────────────────────────────────────────
        gen_metrics = None
        if enable_generation and self.chat_manager:
            gen_answers = [
                sr.generated_answer or ""
                for sr in all_results
                if sr.reference_answer
            ]
            ref_answers = [
                sr.reference_answer or ""
                for sr in all_results
                if sr.reference_answer
            ]
            gt_refs = [
                sr.ground_truth_docs
                for sr in all_results
                if sr.reference_answer
            ]
            if gen_answers:
                gen_metrics = compute_generation_metrics(
                    gen_answers, ref_answers, gt_refs,
                )

        # ── Build report ──────────────────────────────────────────────
        report = EvalReport(
            dataset_name=dataset_name,
            dataset_size=len(samples),
            timestamp=datetime.datetime.now().isoformat(),
            stage_metrics=stage_metrics,
            generation_metrics=gen_metrics,
            per_sample_results=all_results,
            config={
                "k_values": self.k_values,
                "fetch_k": self.fetch_k,
                "similarity_threshold": self.pipeline.similarity_threshold,
                "vector_weight": self.pipeline.vector_weight,
                "bm25_weight": self.pipeline.bm25_weight,
                "has_reranker": self.pipeline.reranker is not None,
                "has_bm25": self.pipeline.bm25_retriever is not None,
                "collection_name": collection_name,
            },
        )

        self._log_summary(report)
        return report

    # ── Display helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _stage_display_name(stage: str) -> str:
        return {
            "vector_only": "Vector Only",
            "bm25_only": "BM25 Only",
            "rrf_fusion": "RRF Fusion",
            "context_assembly": "Context Assembly",
            "rerank": "Rerank",
            "deduplicate": "Deduplicate",
        }.get(stage, stage)

    def _log_summary(self, report: EvalReport) -> None:
        """Print a formatted summary table to the logger."""
        sep = "=" * 80
        line = "-" * 80

        logger.info(sep)
        logger.info(
            "  V2 Retrieval Evaluation Report — %s (%d samples)",
            report.dataset_name, report.dataset_size,
        )
        logger.info(sep)

        # Header
        k_main = 5 if 5 in self.k_values else self.k_values[-1]
        header = (
            f"{'Stage':<20} | {'Recall@' + str(k_main):>10} | "
            f"{'Prec@' + str(k_main):>10} | {'MRR':>8} | "
            f"{'NDCG@' + str(k_main):>10} | {'Hit@' + str(k_main):>8} | "
            f"{'Latency':>10}"
        )
        logger.info(header)
        logger.info(line)

        for stage in self.STAGE_ORDER:
            m = report.stage_metrics.get(stage)
            if not m or m.num_samples == 0:
                logger.info(f"{self._stage_display_name(stage):<20} | {'N/A':^10} | {'N/A':^10} | {'N/A':^8} | {'N/A':^10} | {'N/A':^8} | {'N/A':^10}")
                continue

            logger.info(
                f"{m.stage_name:<20} | "
                f"{m.recall_at_k.get(k_main, 0):>10.4f} | "
                f"{m.precision_at_k.get(k_main, 0):>10.4f} | "
                f"{m.mrr:>8.4f} | "
                f"{m.ndcg_at_k.get(k_main, 0):>10.4f} | "
                f"{m.hit_rate_at_k.get(k_main, 0):>8.4f} | "
                f"{m.avg_latency_ms:>8.1f}ms"
            )

        logger.info(line)

        if report.generation_metrics:
            gm = report.generation_metrics
            logger.info(
                "Generation: ROUGE-L=%.4f  Jaccard=%.4f  Citation=%.4f  (n=%d)",
                gm.avg_rouge_l, gm.avg_jaccard, gm.avg_citation_accuracy,
                gm.num_samples,
            )

        logger.info(sep)

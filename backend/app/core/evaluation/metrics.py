"""Retrieval evaluation metrics for the V2 benchmark.

Provides functions to compute standard IR metrics given a ranked list of
retrieved documents and a set of ground-truth relevant documents.

All ``compute_*`` functions accept pre-tokenised / pre-normalised inputs so
the caller can plug in different relevance-matching strategies (substring,
embedding cosine, exact-id, …).
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import jieba

logger = logging.getLogger(__name__)

# ── Default K values for multi-cutoff metrics ─────────────────────────────────

DEFAULT_K_VALUES: List[int] = [1, 3, 5, 10]


# ── Relevance matching helpers ────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Lowercase + strip whitespace for comparison."""
    return text.strip().lower()


def is_relevant_substring(retrieved: str, ground_truth: str, min_overlap: float = 0.3) -> bool:
    """Check whether *retrieved* is relevant to *ground_truth* using
    bidirectional substring matching and token-level Jaccard overlap.

    The strategy is deliberately lenient for legal text because:
      - A retrieved chunk may contain the ground-truth article **plus** extra
        context (parent-chunk expansion), so simple ``gt in retrieved`` works.
      - Conversely the ground-truth string may be a short article number
        (e.g. "第二百九十四条 …") that appears inside a longer retrieved chunk.

    Falls back to Jaccard token overlap when neither direction matches,
    with a configurable ``min_overlap`` threshold (default 30 %).
    """
    r_norm = _normalise(retrieved)
    g_norm = _normalise(ground_truth)

    # Direct substring match (either direction)
    if g_norm in r_norm or r_norm in g_norm:
        return True

    # Token-level Jaccard as fallback
    r_tokens = set(jieba.lcut(r_norm))
    g_tokens = set(jieba.lcut(g_norm))
    if not g_tokens:
        return False
    overlap = len(r_tokens & g_tokens) / len(g_tokens)
    return overlap >= min_overlap


def relevance_vector(
    retrieved_docs: Sequence[str],
    ground_truth_docs: Sequence[str],
    match_fn=is_relevant_substring,
) -> List[int]:
    """Build a binary relevance vector for a ranked list.

    ``relevance_vector[i]`` is 1 if ``retrieved_docs[i]`` matches **any**
    ground-truth document (according to *match_fn*), else 0.
    """
    rel = []
    for doc in retrieved_docs:
        hit = any(match_fn(doc, gt) for gt in ground_truth_docs)
        rel.append(1 if hit else 0)
    return rel


# ── Per-query metrics ─────────────────────────────────────────────────────────

def recall_at_k(rel: List[int], total_relevant: int, k: int) -> float:
    """Recall@K — fraction of relevant docs found in top-K results."""
    if total_relevant == 0:
        return 0.0
    hits = sum(rel[:k])
    return hits / total_relevant


def precision_at_k(rel: List[int], k: int) -> float:
    """Precision@K — fraction of top-K results that are relevant."""
    if k == 0:
        return 0.0
    return sum(rel[:k]) / k


def f1_at_k(prec: float, rec: float) -> float:
    """F1@K — harmonic mean of precision and recall at K."""
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def mrr(rel: List[int]) -> float:
    """Mean Reciprocal Rank — 1 / rank of first relevant result."""
    for i, r in enumerate(rel, 1):
        if r:
            return 1.0 / i
    return 0.0


def ndcg_at_k(rel: List[int], k: int) -> float:
    """Normalised Discounted Cumulative Gain @ K.

    Uses binary relevance (0/1).  The ideal ordering places all 1s first.
    """
    if k == 0:
        return 0.0

    # DCG
    dcg = sum(rel[i] / math.log2(i + 2) for i in range(min(k, len(rel))))

    # Ideal DCG — all relevant docs first
    ideal_rel = sorted(rel[:k], reverse=True)
    idcg = sum(ideal_rel[i] / math.log2(i + 2) for i in range(len(ideal_rel)))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def hit_at_k(rel: List[int], k: int) -> int:
    """Hit@K — 1 if at least one relevant doc in top-K, else 0."""
    return 1 if any(rel[:k]) else 0


# ── Aggregated stage metrics ──────────────────────────────────────────────────

@dataclass
class StageMetrics:
    """Aggregated metrics for one pipeline stage across all evaluation samples."""

    stage_name: str
    num_samples: int = 0

    recall_at_k: Dict[int, float] = field(default_factory=dict)
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    f1_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    hit_rate_at_k: Dict[int, float] = field(default_factory=dict)
    avg_latency_ms: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "stage_name": self.stage_name,
            "num_samples": self.num_samples,
            "recall_at_k": self.recall_at_k,
            "precision_at_k": self.precision_at_k,
            "f1_at_k": self.f1_at_k,
            "mrr": round(self.mrr, 4),
            "ndcg_at_k": self.ndcg_at_k,
            "hit_rate_at_k": self.hit_rate_at_k,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
        }


def compute_stage_metrics(
    all_retrieved: List[List[str]],
    all_ground_truth: List[List[str]],
    all_latencies_ms: List[float],
    stage_name: str,
    k_values: List[int] = None,
) -> StageMetrics:
    """Compute aggregated metrics for a single pipeline stage.

    Parameters
    ----------
    all_retrieved : list of list of str
        For each sample, the ranked list of retrieved document contents.
    all_ground_truth : list of list of str
        For each sample, the ground-truth document contents.
    all_latencies_ms : list of float
        Per-sample latency in milliseconds.
    stage_name : str
        Human-readable stage label (e.g. ``"Vector Only"``).
    k_values : list of int
        Cutoff values for multi-cutoff metrics.

    Returns
    -------
    StageMetrics
    """
    if k_values is None:
        k_values = DEFAULT_K_VALUES

    n = len(all_retrieved)
    if n == 0:
        return StageMetrics(stage_name=stage_name)

    # Accumulators
    sum_recall: Dict[int, float] = {k: 0.0 for k in k_values}
    sum_precision: Dict[int, float] = {k: 0.0 for k in k_values}
    sum_ndcg: Dict[int, float] = {k: 0.0 for k in k_values}
    sum_hit: Dict[int, int] = {k: 0 for k in k_values}
    sum_mrr = 0.0

    for retrieved, gt in zip(all_retrieved, all_ground_truth):
        if not gt:
            # Skip samples without ground truth
            continue

        rel = relevance_vector(retrieved, gt)
        total_rel = len(gt)

        for k in k_values:
            r = recall_at_k(rel, total_rel, k)
            p = precision_at_k(rel, k)
            sum_recall[k] += r
            sum_precision[k] += p
            sum_ndcg[k] += ndcg_at_k(rel, k)
            sum_hit[k] += hit_at_k(rel, k)

        sum_mrr += mrr(rel)

    # Average
    metrics = StageMetrics(
        stage_name=stage_name,
        num_samples=n,
        recall_at_k={k: round(sum_recall[k] / n, 4) for k in k_values},
        precision_at_k={k: round(sum_precision[k] / n, 4) for k in k_values},
        f1_at_k={
            k: round(
                f1_at_k(sum_precision[k] / n, sum_recall[k] / n), 4
            )
            for k in k_values
        },
        mrr=sum_mrr / n,
        ndcg_at_k={k: round(sum_ndcg[k] / n, 4) for k in k_values},
        hit_rate_at_k={k: round(sum_hit[k] / n, 4) for k in k_values},
        avg_latency_ms=sum(all_latencies_ms) / n if all_latencies_ms else 0.0,
    )

    return metrics


# ── Generation quality metrics ────────────────────────────────────────────────

def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Token-level Jaccard similarity using jieba segmentation."""
    tokens_a = set(jieba.lcut(text_a.strip()))
    tokens_b = set(jieba.lcut(text_b.strip()))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def rouge_l(hypothesis: str, reference: str) -> float:
    """ROUGE-L F1 score (longest common subsequence based).

    Uses character-level LCS for Chinese text.
    """
    h = hypothesis.strip()
    r = reference.strip()
    if not h or not r:
        return 0.0

    # Character-level LCS
    m, n = len(h), len(r)
    # Optimised space: only keep two rows
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if h[i - 1] == r[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr

    lcs_len = prev[n]
    if lcs_len == 0:
        return 0.0

    precision = lcs_len / m
    recall = lcs_len / n
    return 2 * precision * recall / (precision + recall)


def law_citation_accuracy(generated: str, ground_truth_refs: List[str]) -> float:
    """Fraction of ground-truth law citations mentioned in the generated answer.

    Uses substring matching on normalised text.
    """
    if not ground_truth_refs:
        return 0.0

    gen_norm = _normalise(generated)
    hits = 0
    for ref in ground_truth_refs:
        ref_norm = _normalise(ref)
        # Check if the article number appears in the generated text
        if ref_norm in gen_norm:
            hits += 1
            continue
        # Partial: check article number portion (e.g. "第二百九十四条")
        import re as _re
        article_match = _re.search(r"第[零一二三四五六七八九十百千万\d]+条", ref_norm)
        if article_match and article_match.group() in gen_norm:
            hits += 1

    return hits / len(ground_truth_refs)


@dataclass
class GenerationMetrics:
    """Aggregated generation quality metrics."""

    num_samples: int = 0
    avg_jaccard: float = 0.0
    avg_rouge_l: float = 0.0
    avg_citation_accuracy: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "num_samples": self.num_samples,
            "avg_jaccard_similarity": round(self.avg_jaccard, 4),
            "avg_rouge_l": round(self.avg_rouge_l, 4),
            "avg_citation_accuracy": round(self.avg_citation_accuracy, 4),
        }


def compute_generation_metrics(
    generated_answers: List[str],
    reference_answers: List[str],
    ground_truth_refs: List[List[str]],
) -> GenerationMetrics:
    """Compute aggregated generation quality metrics.

    Parameters
    ----------
    generated_answers : list of str
        LLM-generated answers.
    reference_answers : list of str
        Gold-standard reference answers.
    ground_truth_refs : list of list of str
        Ground-truth law article references per sample (for citation accuracy).
    """
    n = len(generated_answers)
    if n == 0:
        return GenerationMetrics()

    sum_jaccard = 0.0
    sum_rouge = 0.0
    sum_citation = 0.0

    for gen, ref, gt_refs in zip(generated_answers, reference_answers, ground_truth_refs):
        if ref:
            sum_jaccard += jaccard_similarity(gen, ref)
            sum_rouge += rouge_l(gen, ref)
        sum_citation += law_citation_accuracy(gen, gt_refs)

    return GenerationMetrics(
        num_samples=n,
        avg_jaccard=sum_jaccard / n,
        avg_rouge_l=sum_rouge / n,
        avg_citation_accuracy=sum_citation / n,
    )

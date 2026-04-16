"""V2 Retrieval Pipeline Evaluation.

Public API::

    from backend.app.core.evaluation import (
        EvalDatasetLoader,
        EvalSample,
        PipelineEvaluator,
        EvalReport,
        save_report_json,
        save_report_markdown,
    )

CLI::

    python -m backend.app.core.evaluation.run_evaluation \\
        --dataset path/to/qa.json \\
        --dataset path/to/mcq.json \\
        --output-dir ./evaluation_results
"""

from backend.app.core.evaluation.datasets import EvalDatasetLoader, EvalSample
from backend.app.core.evaluation.metrics import (
    StageMetrics,
    GenerationMetrics,
    compute_stage_metrics,
    compute_generation_metrics,
)
from backend.app.core.evaluation.pipeline_evaluator import (
    PipelineEvaluator,
    EvalReport,
    SampleResult,
)
from backend.app.core.evaluation.report import (
    save_report_json,
    save_report_markdown,
    report_to_markdown,
)

__all__ = [
    "EvalDatasetLoader",
    "EvalSample",
    "PipelineEvaluator",
    "EvalReport",
    "SampleResult",
    "StageMetrics",
    "GenerationMetrics",
    "compute_stage_metrics",
    "compute_generation_metrics",
    "save_report_json",
    "save_report_markdown",
    "report_to_markdown",
]

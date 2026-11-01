"""Phase 8: unified end-to-end AI pipeline.

    from src.pipeline import AIPipeline, PipelineSession

    pipeline = AIPipeline()
    session = pipeline.new_session()
    r1 = pipeline.run_text("Bring me the bottle.", session=session)
    r2 = pipeline.run_text("From the kitchen.", session=session, mode="deterministic")
"""

from src.pipeline.orchestrator import AIPipeline
from src.pipeline.session import PipelineSession
from src.pipeline.types import PipelineResult, PipelineStatus, StageTimings

__all__ = [
    "AIPipeline",
    "PipelineResult",
    "PipelineSession",
    "PipelineStatus",
    "StageTimings",
]

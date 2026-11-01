"""Phase 7: constrained LLM-assisted high-level task planning.

    from src.dialogue import DialogueManager
    from src.llm_planning import LLMAssistedPlanner, FakeLLMProvider

    manager = DialogueManager()
    ready = manager.process("Take me to the kitchen.")
    provider = FakeLLMProvider([valid_json_plan])
    result = LLMAssistedPlanner(provider=provider).plan(ready)
    # result.plan is a Phase 6 TaskPlan after parse + grounding + validate_plan
"""

from src.llm_planning.planner import LLMAssistedPlanner
from src.llm_planning.provider import FakeLLMProvider, OpenAICompatibleProvider
from src.llm_planning.types import LLMPlanningResult, PlanningMode

__all__ = [
    "FakeLLMProvider",
    "LLMAssistedPlanner",
    "LLMPlanningResult",
    "OpenAICompatibleProvider",
    "PlanningMode",
]

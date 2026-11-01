"""Unit and integration tests for the Phase 6 TaskPlanner."""

from __future__ import annotations

import pytest

from src.dialogue.types import DialogueStatus
from src.planning.plan_validator import validate_plan
from src.planning.types import ActionType, PlannedAction, PlanningStatus, TaskPlan
from src.validation.types import ValidationStatus
from tests.planning.conftest import make_dialogue, make_ready


class TestValidIntents:
    def test_navigate(self, planner):
        plan = planner.plan(make_ready("NAVIGATE", {"destination": "kitchen"}))
        assert plan.status == PlanningStatus.PLANNED
        assert len(plan.actions) == 1
        assert plan.actions[0].type == ActionType.NAVIGATE_TO
        assert plan.actions[0].parameters == {"destination": "kitchen"}
        assert plan.actions[0].depends_on == []

    def test_navigate_to_user(self, planner):
        plan = planner.plan(make_ready("NAVIGATE", {"destination": "user"}))
        assert plan.status == PlanningStatus.PLANNED
        assert plan.actions[0].type == ActionType.NAVIGATE_TO_USER
        assert plan.actions[0].parameters == {}

    def test_return(self, planner):
        plan = planner.plan(
            make_ready("RETURN", {"destination": "charging_station"})
        )
        assert plan.status == PlanningStatus.PLANNED
        assert plan.actions[0].type == ActionType.NAVIGATE_TO
        assert plan.actions[0].parameters["destination"] == "charging_station"

    def test_follow(self, planner):
        plan = planner.plan(make_ready("FOLLOW", {"person": "user"}))
        assert plan.status == PlanningStatus.PLANNED
        assert plan.actions[0].type == ActionType.FOLLOW_PERSON
        assert plan.actions[0].parameters == {"person": "user"}

    def test_stop(self, planner):
        plan = planner.plan(make_ready("STOP", {}))
        assert plan.status == PlanningStatus.PLANNED
        assert plan.actions[0].type == ActionType.STOP

    def test_retrieve_to_user(self, planner):
        plan = planner.plan(
            make_ready(
                "RETRIEVE",
                {
                    "object": "bottle",
                    "source_location": "kitchen",
                    "destination": "user",
                },
            )
        )
        assert plan.status == PlanningStatus.PLANNED
        types = [action.type for action in plan.actions]
        assert types == [
            ActionType.NAVIGATE_TO,
            ActionType.SEARCH_OBJECT,
            ActionType.APPROACH_OBJECT,
            ActionType.GRASP_OBJECT,
            ActionType.NAVIGATE_TO_USER,
            ActionType.DELIVER_OBJECT,
        ]
        assert plan.actions[0].parameters == {"destination": "kitchen"}
        assert plan.actions[1].parameters == {
            "object": "bottle",
            "location": "kitchen",
        }
        assert plan.actions[5].parameters == {
            "object": "bottle",
            "destination": "user",
        }
        # Sequential dependencies
        for index, action in enumerate(plan.actions):
            if index == 0:
                assert action.depends_on == []
            else:
                assert action.depends_on == [plan.actions[index - 1].id]

    def test_retrieve_to_location(self, planner):
        plan = planner.plan(
            make_ready(
                "RETRIEVE",
                {
                    "object": "cup",
                    "source_location": "office",
                    "destination": "living_room",
                },
            )
        )
        assert plan.status == PlanningStatus.PLANNED
        assert plan.actions[4].type == ActionType.NAVIGATE_TO
        assert plan.actions[4].parameters == {"destination": "living_room"}
        assert plan.actions[5].parameters["destination"] == "living_room"

    def test_deliver_to_person(self, planner):
        plan = planner.plan(
            make_ready("DELIVER", {"object": "bottle", "person": "john"})
        )
        assert plan.status == PlanningStatus.PLANNED
        assert [a.type for a in plan.actions] == [
            ActionType.NAVIGATE_TO_PERSON,
            ActionType.DELIVER_OBJECT,
        ]
        assert any("already held" in w for w in plan.warnings)

    def test_deliver_to_destination(self, planner):
        plan = planner.plan(
            make_ready("DELIVER", {"object": "book", "destination": "kitchen_table"})
        )
        assert plan.status == PlanningStatus.PLANNED
        assert plan.actions[0].type == ActionType.NAVIGATE_TO
        assert plan.actions[0].parameters["destination"] == "kitchen_table"


class TestInvalidInputs:
    @pytest.mark.parametrize(
        "status",
        [
            DialogueStatus.CLARIFICATION_REQUIRED,
            DialogueStatus.CLARIFICATION_FAILED,
            DialogueStatus.CANCELLED,
            DialogueStatus.UNSUPPORTED,
        ],
    )
    def test_rejects_non_ready_dialogue(self, planner, status):
        plan = planner.plan(make_dialogue(status, "NAVIGATE", {"destination": "kitchen"}))
        assert plan.status == PlanningStatus.REJECTED
        assert plan.actions == []

    def test_rejects_ready_with_non_valid_validation(self, planner):
        result = make_ready(
            "NAVIGATE",
            {"destination": "kitchen"},
            validation_status=ValidationStatus.INCOMPLETE,
        )
        plan = planner.plan(result)
        assert plan.status == PlanningStatus.REJECTED

    def test_unsupported_intent(self, planner):
        plan = planner.plan_task("UNSUPPORTED", {})
        assert plan.status == PlanningStatus.UNSUPPORTED

    def test_missing_retrieve_source(self, planner):
        plan = planner.plan_task(
            "RETRIEVE",
            {"object": "bottle", "destination": "user"},
        )
        assert plan.status == PlanningStatus.FAILED
        assert any("source_location" in err for err in plan.errors)

    def test_deliver_missing_target(self, planner):
        plan = planner.plan_task("DELIVER", {"object": "bottle"})
        assert plan.status == PlanningStatus.FAILED


class TestDeterminism:
    def test_retrieve_repeatable(self, planner):
        entities = {
            "object": "bottle",
            "source_location": "kitchen",
            "destination": "user",
        }
        first = planner.plan_task("RETRIEVE", entities).to_dict()
        second = planner.plan_task("RETRIEVE", entities).to_dict()
        assert first == second

    def test_navigate_repeatable(self, planner):
        first = planner.plan(make_ready("NAVIGATE", {"destination": "office"})).to_dict()
        second = planner.plan(make_ready("NAVIGATE", {"destination": "office"})).to_dict()
        assert first == second


class TestPlanValidation:
    def test_valid_plan_passes(self, planner):
        plan = planner.plan(make_ready("FOLLOW", {"person": "mary"}))
        validate_plan(plan)  # does not raise

    def test_duplicate_ids(self):
        from src.planning.exceptions import PlanValidationError

        plan = TaskPlan(
            status=PlanningStatus.PLANNED,
            intent="STOP",
            entities={},
            actions=[
                PlannedAction(id="a1", type=ActionType.STOP, order=1),
                PlannedAction(id="a1", type=ActionType.STOP, order=2),
            ],
        )
        with pytest.raises(PlanValidationError):
            validate_plan(plan)

    def test_self_dependency(self):
        from src.planning.exceptions import PlanValidationError

        plan = TaskPlan(
            status=PlanningStatus.PLANNED,
            intent="STOP",
            entities={},
            actions=[
                PlannedAction(
                    id="a1",
                    type=ActionType.STOP,
                    order=1,
                    depends_on=["a1"],
                )
            ],
        )
        with pytest.raises(PlanValidationError):
            validate_plan(plan)

    def test_unknown_dependency(self):
        from src.planning.exceptions import PlanValidationError

        plan = TaskPlan(
            status=PlanningStatus.PLANNED,
            intent="STOP",
            entities={},
            actions=[
                PlannedAction(
                    id="a1",
                    type=ActionType.STOP,
                    order=1,
                    depends_on=["missing"],
                )
            ],
        )
        with pytest.raises(PlanValidationError):
            validate_plan(plan)

    def test_cycle(self):
        from src.planning.exceptions import PlanValidationError

        plan = TaskPlan(
            status=PlanningStatus.PLANNED,
            intent="FOLLOW",
            entities={"person": "john"},
            actions=[
                PlannedAction(
                    id="a1",
                    type=ActionType.FOLLOW_PERSON,
                    parameters={"person": "john"},
                    order=1,
                    depends_on=["a2"],
                ),
                PlannedAction(
                    id="a2",
                    type=ActionType.FOLLOW_PERSON,
                    parameters={"person": "john"},
                    order=2,
                    depends_on=["a1"],
                ),
            ],
        )
        with pytest.raises(PlanValidationError):
            validate_plan(plan)

    def test_missing_parameter(self):
        from src.planning.exceptions import PlanValidationError

        plan = TaskPlan(
            status=PlanningStatus.PLANNED,
            intent="NAVIGATE",
            entities={},
            actions=[
                PlannedAction(
                    id="a1",
                    type=ActionType.NAVIGATE_TO,
                    parameters={},
                    order=1,
                )
            ],
        )
        with pytest.raises(PlanValidationError):
            validate_plan(plan)

    def test_dependency_order_violation(self):
        from src.planning.exceptions import PlanValidationError

        plan = TaskPlan(
            status=PlanningStatus.PLANNED,
            intent="FOLLOW",
            entities={"person": "john"},
            actions=[
                PlannedAction(
                    id="a1",
                    type=ActionType.FOLLOW_PERSON,
                    parameters={"person": "john"},
                    order=1,
                    depends_on=["a2"],
                ),
                PlannedAction(
                    id="a2",
                    type=ActionType.FOLLOW_PERSON,
                    parameters={"person": "john"},
                    order=2,
                ),
            ],
        )
        with pytest.raises(PlanValidationError):
            validate_plan(plan)


class TestDialogueIntegration:
    def test_navigate_through_dialogue(self, manager, planner):
        dialogue = manager.process("Take me to the kitchen.")
        assert dialogue.status == DialogueStatus.READY
        plan = planner.plan(dialogue)
        assert plan.status == PlanningStatus.PLANNED
        assert plan.actions[0].type == ActionType.NAVIGATE_TO
        assert plan.actions[0].parameters["destination"] == "kitchen"

    def test_incomplete_not_planned(self, manager, planner):
        dialogue = manager.process("Bring me the bottle.")
        assert dialogue.status == DialogueStatus.CLARIFICATION_REQUIRED
        plan = planner.plan(dialogue)
        assert plan.status == PlanningStatus.REJECTED

    def test_retrieve_after_clarification(self, manager, planner):
        first = manager.process("Bring me the bottle.")
        assert first.status == DialogueStatus.CLARIFICATION_REQUIRED
        ready = manager.process("From the kitchen.")
        assert ready.status == DialogueStatus.READY
        plan = planner.plan(ready)
        assert plan.status == PlanningStatus.PLANNED
        assert plan.intent == "RETRIEVE"
        assert len(plan.actions) == 6

    def test_stop_through_dialogue(self, manager, planner):
        dialogue = manager.process("Stop.")
        assert dialogue.status == DialogueStatus.READY
        plan = planner.plan(dialogue)
        assert plan.status == PlanningStatus.PLANNED
        assert plan.actions[0].type == ActionType.STOP

    def test_to_dict_json_compatible(self, planner):
        plan = planner.plan(make_ready("NAVIGATE", {"destination": "hallway"}))
        payload = plan.to_dict()
        assert payload["status"] == "PLANNED"
        assert isinstance(payload["actions"][0]["type"], str)
        assert isinstance(payload["actions"][0]["parameters"], dict)

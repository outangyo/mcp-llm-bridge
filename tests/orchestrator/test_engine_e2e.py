from __future__ import annotations

import io
import pytest

from src.orchestrator.adapters.mock_builder import MockBuilder
from src.orchestrator.adapters.mock_reviewer import MockReviewer
from src.orchestrator.cli.renderer import CLIRenderer
from src.orchestrator.contracts.event import EventType
from src.orchestrator.contracts.report import ReviewerVerdict
from src.orchestrator.contracts.task import TaskBudget, TaskContract
from src.orchestrator.engine import OrchestratorEngine
from src.orchestrator.policy.execution_policy import ExecutionPolicy, PolicyMode
from src.orchestrator.state.states import WorkflowState


def create_task(max_iterations: int = 3, max_retries: int = 2) -> TaskContract:
    return TaskContract(
        title="Implement Feature X",
        description="Write and verify feature X implementation",
        allowed_paths=["src/feature_x.py", "tests/test_feature_x.py"],
        budget=TaskBudget(max_iterations=max_iterations, max_retries=max_retries),
    )


def test_engine_happy_path_1_iteration():
    """Scenario 1: Happy path with 1 iteration (PASS -> PO approval -> COMPLETED)."""
    task = create_task()
    builder = MockBuilder(agent_name="MockAGY")
    reviewer = MockReviewer(agent_name="MockGemini")
    policy = ExecutionPolicy(mode=PolicyMode.SUPERVISED, require_final_approval=True)

    stream = io.StringIO()
    renderer = CLIRenderer(stream=stream, use_icons=False)

    approval_called = False

    def po_approval(reason: str, ctx) -> bool:
        nonlocal approval_called
        approval_called = True
        assert "PO final approval required" in reason
        return True

    engine = OrchestratorEngine(
        task=task,
        builder=builder,
        reviewer=reviewer,
        policy=policy,
        renderer=renderer,
        approval_handler=po_approval,
    )

    summary = engine.run()

    assert engine.state_machine.is_completed is True
    assert summary.status == "COMPLETED"
    assert summary.iterations == 1
    assert summary.retries == 0
    assert summary.final_review_verdict == "PASS"
    assert approval_called is True
    assert len(summary.human_approvals) == 1
    assert summary.human_approvals[0]["granted"] is True

    # Verify event stream captured AGENT_PROGRESS
    events = [e.event_type for e in renderer.recorded_events]
    assert EventType.WORKFLOW_STARTED in events
    assert EventType.ITERATION_STARTED in events
    assert EventType.AGENT_PROGRESS in events
    assert EventType.REVIEW_PASSED in events
    assert EventType.APPROVAL_REQUESTED in events
    assert EventType.APPROVAL_GRANTED in events
    assert EventType.WORKFLOW_COMPLETED in events


def test_engine_multi_iteration_retry():
    """Scenario 2: Multi-iteration retry path (REJECT -> address feedback -> PASS -> COMPLETED)."""
    task = create_task(max_iterations=4, max_retries=3)
    builder = MockBuilder(agent_name="MockAGY")
    reviewer = MockReviewer(
        agent_name="MockGemini",
        scripted_verdicts=[ReviewerVerdict.REJECT, ReviewerVerdict.PASS],
    )
    policy = ExecutionPolicy(mode=PolicyMode.SUPERVISED, require_final_approval=True)

    engine = OrchestratorEngine(
        task=task,
        builder=builder,
        reviewer=reviewer,
        policy=policy,
        approval_handler=lambda reason, ctx: True,
    )

    summary = engine.run()

    assert engine.state_machine.is_completed is True
    assert summary.status == "COMPLETED"
    assert summary.iterations == 2
    assert summary.retries == 1
    assert builder.call_count == 2
    assert reviewer.call_count == 2
    assert summary.final_review_verdict == "PASS"


def test_engine_retry_budget_exhaustion():
    """Scenario 3: Retry budget exhaustion (retries exceed max_retries -> FAILED)."""
    # max_retries = 1, so 1st rejection brings retry_count=1 (within budget).
    # 2nd rejection brings retry_count=2 (> 1), exhausting budget.
    task = create_task(max_iterations=10, max_retries=1)
    builder = MockBuilder(agent_name="MockAGY")
    reviewer = MockReviewer(
        agent_name="MockGemini",
        scripted_verdicts=[ReviewerVerdict.REJECT, ReviewerVerdict.REJECT],
    )
    policy = ExecutionPolicy(mode=PolicyMode.SUPERVISED)

    engine = OrchestratorEngine(
        task=task,
        builder=builder,
        reviewer=reviewer,
        policy=policy,
    )

    summary = engine.run()

    assert engine.state_machine.is_failed is True
    assert summary.status == "FAILED"
    assert summary.retries == 2
    assert "Retry budget exhausted" in engine.state_machine.events[-1].message


def test_engine_iteration_budget_exhaustion():
    """Scenario 4: Iteration budget exhaustion (iterations reach max_iterations -> FAILED)."""
    # max_iterations = 2, max_retries = 10
    # Iteration 1 rejects. Iteration 2 rejects. At end of iteration 2, iteration_index == 2 >= max_iterations.
    task = create_task(max_iterations=2, max_retries=10)
    builder = MockBuilder(agent_name="MockAGY")
    reviewer = MockReviewer(
        agent_name="MockGemini",
        scripted_verdicts=[ReviewerVerdict.REJECT, ReviewerVerdict.REJECT],
    )
    policy = ExecutionPolicy(mode=PolicyMode.SUPERVISED)

    engine = OrchestratorEngine(
        task=task,
        builder=builder,
        reviewer=reviewer,
        policy=policy,
    )

    summary = engine.run()

    assert engine.state_machine.is_failed is True
    assert summary.status == "FAILED"
    assert summary.iterations == 2
    assert "budget exhausted" in engine.state_machine.events[-1].message.lower()


def test_engine_po_rejection_at_gate():
    """Scenario 5: PO rejects changes during approval gate -> STOPPED."""
    task = create_task()
    builder = MockBuilder(agent_name="MockAGY")
    reviewer = MockReviewer(agent_name="MockGemini")
    policy = ExecutionPolicy(mode=PolicyMode.SUPERVISED, require_final_approval=True)

    def po_rejects(reason: str, ctx) -> bool:
        return False

    engine = OrchestratorEngine(
        task=task,
        builder=builder,
        reviewer=reviewer,
        policy=policy,
        approval_handler=po_rejects,
    )

    summary = engine.run()

    assert engine.state_machine.is_stopped is True
    assert summary.status == "STOPPED"
    assert summary.human_approvals[0]["granted"] is False


def test_engine_builder_fatal_error():
    """Scenario 6: Fatal builder error -> FAILED."""
    task = create_task()
    builder = MockBuilder(fatal_error=RuntimeError("Subprocess failed unexpectedly"))
    reviewer = MockReviewer()

    engine = OrchestratorEngine(task=task, builder=builder, reviewer=reviewer)
    summary = engine.run()

    assert engine.state_machine.is_failed is True
    assert summary.status == "FAILED"
    assert "Subprocess failed unexpectedly" in engine.state_machine.events[-1].message


def test_engine_reviewer_fatal_error():
    """Scenario 7: Fatal reviewer error -> FAILED."""
    task = create_task()
    builder = MockBuilder()
    reviewer = MockReviewer(fatal_error=RuntimeError("Reviewer timeout"))

    engine = OrchestratorEngine(task=task, builder=builder, reviewer=reviewer)
    summary = engine.run()

    assert engine.state_machine.is_failed is True
    assert summary.status == "FAILED"
    assert "Reviewer timeout" in engine.state_machine.events[-1].message


def test_engine_autonomous_mode():
    """Scenario 8: AUTONOMOUS mode auto-completes without approval gate."""
    task = create_task()
    builder = MockBuilder()
    reviewer = MockReviewer()
    policy = ExecutionPolicy(mode=PolicyMode.AUTONOMOUS)

    engine = OrchestratorEngine(
        task=task,
        builder=builder,
        reviewer=reviewer,
        policy=policy,
        approval_handler=None,
    )

    summary = engine.run()

    assert engine.state_machine.is_completed is True
    assert summary.status == "COMPLETED"
    assert summary.iterations == 1
    # No human approvals solicited in autonomous mode
    assert len(summary.human_approvals) == 0

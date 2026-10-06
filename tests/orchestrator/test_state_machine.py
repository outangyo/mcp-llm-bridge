import pytest

from src.orchestrator.contracts.context import WorkflowContext
from src.orchestrator.contracts.event import EventType, WorkflowEvent
from src.orchestrator.contracts.report import (
    AgentReport,
    AgentRole,
    BuilderResult,
    ReviewerVerdict,
)
from src.orchestrator.contracts.task import TaskBudget, TaskContract
from src.orchestrator.state.machine import WorkflowStateMachine
from src.orchestrator.state.states import WorkflowState
from src.orchestrator.state.transitions import InvalidStateTransitionError


def create_test_context(max_iterations: int = 3, max_retries: int = 2) -> WorkflowContext:
    task = TaskContract(
        title="Test Task",
        description="Test Desc",
        budget=TaskBudget(max_iterations=max_iterations, max_retries=max_retries),
    )
    return WorkflowContext(task=task)


def test_state_machine_initial_state():
    ctx = create_test_context()
    sm = WorkflowStateMachine(ctx)

    assert sm.current_state == WorkflowState.PENDING
    assert sm.is_terminal is False
    assert sm.is_completed is False
    assert sm.is_failed is False
    assert sm.is_stopped is False
    assert len(sm.events) == 0


def test_invalid_state_transitions_raise_error():
    ctx = create_test_context()
    sm = WorkflowStateMachine(ctx)

    # Cannot transition directly from PENDING to COMPLETED
    with pytest.raises(InvalidStateTransitionError, match="Cannot transition from PENDING to COMPLETED"):
        sm.transition(WorkflowState.COMPLETED, EventType.WORKFLOW_COMPLETED)

    # Cannot transition from PENDING to BUILDER_EXECUTING
    with pytest.raises(InvalidStateTransitionError):
        sm.transition(WorkflowState.BUILDER_EXECUTING, EventType.AGENT_STARTED)


def test_terminal_states_reject_all_outgoing_transitions():
    ctx = create_test_context()
    sm = WorkflowStateMachine(ctx)
    sm.start()
    sm.fail("Fatal crash")

    assert sm.current_state == WorkflowState.FAILED
    assert sm.is_terminal is True

    # No transition allowed from FAILED
    with pytest.raises(InvalidStateTransitionError, match="Allowed destination states from FAILED are: None"):
        sm.transition(WorkflowState.RUNNING, EventType.WORKFLOW_STARTED)


def test_happy_path_with_human_approval():
    captured_events: list[WorkflowEvent] = []
    ctx = create_test_context()
    sm = WorkflowStateMachine(ctx, event_listener=captured_events.append)

    # 1. Start workflow
    sm.start()
    assert sm.current_state == WorkflowState.RUNNING

    # 2. Start iteration 1 -> BUILDER_EXECUTING
    sm.start_iteration()
    assert sm.current_state == WorkflowState.BUILDER_EXECUTING
    assert ctx.iteration_index == 1

    # 3. Emit progress event without changing state
    sm.record_progress(AgentRole.BUILDER, "AGY", "Refactoring code in progress...")
    assert sm.current_state == WorkflowState.BUILDER_EXECUTING

    # 4. Builder completes
    builder_report = AgentReport(
        agent_role=AgentRole.BUILDER,
        agent_name="AGY",
        summary="Built feature",
        findings="All clean",
        builder_result=BuilderResult.SUCCESS,
        next_step="Send to review",
    )
    sm.complete_builder(builder_report)
    assert sm.current_state == WorkflowState.BUILDER_COMPLETED

    # 5. Reviewer starts
    sm.start_review("Gemini")
    assert sm.current_state == WorkflowState.REVIEWING

    # 6. Reviewer PASS + requires human approval -> AWAITING_APPROVAL
    reviewer_report = AgentReport(
        agent_role=AgentRole.REVIEWER,
        agent_name="Gemini",
        summary="Review passed",
        findings="Code conforms to spec",
        reviewer_verdict=ReviewerVerdict.PASS,
        next_step="Sign off",
    )
    sm.evaluate_reviewer_verdict(reviewer_report, requires_human_approval=True)
    assert sm.current_state == WorkflowState.AWAITING_APPROVAL

    # 7. PO grants approval -> COMPLETED
    sm.grant_approval("Looks great, PO accepted")
    assert sm.current_state == WorkflowState.COMPLETED
    assert sm.is_completed is True
    assert sm.is_terminal is True

    # Verify event stream
    event_types = [e.event_type for e in captured_events]
    assert EventType.WORKFLOW_STARTED in event_types
    assert EventType.ITERATION_STARTED in event_types
    assert EventType.AGENT_PROGRESS in event_types
    assert EventType.AGENT_COMPLETED in event_types
    assert EventType.REVIEW_STARTED in event_types
    assert EventType.REVIEW_PASSED in event_types
    assert EventType.APPROVAL_REQUESTED in event_types
    assert EventType.APPROVAL_GRANTED in event_types
    assert EventType.WORKFLOW_COMPLETED in event_types


def test_happy_path_without_human_approval_auto_completes():
    ctx = create_test_context()
    sm = WorkflowStateMachine(ctx)

    sm.start()
    sm.start_iteration()
    sm.complete_builder(
        AgentReport(
            agent_role=AgentRole.BUILDER,
            agent_name="AGY",
            summary="Code ready",
            findings="Clean",
            builder_result=BuilderResult.SUCCESS,
            next_step="Review",
        )
    )
    sm.start_review("Gemini")

    # Evaluate PASS with requires_human_approval=False
    sm.evaluate_reviewer_verdict(
        AgentReport(
            agent_role=AgentRole.REVIEWER,
            agent_name="Gemini",
            summary="Review passed",
            findings="Passed",
            reviewer_verdict=ReviewerVerdict.PASS,
            next_step="Done",
        ),
        requires_human_approval=False,
    )

    assert sm.current_state == WorkflowState.COMPLETED
    assert sm.is_completed is True


def test_reviewer_rejection_auto_retries_until_budget_exhausted():
    ctx = create_test_context(max_iterations=2, max_retries=1)
    sm = WorkflowStateMachine(ctx)

    sm.start()
    sm.start_iteration()  # Iteration 1

    sm.complete_builder(
        AgentReport(
            agent_role=AgentRole.BUILDER,
            agent_name="AGY",
            summary="Attempt 1",
            findings="Findings 1",
            builder_result=BuilderResult.SUCCESS,
            next_step="Review",
        )
    )
    sm.start_review("Gemini")

    # Rejection 1 (auto-retry, no approval needed) -> launches Iteration 2
    sm.evaluate_reviewer_verdict(
        AgentReport(
            agent_role=AgentRole.REVIEWER,
            agent_name="Gemini",
            summary="Bug found",
            findings="Fails edge case",
            reviewer_verdict=ReviewerVerdict.REJECT,
            next_step="Retry",
        ),
        requires_human_approval=False,
    )

    assert sm.current_state == WorkflowState.BUILDER_EXECUTING
    assert ctx.iteration_index == 2
    assert ctx.retry_count == 1

    # Complete Builder on Iteration 2
    sm.complete_builder(
        AgentReport(
            agent_role=AgentRole.BUILDER,
            agent_name="AGY",
            summary="Attempt 2",
            findings="Findings 2",
            builder_result=BuilderResult.SUCCESS,
            next_step="Review",
        )
    )
    sm.start_review("Gemini")

    # Rejection 2 -> budget exhausted -> FAILED
    sm.evaluate_reviewer_verdict(
        AgentReport(
            agent_role=AgentRole.REVIEWER,
            agent_name="Gemini",
            summary="Still fails",
            findings="Persistent bug",
            reviewer_verdict=ReviewerVerdict.REJECT,
            next_step="Give up",
        ),
        requires_human_approval=False,
    )

    assert sm.current_state == WorkflowState.FAILED
    assert sm.is_failed is True
    assert sm.is_terminal is True


def test_user_stop_aborts_workflow():
    ctx = create_test_context()
    sm = WorkflowStateMachine(ctx)

    sm.start()
    sm.start_iteration()
    sm.stop("PO canceled the run")

    assert sm.current_state == WorkflowState.STOPPED
    assert sm.is_stopped is True
    assert sm.is_terminal is True


def test_po_rejection_in_awaiting_approval():
    ctx = create_test_context()
    sm = WorkflowStateMachine(ctx)

    sm.start()
    sm.start_iteration()
    sm.complete_builder(
        AgentReport(
            agent_role=AgentRole.BUILDER,
            agent_name="AGY",
            summary="Done",
            findings="Good",
            builder_result=BuilderResult.SUCCESS,
            next_step="Review",
        )
    )
    sm.start_review("Gemini")
    sm.evaluate_reviewer_verdict(
        AgentReport(
            agent_role=AgentRole.REVIEWER,
            agent_name="Gemini",
            summary="Pass",
            findings="Passed",
            reviewer_verdict=ReviewerVerdict.PASS,
            next_step="Approval",
        ),
        requires_human_approval=True,
    )
    assert sm.current_state == WorkflowState.AWAITING_APPROVAL

    # PO rejects non-fatally -> STOPPED
    sm.reject_approval(reason="PO not satisfied with approach", fatal=False)
    assert sm.current_state == WorkflowState.STOPPED
    assert sm.is_stopped is True

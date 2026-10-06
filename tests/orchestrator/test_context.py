from src.orchestrator.contracts.context import WorkflowContext
from src.orchestrator.contracts.report import (
    AgentReport,
    AgentRole,
    BuilderResult,
    ReviewerVerdict,
)
from src.orchestrator.contracts.task import TaskBudget, TaskContract


def create_sample_task(max_iterations: int = 3, max_retries: int = 2) -> TaskContract:
    return TaskContract(
        title="Sample Task",
        description="Sample Desc",
        budget=TaskBudget(max_iterations=max_iterations, max_retries=max_retries),
    )


def test_workflow_context_initialization():
    task = create_sample_task()
    context = WorkflowContext(task=task)

    assert context.run_id.startswith("run_")
    assert context.iteration_index == 0
    assert context.retry_count == 0
    assert context.history == []
    assert context.modified_files == []
    assert context.last_builder_report is None
    assert context.last_reviewer_report is None


def test_workflow_context_iteration_and_report_recording():
    task = create_sample_task()
    context = WorkflowContext(task=task)

    # Start iteration 1
    record = context.start_new_iteration()
    assert context.iteration_index == 1
    assert record.iteration_index == 1
    assert len(context.history) == 1

    # Record Builder report
    builder_rep = AgentReport(
        agent_role=AgentRole.BUILDER,
        agent_name="AGY",
        summary="Modified code",
        findings="Updated files",
        builder_result=BuilderResult.SUCCESS,
        next_step="Review",
        artifacts={"modified_files": ["src/a.py", "src/b.py"]},
    )
    context.record_builder_report(builder_rep)
    assert context.last_builder_report == builder_rep
    assert "src/a.py" in context.modified_files
    assert "src/b.py" in context.modified_files
    assert context.history[0].builder_report == builder_rep

    # Record Reviewer report (REJECT)
    reviewer_rep = AgentReport(
        agent_role=AgentRole.REVIEWER,
        agent_name="Gemini",
        summary="Rejected",
        findings="Bug found",
        reviewer_verdict=ReviewerVerdict.REJECT,
        next_step="Fix bug",
    )
    context.record_reviewer_report(reviewer_rep)
    assert context.last_reviewer_report == reviewer_rep
    assert context.retry_count == 1
    assert context.history[0].reviewer_report == reviewer_rep
    assert context.history[0].completed_at is not None


def test_workflow_context_budget_exhaustion():
    task = create_sample_task(max_iterations=2, max_retries=1)
    context = WorkflowContext(task=task)

    assert context.is_iteration_budget_exhausted() is False
    assert context.is_retry_budget_exhausted() is False

    # Iteration 1
    context.start_new_iteration()
    assert context.is_iteration_budget_exhausted() is False

    # Rejection 1 uses 1 allowed retry
    rep1 = AgentReport(
        agent_role=AgentRole.REVIEWER,
        agent_name="Gemini",
        summary="Reject 1",
        findings="Err",
        reviewer_verdict=ReviewerVerdict.REJECT,
        next_step="Fix",
    )
    context.record_reviewer_report(rep1)
    assert context.retry_count == 1
    assert context.is_retry_budget_exhausted() is False

    # Rejection 2 exceeds max_retries (1)
    rep2 = AgentReport(
        agent_role=AgentRole.REVIEWER,
        agent_name="Gemini",
        summary="Reject 2",
        findings="Err 2",
        reviewer_verdict=ReviewerVerdict.REJECT,
        next_step="Fix again",
    )
    context.record_reviewer_report(rep2)
    assert context.retry_count == 2
    assert context.is_retry_budget_exhausted() is True

    # Iteration 2
    context.start_new_iteration()
    assert context.iteration_index == 2
    assert context.is_iteration_budget_exhausted() is True


def test_workflow_context_deterministic_json_roundtrip():
    task = create_sample_task()
    context = WorkflowContext(task=task)
    context.start_new_iteration()

    json_str = context.to_json()
    reconstructed = WorkflowContext.from_json(json_str)

    assert reconstructed.run_id == context.run_id
    assert reconstructed.iteration_index == 1
    assert reconstructed.task.title == task.title
    assert len(reconstructed.history) == 1

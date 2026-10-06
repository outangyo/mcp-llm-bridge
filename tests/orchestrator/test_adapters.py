from __future__ import annotations

import pytest

from src.orchestrator.adapters.mock_builder import MockBuilder
from src.orchestrator.adapters.mock_reviewer import MockReviewer
from src.orchestrator.contracts.context import WorkflowContext
from src.orchestrator.contracts.report import (
    AgentReport,
    AgentRole,
    BuilderResult,
    ReviewerVerdict,
)
from src.orchestrator.contracts.task import TaskBudget, TaskContract


@pytest.fixture
def context() -> WorkflowContext:
    task = TaskContract(
        title="Sample Adapter Task",
        description="Verify adapter behavior",
        allowed_paths=["src/test.py"],
        budget=TaskBudget(max_iterations=3, max_retries=2),
    )
    ctx = WorkflowContext(task=task)
    ctx.start_new_iteration()
    return ctx


def test_mock_builder_default_turn(context: WorkflowContext):
    builder = MockBuilder(agent_name="TestAGY")
    progress_messages = []

    def on_progress(msg: str, payload=None):
        progress_messages.append(msg)

    report = builder.execute(context, progress_callback=on_progress)

    assert builder.call_count == 1
    assert len(progress_messages) == 3
    assert "Inspecting task requirements" in progress_messages[0]
    assert report.agent_role == AgentRole.BUILDER
    assert report.agent_name == "TestAGY"
    assert report.builder_result == BuilderResult.SUCCESS
    assert report.artifacts["modified_files"] == ["src/test.py"]


def test_mock_builder_feedback_incorporation(context: WorkflowContext):
    # Simulate a prior reviewer rejection
    rejection = AgentReport(
        agent_role=AgentRole.REVIEWER,
        agent_name="Reviewer",
        summary="Found syntax errors",
        findings="Variable undefined on line 42",
        reviewer_verdict=ReviewerVerdict.REJECT,
        blockers=["Syntax error"],
        next_step="Fix undefined variable",
    )
    context.record_reviewer_report(rejection)
    context.start_new_iteration()

    builder = MockBuilder(agent_name="TestAGY")
    report = builder.execute(context)

    assert "Refactored implementation" in report.summary
    assert "Variable undefined on line 42" in report.findings


def test_mock_builder_scripted_reports(context: WorkflowContext):
    scripted = [
        AgentReport(
            agent_role=AgentRole.BUILDER,
            agent_name="ScriptedAGY",
            summary="Custom Script 1",
            findings="Done",
            builder_result=BuilderResult.SUCCESS,
            next_step="Review",
        )
    ]
    builder = MockBuilder(scripted_reports=scripted)
    report = builder.execute(context)
    assert report.summary == "Custom Script 1"


def test_mock_builder_fatal_error(context: WorkflowContext):
    builder = MockBuilder(fatal_error=RuntimeError("Builder crash"))
    with pytest.raises(RuntimeError, match="Builder crash"):
        builder.execute(context)


def test_mock_reviewer_default_turn(context: WorkflowContext):
    reviewer = MockReviewer(agent_name="TestGemini")
    progress_messages = []

    def on_progress(msg: str, payload=None):
        progress_messages.append(msg)

    report = reviewer.execute(context, progress_callback=on_progress)

    assert reviewer.call_count == 1
    assert len(progress_messages) == 3
    assert report.agent_role == AgentRole.REVIEWER
    assert report.agent_name == "TestGemini"
    assert report.reviewer_verdict == ReviewerVerdict.PASS
    assert report.is_pass is True


def test_mock_reviewer_scripted_verdicts(context: WorkflowContext):
    reviewer = MockReviewer(
        agent_name="TestGemini",
        scripted_verdicts=[ReviewerVerdict.REJECT, ReviewerVerdict.PASS],
    )

    report1 = reviewer.execute(context)
    assert report1.reviewer_verdict == ReviewerVerdict.REJECT
    assert report1.is_pass is False

    report2 = reviewer.execute(context)
    assert report2.reviewer_verdict == ReviewerVerdict.PASS
    assert report2.is_pass is True


def test_mock_reviewer_fatal_error(context: WorkflowContext):
    reviewer = MockReviewer(fatal_error=ValueError("Reviewer parse failure"))
    with pytest.raises(ValueError, match="Reviewer parse failure"):
        reviewer.execute(context)

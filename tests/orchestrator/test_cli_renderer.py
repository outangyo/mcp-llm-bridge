from __future__ import annotations

import io
import pytest

from src.orchestrator.cli.renderer import CLIRenderer
from src.orchestrator.contracts.event import EventType, WorkflowEvent
from src.orchestrator.contracts.report import (
    AgentReport,
    AgentRole,
    BuilderResult,
    ReviewerVerdict,
)
from src.orchestrator.contracts.task import TaskBudget, TaskContract
from src.orchestrator.contracts.context import WorkflowContext
from src.orchestrator.reporting.summary import WorkflowSummary


def test_cli_renderer_formats_progress_event_without_internal_reasoning():
    stream = io.StringIO()
    renderer = CLIRenderer(stream=stream, use_icons=False)

    event = WorkflowEvent(
        event_type=EventType.AGENT_PROGRESS,
        run_id="run-1",
        iteration=1,
        agent_role=AgentRole.BUILDER,
        agent_name="MockAGY",
        message="Compiling sources and updating files",
    )

    renderer.handle_event(event)
    output = stream.getvalue()

    assert "[BUILDER:MockAGY]" in output
    assert "Progress: Compiling sources and updating files" in output
    # Ensure no chain of thought or raw reasoning leakage
    assert "chain_of_thought" not in output
    assert len(renderer.recorded_events) == 1


def test_cli_renderer_formats_governance_and_approval_events():
    stream = io.StringIO()
    renderer = CLIRenderer(stream=stream, use_icons=True)

    req_event = WorkflowEvent(
        event_type=EventType.APPROVAL_REQUESTED,
        run_id="run-1",
        iteration=1,
        message="Awaiting PO sign-off",
    )
    renderer.handle_event(req_event)

    grant_event = WorkflowEvent(
        event_type=EventType.APPROVAL_GRANTED,
        run_id="run-1",
        iteration=1,
        message="PO granted approval",
    )
    renderer.handle_event(grant_event)

    output = stream.getvalue()
    assert "[GOVERNANCE]" in output
    assert "Human Gate: Awaiting PO sign-off" in output
    assert "[PO]" in output
    assert "PO granted approval" in output


def test_workflow_summary_terminal_and_markdown():
    task = TaskContract(
        title="Sample Summary Task",
        description="Verify summary format",
        budget=TaskBudget(max_iterations=2, max_retries=1),
    )
    ctx = WorkflowContext(task=task)
    ctx.start_new_iteration()
    ctx.modified_files = ["src/main.py", "tests/test_main.py"]

    summary = WorkflowSummary.from_context(
        context=ctx,
        final_state="COMPLETED",
        policy_mode="SUPERVISED",
        builder_name="AGY",
        reviewer_name="Gemini",
        human_approvals=[{"type": "PO_APPROVAL", "granted": True, "notes": "LGTM"}],
    )

    term_output = summary.format_terminal()
    assert "WORKFLOW EXECUTION SUMMARY" in term_output
    assert "Sample Summary Task" in term_output
    assert "COMPLETED" in term_output
    assert "src/main.py" in term_output
    assert "LGTM" in term_output

    md_output = summary.format_markdown()
    assert "# Workflow Execution Summary: Sample Summary Task" in md_output
    assert "- **Final Status:** **COMPLETED**" in md_output
    assert "`src/main.py`" in md_output


def test_cli_renderer_render_summary_and_report():
    stream = io.StringIO()
    renderer = CLIRenderer(stream=stream, use_icons=False)

    task = TaskContract(title="Render Test", description="Test render", budget=TaskBudget())
    ctx = WorkflowContext(task=task)
    summary = WorkflowSummary.from_context(context=ctx, final_state="COMPLETED", policy_mode="SAFE")

    renderer.render_summary(summary)
    assert "WORKFLOW EXECUTION SUMMARY" in stream.getvalue()

    report = AgentReport(
        agent_role=AgentRole.BUILDER,
        agent_name="Builder1",
        summary="Done task",
        findings="No defects",
        builder_result=BuilderResult.SUCCESS,
        next_step="Review",
    )
    renderer.render_report(report)
    assert "[BUILDER:Builder1]" in stream.getvalue()

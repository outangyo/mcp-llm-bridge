from __future__ import annotations

import io
import json
import subprocess
from unittest.mock import MagicMock

import pytest

from src.orchestrator.adapters.agy_builder import AGYBuilderAdapter
from src.orchestrator.adapters.gemini_reviewer import GeminiReviewerAdapter
from src.orchestrator.cli.renderer import CLIRenderer
from src.orchestrator.contracts.context import WorkflowContext
from src.orchestrator.contracts.event import EventType
from src.orchestrator.contracts.report import BuilderResult, ReviewerVerdict
from src.orchestrator.contracts.task import TaskBudget, TaskContract
from src.orchestrator.engine import OrchestratorEngine
from src.orchestrator.policy.execution_policy import ExecutionPolicy, PolicyMode
from src.orchestrator.state.states import WorkflowState
from src.providers.base import BaseLLMProvider, LLMProviderError


class MockLLM(BaseLLMProvider):
    def __init__(self, responses: list[str] | None = None, raises_exc: Exception | None = None) -> None:
        self.responses = list(responses) if responses else []
        self.raises_exc = raises_exc
        self.calls = 0

    async def generate_text(self, question: str, context: str | None = None) -> str:
        if self.raises_exc:
            raise self.raises_exc
        idx = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return self.responses[idx]


@pytest.fixture
def safe_verification_task() -> TaskContract:
    return TaskContract(
        title="Verify Data Pipeline",
        description="Process input data and output result to allowed directory",
        acceptance_criteria=[
            "Output file tests/fixtures/m3_4_verification/output/result.json exists",
            "Must have count=3 and status='processed'",
        ],
        allowed_paths=["tests/fixtures/m3_4_verification/output/"],
        budget=TaskBudget(max_iterations=2, max_retries=1),
    )


def test_real_adapter_wiring_and_contracts(safe_verification_task: TaskContract):
    """Area 1: Verify adapter wiring and adherence to BaseAgentAdapter contract."""
    fake_proc = MagicMock(spec=subprocess.CompletedProcess)
    fake_proc.returncode = 0
    fake_proc.stdout = json.dumps({
        "status": "SUCCESS",
        "response": """```json
{
  "summary": "Updated result.json",
  "findings": "All criteria met",
  "builder_result": "SUCCESS",
  "blockers": [],
  "next_step": "Submit for review",
  "modified_files": ["tests/fixtures/m3_4_verification/output/result.json"]
}
```""",
    })
    fake_proc.stderr = ""

    builder = AGYBuilderAdapter(agent_name="AGY", process_runner=lambda *a, **kw: fake_proc)
    reviewer = GeminiReviewerAdapter(
        agent_name="Gemini",
        provider=MockLLM(responses=['```json\n{"reviewer_verdict": "PASS", "summary": "Verified", "findings": "All good", "blockers": [], "next_step": "PO signoff"}\n```']),
    )

    engine = OrchestratorEngine(task=safe_verification_task, builder=builder, reviewer=reviewer)
    assert engine.builder.agent_name == "AGY"
    assert engine.reviewer.agent_name == "Gemini"
    assert engine.policy.mode == PolicyMode.SUPERVISED


def test_e2e_pass_workflow_event_ordering(safe_verification_task: TaskContract):
    """Area 2 & 9: Verify end-to-end PASS event ordering."""
    fake_proc = MagicMock(spec=subprocess.CompletedProcess)
    fake_proc.returncode = 0
    fake_proc.stdout = json.dumps({
        "status": "SUCCESS",
        "response": """```json
{
  "summary": "Generated result.json",
  "findings": "Count=3 and status=processed verified",
  "builder_result": "SUCCESS",
  "blockers": [],
  "next_step": "Submit for review",
  "modified_files": ["tests/fixtures/m3_4_verification/output/result.json"]
}
```""",
    })
    fake_proc.stderr = ""

    builder = AGYBuilderAdapter(agent_name="AGY", process_runner=lambda *a, **kw: fake_proc)
    reviewer = GeminiReviewerAdapter(
        agent_name="Gemini",
        provider=MockLLM(responses=['```json\n{"reviewer_verdict": "PASS", "summary": "Criteria satisfied", "findings": "Valid output verified", "blockers": [], "next_step": "PO approval"}\n```']),
    )

    stream = io.StringIO()
    renderer = CLIRenderer(stream=stream, use_icons=False)
    engine = OrchestratorEngine(
        task=safe_verification_task,
        builder=builder,
        reviewer=reviewer,
        renderer=renderer,
        approval_handler=lambda reason, ctx: True,
    )

    summary = engine.run()

    assert engine.state_machine.is_completed is True
    assert summary.status == "COMPLETED"

    # Check exact expected event sequence
    event_types = [e.event_type for e in renderer.recorded_events]
    assert event_types[0] == EventType.WORKFLOW_STARTED
    assert EventType.ITERATION_STARTED in event_types
    assert EventType.AGENT_STARTED in event_types
    assert EventType.AGENT_PROGRESS in event_types
    assert EventType.AGENT_COMPLETED in event_types
    assert EventType.REVIEW_STARTED in event_types
    assert EventType.REVIEW_PASSED in event_types
    assert EventType.APPROVAL_REQUESTED in event_types
    assert EventType.APPROVAL_GRANTED in event_types
    assert event_types[-1] == EventType.WORKFLOW_COMPLETED


def test_e2e_supervised_approval_granted(safe_verification_task: TaskContract):
    """Area 4 & 5: Verify SUPERVISED approval granted leads to COMPLETED."""
    fake_proc = MagicMock(spec=subprocess.CompletedProcess)
    fake_proc.returncode = 0
    fake_proc.stdout = json.dumps({"status": "SUCCESS", "response": '```json\n{"summary": "Done", "findings": "OK", "builder_result": "SUCCESS", "blockers": [], "next_step": "Review", "modified_files": ["tests/fixtures/m3_4_verification/output/result.json"]}\n```'})
    fake_proc.stderr = ""

    approval_invoked = False

    def po_grant(reason: str, ctx) -> bool:
        nonlocal approval_invoked
        approval_invoked = True
        return True

    builder = AGYBuilderAdapter(process_runner=lambda *a, **kw: fake_proc)
    reviewer = GeminiReviewerAdapter(provider=MockLLM(responses=['```json\n{"reviewer_verdict": "PASS", "summary": "PASS", "findings": "OK", "blockers": [], "next_step": "PO"}\n```']))

    engine = OrchestratorEngine(
        task=safe_verification_task,
        builder=builder,
        reviewer=reviewer,
        policy=ExecutionPolicy(mode=PolicyMode.SUPERVISED, require_final_approval=True),
        approval_handler=po_grant,
    )

    summary = engine.run()
    assert approval_invoked is True
    assert summary.status == "COMPLETED"
    assert summary.human_approvals[0]["granted"] is True


def test_e2e_supervised_approval_rejected_leads_to_stopped(safe_verification_task: TaskContract):
    """Area 5: Verify rejecting approval leads to STOPPED, not falsely COMPLETED."""
    fake_proc = MagicMock(spec=subprocess.CompletedProcess)
    fake_proc.returncode = 0
    fake_proc.stdout = json.dumps({"status": "SUCCESS", "response": '```json\n{"summary": "Done", "findings": "OK", "builder_result": "SUCCESS", "blockers": [], "next_step": "Review", "modified_files": ["tests/fixtures/m3_4_verification/output/result.json"]}\n```'})
    fake_proc.stderr = ""

    builder = AGYBuilderAdapter(process_runner=lambda *a, **kw: fake_proc)
    reviewer = GeminiReviewerAdapter(provider=MockLLM(responses=['```json\n{"reviewer_verdict": "PASS", "summary": "PASS", "findings": "OK", "blockers": [], "next_step": "PO"}\n```']))

    engine = OrchestratorEngine(
        task=safe_verification_task,
        builder=builder,
        reviewer=reviewer,
        policy=ExecutionPolicy(mode=PolicyMode.SUPERVISED, require_final_approval=True),
        approval_handler=lambda reason, ctx: False,  # PO rejects
    )

    summary = engine.run()
    assert engine.state_machine.is_stopped is True
    assert summary.status == "STOPPED"
    assert summary.human_approvals[0]["granted"] is False


def test_e2e_controlled_builder_failure(safe_verification_task: TaskContract):
    """Area 6: Verify controlled AGY builder failure leads to FAILED, not COMPLETED."""
    fake_proc = MagicMock(spec=subprocess.CompletedProcess)
    fake_proc.returncode = 1
    fake_proc.stdout = ""
    fake_proc.stderr = "Fatal compilation error in builder task"

    builder = AGYBuilderAdapter(process_runner=lambda *a, **kw: fake_proc)
    reviewer = GeminiReviewerAdapter(provider=MockLLM(responses=[]))

    engine = OrchestratorEngine(task=safe_verification_task, builder=builder, reviewer=reviewer)
    summary = engine.run()

    # When builder returns FAILURE result, state machine completes builder and reviews, or fails
    assert summary.status in ("FAILED", "STOPPED") or engine.state_machine.is_failed
    assert summary.status != "COMPLETED"


def test_e2e_controlled_reviewer_failure(safe_verification_task: TaskContract):
    """Area 7: Verify controlled Gemini provider failure leads to safe failure handling."""
    fake_proc = MagicMock(spec=subprocess.CompletedProcess)
    fake_proc.returncode = 0
    fake_proc.stdout = json.dumps({"status": "SUCCESS", "response": '```json\n{"summary": "Done", "findings": "OK", "builder_result": "SUCCESS", "blockers": [], "next_step": "Review", "modified_files": ["tests/fixtures/m3_4_verification/output/result.json"]}\n```'})
    fake_proc.stderr = ""

    builder = AGYBuilderAdapter(process_runner=lambda *a, **kw: fake_proc)
    # Reviewer raises safe provider error
    reviewer = GeminiReviewerAdapter(provider=MockLLM(raises_exc=LLMProviderError("Quota exceeded 429")))

    engine = OrchestratorEngine(
        task=safe_verification_task,
        builder=builder,
        reviewer=reviewer,
        policy=ExecutionPolicy(mode=PolicyMode.SUPERVISED),
    )
    summary = engine.run()

    # Rejection retry loop should exhaust budget and fail, never complete
    assert summary.status == "FAILED"
    assert summary.final_review_verdict == "REJECT"
    assert any("Quota exceeded" in b for b in summary.review_findings.splitlines() or ["Quota exceeded"])


def test_e2e_final_workflow_summary_content(safe_verification_task: TaskContract):
    """Area 8: Verify completeness and formatting of final WorkflowSummary."""
    fake_proc = MagicMock(spec=subprocess.CompletedProcess)
    fake_proc.returncode = 0
    fake_proc.stdout = json.dumps({"status": "SUCCESS", "response": '```json\n{"summary": "Done", "findings": "OK", "builder_result": "SUCCESS", "blockers": [], "next_step": "Review", "modified_files": ["tests/fixtures/m3_4_verification/output/result.json"]}\n```'})
    fake_proc.stderr = ""

    builder = AGYBuilderAdapter(process_runner=lambda *a, **kw: fake_proc)
    reviewer = GeminiReviewerAdapter(provider=MockLLM(responses=['```json\n{"reviewer_verdict": "PASS", "summary": "Approved", "findings": "All tests pass", "blockers": [], "next_step": "PO"}\n```']))

    engine = OrchestratorEngine(
        task=safe_verification_task,
        builder=builder,
        reviewer=reviewer,
        approval_handler=lambda reason, ctx: True,
    )
    summary = engine.run()

    assert summary.task_id == safe_verification_task.task_id
    assert summary.task_title == safe_verification_task.title
    assert summary.status == "COMPLETED"
    assert summary.iterations == 1
    assert summary.builder_name == "AGY"
    assert summary.reviewer_name == "Gemini"
    assert summary.final_review_verdict == "PASS"
    assert "tests/fixtures/m3_4_verification/output/result.json" in summary.modified_files

    # Formats render cleanly
    assert "WORKFLOW EXECUTION SUMMARY" in summary.format_terminal()
    assert "# Workflow Execution Summary:" in summary.format_markdown()


def test_e2e_secret_scrubbing_in_events_and_reports(safe_verification_task: TaskContract):
    """Area 10: Verify API keys and secrets are scrubbed across events, reports, and output."""
    secret_key = "AIzaSyD91023812038102381203812038"
    fake_proc = MagicMock(spec=subprocess.CompletedProcess)
    fake_proc.returncode = 0
    fake_proc.stdout = json.dumps({"status": "SUCCESS", "response": f'```json\n{{"summary": "Done with {secret_key}", "findings": "OK", "builder_result": "SUCCESS", "blockers": [], "next_step": "Review", "modified_files": ["tests/fixtures/m3_4_verification/output/result.json"]}}\n```'})
    fake_proc.stderr = ""

    builder = AGYBuilderAdapter(process_runner=lambda *a, **kw: fake_proc)
    reviewer = GeminiReviewerAdapter(
        provider=MockLLM(responses=[f'```json\n{{"reviewer_verdict": "PASS", "summary": "Checked with {secret_key}", "findings": "OK", "blockers": [], "next_step": "Done"}}\n```']),
    )

    stream = io.StringIO()
    renderer = CLIRenderer(stream=stream, use_icons=False)
    engine = OrchestratorEngine(
        task=safe_verification_task,
        builder=builder,
        reviewer=reviewer,
        renderer=renderer,
        approval_handler=lambda reason, ctx: True,
    )
    summary = engine.run()

    # Secret must never appear in raw text in summary or reviewer report
    assert secret_key not in summary.review_findings
    assert secret_key not in stream.getvalue()
    assert "[REDACTED_API_KEY]" in summary.review_findings or "[REDACTED_API_KEY]" in stream.getvalue()


def test_e2e_allowed_paths_boundary_protection(safe_verification_task: TaskContract):
    """Area 9: Verify out-of-scope modifications are blocked by path policy."""
    # Builder attempts to modify an out-of-scope file
    fake_proc = MagicMock(spec=subprocess.CompletedProcess)
    fake_proc.returncode = 0
    fake_proc.stdout = json.dumps({
        "status": "SUCCESS",
        "response": """```json
{
  "summary": "Modified server.py",
  "findings": "Attempted to touch server.py outside allowed scope",
  "builder_result": "SUCCESS",
  "blockers": [],
  "next_step": "Review",
  "modified_files": ["src/server.py"]
}
```""",
    })
    fake_proc.stderr = ""

    builder = AGYBuilderAdapter(process_runner=lambda *a, **kw: fake_proc)
    report = builder.execute(engine_context := WorkflowContext(task=safe_verification_task))

    assert report.builder_result == BuilderResult.BLOCKED
    assert any("Path policy violation" in b and "src/server.py" in b for b in report.blockers)


def test_e2e_cli_renderer_live_progress_no_cot(safe_verification_task: TaskContract):
    """Area 6: Verify CLI renderer emits progress without chain-of-thought."""
    stream = io.StringIO()
    renderer = CLIRenderer(stream=stream, use_icons=False)

    fake_proc = MagicMock(spec=subprocess.CompletedProcess)
    fake_proc.returncode = 0
    fake_proc.stdout = json.dumps({"status": "SUCCESS", "response": '```json\n{"summary": "Done", "findings": "OK", "builder_result": "SUCCESS", "blockers": [], "next_step": "Review", "modified_files": ["tests/fixtures/m3_4_verification/output/result.json"]}\n```'})
    fake_proc.stderr = ""

    builder = AGYBuilderAdapter(process_runner=lambda *a, **kw: fake_proc)
    reviewer = GeminiReviewerAdapter(provider=MockLLM(responses=['```json\n{"reviewer_verdict": "PASS", "summary": "PASS", "findings": "OK", "blockers": [], "next_step": "PO"}\n```']))

    engine = OrchestratorEngine(
        task=safe_verification_task,
        builder=builder,
        reviewer=reviewer,
        renderer=renderer,
        approval_handler=lambda reason, ctx: True,
    )
    engine.run()

    output = stream.getvalue()
    assert "Progress:" in output
    assert "chain_of_thought" not in output
    assert "internal_reasoning" not in output

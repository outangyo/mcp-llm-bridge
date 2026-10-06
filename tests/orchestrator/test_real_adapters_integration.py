from __future__ import annotations

import io
import json
import subprocess
from unittest.mock import MagicMock

import pytest

from src.orchestrator.adapters.agy_builder import AGYBuilderAdapter
from src.orchestrator.adapters.gemini_reviewer import GeminiReviewerAdapter
from src.orchestrator.cli.renderer import CLIRenderer
from src.orchestrator.contracts.task import TaskBudget, TaskContract
from src.orchestrator.engine import OrchestratorEngine
from src.orchestrator.policy.execution_policy import ExecutionPolicy, PolicyMode
from src.providers.base import BaseLLMProvider


class FakeLLMProvider(BaseLLMProvider):
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = 0

    async def generate_text(self, question: str, context: str | None = None) -> str:
        idx = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return self.responses[idx]


def test_orchestrator_integration_with_real_adapters():
    """End-to-end integration test of OrchestratorEngine with AGYBuilderAdapter and GeminiReviewerAdapter."""
    task = TaskContract(
        title="Implement Feature Auth",
        description="Add authentication module",
        acceptance_criteria=["Module exports authenticate() function", "Zero vulnerabilities"],
        allowed_paths=["src/auth.py", "tests/test_auth.py"],
        budget=TaskBudget(max_iterations=3, max_retries=2),
    )

    # Simulated AGY CLI output
    agy_response = json.dumps({
        "status": "SUCCESS",
        "response": """```json
{
  "summary": "Implemented authentication function in src/auth.py",
  "findings": "Created secure hash verification and export",
  "builder_result": "SUCCESS",
  "blockers": [],
  "next_step": "Submit for Reviewer evaluation",
  "modified_files": ["src/auth.py", "tests/test_auth.py"]
}
```""",
    })
    fake_proc = MagicMock(spec=subprocess.CompletedProcess)
    fake_proc.returncode = 0
    fake_proc.stdout = agy_response
    fake_proc.stderr = ""

    builder = AGYBuilderAdapter(
        agent_name="AGY",
        process_runner=lambda *args, **kwargs: fake_proc,
    )

    # Simulated Gemini response
    gemini_response = """```json
{
  "reviewer_verdict": "PASS",
  "summary": "Authentication implementation verified",
  "findings": "Clean code structure, all criteria met",
  "blockers": [],
  "next_step": "Ready for PO sign-off"
}
```"""
    provider = FakeLLMProvider(responses=[gemini_response])
    reviewer = GeminiReviewerAdapter(
        agent_name="Gemini",
        provider=provider,
    )

    stream = io.StringIO()
    renderer = CLIRenderer(stream=stream, use_icons=False)
    policy = ExecutionPolicy(mode=PolicyMode.SUPERVISED, require_final_approval=True)

    engine = OrchestratorEngine(
        task=task,
        builder=builder,
        reviewer=reviewer,
        policy=policy,
        renderer=renderer,
        approval_handler=lambda reason, ctx: True,
    )

    summary = engine.run()

    # Assertions
    assert engine.state_machine.is_completed is True
    assert summary.status == "COMPLETED"
    assert summary.iterations == 1
    assert summary.retries == 0
    assert summary.final_review_verdict == "PASS"
    assert summary.builder_name == "AGY"
    assert summary.reviewer_name == "Gemini"
    assert "src/auth.py" in summary.modified_files
    assert len(summary.human_approvals) == 1
    assert summary.human_approvals[0]["granted"] is True

    # Check CLI output rendered progress without raw chain of thought
    terminal_out = stream.getvalue()
    assert "WORKFLOW EXECUTION SUMMARY" in terminal_out
    assert "Implement Feature Auth" in terminal_out
    assert "COMPLETED" in terminal_out
    assert "Progress:" in terminal_out
    assert "chain_of_thought" not in terminal_out

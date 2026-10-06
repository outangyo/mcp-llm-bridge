from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from src.orchestrator.adapters.agy_builder import (
    AGYBuilderAdapter,
    find_agy_binary,
)
from src.orchestrator.contracts.context import WorkflowContext
from src.orchestrator.contracts.report import (
    AgentReport,
    AgentRole,
    BuilderResult,
    ReviewerVerdict,
)
from src.orchestrator.contracts.task import TaskBudget, TaskContract


@pytest.fixture
def sample_task() -> TaskContract:
    return TaskContract(
        title="Implement Feature Y",
        description="Write feature Y code in src/y.py",
        acceptance_criteria=["Must export Y", "Must have unit test"],
        allowed_paths=["src/y.py", "tests/test_y.py"],
        budget=TaskBudget(max_iterations=3, max_retries=2),
    )


@pytest.fixture
def workflow_context(sample_task: TaskContract) -> WorkflowContext:
    ctx = WorkflowContext(task=sample_task)
    ctx.start_new_iteration()
    return ctx


def test_find_agy_binary_env_override(monkeypatch):
    monkeypatch.setenv("AGY_BIN_PATH", "C:\\custom\\path\\agy.exe")
    with patch("os.path.isfile", return_value=True):
        bin_path = find_agy_binary()
        assert bin_path == "C:\\custom\\path\\agy.exe"


def test_find_agy_binary_path_lookup(monkeypatch):
    monkeypatch.delenv("AGY_BIN_PATH", raising=False)
    with patch("shutil.which", return_value="C:\\system\\agy.exe"):
        bin_path = find_agy_binary()
        assert bin_path == "C:\\system\\agy.exe"


def test_agy_builder_prompt_construction(workflow_context: WorkflowContext):
    adapter = AGYBuilderAdapter(agent_name="AGY")
    prompt = adapter.build_prompt(workflow_context)

    assert "Implement Feature Y" in prompt
    assert "src/y.py, tests/test_y.py" in prompt
    assert "Must export Y" in prompt
    assert "builder_result" in prompt


def test_agy_builder_prompt_with_prior_reviewer_feedback(workflow_context: WorkflowContext):
    # Simulate prior reviewer rejection
    rejection = AgentReport(
        agent_role=AgentRole.REVIEWER,
        agent_name="Gemini",
        summary="Type mismatch in Y",
        findings="Expected int but got str",
        reviewer_verdict=ReviewerVerdict.REJECT,
        blockers=["Type mismatch"],
        next_step="Fix type annotation",
    )
    workflow_context.record_reviewer_report(rejection)
    workflow_context.start_new_iteration()

    adapter = AGYBuilderAdapter()
    prompt = adapter.build_prompt(workflow_context)

    assert "PRIOR REVIEWER FEEDBACK" in prompt
    assert "Type mismatch in Y" in prompt
    assert "Expected int but got str" in prompt


def test_agy_builder_successful_execution_json_codeblock(workflow_context: WorkflowContext):
    raw_agy_json = json.dumps({
        "status": "SUCCESS",
        "response": """Here is the work completed:
```json
{
  "summary": "Implemented Feature Y successfully",
  "findings": "Created src/y.py with proper exports",
  "builder_result": "SUCCESS",
  "blockers": [],
  "next_step": "Submit for Gemini review",
  "modified_files": ["src/y.py", "tests/test_y.py"]
}
```
All criteria satisfied!""",
    })

    fake_proc = MagicMock(spec=subprocess.CompletedProcess)
    fake_proc.returncode = 0
    fake_proc.stdout = raw_agy_json
    fake_proc.stderr = ""

    adapter = AGYBuilderAdapter(process_runner=lambda *args, **kwargs: fake_proc)
    report = adapter.execute(workflow_context)

    assert report.agent_role == AgentRole.BUILDER
    assert report.builder_result == BuilderResult.SUCCESS
    assert report.summary == "Implemented Feature Y successfully"
    assert report.artifacts["modified_files"] == ["src/y.py", "tests/test_y.py"]
    assert report.blockers == []


def test_agy_builder_fallback_raw_response(workflow_context: WorkflowContext):
    raw_agy_json = json.dumps({
        "status": "SUCCESS",
        "response": "I made the code changes and all tests passed without issues.",
    })

    fake_proc = MagicMock(spec=subprocess.CompletedProcess)
    fake_proc.returncode = 0
    fake_proc.stdout = raw_agy_json
    fake_proc.stderr = ""

    adapter = AGYBuilderAdapter(process_runner=lambda *args, **kwargs: fake_proc)
    report = adapter.execute(workflow_context)

    assert report.builder_result == BuilderResult.SUCCESS
    assert "I made the code changes" in report.summary


def test_agy_builder_nonzero_exit_code(workflow_context: WorkflowContext):
    fake_proc = MagicMock(spec=subprocess.CompletedProcess)
    fake_proc.returncode = 2
    fake_proc.stdout = ""
    fake_proc.stderr = "Error: Invalid argument passed to agy"

    adapter = AGYBuilderAdapter(process_runner=lambda *args, **kwargs: fake_proc)
    report = adapter.execute(workflow_context)

    assert report.builder_result == BuilderResult.FAILURE
    assert "exited with code 2" in report.summary
    assert "Invalid argument passed to agy" in report.findings
    assert "ExitCode: 2" in report.blockers[0]


def test_agy_builder_timeout_handling(workflow_context: WorkflowContext):
    def timeout_runner(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["agy"], timeout=30)

    adapter = AGYBuilderAdapter(timeout_seconds=30, process_runner=timeout_runner)
    report = adapter.execute(workflow_context)

    assert report.builder_result == BuilderResult.FAILURE
    assert "timed out after 30 seconds" in report.summary
    assert "TimeoutExpired: 30s" in report.blockers[0]


def test_agy_builder_binary_not_found(workflow_context: WorkflowContext):
    def missing_runner(*args, **kwargs):
        raise FileNotFoundError("System cannot find the file specified")

    adapter = AGYBuilderAdapter(agy_bin="non_existent_agy", process_runner=missing_runner)
    report = adapter.execute(workflow_context)

    assert report.builder_result == BuilderResult.FAILURE
    assert "not found" in report.summary
    assert "FileNotFoundError" in report.blockers[0]


def test_agy_builder_unexpected_exception(workflow_context: WorkflowContext):
    def error_runner(*args, **kwargs):
        raise OSError("Permission denied to launch process")

    adapter = AGYBuilderAdapter(process_runner=error_runner)
    report = adapter.execute(workflow_context)

    assert report.builder_result == BuilderResult.FAILURE
    assert "Failed to execute AGY CLI subprocess" in report.summary
    assert "SubprocessError" in report.blockers[0]


def test_agy_builder_path_policy_enforcement(workflow_context: WorkflowContext):
    # Allowed paths: ["src/y.py", "tests/test_y.py"]
    # Attempting to modify outside path "secret/config.env"
    raw_agy_json = json.dumps({
        "status": "SUCCESS",
        "response": """```json
{
  "summary": "Modified files including configuration",
  "findings": "Altered secret/config.env and src/y.py",
  "builder_result": "SUCCESS",
  "blockers": [],
  "next_step": "Review",
  "modified_files": ["src/y.py", "secret/config.env"]
}
```""",
    })

    fake_proc = MagicMock(spec=subprocess.CompletedProcess)
    fake_proc.returncode = 0
    fake_proc.stdout = raw_agy_json
    fake_proc.stderr = ""

    adapter = AGYBuilderAdapter(process_runner=lambda *args, **kwargs: fake_proc)
    report = adapter.execute(workflow_context)

    # Path policy violation must change builder_result to BLOCKED
    assert report.builder_result == BuilderResult.BLOCKED
    assert any("Path policy violation" in b and "secret/config.env" in b for b in report.blockers)


def test_agy_builder_progress_reporting(workflow_context: WorkflowContext):
    progress_updates = []

    def on_progress(msg: str, payload=None):
        progress_updates.append(msg)

    fake_proc = MagicMock(spec=subprocess.CompletedProcess)
    fake_proc.returncode = 0
    fake_proc.stdout = json.dumps({"status": "SUCCESS", "response": "ok"})
    fake_proc.stderr = ""

    adapter = AGYBuilderAdapter(process_runner=lambda *args, **kwargs: fake_proc)
    adapter.execute(workflow_context, progress_callback=on_progress)

    assert len(progress_updates) >= 3
    assert any("Preparing AGY" in m for m in progress_updates)
    assert any("Launching AGY CLI" in m for m in progress_updates)
    assert any("Parsing AGY response" in m for m in progress_updates)

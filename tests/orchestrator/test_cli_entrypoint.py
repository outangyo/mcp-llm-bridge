from __future__ import annotations

import json
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from src.orchestrator.adapters.mock_builder import MockBuilder
from src.orchestrator.adapters.mock_reviewer import MockReviewer
from src.orchestrator.cli.main import (
    EXIT_FAILED,
    EXIT_INVALID_INPUT,
    EXIT_STOPPED,
    EXIT_SUCCESS,
    build_parser,
    main,
    parse_task_input,
)
from src.orchestrator.contracts.context import WorkflowContext
from src.orchestrator.contracts.task import TaskBudget, TaskContract
from src.orchestrator.policy.execution_policy import ExecutionPolicy, PolicyMode
from src.orchestrator.reporting.summary import WorkflowSummary


def _create_dummy_summary(status: str = "COMPLETED") -> WorkflowSummary:
    task = TaskContract(title="Test Task", description="Testing summary")
    ctx = WorkflowContext(task=task)
    return WorkflowSummary.from_context(
        context=ctx,
        final_state=status,
        policy_mode="SUPERVISED",
    )



class FakeEngine:
    """Fake OrchestratorEngine for testing CLI entrypoint wiring."""

    def __init__(
        self,
        task: TaskContract,
        builder: Any,
        reviewer: Any,
        policy: Optional[ExecutionPolicy] = None,
        renderer: Any = None,
        approval_handler: Optional[Callable[[str, WorkflowContext], bool]] = None,
        summary_status: str = "COMPLETED",
        raise_exc: Optional[Exception] = None,
    ) -> None:
        self.task = task
        self.builder = builder
        self.reviewer = reviewer
        self.policy = policy
        self.renderer = renderer
        self.approval_handler = approval_handler
        self.summary_status = summary_status
        self.raise_exc = raise_exc

    def run(self) -> WorkflowSummary:
        if self.raise_exc is not None:
            raise self.raise_exc
        return _create_dummy_summary(status=self.summary_status)


# ---------------------------------------------------------------------------
# Argument Parser & Task Parsing Tests
# ---------------------------------------------------------------------------

def test_build_parser_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["run", "task.json"])
    assert args.subcommand == "run"
    assert args.task_input == "task.json"
    assert args.policy == "SUPERVISED"
    assert args.max_iterations is None
    assert args.max_retries is None
    assert not args.mock
    assert not args.auto_approve
    assert not args.no_icons



def test_parse_task_input_from_valid_file(tmp_path: Any) -> None:
    task_file = tmp_path / "task.json"
    task_data = {
        "title": "File-based Task",
        "description": "Task from temporary file",
        "acceptance_criteria": ["Criterion 1"],
        "allowed_paths": ["src/"],
    }
    task_file.write_text(json.dumps(task_data), encoding="utf-8")

    contract = parse_task_input(str(task_file))
    assert isinstance(contract, TaskContract)
    assert contract.title == "File-based Task"
    assert contract.description == "Task from temporary file"
    assert contract.acceptance_criteria == ["Criterion 1"]


def test_parse_task_input_from_inline_json() -> None:
    raw_json = json.dumps({
        "title": "Inline Task",
        "description": "Task directly from JSON string",
        "acceptance_criteria": ["Criteria 1"],
    })
    contract = parse_task_input(raw_json)
    assert isinstance(contract, TaskContract)
    assert contract.title == "Inline Task"
    assert contract.description == "Task directly from JSON string"
    assert contract.acceptance_criteria == ["Criteria 1"]


def test_parse_task_input_invalid_json() -> None:
    with pytest.raises(ValueError, match="Input is neither a valid file path nor a valid TaskContract"):
        parse_task_input("{not: valid json}")


def test_parse_task_input_missing_required_fields() -> None:
    with pytest.raises(ValueError):
        parse_task_input('{"title": "Only Title"}')


# ---------------------------------------------------------------------------
# CLI Invocation and Exit Code Tests
# ---------------------------------------------------------------------------

def test_cli_empty_args(capsys: pytest.CaptureFixture[str]) -> None:
    code = main([])
    assert code == EXIT_INVALID_INPUT
    captured = capsys.readouterr()
    assert "usage:" in captured.err.lower() or "orchestrator" in captured.err.lower()


def test_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--help"])
    assert code == EXIT_SUCCESS
    captured = capsys.readouterr()
    assert "orchestrator" in captured.out.lower()


def test_cli_run_help(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["run", "--help"])
    assert code == EXIT_SUCCESS
    captured = capsys.readouterr()
    assert "orchestrator run" in captured.out.lower() or "help" in captured.out.lower()


def test_cli_unknown_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["unknown_cmd"])
    assert code == EXIT_INVALID_INPUT


def test_cli_missing_task_definition(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["run"])
    assert code == EXIT_INVALID_INPUT
    captured = capsys.readouterr()
    assert "error" in captured.err.lower()


def test_cli_partial_flags_missing_description(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["run", "--title", "Task Without Description"])
    assert code == EXIT_INVALID_INPUT
    captured = capsys.readouterr()
    assert "error" in captured.err.lower()


def test_cli_run_from_file_success(tmp_path: Any) -> None:
    task_file = tmp_path / "valid_task.json"
    task_file.write_text(
        json.dumps({
            "title": "File Task",
            "description": "Run from file test",
            "acceptance_criteria": ["Criteria 1"],
        }),
        encoding="utf-8",
    )

    captured_engine: Dict[str, Any] = {}

    def factory(**kwargs: Any) -> FakeEngine:
        captured_engine.update(kwargs)
        return FakeEngine(**kwargs, summary_status="COMPLETED")

    code = main(["run", str(task_file)], engine_factory=factory)
    assert code == EXIT_SUCCESS
    assert captured_engine["task"].title == "File Task"
    assert captured_engine["policy"].mode == PolicyMode.SUPERVISED


def test_cli_run_from_inline_json_success() -> None:
    inline_json = json.dumps({
        "title": "Inline Task",
        "description": "Run from inline",
        "acceptance_criteria": ["Criteria 1"],
    })

    captured_engine: Dict[str, Any] = {}

    def factory(**kwargs: Any) -> FakeEngine:
        captured_engine.update(kwargs)
        return FakeEngine(**kwargs, summary_status="COMPLETED")

    code = main(["run", inline_json], engine_factory=factory)
    assert code == EXIT_SUCCESS
    assert captured_engine["task"].title == "Inline Task"



def test_cli_run_from_flags_success() -> None:
    captured_engine: Dict[str, Any] = {}

    def factory(**kwargs: Any) -> FakeEngine:
        captured_engine.update(kwargs)
        return FakeEngine(**kwargs, summary_status="COMPLETED")

    argv = [
        "run",
        "--title", "CLI Flag Task",
        "--description", "Built via command line arguments",
        "--criteria", "Test passes",
        "--criteria", "Code clean",
        "--allowed-paths", "src/",
        "--max-iterations", "5",
        "--max-retries", "4",
        "--policy", "SAFE",
        "--mock",
    ]
    code = main(argv, engine_factory=factory)
    assert code == EXIT_SUCCESS

    task: TaskContract = captured_engine["task"]
    assert task.title == "CLI Flag Task"
    assert task.description == "Built via command line arguments"
    assert task.acceptance_criteria == ["Test passes", "Code clean"]
    assert task.allowed_paths == ["src/"]
    assert task.budget.max_iterations == 5
    assert task.budget.max_retries == 4

    assert captured_engine["policy"].mode == PolicyMode.SAFE
    assert isinstance(captured_engine["builder"], MockBuilder)
    assert isinstance(captured_engine["reviewer"], MockReviewer)


def test_cli_workflow_failure_exit_code() -> None:
    def factory(**kwargs: Any) -> FakeEngine:
        return FakeEngine(**kwargs, summary_status="FAILED")

    argv = ["run", "--title", "Failing Task", "--description", "Desc", "--criteria", "Test criteria"]
    code = main(argv, engine_factory=factory)
    assert code == EXIT_FAILED


def test_cli_workflow_stopped_exit_code() -> None:
    def factory(**kwargs: Any) -> FakeEngine:
        return FakeEngine(**kwargs, summary_status="STOPPED")

    argv = ["run", "--title", "Stopped Task", "--description", "Desc", "--criteria", "Test criteria"]
    code = main(argv, engine_factory=factory)
    assert code == EXIT_STOPPED


def test_cli_workflow_runtime_exception(capsys: pytest.CaptureFixture[str]) -> None:
    def factory(**kwargs: Any) -> FakeEngine:
        return FakeEngine(**kwargs, raise_exc=RuntimeError("Engine crashed unexpectedly"))

    argv = ["run", "--title", "Crash Task", "--description", "Desc", "--criteria", "Test criteria"]
    code = main(argv, engine_factory=factory)
    assert code == EXIT_FAILED
    captured = capsys.readouterr()
    assert "workflow execution failure: engine crashed unexpectedly" in captured.err.lower()


def test_cli_keyboard_interrupt(capsys: pytest.CaptureFixture[str]) -> None:
    def factory(**kwargs: Any) -> FakeEngine:
        return FakeEngine(**kwargs, raise_exc=KeyboardInterrupt())

    argv = ["run", "--title", "Interrupt Task", "--description", "Desc", "--criteria", "Test criteria"]
    code = main(argv, engine_factory=factory)
    assert code == EXIT_STOPPED
    captured = capsys.readouterr()
    assert "interrupted by user" in captured.err.lower()


# ---------------------------------------------------------------------------
# Human Governance Approval Hook Tests
# ---------------------------------------------------------------------------

def test_cli_approval_handler_auto_approve() -> None:
    captured_engine: Dict[str, Any] = {}

    def factory(**kwargs: Any) -> FakeEngine:
        captured_engine.update(kwargs)
        return FakeEngine(**kwargs, summary_status="COMPLETED")

    argv = ["run", "--title", "Task", "--description", "Desc", "--criteria", "Test criteria", "--auto-approve"]
    main(argv, engine_factory=factory)

    handler = captured_engine["approval_handler"]
    dummy_ctx = WorkflowContext(task=captured_engine["task"])
    assert handler("Requires approval", dummy_ctx) is True


def test_cli_approval_handler_autonomous_policy() -> None:
    captured_engine: Dict[str, Any] = {}

    def factory(**kwargs: Any) -> FakeEngine:
        captured_engine.update(kwargs)
        return FakeEngine(**kwargs, summary_status="COMPLETED")

    argv = ["run", "--title", "Task", "--description", "Desc", "--criteria", "Test criteria", "--policy", "AUTONOMOUS"]
    main(argv, engine_factory=factory)

    handler = captured_engine["approval_handler"]
    dummy_ctx = WorkflowContext(task=captured_engine["task"])
    assert handler("Autonomous mode approval", dummy_ctx) is True


def test_cli_approval_handler_interactive_yes() -> None:
    captured_engine: Dict[str, Any] = {}

    def factory(**kwargs: Any) -> FakeEngine:
        captured_engine.update(kwargs)
        return FakeEngine(**kwargs, summary_status="COMPLETED")

    argv = ["run", "--title", "Task", "--description", "Desc", "--criteria", "Test criteria"]
    main(argv, engine_factory=factory, input_func=lambda prompt: "y")

    handler = captured_engine["approval_handler"]
    dummy_ctx = WorkflowContext(task=captured_engine["task"])
    assert handler("Reviewer PASS - Final PO Signoff", dummy_ctx) is True


def test_cli_approval_handler_interactive_no() -> None:
    captured_engine: Dict[str, Any] = {}

    def factory(**kwargs: Any) -> FakeEngine:
        captured_engine.update(kwargs)
        return FakeEngine(**kwargs, summary_status="COMPLETED")

    argv = ["run", "--title", "Task", "--description", "Desc", "--criteria", "Test criteria"]
    main(argv, engine_factory=factory, input_func=lambda prompt: "n")

    handler = captured_engine["approval_handler"]
    dummy_ctx = WorkflowContext(task=captured_engine["task"])
    assert handler("Reviewer PASS - Final PO Signoff", dummy_ctx) is False


def test_cli_approval_handler_interactive_eof() -> None:
    captured_engine: Dict[str, Any] = {}

    def factory(**kwargs: Any) -> FakeEngine:
        captured_engine.update(kwargs)
        return FakeEngine(**kwargs, summary_status="COMPLETED")

    def raise_eof(prompt: str) -> str:
        raise EOFError("No stdin")

    argv = ["run", "--title", "Task", "--description", "Desc", "--criteria", "Test criteria"]
    main(argv, engine_factory=factory, input_func=raise_eof)

    handler = captured_engine["approval_handler"]
    dummy_ctx = WorkflowContext(task=captured_engine["task"])
    assert handler("Reviewer PASS - Final PO Signoff", dummy_ctx) is False


# ---------------------------------------------------------------------------
# Secret Sanitization in Error Outputs
# ---------------------------------------------------------------------------

def test_cli_secret_sanitized_on_task_load_error(capsys: pytest.CaptureFixture[str]) -> None:
    secret_key = "AIzaSyDummySecretKey12345678901234"
    invalid_input = f'{{"title": "Secret Task", "description": "Desc", "acceptance_criteria": ["C1"], "budget": "{secret_key}"}}'

    code = main(["run", invalid_input])
    assert code == EXIT_INVALID_INPUT

    captured = capsys.readouterr()
    assert secret_key not in captured.err
    assert "[REDACTED_API_KEY]" in captured.err


def test_cli_secret_sanitized_on_runtime_error(capsys: pytest.CaptureFixture[str]) -> None:
    secret_key = "sk-ant-api03-DummyAnthropicSecretKey123456789"

    def factory(**kwargs: Any) -> FakeEngine:
        return FakeEngine(**kwargs, raise_exc=RuntimeError(f"API Failed: {secret_key}"))

    argv = ["run", "--title", "Task", "--description", "Desc", "--criteria", "Test criteria"]
    code = main(argv, engine_factory=factory)
    assert code == EXIT_FAILED


    captured = capsys.readouterr()
    assert secret_key not in captured.err
    assert "[REDACTED_API_KEY]" in captured.err



# ---------------------------------------------------------------------------
# Module Invocation Subprocess Tests
# ---------------------------------------------------------------------------

def test_python_module_orchestrator_invocation() -> None:
    """Verify `python -m src.orchestrator --help` returns 0 via subprocess."""
    result = subprocess.run(
        [sys.executable, "-m", "src.orchestrator", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "orchestrator" in result.stdout.lower()


def test_python_module_orchestrator_cli_invocation() -> None:
    """Verify `python -m src.orchestrator.cli --help` returns 0 via subprocess."""
    result = subprocess.run(
        [sys.executable, "-m", "src.orchestrator.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "orchestrator" in result.stdout.lower()

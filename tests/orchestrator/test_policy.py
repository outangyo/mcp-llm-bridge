from __future__ import annotations

import pytest

from src.orchestrator.contracts.context import WorkflowContext
from src.orchestrator.contracts.report import (
    AgentReport,
    AgentRole,
    BuilderResult,
    ReviewerVerdict,
)
from src.orchestrator.contracts.task import TaskBudget, TaskContract
from src.orchestrator.policy.execution_policy import (
    ExecutionPolicy,
    PolicyMode,
)


@pytest.fixture
def sample_task() -> TaskContract:
    return TaskContract(
        title="Test Task",
        description="Verify policy logic",
        allowed_paths=["src/orchestrator/"],
        budget=TaskBudget(max_iterations=3, max_retries=2),
    )


@pytest.fixture
def pass_report() -> AgentReport:
    return AgentReport(
        agent_role=AgentRole.REVIEWER,
        agent_name="Gemini",
        summary="Code looks good",
        findings="All criteria met",
        reviewer_verdict=ReviewerVerdict.PASS,
        next_step="Deploy",
    )


@pytest.fixture
def reject_report() -> AgentReport:
    return AgentReport(
        agent_role=AgentRole.REVIEWER,
        agent_name="Gemini",
        summary="Code rejected",
        findings="Missing tests",
        reviewer_verdict=ReviewerVerdict.REJECT,
        blockers=["Test coverage low"],
        next_step="Add tests",
    )


def test_default_policy_mode(sample_task: TaskContract):
    policy = ExecutionPolicy()
    assert policy.mode == PolicyMode.SUPERVISED
    assert policy.require_final_approval is True
    assert policy.require_retry_approval is False


def test_supervised_mode_reviewer_pass(sample_task: TaskContract, pass_report: AgentReport):
    context = WorkflowContext(task=sample_task)
    policy = ExecutionPolicy(mode=PolicyMode.SUPERVISED, require_final_approval=True)

    decision = policy.evaluate_reviewer_pass(context, pass_report)
    assert decision.allowed is True
    assert decision.requires_human_approval is True
    assert "PO final approval" in decision.reason


def test_supervised_mode_reviewer_reject_within_budget(sample_task: TaskContract, reject_report: AgentReport):
    context = WorkflowContext(task=sample_task)
    context.start_new_iteration()
    policy = ExecutionPolicy(mode=PolicyMode.SUPERVISED, require_retry_approval=False)

    decision = policy.evaluate_reviewer_reject(context, reject_report)
    assert decision.allowed is True
    assert decision.requires_human_approval is False


def test_supervised_mode_reviewer_reject_with_approval_configured(sample_task: TaskContract, reject_report: AgentReport):
    context = WorkflowContext(task=sample_task)
    context.start_new_iteration()
    policy = ExecutionPolicy(mode=PolicyMode.SUPERVISED, require_retry_approval=True)

    decision = policy.evaluate_reviewer_reject(context, reject_report)
    assert decision.allowed is True
    assert decision.requires_human_approval is True


def test_safe_mode_always_requires_human_approval(sample_task: TaskContract, pass_report: AgentReport, reject_report: AgentReport):
    context = WorkflowContext(task=sample_task)
    context.start_new_iteration()
    policy = ExecutionPolicy(mode=PolicyMode.SAFE)

    pass_decision = policy.evaluate_reviewer_pass(context, pass_report)
    assert pass_decision.allowed is True
    assert pass_decision.requires_human_approval is True

    reject_decision = policy.evaluate_reviewer_reject(context, reject_report)
    assert reject_decision.allowed is True
    assert reject_decision.requires_human_approval is True


def test_autonomous_mode_auto_proceeds(sample_task: TaskContract, pass_report: AgentReport, reject_report: AgentReport):
    context = WorkflowContext(task=sample_task)
    context.start_new_iteration()
    policy = ExecutionPolicy(mode=PolicyMode.AUTONOMOUS)

    pass_decision = policy.evaluate_reviewer_pass(context, pass_report)
    assert pass_decision.allowed is True
    assert pass_decision.requires_human_approval is False

    reject_decision = policy.evaluate_reviewer_reject(context, reject_report)
    assert reject_decision.allowed is True
    assert reject_decision.requires_human_approval is False


def test_reviewer_reject_blocks_when_budget_exhausted(sample_task: TaskContract, reject_report: AgentReport):
    context = WorkflowContext(task=sample_task)
    context.iteration_index = 3  # max_iterations is 3
    policy = ExecutionPolicy(mode=PolicyMode.AUTONOMOUS)

    decision = policy.evaluate_reviewer_reject(context, reject_report)
    assert decision.allowed is False
    assert decision.requires_human_approval is False
    assert "Budget exhausted" in decision.reason


def test_evaluate_file_access(sample_task: TaskContract):
    policy = ExecutionPolicy()
    allowed_paths = ["src/orchestrator/", "tests/"]

    decision_ok = policy.evaluate_file_access("src/orchestrator/engine.py", allowed_paths)
    assert decision_ok.allowed is True

    decision_blocked = policy.evaluate_file_access("config/secrets.env", allowed_paths)
    assert decision_blocked.allowed is False
    assert "outside permitted scopes" in decision_blocked.reason

    decision_empty = policy.evaluate_file_access("any/path.py", [])
    assert decision_empty.allowed is True

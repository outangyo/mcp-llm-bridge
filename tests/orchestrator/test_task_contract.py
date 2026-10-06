import pytest
from pydantic import ValidationError

from src.orchestrator.contracts.task import TaskBudget, TaskContract


def test_task_contract_defaults():
    task = TaskContract(
        title="Refactor Connection Pool",
        description="Add timeout handling to connection pool",
    )
    assert task.title == "Refactor Connection Pool"
    assert task.description == "Add timeout handling to connection pool"
    assert task.budget.max_iterations == 3
    assert task.budget.max_retries == 2
    assert task.budget.timeout_seconds == 300
    assert task.acceptance_criteria == []
    assert task.allowed_paths == []
    assert task.task_id is not None
    assert task.created_at is not None


def test_task_contract_custom_values():
    task = TaskContract(
        title="Custom Task",
        description="Custom Desc",
        acceptance_criteria=["Criterion 1", "Criterion 2"],
        budget=TaskBudget(max_iterations=5, max_retries=4, timeout_seconds=600),
        allowed_paths=["src/"],
    )
    assert task.budget.max_iterations == 5
    assert task.budget.max_retries == 4
    assert len(task.acceptance_criteria) == 2
    assert task.allowed_paths == ["src/"]


def test_task_contract_validation_empty_fields():
    with pytest.raises(ValidationError):
        TaskContract(title="", description="Valid description")

    with pytest.raises(ValidationError):
        TaskContract(title="Valid title", description="")


def test_task_contract_deterministic_json_roundtrip():
    original = TaskContract(
        title="Test JSON",
        description="Test Desc",
        acceptance_criteria=["Unit test passes"],
        budget=TaskBudget(max_iterations=2, max_retries=1, timeout_seconds=120),
    )
    json_str = original.to_json()
    reconstructed = TaskContract.from_json(json_str)

    assert reconstructed.task_id == original.task_id
    assert reconstructed.title == original.title
    assert reconstructed.description == original.description
    assert reconstructed.budget.max_iterations == original.budget.max_iterations
    assert reconstructed.acceptance_criteria == original.acceptance_criteria

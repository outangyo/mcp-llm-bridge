from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from src.orchestrator.contracts.task import (
    TaskBudget,
    TaskContract,
    normalize_task_input,
    validate_task_for_execution,
)
from src.orchestrator.security.paths import (
    is_path_allowed,
    is_subpath,
    normalize_path,
    validate_path_string,
)


# ---------------------------------------------------------------------------
# 1. Valid Task Tests
# ---------------------------------------------------------------------------

def test_valid_task_minimal() -> None:
    task = TaskContract(
        title="Valid Minimal Task",
        description="A task with minimal valid parameters",
        acceptance_criteria=["Criterion 1"],
    )
    assert task.title == "Valid Minimal Task"
    assert task.description == "A task with minimal valid parameters"
    assert task.acceptance_criteria == ["Criterion 1"]
    assert task.allowed_paths == []
    assert task.budget.max_iterations == 3
    assert task.budget.max_retries == 2
    task.validate_for_execution()


def test_valid_task_full(tmp_path: Path) -> None:
    task = TaskContract(
        title="Valid Full Task",
        description="Comprehensive task specification",
        acceptance_criteria=["Criterion 1", "Criterion 2"],
        allowed_paths=["src/core/", "tests/"],
        budget=TaskBudget(max_iterations=5, max_retries=3, timeout_seconds=600),
        metadata={"priority": "high", "env": "testing"},
    )
    assert task.budget.max_iterations == 5
    assert task.budget.max_retries == 3
    assert task.budget.timeout_seconds == 600
    assert len(task.acceptance_criteria) == 2
    assert len(task.allowed_paths) == 2
    task.validate_for_execution()


# ---------------------------------------------------------------------------
# 2. Missing / Empty Required Fields
# ---------------------------------------------------------------------------

def test_empty_title_fails() -> None:
    with pytest.raises((ValueError, ValidationError), match=r"(at least 1 character|empty or blank)"):
        TaskContract(title="", description="Valid desc", acceptance_criteria=["Crit"])

    with pytest.raises((ValueError, ValidationError), match=r"(at least 1 character|empty or blank)"):
        TaskContract(title="   ", description="Valid desc", acceptance_criteria=["Crit"])


def test_empty_description_fails() -> None:
    with pytest.raises((ValueError, ValidationError), match=r"(at least 1 character|empty or blank)"):
        TaskContract(title="Valid title", description="", acceptance_criteria=["Crit"])

    with pytest.raises((ValueError, ValidationError), match=r"(at least 1 character|empty or blank)"):
        TaskContract(title="Valid title", description="   ", acceptance_criteria=["Crit"])



def test_empty_acceptance_criteria_fails_validation() -> None:
    task = TaskContract(title="Title", description="Desc", acceptance_criteria=[])
    with pytest.raises(ValueError, match="at least one acceptance criterion"):
        validate_task_for_execution(task)


def test_blank_acceptance_criterion_fails_validation() -> None:
    task = TaskContract(title="Title", description="Desc", acceptance_criteria=[""])
    with pytest.raises(ValueError, match="must not be empty or blank"):
        validate_task_for_execution(task)

    task_whitespace = TaskContract(title="Title", description="Desc", acceptance_criteria=["Valid", "   "])
    with pytest.raises(ValueError, match="must not be empty or blank"):
        validate_task_for_execution(task_whitespace)


def test_missing_required_fields_in_normalization() -> None:
    with pytest.raises(ValueError, match="Task title is required"):
        normalize_task_input(description="Desc only", criteria=["Crit"])

    with pytest.raises(ValueError, match="Task description is required"):
        normalize_task_input(title="Title only", criteria=["Crit"])


# ---------------------------------------------------------------------------
# 3. Invalid Iteration / Retry / Budget Limits
# ---------------------------------------------------------------------------

def test_invalid_max_iterations_zero() -> None:
    with pytest.raises(ValidationError):
        TaskBudget(max_iterations=0)


def test_invalid_max_iterations_negative() -> None:
    with pytest.raises(ValidationError):
        TaskBudget(max_iterations=-1)


def test_invalid_max_retries_negative() -> None:
    with pytest.raises(ValidationError):
        TaskBudget(max_retries=-1)


def test_invalid_timeout_seconds_too_low() -> None:
    with pytest.raises(ValidationError):
        TaskBudget(timeout_seconds=5)


def test_invalid_budget_in_normalization() -> None:
    with pytest.raises((ValueError, ValidationError)):
        normalize_task_input(
            title="Title",
            description="Desc",
            criteria=["Crit"],
            max_iterations=0,
        )

    with pytest.raises((ValueError, ValidationError)):
        normalize_task_input(
            title="Title",
            description="Desc",
            criteria=["Crit"],
            max_retries=-1,
        )


# ---------------------------------------------------------------------------
# 4. Invalid Allowed Paths
# ---------------------------------------------------------------------------

def test_invalid_allowed_path_blank() -> None:
    task = TaskContract(title="Title", description="Desc", acceptance_criteria=["Crit"], allowed_paths=["   "])
    with pytest.raises(ValueError, match=r"(non-empty string|empty or blank)"):
        validate_task_for_execution(task)



def test_invalid_allowed_path_null_byte() -> None:
    task = TaskContract(title="Title", description="Desc", acceptance_criteria=["Crit"], allowed_paths=["src/\0bad"])
    with pytest.raises(ValueError, match="null byte"):
        validate_task_for_execution(task)


def test_validate_path_string() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        validate_path_string("")
    with pytest.raises(ValueError, match="non-empty string"):
        validate_path_string("   ")
    with pytest.raises(ValueError, match="null byte"):
        validate_path_string("dir/\0file")


# ---------------------------------------------------------------------------
# 5. Path Normalization
# ---------------------------------------------------------------------------

def test_normalize_relative_path(tmp_path: Path) -> None:
    resolved = normalize_path("src/foo", base_dir=tmp_path)
    expected = (tmp_path / "src" / "foo").resolve()
    assert resolved == expected


def test_normalize_path_trailing_separator(tmp_path: Path) -> None:
    p1 = normalize_path("src/foo/", base_dir=tmp_path)
    p2 = normalize_path("src/foo", base_dir=tmp_path)
    assert p1 == p2


def test_normalize_path_single_dot(tmp_path: Path) -> None:
    p = normalize_path("src/./foo", base_dir=tmp_path)
    expected = (tmp_path / "src" / "foo").resolve()
    assert p == expected


# ---------------------------------------------------------------------------
# 6. '..' Traversal Safety
# ---------------------------------------------------------------------------

def test_dot_dot_traversal_escape(tmp_path: Path) -> None:
    scope = tmp_path / "project" / "allowed"
    target = tmp_path / "project" / "allowed" / ".." / "secret.env"
    assert not is_subpath(target, scope, base_dir=tmp_path)


def test_dot_dot_traversal_within_scope(tmp_path: Path) -> None:
    scope = tmp_path / "project" / "allowed"
    target = tmp_path / "project" / "allowed" / "sub" / ".." / "file.py"
    assert is_subpath(target, scope, base_dir=tmp_path)


# ---------------------------------------------------------------------------
# 7. Path-Prefix Ambiguity Safety
# ---------------------------------------------------------------------------

def test_prefix_ambiguity_prevented(tmp_path: Path) -> None:
    """Ensure /project/foo does NOT match /project/foobar (true boundary protection)."""
    scope = tmp_path / "project" / "foo"
    target_sibling = tmp_path / "project" / "foobar"
    target_sibling_file = tmp_path / "project" / "foobar" / "evil.py"

    assert not is_subpath(target_sibling, scope, base_dir=tmp_path)
    assert not is_subpath(target_sibling_file, scope, base_dir=tmp_path)


def test_prefix_ambiguity_legitimate_child(tmp_path: Path) -> None:
    scope = tmp_path / "project" / "foo"
    target_child = tmp_path / "project" / "foo" / "bar.py"
    assert is_subpath(target_child, scope, base_dir=tmp_path)


def test_is_path_allowed_unrestricted_when_empty() -> None:
    assert is_path_allowed("any/path/file.py", [])


def test_is_path_allowed_matches_one_of_many(tmp_path: Path) -> None:
    allowed = ["src/module_a/", "tests/"]
    assert is_path_allowed("src/module_a/code.py", allowed, base_dir=tmp_path)
    assert is_path_allowed("tests/test_code.py", allowed, base_dir=tmp_path)
    assert not is_path_allowed("src/module_b/code.py", allowed, base_dir=tmp_path)


# ---------------------------------------------------------------------------
# 8. Conflicting Input Sources (Deterministic Normalization Precedence)
# ---------------------------------------------------------------------------

def test_conflicting_sources_cli_overrides_file(tmp_path: Path) -> None:
    task_file = tmp_path / "task.json"
    file_data = {
        "title": "File Title",
        "description": "File Description",
        "acceptance_criteria": ["File Criterion"],
        "allowed_paths": ["src/file/"],
        "budget": {"max_iterations": 2, "max_retries": 1},
    }
    task_file.write_text(json.dumps(file_data), encoding="utf-8")

    # CLI explicitly overrides title and max_iterations, but leaves description and criteria
    normalized = normalize_task_input(
        task_input=str(task_file),
        title="CLI Title Override",
        max_iterations=10,
    )

    assert normalized.title == "CLI Title Override"
    assert normalized.description == "File Description"  # preserved from file
    assert normalized.acceptance_criteria == ["File Criterion"]  # preserved from file
    assert normalized.allowed_paths == ["src/file/"]  # preserved from file
    assert normalized.budget.max_iterations == 10  # overridden by CLI
    assert normalized.budget.max_retries == 1  # preserved from file


def test_conflicting_sources_cli_overrides_criteria_and_paths(tmp_path: Path) -> None:
    task_file = tmp_path / "task.json"
    file_data = {
        "title": "Base Title",
        "description": "Base Description",
        "acceptance_criteria": ["Base Crit"],
        "allowed_paths": ["base/path/"],
    }
    task_file.write_text(json.dumps(file_data), encoding="utf-8")

    normalized = normalize_task_input(
        task_input=str(task_file),
        criteria=["CLI Crit Override"],
        allowed_paths=["cli/path/"],
    )

    assert normalized.acceptance_criteria == ["CLI Crit Override"]
    assert normalized.allowed_paths == ["cli/path/"]


# ---------------------------------------------------------------------------
# 9. Task Immutability
# ---------------------------------------------------------------------------

def test_task_immutability_field_reassignment() -> None:
    task = TaskContract(
        title="Immutable Task",
        description="Task testing immutability",
        acceptance_criteria=["Criterion 1"],
        allowed_paths=["src/"],
    )

    with pytest.raises(ValidationError):
        task.title = "Hacked Title"  # type: ignore

    with pytest.raises(ValidationError):
        task.description = "Hacked Desc"  # type: ignore

    with pytest.raises(ValidationError):
        task.allowed_paths = ["hacked/"]  # type: ignore

    with pytest.raises(ValidationError):
        task.acceptance_criteria = ["Hacked Crit"]  # type: ignore

    with pytest.raises(ValidationError):
        task.task_id = "hacked_id"  # type: ignore

    with pytest.raises(ValidationError):
        task.budget = TaskBudget(max_iterations=100)  # type: ignore

    with pytest.raises(ValidationError):
        task.budget.max_iterations = 100  # type: ignore


def test_task_immutability_list_in_place_mutation() -> None:
    task = TaskContract(
        title="Immutable Task",
        description="Testing in-place list mutation protection",
        acceptance_criteria=["Criterion 1"],
        allowed_paths=["src/"],
    )

    with pytest.raises(TypeError, match="TaskContract lists are immutable"):
        task.allowed_paths.append("evil/path")

    with pytest.raises(TypeError, match="TaskContract lists are immutable"):
        task.allowed_paths.extend(["evil/path"])

    with pytest.raises(TypeError, match="TaskContract lists are immutable"):
        task.allowed_paths.insert(0, "evil/path")

    with pytest.raises(TypeError, match="TaskContract lists are immutable"):
        task.allowed_paths[0] = "evil/path"

    with pytest.raises(TypeError, match="TaskContract lists are immutable"):
        task.acceptance_criteria.append("evil criterion")

    with pytest.raises(TypeError, match="TaskContract lists are immutable"):
        task.acceptance_criteria.clear()


def test_task_immutability_metadata_in_place_mutation() -> None:
    task = TaskContract(
        title="Immutable Task",
        description="Testing metadata immutability",
        acceptance_criteria=["Criterion 1"],
        metadata={"key": "initial"},
    )

    with pytest.raises(TypeError, match="TaskContract metadata is immutable"):
        task.metadata["key"] = "hacked"

    with pytest.raises(TypeError, match="TaskContract metadata is immutable"):
        task.metadata["new_key"] = "val"

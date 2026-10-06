from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.orchestrator.security.paths import validate_path_string


class FrozenList(list):
    """Immutable list implementation protecting TaskContract fields from mutation."""

    def append(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("TaskContract lists are immutable during workflow run")

    def extend(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("TaskContract lists are immutable during workflow run")

    def insert(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("TaskContract lists are immutable during workflow run")

    def __setitem__(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("TaskContract lists are immutable during workflow run")

    def __delitem__(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("TaskContract lists are immutable during workflow run")

    def pop(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("TaskContract lists are immutable during workflow run")

    def remove(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("TaskContract lists are immutable during workflow run")

    def clear(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("TaskContract lists are immutable during workflow run")


class FrozenDict(dict):
    """Immutable dict implementation protecting TaskContract metadata from mutation."""

    def __setitem__(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("TaskContract metadata is immutable during workflow run")

    def __delitem__(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("TaskContract metadata is immutable during workflow run")

    def pop(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("TaskContract metadata is immutable during workflow run")

    def clear(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("TaskContract metadata is immutable during workflow run")

    def update(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("TaskContract metadata is immutable during workflow run")

    def setdefault(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("TaskContract metadata is immutable during workflow run")


class TaskBudget(BaseModel):
    """Resource and retry constraints for a workflow task."""

    model_config = ConfigDict(frozen=True)

    max_iterations: int = Field(default=3, ge=1, description="Maximum allowed iteration loops")
    max_retries: int = Field(default=2, ge=0, description="Maximum allowed retries on reviewer rejection")
    timeout_seconds: int = Field(default=300, ge=10, description="Workflow timeout in seconds")


class TaskContract(BaseModel):
    """Formal specification defining an engineering task for multi-agent execution."""

    model_config = ConfigDict(frozen=True)

    task_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the task",
    )
    title: str = Field(..., min_length=1, description="Short human-readable task title")
    description: str = Field(..., min_length=1, description="Detailed problem description and requirements")
    acceptance_criteria: List[str] = Field(
        default_factory=list,
        description="List of verifiable criteria that must be satisfied for completion",
    )
    budget: TaskBudget = Field(
        default_factory=TaskBudget,
        description="Execution budget and retry boundaries",
    )
    allowed_paths: List[str] = Field(
        default_factory=list,
        description="Scoped filesystem paths the builder is permitted to modify",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of creation",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional task metadata and tags",
    )

    @field_validator("title", "description")
    @classmethod
    def _validate_non_blank_string(cls, v: str, info: Any) -> str:
        if not v or not v.strip():
            raise ValueError(f"Task {info.field_name} must not be empty or blank")
        return v.strip()

    @field_validator("acceptance_criteria", "allowed_paths", mode="after")
    @classmethod
    def _freeze_list(cls, v: List[str]) -> FrozenList:
        return FrozenList(v)

    @field_validator("metadata", mode="after")
    @classmethod
    def _freeze_dict(cls, v: dict[str, Any]) -> FrozenDict:
        return FrozenDict(v)

    def validate_for_execution(self) -> None:
        """Validate that this task contract is fully valid to start an orchestrated workflow."""
        validate_task_for_execution(self)

    def to_json(self, indent: int | None = None) -> str:
        """Serialize task contract to deterministic JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> TaskContract:
        """Deserialize task contract from JSON string."""
        return cls.model_validate_json(json_str)


def validate_task_for_execution(task: TaskContract) -> None:
    """
    Validate all required constraints on TaskContract before workflow begins.
    Fails before any agent execution begins:
    - title is non-empty
    - description is valid
    - acceptance criteria is present and non-empty
    - allowed_paths are valid syntax (no null bytes, non-empty strings)
    - max_iterations >= 1
    - max_retries >= 0
    - budget limits are sane
    """
    if not task.title or not task.title.strip():
        raise ValueError("Task title must not be empty or blank")

    if not task.description or not task.description.strip():
        raise ValueError("Task description must not be empty or blank")

    if not task.acceptance_criteria:
        raise ValueError("Task must include at least one acceptance criterion")

    for idx, criterion in enumerate(task.acceptance_criteria):
        if not isinstance(criterion, str) or not criterion.strip():
            raise ValueError(f"Acceptance criterion at index {idx} must not be empty or blank")

    for idx, path_str in enumerate(task.allowed_paths):
        try:
            validate_path_string(path_str)
        except ValueError as exc:
            raise ValueError(f"Invalid allowed_path at index {idx}: {exc}") from exc

    if task.budget.max_iterations < 1:
        raise ValueError(f"Task budget max_iterations must be >= 1, got {task.budget.max_iterations}")

    if task.budget.max_retries < 0:
        raise ValueError(f"Task budget max_retries must be >= 0, got {task.budget.max_retries}")

    if task.budget.timeout_seconds < 10:
        raise ValueError(f"Task budget timeout_seconds must be >= 10, got {task.budget.timeout_seconds}")


def normalize_task_input(
    task_input: Optional[str] = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
    criteria: Optional[List[str]] = None,
    allowed_paths: Optional[List[str]] = None,
    max_iterations: Optional[int] = None,
    max_retries: Optional[int] = None,
    timeout_seconds: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> TaskContract:
    """
    Normalize task input from file, inline JSON, and/or CLI flags into one validated, immutable TaskContract.

    Deterministic precedence rule:
    1. Base values loaded from task_input (JSON file path or inline JSON string) if provided.
    2. CLI flags override base values if explicitly provided (not None).
    3. If neither provides required fields, raises ValueError.
    4. Validates full contract before returning.
    """
    base_data: Dict[str, Any] = {}

    if task_input:
        cleaned_input = task_input.strip()
        raw_json: str
        if os.path.isfile(cleaned_input):
            try:
                with open(cleaned_input, "r", encoding="utf-8") as f:
                    raw_json = f.read()
            except Exception as exc:
                raise ValueError(f"Failed to load task file '{cleaned_input}': {exc}") from exc
        else:
            raw_json = cleaned_input

        try:
            parsed = json.loads(raw_json)
            if not isinstance(parsed, dict):
                raise ValueError("Task definition JSON must be an object/dictionary")
            base_data = parsed
        except Exception as exc:
            raise ValueError(
                f"Input is neither a valid file path nor a valid TaskContract JSON string: {exc}"
            ) from exc

    # Merge budget
    budget_raw = base_data.get("budget")
    if budget_raw is None:
        budget_dict: Dict[str, Any] = {}
    elif isinstance(budget_raw, dict):
        budget_dict = dict(budget_raw)
    else:
        raise ValueError(f"Task budget must be a dictionary/object, got {type(budget_raw).__name__}: {budget_raw}")

    if max_iterations is not None:
        budget_dict["max_iterations"] = max_iterations
    if max_retries is not None:
        budget_dict["max_retries"] = max_retries
    if timeout_seconds is not None:
        budget_dict["timeout_seconds"] = timeout_seconds


    # Merge top-level fields (CLI explicit overrides file/JSON)
    final_title = title if title is not None else base_data.get("title")
    final_desc = description if description is not None else base_data.get("description")

    final_criteria = (
        criteria if (criteria is not None and len(criteria) > 0)
        else base_data.get("acceptance_criteria", [])
    )
    final_paths = (
        allowed_paths if (allowed_paths is not None and len(allowed_paths) > 0)
        else base_data.get("allowed_paths", [])
    )
    final_meta = (
        metadata if metadata is not None
        else base_data.get("metadata", {})
    )

    if not final_title or not str(final_title).strip():
        raise ValueError("Task title is required and must not be empty")

    if not final_desc or not str(final_desc).strip():
        raise ValueError("Task description is required and must not be empty")

    budget_obj = TaskBudget(**budget_dict)

    task = TaskContract(
        task_id=base_data.get("task_id", str(uuid.uuid4())),
        title=str(final_title).strip(),
        description=str(final_desc).strip(),
        acceptance_criteria=final_criteria or [],
        budget=budget_obj,
        allowed_paths=final_paths or [],
        metadata=final_meta or {},
    )

    # Validate before returning
    validate_task_for_execution(task)
    return task

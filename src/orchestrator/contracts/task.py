from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, List

from pydantic import BaseModel, Field


class TaskBudget(BaseModel):
    """Resource and retry constraints for a workflow task."""

    max_iterations: int = Field(default=3, ge=1, description="Maximum allowed iteration loops")
    max_retries: int = Field(default=2, ge=0, description="Maximum allowed retries on reviewer rejection")
    timeout_seconds: int = Field(default=300, ge=10, description="Workflow timeout in seconds")


class TaskContract(BaseModel):
    """Formal specification defining an engineering task for multi-agent execution."""

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

    def to_json(self, indent: int | None = None) -> str:
        """Serialize task contract to deterministic JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> TaskContract:
        """Deserialize task contract from JSON string."""
        return cls.model_validate_json(json_str)

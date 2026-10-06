from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.orchestrator.contracts.report import AgentRole


class EventType(str, Enum):
    """Exhaustive, machine-readable event types emitted during orchestration."""

    # Workflow lifecycle
    TASK_CREATED = "TASK_CREATED"
    WORKFLOW_STARTED = "WORKFLOW_STARTED"

    # Agent execution lifecycle
    AGENT_STARTED = "AGENT_STARTED"
    AGENT_PROGRESS = "AGENT_PROGRESS"
    AGENT_COMPLETED = "AGENT_COMPLETED"
    AGENT_FAILED = "AGENT_FAILED"

    # Verification / Testing lifecycle
    TEST_STARTED = "TEST_STARTED"
    TEST_PASSED = "TEST_PASSED"
    TEST_FAILED = "TEST_FAILED"

    # Review lifecycle
    REVIEW_STARTED = "REVIEW_STARTED"
    REVIEW_PASSED = "REVIEW_PASSED"
    REVIEW_REJECTED = "REVIEW_REJECTED"

    # Governance & Approval lifecycle
    POLICY_BLOCKED = "POLICY_BLOCKED"
    APPROVAL_REQUESTED = "APPROVAL_REQUESTED"
    APPROVAL_GRANTED = "APPROVAL_GRANTED"
    APPROVAL_REJECTED = "APPROVAL_REJECTED"

    # Iteration lifecycle
    ITERATION_STARTED = "ITERATION_STARTED"
    ITERATION_COMPLETED = "ITERATION_COMPLETED"

    # Terminal workflow states
    WORKFLOW_COMPLETED = "WORKFLOW_COMPLETED"
    WORKFLOW_FAILED = "WORKFLOW_FAILED"
    WORKFLOW_STOPPED = "WORKFLOW_STOPPED"


class WorkflowEvent(BaseModel):
    """Discrete, immutable event emitted by the orchestrator engine or agent adapters."""

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this event instance",
    )
    event_type: EventType = Field(..., description="Machine-readable event type")
    run_id: str = Field(..., description="ID of the workflow run session")
    iteration: int = Field(default=0, ge=0, description="Iteration number (0 for global setup)")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of occurrence",
    )
    agent_role: Optional[AgentRole] = Field(default=None, description="Role of the associated agent, if any")
    agent_name: Optional[str] = Field(default=None, description="Name of the associated agent, if any")
    message: str = Field(
        default="",
        description="Human-readable progress narrative for live CLI streaming",
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured machine data payload associated with this event",
    )

    def format_cli_line(self) -> str:
        """Format event into a concise terminal line for human display."""
        time_part = self.timestamp.split("T")[-1][:8] if "T" in self.timestamp else self.timestamp
        role_part = f"[{self.agent_role.value}] " if self.agent_role else ""
        msg = self.message or f"Event {self.event_type.value}"
        return f"[{time_part}] {role_part}{self.event_type.value}: {msg}"

    def to_json(self, indent: int | None = None) -> str:
        """Serialize event to deterministic JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> WorkflowEvent:
        """Deserialize event from JSON string."""
        return cls.model_validate_json(json_str)

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field, model_validator


class AgentRole(str, Enum):
    """Assigned functional role of an agent in the workflow."""

    BUILDER = "BUILDER"
    REVIEWER = "REVIEWER"
    LEAD = "LEAD"
    ORCHESTRATOR = "ORCHESTRATOR"


class BuilderResult(str, Enum):
    """Strongly-typed machine-readable outcome produced by a Builder."""

    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    INCOMPLETE = "INCOMPLETE"
    BLOCKED = "BLOCKED"


class ReviewerVerdict(str, Enum):
    """Strongly-typed machine-readable decision produced by a Reviewer."""

    PASS = "PASS"
    REJECT = "REJECT"
    REQUEST_CHANGES = "REQUEST_CHANGES"


class AgentReport(BaseModel):
    """Human-readable and machine-typed structured report returned by any agent turn."""

    report_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the report",
    )
    agent_role: AgentRole = Field(..., description="Role of the reporting agent")
    agent_name: str = Field(..., min_length=1, description="Identifier/name of the reporting agent (e.g., AGY, Gemini)")
    summary: str = Field(..., min_length=1, description="Human-readable summary of actions taken")
    findings: str = Field(..., min_length=1, description="Key discoveries, observations, or diff analysis")
    builder_result: Optional[BuilderResult] = Field(
        default=None,
        description="Strongly-typed decision if reporting agent is a BUILDER",
    )
    reviewer_verdict: Optional[ReviewerVerdict] = Field(
        default=None,
        description="Strongly-typed decision if reporting agent is a REVIEWER",
    )
    blockers: List[str] = Field(
        default_factory=list,
        description="Identified obstacles, errors, or dependencies preventing progress",
    )
    next_step: str = Field(
        ...,
        min_length=1,
        description="Recommended immediate next action or instruction",
    )
    artifacts: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured data payloads: modified files, git diffs, test summaries",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp when report was compiled",
    )

    @model_validator(mode="after")
    def validate_typed_outcome(self) -> AgentReport:
        """Enforce that BUILDER provides builder_result and REVIEWER provides reviewer_verdict."""
        if self.agent_role == AgentRole.BUILDER and self.builder_result is None:
            raise ValueError("AgentReport for BUILDER must specify 'builder_result'")
        if self.agent_role == AgentRole.REVIEWER and self.reviewer_verdict is None:
            raise ValueError("AgentReport for REVIEWER must specify 'reviewer_verdict'")
        return self

    @property
    def is_pass(self) -> bool:
        """Helper to quickly check if a Reviewer issued a PASS verdict."""
        return self.reviewer_verdict == ReviewerVerdict.PASS

    @property
    def is_success(self) -> bool:
        """Helper to quickly check if a Builder reported SUCCESS."""
        return self.builder_result == BuilderResult.SUCCESS

    def format_human_summary(self) -> str:
        """Render report into a human-readable summary block for terminal display."""
        role_tag = f"[{self.agent_role.value}:{self.agent_name}]"
        decision = (
            f"Result: {self.builder_result.value}"
            if self.builder_result
            else f"Verdict: {self.reviewer_verdict.value}"
            if self.reviewer_verdict
            else "N/A"
        )
        blockers_str = ", ".join(self.blockers) if self.blockers else "None"

        return (
            f"{role_tag} - {decision}\n"
            f"  • Summary:   {self.summary}\n"
            f"  • Findings:  {self.findings}\n"
            f"  • Blockers:  {blockers_str}\n"
            f"  • Next Step: {self.next_step}"
        )

    def to_json(self, indent: int | None = None) -> str:
        """Serialize report to deterministic JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> AgentReport:
        """Deserialize report from JSON string."""
        return cls.model_validate_json(json_str)

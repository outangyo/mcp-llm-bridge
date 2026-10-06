from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from src.orchestrator.contracts.context import WorkflowContext
from src.orchestrator.contracts.report import AgentReport


class PolicyMode(str, Enum):
    """Execution governance modes controlling human approval boundaries."""

    SAFE = "SAFE"
    SUPERVISED = "SUPERVISED"
    AUTONOMOUS = "AUTONOMOUS"


class PolicyDecision(BaseModel):
    """Outcome of a policy evaluation determining execution progression and human gating."""

    allowed: bool = Field(default=True, description="Whether the workflow action is permitted to proceed")
    requires_human_approval: bool = Field(
        default=False,
        description="Whether human (PO) confirmation is mandatory before transitioning",
    )
    reason: str = Field(default="", description="Diagnostic explanation of the policy decision")


class ExecutionPolicy:
    """Deterministic governance engine enforcing execution modes and approval boundaries."""

    def __init__(
        self,
        mode: PolicyMode = PolicyMode.SUPERVISED,
        require_final_approval: bool = True,
        require_retry_approval: bool = False,
    ) -> None:
        self.mode = mode
        self.require_final_approval = require_final_approval
        self.require_retry_approval = require_retry_approval

    def evaluate_reviewer_pass(
        self,
        context: WorkflowContext,
        report: AgentReport,
    ) -> PolicyDecision:
        """
        Evaluate policy when a Reviewer issues a PASS verdict.
        Determines whether the workflow auto-completes or halts for final PO approval.
        """
        if self.mode == PolicyMode.SAFE:
            return PolicyDecision(
                allowed=True,
                requires_human_approval=True,
                reason="SAFE mode requires explicit human confirmation for final sign-off.",
            )

        if self.mode == PolicyMode.SUPERVISED:
            return PolicyDecision(
                allowed=True,
                requires_human_approval=self.require_final_approval,
                reason="SUPERVISED mode requires PO final approval gate before completion."
                if self.require_final_approval
                else "SUPERVISED mode configured to auto-complete.",
            )

        if self.mode == PolicyMode.AUTONOMOUS:
            return PolicyDecision(
                allowed=True,
                requires_human_approval=False,
                reason="AUTONOMOUS mode automatically accepts Reviewer PASS without human gate.",
            )

        return PolicyDecision(allowed=True, requires_human_approval=True, reason="Default fallback gate.")

    def evaluate_reviewer_reject(
        self,
        context: WorkflowContext,
        report: AgentReport,
    ) -> PolicyDecision:
        """
        Evaluate policy when a Reviewer issues a REJECT or REQUEST_CHANGES verdict.
        Determines whether the retry is permitted, blocked, or requires PO approval.
        """
        if context.is_retry_budget_exhausted() or context.is_iteration_budget_exhausted():
            return PolicyDecision(
                allowed=False,
                requires_human_approval=False,
                reason=f"Budget exhausted (retries: {context.retry_count}/{context.task.budget.max_retries}, iterations: {context.iteration_index}/{context.task.budget.max_iterations}).",
            )

        if self.mode == PolicyMode.SAFE:
            return PolicyDecision(
                allowed=True,
                requires_human_approval=True,
                reason="SAFE mode requires human approval before launching next retry iteration.",
            )

        if self.mode == PolicyMode.SUPERVISED:
            return PolicyDecision(
                allowed=True,
                requires_human_approval=self.require_retry_approval,
                reason="SUPERVISED mode requires human approval for retry."
                if self.require_retry_approval
                else "SUPERVISED mode auto-continues to next iteration within budget.",
            )

        if self.mode == PolicyMode.AUTONOMOUS:
            return PolicyDecision(
                allowed=True,
                requires_human_approval=False,
                reason="AUTONOMOUS mode auto-retries next iteration within budget.",
            )

        return PolicyDecision(allowed=True, requires_human_approval=True, reason="Default fallback gate.")

    def evaluate_file_access(
        self,
        target_path: str,
        allowed_paths: List[str],
    ) -> PolicyDecision:
        """Enforce path-level boundary restrictions defined in TaskContract."""
        if not allowed_paths:
            return PolicyDecision(allowed=True, requires_human_approval=False, reason="No path restrictions active.")

        normalized_target = target_path.replace("\\", "/").strip().lower()
        is_allowed = any(
            normalized_target.startswith(p.replace("\\", "/").strip().lower())
            for p in allowed_paths
        )

        if not is_allowed:
            return PolicyDecision(
                allowed=False,
                requires_human_approval=False,
                reason=f"Path '{target_path}' is outside permitted scopes: {allowed_paths}",
            )

        return PolicyDecision(allowed=True, requires_human_approval=False, reason="Path is within permitted scope.")

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from src.orchestrator.contracts.context import WorkflowContext
from src.orchestrator.contracts.event import EventType, WorkflowEvent
from src.orchestrator.contracts.report import AgentReport, AgentRole, ReviewerVerdict
from src.orchestrator.state.states import WorkflowState, is_terminal_state
from src.orchestrator.state.transitions import validate_transition

logger = logging.getLogger(__name__)


class WorkflowStateMachine:
    """Pure, deterministic state machine controller driving the orchestration loop."""

    def __init__(
        self,
        context: WorkflowContext,
        event_listener: Optional[Callable[[WorkflowEvent], None]] = None,
    ) -> None:
        self.context = context
        self.current_state = WorkflowState.PENDING
        self.event_listener = event_listener
        self._events: List[WorkflowEvent] = []

    @property
    def is_terminal(self) -> bool:
        """Return True if workflow has reached a terminal state."""
        return is_terminal_state(self.current_state)

    @property
    def is_completed(self) -> bool:
        """Return True if workflow completed successfully."""
        return self.current_state == WorkflowState.COMPLETED

    @property
    def is_failed(self) -> bool:
        """Return True if workflow failed."""
        return self.current_state == WorkflowState.FAILED

    @property
    def is_stopped(self) -> bool:
        """Return True if workflow was stopped or aborted."""
        return self.current_state == WorkflowState.STOPPED

    @property
    def events(self) -> List[WorkflowEvent]:
        """Return immutable copy of all emitted workflow events."""
        return list(self._events)

    def emit_event(
        self,
        event_type: EventType,
        message: str = "",
        payload: Optional[Dict[str, Any]] = None,
        agent_role: Optional[AgentRole] = None,
        agent_name: Optional[str] = None,
    ) -> WorkflowEvent:
        """Create, store, and publish a WorkflowEvent without altering state."""
        event = WorkflowEvent(
            event_type=event_type,
            run_id=self.context.run_id,
            iteration=self.context.iteration_index,
            agent_role=agent_role,
            agent_name=agent_name,
            message=message,
            payload=payload or {},
        )
        self._events.append(event)
        if self.event_listener:
            try:
                self.event_listener(event)
            except Exception as exc:
                logger.warning("Event listener raised unexpected exception: %s", exc)
        return event

    def transition(
        self,
        target_state: WorkflowState,
        event_type: EventType,
        message: str = "",
        payload: Optional[Dict[str, Any]] = None,
        agent_role: Optional[AgentRole] = None,
        agent_name: Optional[str] = None,
    ) -> WorkflowEvent:
        """Validate and execute a deterministic state transition, emitting the associated event."""
        validate_transition(self.current_state, target_state)
        prev_state = self.current_state
        self.current_state = target_state

        event_payload = {
            "from_state": prev_state.value,
            "to_state": target_state.value,
            **(payload or {}),
        }

        return self.emit_event(
            event_type=event_type,
            message=message or f"State transitioned from {prev_state.value} to {target_state.value}",
            payload=event_payload,
            agent_role=agent_role,
            agent_name=agent_name,
        )

    # -------------------------------------------------------------------------
    # Core Deterministic Transition Actions
    # -------------------------------------------------------------------------

    def start(self) -> WorkflowEvent:
        """Transition from PENDING to RUNNING, emitting WORKFLOW_STARTED."""
        return self.transition(
            target_state=WorkflowState.RUNNING,
            event_type=EventType.WORKFLOW_STARTED,
            message=f"Workflow run '{self.context.run_id}' started for task '{self.context.task.title}'",
            payload={"task_id": self.context.task.task_id},
        )

    def start_iteration(self) -> WorkflowEvent:
        """Advance iteration counter and transition into BUILDER_EXECUTING."""
        record = self.context.start_new_iteration()
        self.emit_event(
            event_type=EventType.ITERATION_STARTED,
            message=f"Starting iteration {self.context.iteration_index} (Budget: max {self.context.task.budget.max_iterations})",
            payload={"iteration_index": self.context.iteration_index},
        )
        return self.transition(
            target_state=WorkflowState.BUILDER_EXECUTING,
            event_type=EventType.AGENT_STARTED,
            message=f"Builder executing iteration {self.context.iteration_index}",
            agent_role=AgentRole.BUILDER,
            payload={"iteration_index": self.context.iteration_index},
        )

    def record_progress(
        self,
        agent_role: AgentRole,
        agent_name: str,
        message: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> WorkflowEvent:
        """Emit AGENT_PROGRESS event for live human CLI feedback without mutating state."""
        return self.emit_event(
            event_type=EventType.AGENT_PROGRESS,
            message=message,
            agent_role=agent_role,
            agent_name=agent_name,
            payload=payload,
        )

    def complete_builder(self, report: AgentReport) -> WorkflowEvent:
        """Record Builder report and transition from BUILDER_EXECUTING to BUILDER_COMPLETED."""
        self.context.record_builder_report(report)
        return self.transition(
            target_state=WorkflowState.BUILDER_COMPLETED,
            event_type=EventType.AGENT_COMPLETED,
            message=f"Builder [{report.agent_name}] completed with result {report.builder_result.value}: {report.summary}",
            agent_role=AgentRole.BUILDER,
            agent_name=report.agent_name,
            payload={
                "builder_result": report.builder_result.value,
                "report_id": report.report_id,
            },
        )

    def fail_builder(self, agent_name: str, reason: str) -> WorkflowEvent:
        """Handle fatal Builder failure, transitioning to FAILED."""
        self.emit_event(
            event_type=EventType.AGENT_FAILED,
            message=f"Builder [{agent_name}] failed: {reason}",
            agent_role=AgentRole.BUILDER,
            agent_name=agent_name,
            payload={"reason": reason},
        )
        return self.transition(
            target_state=WorkflowState.FAILED,
            event_type=EventType.WORKFLOW_FAILED,
            message=f"Workflow failed due to Builder failure: {reason}",
            payload={"reason": reason},
        )

    def start_review(self, agent_name: str) -> WorkflowEvent:
        """Transition from BUILDER_COMPLETED to REVIEWING, emitting REVIEW_STARTED."""
        return self.transition(
            target_state=WorkflowState.REVIEWING,
            event_type=EventType.REVIEW_STARTED,
            message=f"Reviewer [{agent_name}] started evaluation for iteration {self.context.iteration_index}",
            agent_role=AgentRole.REVIEWER,
            agent_name=agent_name,
        )

    def evaluate_reviewer_verdict(
        self,
        report: AgentReport,
        requires_human_approval: bool,
    ) -> WorkflowEvent:
        """
        Evaluate reviewer report. Emits REVIEW_PASSED or REVIEW_REJECTED.
        Then transitions according to policy:
          - PASS + approval required -> AWAITING_APPROVAL
          - PASS + no approval required -> COMPLETED
          - REJECT + budget exhausted -> FAILED
          - REJECT + approval required -> AWAITING_APPROVAL
          - REJECT + no approval required -> BUILDER_EXECUTING (next iteration)
        """
        self.context.record_reviewer_report(report)

        if report.is_pass:
            self.emit_event(
                event_type=EventType.REVIEW_PASSED,
                message=f"Reviewer [{report.agent_name}] issued PASS verdict: {report.summary}",
                agent_role=AgentRole.REVIEWER,
                agent_name=report.agent_name,
                payload={"verdict": report.reviewer_verdict.value, "report_id": report.report_id},
            )
            if requires_human_approval:
                return self.transition(
                    target_state=WorkflowState.AWAITING_APPROVAL,
                    event_type=EventType.APPROVAL_REQUESTED,
                    message="Review passed; awaiting PO final approval sign-off",
                    payload={"verdict": report.reviewer_verdict.value},
                )
            else:
                return self.transition(
                    target_state=WorkflowState.COMPLETED,
                    event_type=EventType.WORKFLOW_COMPLETED,
                    message="Review passed and policy auto-completed workflow",
                )
        else:
            # Reviewer rejected or requested changes
            verdict_val = report.reviewer_verdict.value if report.reviewer_verdict else "REJECT"
            self.emit_event(
                event_type=EventType.REVIEW_REJECTED,
                message=f"Reviewer [{report.agent_name}] issued {verdict_val}: {report.summary}",
                agent_role=AgentRole.REVIEWER,
                agent_name=report.agent_name,
                payload={"verdict": verdict_val, "report_id": report.report_id},
            )

            # Check budgets
            if self.context.is_retry_budget_exhausted():
                return self.transition(
                    target_state=WorkflowState.FAILED,
                    event_type=EventType.WORKFLOW_FAILED,
                    message=f"Workflow failed: Retry budget exhausted ({self.context.retry_count}/{self.context.task.budget.max_retries})",
                    payload={"retries": self.context.retry_count},
                )

            if self.context.is_iteration_budget_exhausted():
                return self.transition(
                    target_state=WorkflowState.FAILED,
                    event_type=EventType.WORKFLOW_FAILED,
                    message=f"Workflow failed: Iteration budget exhausted ({self.context.iteration_index}/{self.context.task.budget.max_iterations})",
                    payload={"iterations": self.context.iteration_index},
                )

            if requires_human_approval:
                return self.transition(
                    target_state=WorkflowState.AWAITING_APPROVAL,
                    event_type=EventType.APPROVAL_REQUESTED,
                    message="Review rejected changes; awaiting PO approval before launching next iteration",
                    payload={"verdict": verdict_val},
                )
            else:
                # Auto-continue next iteration
                return self.start_iteration()

    def grant_approval(self, notes: str = "") -> WorkflowEvent:
        """Handle PO approval grant while in AWAITING_APPROVAL."""
        self.emit_event(
            event_type=EventType.APPROVAL_GRANTED,
            message=f"PO granted approval. {notes}".strip(),
            payload={"notes": notes},
        )

        last_rev = self.context.last_reviewer_report
        if last_rev and last_rev.is_pass:
            return self.transition(
                target_state=WorkflowState.COMPLETED,
                event_type=EventType.WORKFLOW_COMPLETED,
                message="Workflow COMPLETED upon PO approval grant",
            )
        else:
            # Approval granted to retry / start next iteration
            return self.start_iteration()

    def reject_approval(self, reason: str = "", fatal: bool = False) -> WorkflowEvent:
        """Handle PO approval rejection while in AWAITING_APPROVAL."""
        self.emit_event(
            event_type=EventType.APPROVAL_REJECTED,
            message=f"PO rejected approval: {reason}".strip(),
            payload={"reason": reason, "fatal": fatal},
        )
        if fatal:
            return self.transition(
                target_state=WorkflowState.FAILED,
                event_type=EventType.WORKFLOW_FAILED,
                message=f"Workflow failed by PO decision: {reason}",
                payload={"reason": reason},
            )
        else:
            return self.transition(
                target_state=WorkflowState.STOPPED,
                event_type=EventType.WORKFLOW_STOPPED,
                message=f"Workflow stopped by PO: {reason}",
                payload={"reason": reason},
            )

    def stop(self, reason: str = "Stopped by user") -> WorkflowEvent:
        """Halt workflow and transition to STOPPED from any valid active state."""
        return self.transition(
            target_state=WorkflowState.STOPPED,
            event_type=EventType.WORKFLOW_STOPPED,
            message=f"Workflow stopped: {reason}",
            payload={"reason": reason},
        )

    def fail(self, reason: str) -> WorkflowEvent:
        """Fail workflow and transition to FAILED from any valid active state."""
        return self.transition(
            target_state=WorkflowState.FAILED,
            event_type=EventType.WORKFLOW_FAILED,
            message=f"Workflow failed: {reason}",
            payload={"reason": reason},
        )

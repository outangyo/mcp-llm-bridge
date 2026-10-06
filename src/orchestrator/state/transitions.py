from __future__ import annotations

from typing import Dict, Set

from src.orchestrator.state.states import WorkflowState


class InvalidStateTransitionError(Exception):
    """Raised when an illegal or out-of-order state transition is attempted."""

    def __init__(
        self,
        current_state: WorkflowState,
        target_state: WorkflowState,
        reason: str = "",
    ) -> None:
        self.current_state = current_state
        self.target_state = target_state
        self.reason = reason
        msg = f"Cannot transition from {current_state.value} to {target_state.value}"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


VALID_TRANSITIONS: Dict[WorkflowState, Set[WorkflowState]] = {
    WorkflowState.PENDING: {
        WorkflowState.RUNNING,
        WorkflowState.STOPPED,
    },
    WorkflowState.RUNNING: {
        WorkflowState.BUILDER_EXECUTING,
        WorkflowState.STOPPED,
        WorkflowState.FAILED,
    },
    WorkflowState.BUILDER_EXECUTING: {
        WorkflowState.BUILDER_COMPLETED,
        WorkflowState.STOPPED,
        WorkflowState.FAILED,
    },
    WorkflowState.BUILDER_COMPLETED: {
        WorkflowState.REVIEWING,
        WorkflowState.AWAITING_APPROVAL,
        WorkflowState.BUILDER_EXECUTING,  # Re-dispatch builder
        WorkflowState.STOPPED,
        WorkflowState.FAILED,
    },
    WorkflowState.REVIEWING: {
        WorkflowState.AWAITING_APPROVAL,
        WorkflowState.BUILDER_EXECUTING,  # Direct retry if policy allows auto-retry
        WorkflowState.COMPLETED,          # Direct complete if policy allows auto-complete
        WorkflowState.STOPPED,
        WorkflowState.FAILED,
    },
    WorkflowState.AWAITING_APPROVAL: {
        WorkflowState.COMPLETED,          # PO approves final pass
        WorkflowState.BUILDER_EXECUTING,  # PO approves retry / iteration
        WorkflowState.STOPPED,          # PO aborts
        WorkflowState.FAILED,           # PO fatal rejection
    },
    WorkflowState.COMPLETED: set(),  # Terminal
    WorkflowState.FAILED: set(),     # Terminal
    WorkflowState.STOPPED: set(),    # Terminal
}


def validate_transition(current_state: WorkflowState, target_state: WorkflowState) -> None:
    """Validate whether transitioning from current_state to target_state is permitted."""
    allowed = VALID_TRANSITIONS.get(current_state, set())
    if target_state not in allowed:
        raise InvalidStateTransitionError(
            current_state=current_state,
            target_state=target_state,
            reason=f"Allowed destination states from {current_state.value} are: {[s.value for s in allowed] or 'None (Terminal)'}",
        )

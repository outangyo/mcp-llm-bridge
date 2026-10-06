from __future__ import annotations

from enum import Enum


class WorkflowState(str, Enum):
    """Discrete, deterministic states of the orchestration lifecycle."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    BUILDER_EXECUTING = "BUILDER_EXECUTING"
    BUILDER_COMPLETED = "BUILDER_COMPLETED"
    REVIEWING = "REVIEWING"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    STOPPED = "STOPPED"


TERMINAL_STATES = {
    WorkflowState.COMPLETED,
    WorkflowState.FAILED,
    WorkflowState.STOPPED,
}


def is_terminal_state(state: WorkflowState) -> bool:
    """Return True if state is an absorbing terminal state with no outgoing transitions."""
    return state in TERMINAL_STATES

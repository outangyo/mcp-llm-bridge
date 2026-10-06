from __future__ import annotations

from src.orchestrator.adapters import (
    BaseAgentAdapter,
    MockBuilder,
    MockReviewer,
    ProgressCallback,
)
from src.orchestrator.cli import CLIRenderer
from src.orchestrator.contracts import (
    AgentReport,
    AgentRole,
    BuilderResult,
    EventType,
    IterationRecord,
    ReviewerVerdict,
    TaskBudget,
    TaskContract,
    WorkflowContext,
    WorkflowEvent,
)
from src.orchestrator.engine import OrchestratorEngine
from src.orchestrator.policy import (
    ExecutionPolicy,
    PolicyDecision,
    PolicyMode,
)
from src.orchestrator.reporting import WorkflowSummary
from src.orchestrator.state import (
    TERMINAL_STATES,
    InvalidStateTransitionError,
    WorkflowState,
    WorkflowStateMachine,
    is_terminal_state,
    validate_transition,
)

__all__ = [
    # Contracts
    "TaskBudget",
    "TaskContract",
    "AgentRole",
    "BuilderResult",
    "ReviewerVerdict",
    "AgentReport",
    "EventType",
    "WorkflowEvent",
    "IterationRecord",
    "WorkflowContext",
    # State Machine
    "WorkflowState",
    "TERMINAL_STATES",
    "is_terminal_state",
    "InvalidStateTransitionError",
    "validate_transition",
    "WorkflowStateMachine",
    # Policy
    "PolicyMode",
    "PolicyDecision",
    "ExecutionPolicy",
    # Adapters
    "BaseAgentAdapter",
    "ProgressCallback",
    "MockBuilder",
    "MockReviewer",
    # CLI & Reporting
    "CLIRenderer",
    "WorkflowSummary",
    # Engine
    "OrchestratorEngine",
]

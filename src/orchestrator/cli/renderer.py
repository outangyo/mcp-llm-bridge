from __future__ import annotations

import sys
from typing import IO, List, Optional

from src.orchestrator.contracts.event import EventType, WorkflowEvent
from src.orchestrator.contracts.report import AgentReport
from src.orchestrator.reporting.summary import WorkflowSummary


class CLIRenderer:
    """
    Live terminal renderer for CMD and PowerShell.
    Consumes strongly-typed WorkflowEvents and renders clean, structured,
    human-readable progress output without dumping internal chain-of-thought.
    """

    def __init__(self, stream: Optional[IO[str]] = None, use_icons: bool = True) -> None:
        self.stream: IO[str] = stream if stream is not None else sys.stdout
        self.use_icons = use_icons
        self.recorded_events: List[WorkflowEvent] = []

    def handle_event(self, event: WorkflowEvent) -> None:
        """Listener callback entry-point for state machine event emissions."""
        self.recorded_events.append(event)
        line = self.format_event(event)
        self.stream.write(line + "\n")
        self.stream.flush()

    def format_event(self, event: WorkflowEvent) -> str:
        """Format a single WorkflowEvent into a styled, human-readable terminal line."""
        time_tag = event.timestamp.split("T")[-1][:8] if "T" in event.timestamp else event.timestamp
        role_label = (
            f"[{event.agent_role.value}:{event.agent_name}]"
            if event.agent_role and event.agent_name
            else f"[{event.agent_role.value}]"
            if event.agent_role
            else "[ORCHESTRATOR]"
        )

        icon_map = {
            EventType.WORKFLOW_STARTED: "🚀" if self.use_icons else "*",
            EventType.ITERATION_STARTED: "🔄" if self.use_icons else ">",
            EventType.AGENT_STARTED: "🛠️" if self.use_icons else ">",
            EventType.AGENT_PROGRESS: "  ↳" if self.use_icons else "   -",
            EventType.AGENT_COMPLETED: "✅" if self.use_icons else "+",
            EventType.AGENT_FAILED: "❌" if self.use_icons else "!",
            EventType.REVIEW_STARTED: "🔍" if self.use_icons else ">",
            EventType.REVIEW_PASSED: "✅" if self.use_icons else "+",
            EventType.REVIEW_REJECTED: "⚠️" if self.use_icons else "!",
            EventType.APPROVAL_REQUESTED: "✋" if self.use_icons else "?",
            EventType.APPROVAL_GRANTED: "👍" if self.use_icons else "+",
            EventType.APPROVAL_REJECTED: "🛑" if self.use_icons else "!",
            EventType.WORKFLOW_COMPLETED: "🎉" if self.use_icons else "*",
            EventType.WORKFLOW_FAILED: "💥" if self.use_icons else "!",
            EventType.WORKFLOW_STOPPED: "⏹️" if self.use_icons else "-",
        }

        icon = icon_map.get(event.event_type, "•")

        if event.event_type == EventType.AGENT_PROGRESS:
            return f"[{time_tag}] {role_label} {icon} Progress: {event.message}"

        if event.event_type == EventType.APPROVAL_REQUESTED:
            return f"[{time_tag}] [GOVERNANCE] {icon} Human Gate: {event.message}"

        if event.event_type in (EventType.APPROVAL_GRANTED, EventType.APPROVAL_REJECTED):
            return f"[{time_tag}] [PO] {icon} {event.message}"

        return f"[{time_tag}] {role_label} {icon} {event.message}"

    def render_report(self, report: AgentReport) -> None:
        """Render an agent report human-summary directly to the stream."""
        self.stream.write("\n" + report.format_human_summary() + "\n")
        self.stream.flush()

    def render_summary(self, summary: WorkflowSummary) -> None:
        """Render final workflow summary to the stream."""
        self.stream.write(summary.format_terminal() + "\n")
        self.stream.flush()

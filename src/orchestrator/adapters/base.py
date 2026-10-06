from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

from src.orchestrator.contracts.context import WorkflowContext
from src.orchestrator.contracts.report import AgentReport, AgentRole

ProgressCallback = Callable[[str, Optional[Dict[str, Any]]], None]


def sanitize_secrets(text: str) -> str:
    """Mask potential API keys or tokens from report narratives and error logs."""
    text = re.sub(r"AIza[0-9A-Za-z-_]{10,}", "[REDACTED_API_KEY]", text)
    text = re.sub(r"sk-[0-9A-Za-z-_]{10,}", "[REDACTED_API_KEY]", text)
    return text


class BaseAgentAdapter(ABC):
    """
    Abstract interface for all agent adapters in the orchestrator.
    Decouples the orchestrator loop from specific agent runtimes (AGY, Gemini, Ray, or Mocks).
    """

    def __init__(self, agent_name: str, agent_role: AgentRole) -> None:
        self.agent_name = agent_name
        self.agent_role = agent_role

    def notify_progress(
        self,
        message: str,
        payload: Optional[Dict[str, Any]] = None,
        callback: Optional[ProgressCallback] = None,
    ) -> None:
        """Helper to invoke the progress callback if one was provided."""
        if callback:
            callback(message, payload)

    @abstractmethod
    def execute(
        self,
        context: WorkflowContext,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> AgentReport:
        """
        Execute an agent turn within the given workflow context.
        Subclasses may emit one or more progress updates via progress_callback before returning the final AgentReport.
        """
        pass

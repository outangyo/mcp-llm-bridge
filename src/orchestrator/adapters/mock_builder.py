from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from src.orchestrator.adapters.base import BaseAgentAdapter, ProgressCallback
from src.orchestrator.contracts.context import WorkflowContext
from src.orchestrator.contracts.report import AgentReport, AgentRole, BuilderResult


class MockBuilder(BaseAgentAdapter):
    """
    Mock implementation of a Builder agent (simulating AGY).
    Can run with scripted reports, customizable progress events, or dynamic context-driven responses.
    """

    def __init__(
        self,
        agent_name: str = "MockAGY",
        scripted_reports: Optional[List[AgentReport]] = None,
        progress_steps: Optional[List[str]] = None,
        fatal_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(agent_name=agent_name, agent_role=AgentRole.BUILDER)
        self.scripted_reports = list(scripted_reports) if scripted_reports else []
        self.progress_steps = (
            progress_steps
            if progress_steps is not None
            else [
                "Inspecting task requirements and allowed scopes",
                "Applying source modifications to target files",
                "Running local validation and syntax checks",
            ]
        )
        self.fatal_error = fatal_error
        self.call_count = 0

    def execute(
        self,
        context: WorkflowContext,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> AgentReport:
        """Execute Builder turn, emitting progress events before returning report."""
        self.call_count += 1

        if self.fatal_error:
            raise self.fatal_error

        # Emit simulated progress updates
        for step in self.progress_steps:
            self.notify_progress(
                message=step,
                payload={"step": step, "iteration": context.iteration_index},
                callback=progress_callback,
            )

        # Return scripted report if available
        if self.scripted_reports:
            report_index = min(self.call_count - 1, len(self.scripted_reports) - 1)
            report = self.scripted_reports[report_index]
            # If report is already provided, ensure agent_name/agent_role are consistent
            return report

        # Default dynamic response based on iteration and reviewer feedback
        last_rev = context.last_reviewer_report
        modified_files = context.task.allowed_paths or ["src/example.py"]

        if last_rev and not last_rev.is_pass:
            summary = f"Refactored implementation to address reviewer findings from iteration {context.iteration_index - 1}"
            findings = f"Fixed issues identified by reviewer: {last_rev.findings}"
            next_step = "Request reviewer re-evaluation of updated changes."
        else:
            summary = f"Completed implementation for task '{context.task.title}' (Iteration {context.iteration_index})"
            findings = f"Created/updated files satisfying acceptance criteria for '{context.task.title}'"
            next_step = "Hand off to Reviewer for verification."

        return AgentReport(
            agent_role=AgentRole.BUILDER,
            agent_name=self.agent_name,
            summary=summary,
            findings=findings,
            builder_result=BuilderResult.SUCCESS,
            blockers=[],
            next_step=next_step,
            artifacts={"modified_files": modified_files},
        )

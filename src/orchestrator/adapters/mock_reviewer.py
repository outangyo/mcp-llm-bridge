from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from src.orchestrator.adapters.base import BaseAgentAdapter, ProgressCallback
from src.orchestrator.contracts.context import WorkflowContext
from src.orchestrator.contracts.report import AgentReport, AgentRole, ReviewerVerdict


class MockReviewer(BaseAgentAdapter):
    """
    Mock implementation of a Reviewer agent (simulating Gemini).
    Supports scripted verdicts (e.g., REJECT then PASS), scripted reports, or dynamic verification.
    """

    def __init__(
        self,
        agent_name: str = "MockGemini",
        scripted_reports: Optional[List[AgentReport]] = None,
        scripted_verdicts: Optional[List[ReviewerVerdict]] = None,
        progress_steps: Optional[List[str]] = None,
        fatal_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(agent_name=agent_name, agent_role=AgentRole.REVIEWER)
        self.scripted_reports = list(scripted_reports) if scripted_reports else []
        self.scripted_verdicts = list(scripted_verdicts) if scripted_verdicts else []
        self.progress_steps = (
            progress_steps
            if progress_steps is not None
            else [
                "Reviewing Builder artifact and changed files",
                "Executing test suite verification",
                "Analyzing code against acceptance criteria",
            ]
        )
        self.fatal_error = fatal_error
        self.call_count = 0

    def execute(
        self,
        context: WorkflowContext,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> AgentReport:
        """Execute Reviewer turn, emitting progress events before returning report."""
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

        # 1. Check scripted reports
        if self.scripted_reports:
            report_index = min(self.call_count - 1, len(self.scripted_reports) - 1)
            return self.scripted_reports[report_index]

        # 2. Check scripted verdicts
        if self.scripted_verdicts:
            verdict_index = min(self.call_count - 1, len(self.scripted_verdicts) - 1)
            verdict = self.scripted_verdicts[verdict_index]
            if verdict == ReviewerVerdict.PASS:
                return AgentReport(
                    agent_role=AgentRole.REVIEWER,
                    agent_name=self.agent_name,
                    summary=f"Reviewer verified iteration {context.iteration_index} successfully.",
                    findings="All acceptance criteria verified; tests passed without regression.",
                    reviewer_verdict=ReviewerVerdict.PASS,
                    blockers=[],
                    next_step="Proceed to final PO approval or completion.",
                )
            else:
                return AgentReport(
                    agent_role=AgentRole.REVIEWER,
                    agent_name=self.agent_name,
                    summary=f"Reviewer rejected iteration {context.iteration_index} changes.",
                    findings="Identified missing edge-case test and formatting discrepancy.",
                    reviewer_verdict=verdict,
                    blockers=["Missing edge-case coverage"],
                    next_step="Builder must address findings and re-submit.",
                )

        # 3. Default: PASS verdict
        return AgentReport(
            agent_role=AgentRole.REVIEWER,
            agent_name=self.agent_name,
            summary=f"Reviewer verified iteration {context.iteration_index} successfully.",
            findings="All task acceptance criteria verified; no defects found.",
            reviewer_verdict=ReviewerVerdict.PASS,
            blockers=[],
            next_step="Proceed to final PO approval or completion.",
        )

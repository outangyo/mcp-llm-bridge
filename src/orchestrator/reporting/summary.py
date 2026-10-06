from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.orchestrator.contracts.context import WorkflowContext


class WorkflowSummary(BaseModel):
    """
    Comprehensive, human-readable and structured final summary report of a completed workflow run.
    Provides complete traceability for the PO and Lead review.
    """

    run_id: str = Field(..., description="Workflow execution run ID")
    task_id: str = Field(..., description="Target task ID")
    task_title: str = Field(..., description="Title of the task")
    status: str = Field(..., description="Final workflow state (e.g., COMPLETED, FAILED, STOPPED)")
    policy_mode: str = Field(..., description="Execution policy mode (SAFE, SUPERVISED, AUTONOMOUS)")
    iterations: int = Field(..., description="Total iterations executed")
    retries: int = Field(..., description="Total reviewer retries requested")
    builder_name: str = Field(..., description="Name/identifier of Builder agent")
    reviewer_name: str = Field(..., description="Name/identifier of Reviewer agent")
    final_review_verdict: str = Field(..., description="Final review verdict (PASS, REJECT, or N/A)")
    modified_files: List[str] = Field(default_factory=list, description="List of all modified files")
    review_findings: str = Field(default="", description="Findings from the final review")
    human_approvals: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Records of human approvals / rejection interactions",
    )
    final_assessment: str = Field(
        default="",
        description="High-level narrative assessment of the workflow outcome",
    )

    @classmethod
    def from_context(
        cls,
        context: WorkflowContext,
        final_state: str,
        policy_mode: str,
        builder_name: str = "AGY",
        reviewer_name: str = "Gemini",
        human_approvals: Optional[List[Dict[str, Any]]] = None,
        assessment: str = "",
    ) -> WorkflowSummary:
        """Construct a WorkflowSummary directly from a completed WorkflowContext."""
        last_rev = context.last_reviewer_report
        verdict = (
            last_rev.reviewer_verdict.value
            if last_rev and last_rev.reviewer_verdict
            else "N/A"
        )
        findings = last_rev.findings if last_rev else ""

        if not assessment:
            if final_state == "COMPLETED":
                assessment = f"Task '{context.task.title}' was successfully implemented and verified in {context.iteration_index} iteration(s)."
            elif final_state == "FAILED":
                assessment = f"Task '{context.task.title}' failed to complete within allocated budget or encountered a fatal error."
            elif final_state == "STOPPED":
                assessment = f"Task '{context.task.title}' was stopped prior to completion."
            else:
                assessment = f"Workflow concluded in state: {final_state}."

        return cls(
            run_id=context.run_id,
            task_id=context.task.task_id,
            task_title=context.task.title,
            status=final_state,
            policy_mode=policy_mode,
            iterations=context.iteration_index,
            retries=context.retry_count,
            builder_name=builder_name,
            reviewer_name=reviewer_name,
            final_review_verdict=verdict,
            modified_files=list(context.modified_files),
            review_findings=findings,
            human_approvals=list(human_approvals or []),
            final_assessment=assessment,
        )

    def format_terminal(self) -> str:
        """Format the summary into a clear terminal banner for CMD and PowerShell."""
        sep = "=" * 70
        sub_sep = "-" * 70
        files_str = "\n".join(f"    - {f}" for f in self.modified_files) if self.modified_files else "    (None)"

        approvals_str = ""
        if self.human_approvals:
            approvals_str = "\n".join(
                f"    - [{app.get('type', 'APPROVAL')}] Granted: {app.get('granted')} | Notes: {app.get('notes', '')}"
                for app in self.human_approvals
            )
        else:
            approvals_str = "    (None)"

        return (
            f"\n{sep}\n"
            f"  WORKFLOW EXECUTION SUMMARY\n"
            f"{sep}\n"
            f"  Task:         [{self.task_id}] {self.task_title}\n"
            f"  Status:       {self.status}\n"
            f"  Policy Mode:  {self.policy_mode}\n"
            f"  Iterations:   {self.iterations}\n"
            f"  Retries:      {self.retries}\n"
            f"  Builder:      {self.builder_name}\n"
            f"  Reviewer:     {self.reviewer_name}\n"
            f"  Verdict:      {self.final_review_verdict}\n"
            f"{sub_sep}\n"
            f"  Modified Files:\n{files_str}\n"
            f"{sub_sep}\n"
            f"  Review Findings:\n    {self.review_findings or '(None)'}\n"
            f"{sub_sep}\n"
            f"  Human Governance:\n{approvals_str}\n"
            f"{sub_sep}\n"
            f"  Final Assessment:\n    {self.final_assessment}\n"
            f"{sep}\n"
        )

    def format_markdown(self) -> str:
        """Format the summary as a GitHub-flavored markdown report."""
        files_list = "\n".join(f"- `{f}`" for f in self.modified_files) if self.modified_files else "- None"
        return (
            f"# Workflow Execution Summary: {self.task_title}\n\n"
            f"- **Run ID:** `{self.run_id}`\n"
            f"- **Task ID:** `{self.task_id}`\n"
            f"- **Final Status:** **{self.status}**\n"
            f"- **Policy Mode:** `{self.policy_mode}`\n"
            f"- **Iterations:** {self.iterations} | **Retries:** {self.retries}\n"
            f"- **Builder:** `{self.builder_name}` | **Reviewer:** `{self.reviewer_name}`\n"
            f"- **Review Verdict:** `{self.final_review_verdict}`\n\n"
            f"## Modified Files\n{files_list}\n\n"
            f"## Review Findings\n{self.review_findings or 'None'}\n\n"
            f"## Final Assessment\n{self.final_assessment}\n"
        )

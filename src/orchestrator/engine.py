from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from src.orchestrator.adapters.base import BaseAgentAdapter
from src.orchestrator.cli.renderer import CLIRenderer
from src.orchestrator.contracts.context import WorkflowContext
from src.orchestrator.contracts.report import AgentRole
from src.orchestrator.contracts.task import TaskContract
from src.orchestrator.policy.execution_policy import ExecutionPolicy, PolicyMode
from src.orchestrator.reporting.summary import WorkflowSummary
from src.orchestrator.state.machine import WorkflowStateMachine
from src.orchestrator.state.states import WorkflowState

logger = logging.getLogger(__name__)

HumanApprovalHandler = Callable[[str, WorkflowContext], bool]


class OrchestratorEngine:
    """
    Deterministic workflow control-plane coordinating Agent Adapters,
    Execution Policy, and State Machine.
    """

    def __init__(
        self,
        task: TaskContract,
        builder: BaseAgentAdapter,
        reviewer: BaseAgentAdapter,
        policy: Optional[ExecutionPolicy] = None,
        renderer: Optional[CLIRenderer] = None,
        approval_handler: Optional[HumanApprovalHandler] = None,
    ) -> None:
        self.task = task
        self.builder = builder
        self.reviewer = reviewer
        self.policy = policy or ExecutionPolicy(mode=PolicyMode.SUPERVISED)
        self.renderer = renderer
        self.approval_handler = approval_handler

        self.context = WorkflowContext(task=self.task)
        self.state_machine = WorkflowStateMachine(
            context=self.context,
            event_listener=self.renderer.handle_event if self.renderer else None,
        )
        self.human_approvals: List[Dict[str, Any]] = []

    def run(self) -> WorkflowSummary:
        """Execute the workflow loop deterministically from start to terminal state."""
        # Start the workflow
        self.state_machine.start()

        max_steps = (self.task.budget.max_iterations + 2) * 10
        steps = 0

        while not self.state_machine.is_terminal:
            steps += 1
            if steps > max_steps:
                self.state_machine.fail(f"Loop guard exceeded maximum allowed steps ({max_steps})")
                break

            current_state = self.state_machine.current_state

            # 1. RUNNING -> Start initial iteration
            if current_state == WorkflowState.RUNNING:
                self.state_machine.start_iteration()
                continue

            # 2. BUILDER_EXECUTING -> Run Builder turn
            if current_state == WorkflowState.BUILDER_EXECUTING:
                try:
                    builder_report = self.builder.execute(
                        context=self.context,
                        progress_callback=lambda msg, payload=None: self.state_machine.record_progress(
                            agent_role=AgentRole.BUILDER,
                            agent_name=self.builder.agent_name,
                            message=msg,
                            payload=payload,
                        ),
                    )
                except Exception as exc:
                    logger.exception("Builder adapter raised unexpected error: %s", exc)
                    self.state_machine.fail_builder(self.builder.agent_name, str(exc))
                    continue

                if self.renderer:
                    self.renderer.render_report(builder_report)

                self.state_machine.complete_builder(builder_report)
                continue

            # 3. BUILDER_COMPLETED -> Hand off to Reviewer
            if current_state == WorkflowState.BUILDER_COMPLETED:
                self.state_machine.start_review(agent_name=self.reviewer.agent_name)
                continue

            # 4. REVIEWING -> Run Reviewer turn and evaluate policy
            if current_state == WorkflowState.REVIEWING:
                try:
                    reviewer_report = self.reviewer.execute(
                        context=self.context,
                        progress_callback=lambda msg, payload=None: self.state_machine.record_progress(
                            agent_role=AgentRole.REVIEWER,
                            agent_name=self.reviewer.agent_name,
                            message=msg,
                            payload=payload,
                        ),
                    )
                except Exception as exc:
                    logger.exception("Reviewer adapter raised unexpected error: %s", exc)
                    self.state_machine.fail(f"Reviewer execution failed: {exc}")
                    continue

                if self.renderer:
                    self.renderer.render_report(reviewer_report)

                # Policy evaluation determines approval requirements
                if reviewer_report.is_pass:
                    decision = self.policy.evaluate_reviewer_pass(self.context, reviewer_report)
                else:
                    decision = self.policy.evaluate_reviewer_reject(self.context, reviewer_report)

                self.state_machine.evaluate_reviewer_verdict(
                    report=reviewer_report,
                    requires_human_approval=decision.requires_human_approval,
                )
                continue

            # 5. AWAITING_APPROVAL -> Solicit human PO decision
            if current_state == WorkflowState.AWAITING_APPROVAL:
                reason = "PO final approval required" if self.context.last_reviewer_report and self.context.last_reviewer_report.is_pass else "PO retry approval required"
                approved = True
                notes = ""

                if self.approval_handler:
                    approved = self.approval_handler(reason, self.context)
                elif self.policy.mode in (PolicyMode.SAFE, PolicyMode.SUPERVISED):
                    # Default: approve if no explicit rejection handler is registered
                    approved = True
                    notes = "Auto-approved by default handler"

                self.human_approvals.append({
                    "type": "PO_APPROVAL",
                    "iteration": self.context.iteration_index,
                    "granted": approved,
                    "notes": notes,
                })

                if approved:
                    self.state_machine.grant_approval(notes=notes)
                else:
                    self.state_machine.reject_approval(reason="PO rejected changes", fatal=False)
                continue

        # Terminal state reached -> generate final summary
        summary = WorkflowSummary.from_context(
            context=self.context,
            final_state=self.state_machine.current_state.value,
            policy_mode=self.policy.mode.value,
            builder_name=self.builder.agent_name,
            reviewer_name=self.reviewer.agent_name,
            human_approvals=self.human_approvals,
        )

        if self.renderer:
            self.renderer.render_summary(summary)

        return summary

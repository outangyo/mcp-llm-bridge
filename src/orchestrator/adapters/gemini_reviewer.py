from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional

from src.orchestrator.adapters.base import BaseAgentAdapter, ProgressCallback
from src.orchestrator.contracts.context import WorkflowContext
from src.orchestrator.contracts.report import (
    AgentReport,
    AgentRole,
    ReviewerVerdict,
)
from src.providers.base import BaseLLMProvider, LLMProviderError
from src.providers.factory import create_llm_provider
from src.providers.gemini import GeminiProvider

logger = logging.getLogger(__name__)


def sanitize_secrets(text: str) -> str:
    """Mask potential API keys or tokens from report narratives and error logs."""
    # Mask Google Gemini keys (AIza...)
    text = re.sub(r"AIza[0-9A-Za-z-_]{10,}", "[REDACTED_API_KEY]", text)
    # Mask OpenAI / Generic sk- keys
    text = re.sub(r"sk-[0-9A-Za-z-_]{10,}", "[REDACTED_API_KEY]", text)
    return text


def run_async_sync(coro: Any) -> str:
    """Safely execute an asynchronous coroutine from a synchronous call-site."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


class GeminiReviewerAdapter(BaseAgentAdapter):
    """
    Real Reviewer adapter delegating code and artifact evaluation to Google Gemini
    via the existing M2 LLM provider abstraction (GeminiProvider / BaseLLMProvider).
    """

    def __init__(
        self,
        agent_name: str = "Gemini",
        provider: Optional[BaseLLMProvider] = None,
        model: Optional[str] = None,
    ) -> None:
        super().__init__(agent_name=agent_name, agent_role=AgentRole.REVIEWER)
        if provider is not None:
            self._provider = provider
        else:
            try:
                self._provider = create_llm_provider()
            except Exception:
                self._provider = GeminiProvider(model=model)

    @property
    def provider(self) -> BaseLLMProvider:
        """The underlying LLM provider instance."""
        return self._provider

    def build_review_context(self, context: WorkflowContext) -> tuple[str, str]:
        """
        Build (question, context_str) tuple formatted for BaseLLMProvider.generate_text.
        Context provides complete task specification and Builder report.
        """
        task = context.task
        last_builder = context.last_builder_report

        builder_summary = last_builder.summary if last_builder else "No builder report available"
        builder_findings = last_builder.findings if last_builder else "None"
        builder_result = last_builder.builder_result.value if last_builder and last_builder.builder_result else "UNKNOWN"
        modified_files = last_builder.artifacts.get("modified_files", []) if last_builder else []

        criteria_str = "\n".join(f"- {c}" for c in task.acceptance_criteria) if task.acceptance_criteria else "- Verify implementation accuracy"
        paths_str = ", ".join(task.allowed_paths) if task.allowed_paths else "No path restrictions"

        context_str = f"""TASK SPECIFICATION:
Task ID: {task.task_id}
Title: {task.title}
Description: {task.description}
Allowed Scopes: {paths_str}
Iteration: {context.iteration_index} (Retries: {context.retry_count})

ACCEPTANCE CRITERIA:
{criteria_str}

BUILDER OUTCOME (Iteration {context.iteration_index}):
Result: {builder_result}
Summary: {builder_summary}
Findings: {builder_findings}
Modified Files: {', '.join(modified_files) if modified_files else 'None'}
"""

        question = """You are the independent REVIEWER / CHALLENGER agent in an automated multi-agent workflow.
Evaluate the Builder's outcome against the task acceptance criteria, architectural integrity, and security constraints.

Decision Options:
- PASS: All criteria are fully satisfied without defects.
- REJECT: Acceptance criteria are failed, severe defect found, or security bounds violated.
- REQUEST_CHANGES: Minor improvements, missing test coverage, or formatting adjustments required.

Respond with a JSON code block matching this schema:
```json
{
  "reviewer_verdict": "PASS" | "REJECT" | "REQUEST_CHANGES",
  "summary": "<Short verdict explanation>",
  "findings": "<Detailed analysis of review and verification findings>",
  "blockers": ["<Any issues blocking acceptance>"],
  "next_step": "<Actionable instruction for PO or Builder>"
}
```
"""
        return question.strip(), context_str.strip()

    def execute(
        self,
        context: WorkflowContext,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> AgentReport:
        """Execute Reviewer evaluation turn through the M2 provider abstraction."""
        self.notify_progress(
            message="Assembling review criteria and builder artifacts",
            payload={"iteration": context.iteration_index},
            callback=progress_callback,
        )

        question, context_str = self.build_review_context(context)

        self.notify_progress(
            message=f"Requesting review from {self.agent_name} via M2 Provider abstraction",
            callback=progress_callback,
        )

        try:
            coro = self._provider.generate_text(question=question, context=context_str)
            raw_response = run_async_sync(coro)
        except LLMProviderError as exc:
            safe_err = sanitize_secrets(str(exc))
            logger.warning("LLM provider error during review turn: %s", safe_err)
            self.notify_progress(
                message=f"Review failed: {safe_err}",
                payload={"error": safe_err},
                callback=progress_callback,
            )
            return AgentReport(
                agent_role=AgentRole.REVIEWER,
                agent_name=self.agent_name,
                summary=f"Review failed due to LLM provider error: {safe_err}",
                findings=safe_err,
                reviewer_verdict=ReviewerVerdict.REJECT,
                blockers=[safe_err],
                next_step="Check provider configuration or credentials and retry.",
            )
        except Exception as exc:
            safe_err = sanitize_secrets(str(exc))
            logger.exception("Unexpected exception in Gemini reviewer adapter: %s", safe_err)
            self.notify_progress(
                message=f"Review encountered unexpected error: {safe_err}",
                callback=progress_callback,
            )
            return AgentReport(
                agent_role=AgentRole.REVIEWER,
                agent_name=self.agent_name,
                summary=f"Reviewer encountered unexpected error: {safe_err}",
                findings=safe_err,
                reviewer_verdict=ReviewerVerdict.REJECT,
                blockers=[safe_err],
                next_step="Investigate reviewer adapter error.",
            )

        self.notify_progress(
            message="Parsing reviewer verdict and structured findings",
            callback=progress_callback,
        )

        return self._parse_reviewer_response(raw_response)

    def _parse_reviewer_response(self, raw_response: str) -> AgentReport:
        """Parse provider response into typed AgentReport."""
        sanitized = sanitize_secrets(raw_response.strip())
        extracted = self._extract_json_block(sanitized)

        verdict = ReviewerVerdict.PASS
        summary = ""
        findings = ""
        blockers: List[str] = []
        next_step = "Proceed to final PO approval or next iteration."

        if extracted:
            v_str = str(extracted.get("reviewer_verdict", "PASS")).upper().strip()
            if v_str == "PASS":
                verdict = ReviewerVerdict.PASS
            elif v_str == "REJECT":
                verdict = ReviewerVerdict.REJECT
            elif v_str in ("REQUEST_CHANGES", "REQUESTCHANGES"):
                verdict = ReviewerVerdict.REQUEST_CHANGES
            else:
                verdict = ReviewerVerdict.REJECT

            summary = str(extracted.get("summary", "")).strip()
            findings = str(extracted.get("findings", "")).strip()
            raw_b = extracted.get("blockers", [])
            if isinstance(raw_b, list):
                blockers = [str(b) for b in raw_b]
            next_step = str(extracted.get("next_step", next_step)).strip()

        if not summary:
            # Fallback heuristic if no structured JSON could be parsed
            upper_raw = sanitized.upper()
            if "REJECT" in upper_raw:
                verdict = ReviewerVerdict.REJECT
                summary = "Reviewer rejected changes based on unstructured response."
            elif "REQUEST_CHANGES" in upper_raw or "CHANGES REQUESTED" in upper_raw:
                verdict = ReviewerVerdict.REQUEST_CHANGES
                summary = "Reviewer requested changes based on unstructured response."
            else:
                verdict = ReviewerVerdict.PASS
                summary = "Reviewer accepted changes based on unstructured response."
            findings = sanitized[:1000]

        if not next_step:
            next_step = (
                "Proceed to PO approval sign-off."
                if verdict == ReviewerVerdict.PASS
                else "Builder must revise implementation according to findings."
            )

        return AgentReport(
            agent_role=AgentRole.REVIEWER,
            agent_name=self.agent_name,
            summary=summary,
            findings=findings,
            reviewer_verdict=verdict,
            blockers=blockers,
            next_step=next_step,
            artifacts={"raw_response": sanitized[:500]},
        )

    def _extract_json_block(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON code block from markdown or string."""
        matches = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        for match in reversed(matches):
            try:
                return json.loads(match)
            except Exception:
                continue

        raw_match = re.search(r"(\{\s*\"reviewer_verdict\":.*?\})", text, re.DOTALL)
        if raw_match:
            try:
                return json.loads(raw_match.group(1))
            except Exception:
                pass

        return None

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

logger = logging.getLogger(__name__)


def sanitize_secrets(text: str) -> str:
    """Mask potential API keys or tokens from report narratives and error logs."""
    # Mask Google Gemini keys (AIza...)
    text = re.sub(r"AIza[0-9A-Za-z-_]{10,}", "[REDACTED_API_KEY]", text)
    # Mask OpenAI / Generic sk- keys
    text = re.sub(r"sk-[0-9A-Za-z-_]{10,}", "[REDACTED_API_KEY]", text)
    return text


import threading

_GLOBAL_LOOP: Optional[asyncio.AbstractEventLoop] = None
_LOOP_THREAD: Optional[threading.Thread] = None
_LOOP_LOCK = threading.Lock()


def _get_background_loop() -> asyncio.AbstractEventLoop:
    global _GLOBAL_LOOP, _LOOP_THREAD
    with _LOOP_LOCK:
        if _GLOBAL_LOOP is None or _GLOBAL_LOOP.is_closed():
            _GLOBAL_LOOP = asyncio.new_event_loop()
            _LOOP_THREAD = threading.Thread(target=_GLOBAL_LOOP.run_forever, daemon=True)
            _LOOP_THREAD.start()
        return _GLOBAL_LOOP


def run_async_sync(coro: Any) -> str:
    """Safely execute an asynchronous coroutine on a persistent background event loop."""
    loop = _get_background_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


class GeminiReviewerAdapter(BaseAgentAdapter):
    """
    Real Reviewer adapter delegating code and artifact evaluation to an LLM
    via the canonical M2 provider factory (create_llm_provider() / BaseLLMProvider).
    Strictly preserves the provider abstraction boundary and enforces fail-closed parsing.
    """

    def __init__(
        self,
        agent_name: str = "Gemini",
        provider: Optional[BaseLLMProvider] = None,
    ) -> None:
        super().__init__(agent_name=agent_name, agent_role=AgentRole.REVIEWER)
        self._provider: Optional[BaseLLMProvider] = provider
        self._provider_init_error: Optional[str] = None

        if self._provider is None:
            try:
                self._provider = create_llm_provider()
            except Exception as exc:
                self._provider_init_error = sanitize_secrets(str(exc))
                logger.warning(
                    "create_llm_provider() failed during Reviewer adapter initialization: %s",
                    self._provider_init_error,
                )

    @property
    def provider(self) -> Optional[BaseLLMProvider]:
        """The underlying LLM provider instance, if successfully resolved."""
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

        # Enforce provider factory resolution without bypassing abstraction
        if self._provider is None:
            # Re-attempt in case environment changed dynamically
            if not self._provider_init_error:
                try:
                    self._provider = create_llm_provider()
                except Exception as exc:
                    self._provider_init_error = sanitize_secrets(str(exc))

            if self._provider is None:
                err_msg = self._provider_init_error or "LLM provider could not be initialized from create_llm_provider()."
                logger.error("Reviewer adapter execution halted due to provider initialization error: %s", err_msg)
                return AgentReport(
                    agent_role=AgentRole.REVIEWER,
                    agent_name=self.agent_name,
                    summary=f"Review failed: Provider initialization error ({err_msg})",
                    findings=err_msg,
                    reviewer_verdict=ReviewerVerdict.REJECT,
                    blockers=[f"ProviderInitializationError: {err_msg}"],
                    next_step="Configure LLM_PROVIDER or supply a valid provider instance.",
                )

        question, context_str = self.build_review_context(context)

        self.notify_progress(
            message=f"Requesting review from {self.agent_name} via M2 Provider abstraction",
            callback=progress_callback,
        )

        raw_response = None
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                coro = self._provider.generate_text(question=question, context=context_str)
                raw_response = run_async_sync(coro)
                break
            except LLMProviderError as exc:
                err_text = str(exc)
                if ("503" in err_text or "429" in err_text or "UNAVAILABLE" in err_text) and attempt < max_attempts - 1:
                    logger.info("Transient provider error (%s), retrying attempt %s/%s in 3s...", err_text, attempt + 2, max_attempts)
                    import time
                    time.sleep(3 * (attempt + 1))
                    continue
                safe_err = sanitize_secrets(err_text)
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

        if raw_response is None:
            raw_response = ""

        self.notify_progress(
            message="Parsing reviewer verdict and structured findings",
            callback=progress_callback,
        )

        return self._parse_reviewer_response(raw_response)

    def _parse_reviewer_response(self, raw_response: str) -> AgentReport:
        """
        Parse provider response into typed AgentReport.
        Strictly enforces 'fail-closed' semantics:
        Any malformed, empty, or ambiguous response defaults to REJECT.
        """
        sanitized = sanitize_secrets(raw_response.strip())
        if not sanitized:
            return AgentReport(
                agent_role=AgentRole.REVIEWER,
                agent_name=self.agent_name,
                summary="Review failed: Provider returned an empty response.",
                findings="Empty response body received from LLM provider.",
                reviewer_verdict=ReviewerVerdict.REJECT,
                blockers=["Reviewer response could not be parsed into a valid verdict."],
                next_step="Reviewer must re-evaluate and provide a valid structured report.",
                artifacts={"raw_response": ""},
            )

        extracted = self._extract_json_block(sanitized)

        verdict: Optional[ReviewerVerdict] = None
        summary = ""
        findings = ""
        blockers: List[str] = []
        next_step = ""

        if extracted and isinstance(extracted, dict):
            raw_v = str(extracted.get("reviewer_verdict", "")).strip().upper()
            if raw_v == "PASS":
                verdict = ReviewerVerdict.PASS
            elif raw_v == "REJECT":
                verdict = ReviewerVerdict.REJECT
            elif raw_v in ("REQUEST_CHANGES", "REQUESTCHANGES"):
                verdict = ReviewerVerdict.REQUEST_CHANGES
            else:
                verdict = ReviewerVerdict.REJECT
                blockers.append(f"Unrecognized structured reviewer verdict: '{raw_v}'.")

            summary = str(extracted.get("summary", "")).strip()
            findings = str(extracted.get("findings", "")).strip()
            raw_b = extracted.get("blockers", [])
            if isinstance(raw_b, list):
                for b in raw_b:
                    b_str = str(b).strip()
                    if b_str and b_str not in blockers:
                        blockers.append(b_str)
            next_step = str(extracted.get("next_step", "")).strip()

        # If no valid structured verdict was extracted, evaluate explicit unstructured format
        if verdict is None:
            # Fail closed: Only accept explicit, unambiguous verdict keywords
            upper_raw = sanitized.upper()
            has_pass = bool(re.search(r"\b(PASS|VERDICT:\s*PASS)\b", upper_raw))
            has_reject = bool(re.search(r"\b(REJECT|VERDICT:\s*REJECT)\b", upper_raw))
            has_changes = bool(re.search(r"\b(REQUEST_CHANGES|REQUEST\s+CHANGES|CHANGES\s+REQUESTED)\b", upper_raw))

            # Exactly one unambiguous verdict must be present
            matched_verdicts = sum([has_pass, has_reject, has_changes])
            if matched_verdicts == 1:
                if has_pass:
                    verdict = ReviewerVerdict.PASS
                    summary = "Reviewer explicitly indicated PASS in unstructured response."
                elif has_reject:
                    verdict = ReviewerVerdict.REJECT
                    summary = "Reviewer explicitly indicated REJECT in unstructured response."
                elif has_changes:
                    verdict = ReviewerVerdict.REQUEST_CHANGES
                    summary = "Reviewer explicitly requested changes in unstructured response."
                findings = sanitized[:1000]
            else:
                # Ambiguous, conflicting, or missing verdict -> FAIL CLOSED TO REJECT
                verdict = ReviewerVerdict.REJECT
                summary = "Review failed: Reviewer response could not be parsed into a valid verdict."
                findings = sanitized[:1000]
                blockers.append("Reviewer response could not be parsed into a valid verdict.")
                if matched_verdicts > 1:
                    blockers.append("Conflicting verdict keywords detected in reviewer response.")

        # Ensure summary is populated
        if not summary:
            if verdict == ReviewerVerdict.REJECT and blockers and "could not be parsed" in blockers[0]:
                summary = "Review failed: Reviewer response could not be parsed into a valid verdict."
            else:
                summary = f"Reviewer completed evaluation with verdict {verdict.value}."

        # Ensure next_step is populated
        if not next_step:
            if verdict == ReviewerVerdict.PASS:
                next_step = "Proceed to PO final approval sign-off."
            elif verdict == ReviewerVerdict.REQUEST_CHANGES:
                next_step = "Builder must revise implementation according to reviewer findings."
            else:
                next_step = "Builder or PO must review blockers and address reviewer rejection."

        return AgentReport(
            agent_role=AgentRole.REVIEWER,
            agent_name=self.agent_name,
            summary=summary,
            findings=findings or sanitized[:1000],
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

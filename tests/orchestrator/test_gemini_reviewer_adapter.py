from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.orchestrator.adapters.gemini_reviewer import (
    GeminiReviewerAdapter,
    sanitize_secrets,
)
from src.orchestrator.contracts.context import WorkflowContext
from src.orchestrator.contracts.report import (
    AgentReport,
    AgentRole,
    BuilderResult,
    ReviewerVerdict,
)
from src.orchestrator.contracts.task import TaskBudget, TaskContract
from src.providers.base import BaseLLMProvider, LLMProviderError


class DummyProvider(BaseLLMProvider):
    def __init__(self, response_text: str = "", raises_exc: Exception | None = None) -> None:
        self.response_text = response_text
        self.raises_exc = raises_exc
        self.call_count = 0
        self.last_question = ""
        self.last_context = ""

    async def generate_text(self, question: str, context: str | None = None) -> str:
        self.call_count += 1
        self.last_question = question
        self.last_context = context or ""
        if self.raises_exc:
            raise self.raises_exc
        return self.response_text


@pytest.fixture
def workflow_context() -> WorkflowContext:
    task = TaskContract(
        title="Review Feature Z",
        description="Verify correctness and security of Feature Z",
        acceptance_criteria=["Tests must pass", "No secrets committed"],
        allowed_paths=["src/z.py"],
        budget=TaskBudget(max_iterations=2, max_retries=1),
    )
    ctx = WorkflowContext(task=task)
    ctx.start_new_iteration()
    # Builder report
    builder_report = AgentReport(
        agent_role=AgentRole.BUILDER,
        agent_name="AGY",
        summary="Implemented Feature Z",
        findings="Created src/z.py with full test coverage",
        builder_result=BuilderResult.SUCCESS,
        artifacts={"modified_files": ["src/z.py"]},
        next_step="Review",
    )
    ctx.record_builder_report(builder_report)
    return ctx


def test_sanitize_secrets():
    raw = "Error with key AIzaSyD91023812038102381203812038 and sk-proj-12345678901234567890"
    sanitized = sanitize_secrets(raw)
    assert "[REDACTED_API_KEY]" in sanitized
    assert "AIzaSy" not in sanitized
    assert "sk-proj" not in sanitized


def test_gemini_reviewer_build_review_context(workflow_context: WorkflowContext):
    adapter = GeminiReviewerAdapter(provider=DummyProvider())
    question, context_str = adapter.build_review_context(workflow_context)

    assert "Review Feature Z" in context_str
    assert "Tests must pass" in context_str
    assert "Implemented Feature Z" in context_str
    assert "src/z.py" in context_str
    assert "reviewer_verdict" in question


def test_gemini_reviewer_verdict_pass(workflow_context: WorkflowContext):
    review_json = json.dumps({
        "reviewer_verdict": "PASS",
        "summary": "All acceptance criteria verified",
        "findings": "Code cleanly implements requirements and tests pass",
        "blockers": [],
        "next_step": "Ready for PO approval",
    })
    provider = DummyProvider(response_text=f"```json\n{review_json}\n```")
    adapter = GeminiReviewerAdapter(provider=provider)

    report = adapter.execute(workflow_context)

    assert report.agent_role == AgentRole.REVIEWER
    assert report.reviewer_verdict == ReviewerVerdict.PASS
    assert report.is_pass is True
    assert report.summary == "All acceptance criteria verified"
    assert report.blockers == []
    assert provider.call_count == 1


def test_gemini_reviewer_verdict_reject(workflow_context: WorkflowContext):
    review_json = json.dumps({
        "reviewer_verdict": "REJECT",
        "summary": "Security violation detected",
        "findings": "Hard-coded credentials found in source file",
        "blockers": ["Hard-coded credentials"],
        "next_step": "Builder must remove sensitive credentials",
    })
    provider = DummyProvider(response_text=f"```json\n{review_json}\n```")
    adapter = GeminiReviewerAdapter(provider=provider)

    report = adapter.execute(workflow_context)

    assert report.reviewer_verdict == ReviewerVerdict.REJECT
    assert report.is_pass is False
    assert "Security violation detected" in report.summary
    assert "Hard-coded credentials" in report.blockers


def test_gemini_reviewer_verdict_request_changes(workflow_context: WorkflowContext):
    review_json = json.dumps({
        "reviewer_verdict": "REQUEST_CHANGES",
        "summary": "Missing unit test for edge cases",
        "findings": "Core logic looks good but empty string case is untested",
        "blockers": ["Missing test case"],
        "next_step": "Add edge-case test in test_z.py",
    })
    provider = DummyProvider(response_text=f"```json\n{review_json}\n```")
    adapter = GeminiReviewerAdapter(provider=provider)

    report = adapter.execute(workflow_context)

    assert report.reviewer_verdict == ReviewerVerdict.REQUEST_CHANGES
    assert report.is_pass is False
    assert "Missing unit test" in report.summary


def test_gemini_reviewer_fallback_unstructured_response(workflow_context: WorkflowContext):
    # Reviewer responds in plain conversational text containing verdict keyword
    provider = DummyProvider(response_text="I have reviewed the changes. All acceptance criteria PASS.")
    adapter = GeminiReviewerAdapter(provider=provider)

    report = adapter.execute(workflow_context)

    assert report.reviewer_verdict == ReviewerVerdict.PASS
    assert report.is_pass is True
    assert "accepted changes" in report.summary.lower()


def test_gemini_reviewer_llm_provider_error_handling(workflow_context: WorkflowContext):
    # Simulate a provider error containing a mock key string
    err = LLMProviderError("Gemini API authentication failed for AIzaSySecretKeyHere1234567890123")
    provider = DummyProvider(raises_exc=err)
    adapter = GeminiReviewerAdapter(provider=provider)

    report = adapter.execute(workflow_context)

    assert report.reviewer_verdict == ReviewerVerdict.REJECT
    assert report.is_pass is False
    assert "LLM provider error" in report.summary
    # Ensure sensitive key was sanitized
    assert "AIzaSySecretKey" not in report.findings
    assert "[REDACTED_API_KEY]" in report.findings


def test_gemini_reviewer_progress_reporting(workflow_context: WorkflowContext):
    progress_updates = []

    def on_progress(msg: str, payload=None):
        progress_updates.append(msg)

    review_json = json.dumps({
        "reviewer_verdict": "PASS",
        "summary": "OK",
        "findings": "OK",
        "blockers": [],
        "next_step": "Done",
    })
    provider = DummyProvider(response_text=review_json)
    adapter = GeminiReviewerAdapter(provider=provider)
    adapter.execute(workflow_context, progress_callback=on_progress)

    assert len(progress_updates) >= 3
    assert any("Assembling review criteria" in m for m in progress_updates)
    assert any("Requesting review" in m for m in progress_updates)
    assert any("Parsing reviewer verdict" in m for m in progress_updates)

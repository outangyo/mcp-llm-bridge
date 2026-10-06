import pytest
from pydantic import ValidationError

from src.orchestrator.contracts.report import (
    AgentReport,
    AgentRole,
    BuilderResult,
    ReviewerVerdict,
)


def test_builder_report_success():
    report = AgentReport(
        agent_role=AgentRole.BUILDER,
        agent_name="AGY",
        summary="Modified database pooling configuration",
        findings="Identified missing connection timeout",
        builder_result=BuilderResult.SUCCESS,
        next_step="Send diff to Reviewer",
        artifacts={"modified_files": ["src/db.py"]},
    )
    assert report.is_success is True
    assert report.builder_result == BuilderResult.SUCCESS
    assert report.reviewer_verdict is None
    assert report.blockers == []
    assert "src/db.py" in report.artifacts["modified_files"]


def test_reviewer_report_pass():
    report = AgentReport(
        agent_role=AgentRole.REVIEWER,
        agent_name="Gemini",
        summary="Code review passed without regressions",
        findings="Timeout logic verified and clean",
        reviewer_verdict=ReviewerVerdict.PASS,
        next_step="Request PO approval for final sign-off",
    )
    assert report.is_pass is True
    assert report.reviewer_verdict == ReviewerVerdict.PASS
    assert report.builder_result is None


def test_reviewer_report_reject():
    report = AgentReport(
        agent_role=AgentRole.REVIEWER,
        agent_name="Gemini",
        summary="Code review rejected due to missing edge case",
        findings="Negative timeout values not handled",
        reviewer_verdict=ReviewerVerdict.REJECT,
        blockers=["Invalid input error not caught"],
        next_step="Builder must handle negative integers",
    )
    assert report.is_pass is False
    assert report.reviewer_verdict == ReviewerVerdict.REJECT
    assert len(report.blockers) == 1


def test_builder_report_missing_result_raises_validation_error():
    with pytest.raises(ValidationError, match="must specify 'builder_result'"):
        AgentReport(
            agent_role=AgentRole.BUILDER,
            agent_name="AGY",
            summary="Done something",
            findings="Found something",
            next_step="Next",
        )


def test_reviewer_report_missing_verdict_raises_validation_error():
    with pytest.raises(ValidationError, match="must specify 'reviewer_verdict'"):
        AgentReport(
            agent_role=AgentRole.REVIEWER,
            agent_name="Gemini",
            summary="Reviewed code",
            findings="Looks okay",
            next_step="Next",
        )


def test_agent_report_format_human_summary():
    report = AgentReport(
        agent_role=AgentRole.BUILDER,
        agent_name="AGY",
        summary="Implemented feature X",
        findings="Found edge case Y",
        builder_result=BuilderResult.SUCCESS,
        blockers=["Waiting on dependency Z"],
        next_step="Run integration test",
    )
    summary_text = report.format_human_summary()
    assert "[BUILDER:AGY]" in summary_text
    assert "Result: SUCCESS" in summary_text
    assert "Implemented feature X" in summary_text
    assert "Found edge case Y" in summary_text
    assert "Waiting on dependency Z" in summary_text
    assert "Run integration test" in summary_text


def test_agent_report_deterministic_json_roundtrip():
    original = AgentReport(
        agent_role=AgentRole.REVIEWER,
        agent_name="Gemini",
        summary="Review completed",
        findings="All criteria met",
        reviewer_verdict=ReviewerVerdict.PASS,
        next_step="Complete workflow",
        artifacts={"coverage": 98.5},
    )
    json_str = original.to_json()
    reconstructed = AgentReport.from_json(json_str)

    assert reconstructed.report_id == original.report_id
    assert reconstructed.agent_role == original.agent_role
    assert reconstructed.agent_name == original.agent_name
    assert reconstructed.reviewer_verdict == original.reviewer_verdict
    assert reconstructed.artifacts["coverage"] == 98.5

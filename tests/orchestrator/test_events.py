from src.orchestrator.contracts.event import EventType, WorkflowEvent
from src.orchestrator.contracts.report import AgentRole


def test_required_event_types_exist():
    # Verify all 20 required EventTypes from LEAD specifications
    required_events = [
        "TASK_CREATED",
        "WORKFLOW_STARTED",
        "AGENT_STARTED",
        "AGENT_PROGRESS",
        "AGENT_COMPLETED",
        "AGENT_FAILED",
        "TEST_STARTED",
        "TEST_PASSED",
        "TEST_FAILED",
        "REVIEW_STARTED",
        "REVIEW_PASSED",
        "REVIEW_REJECTED",
        "POLICY_BLOCKED",
        "APPROVAL_REQUESTED",
        "APPROVAL_GRANTED",
        "APPROVAL_REJECTED",
        "ITERATION_STARTED",
        "ITERATION_COMPLETED",
        "WORKFLOW_COMPLETED",
        "WORKFLOW_FAILED",
        "WORKFLOW_STOPPED",
    ]
    for ev_name in required_events:
        assert hasattr(EventType, ev_name), f"Missing required EventType: {ev_name}"
        assert getattr(EventType, ev_name).value == ev_name


def test_workflow_event_creation_and_cli_formatting():
    event = WorkflowEvent(
        event_type=EventType.AGENT_PROGRESS,
        run_id="run_123",
        iteration=1,
        agent_role=AgentRole.BUILDER,
        agent_name="AGY",
        message="Refactoring src/server.py line 45",
        payload={"file": "src/server.py", "line": 45},
    )
    assert event.event_id is not None
    assert event.event_type == EventType.AGENT_PROGRESS
    assert event.run_id == "run_123"
    assert event.iteration == 1
    assert event.agent_role == AgentRole.BUILDER
    assert event.payload["line"] == 45

    cli_line = event.format_cli_line()
    assert "[BUILDER]" in cli_line
    assert "AGENT_PROGRESS" in cli_line
    assert "Refactoring src/server.py line 45" in cli_line


def test_workflow_event_deterministic_json_roundtrip():
    original = WorkflowEvent(
        event_type=EventType.REVIEW_PASSED,
        run_id="run_456",
        iteration=2,
        agent_role=AgentRole.REVIEWER,
        agent_name="Gemini",
        message="Review PASS issued",
        payload={"confidence": 0.99},
    )
    json_str = original.to_json()
    reconstructed = WorkflowEvent.from_json(json_str)

    assert reconstructed.event_id == original.event_id
    assert reconstructed.event_type == original.event_type
    assert reconstructed.run_id == original.run_id
    assert reconstructed.iteration == original.iteration
    assert reconstructed.agent_role == original.agent_role
    assert reconstructed.payload == original.payload

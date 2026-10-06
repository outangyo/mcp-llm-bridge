import asyncio
import pytest

from mcp.server.mcpserver.exceptions import ToolError
from src.server import mcp, ask_gpt


def test_ask_gpt_is_registered():
    """Verify that ask_gpt is registered as an MCP tool on the server."""
    tools = asyncio.run(mcp.list_tools())
    tool_names = [tool.name for tool in tools]
    assert "ask_gpt" in tool_names


def test_ask_gpt_schema_question_is_required_string():
    """Verify that 'question' is a required string in the tool schema."""
    tools = asyncio.run(mcp.list_tools())
    tool = next(t for t in tools if t.name == "ask_gpt")
    schema = tool.input_schema

    assert "required" in schema
    assert "question" in schema["required"]
    assert "question" in schema["properties"]
    assert schema["properties"]["question"]["type"] == "string"


def test_ask_gpt_schema_context_is_optional_string_or_null():
    """Verify that 'context' is an optional string or null in the tool schema."""
    tools = asyncio.run(mcp.list_tools())
    tool = next(t for t in tools if t.name == "ask_gpt")
    schema = tool.input_schema

    # context must NOT be in required fields
    required = schema.get("required", [])
    assert "context" not in required

    # context property must exist and accept string or null
    assert "context" in schema["properties"]
    context_prop = schema["properties"]["context"]
    assert "anyOf" in context_prop
    types = [item.get("type") for item in context_prop["anyOf"]]
    assert "string" in types
    assert "null" in types


def test_ask_gpt_empty_question_raises_tool_error():
    """Verify that calling ask_gpt with an empty question raises ToolError."""
    with pytest.raises(ToolError, match="question must not be empty"):
        asyncio.run(ask_gpt(question="   "))

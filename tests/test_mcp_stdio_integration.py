import asyncio
import pytest
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp import ClientSession

SERVER_PARAMS = StdioServerParameters(
    command=r"C:\Project\ai-agent-bridge\.venv\Scripts\python.exe",
    args=["-m", "src.server"],
    cwd=r"C:\Project\ai-agent-bridge",
)


def test_mcp_stdio_discovery():
    """Verify that a stdio MCP client can connect, initialize, and discover ask_gpt."""
    async def run():
        async with stdio_client(SERVER_PARAMS) as (read, write):
            async with ClientSession(read, write) as session:
                init_result = await session.initialize()
                assert init_result.server_info.name == "ai-agent-bridge"

                tools_result = await session.list_tools()
                tool_names = [t.name for t in tools_result.tools]
                assert "ask_gpt" in tool_names

                tool = next(t for t in tools_result.tools if t.name == "ask_gpt")
                schema = tool.input_schema
                assert "question" in schema["required"]
                assert schema["properties"]["question"]["type"] == "string"
                assert "context" not in schema.get("required", [])

    asyncio.run(run())


def test_mcp_stdio_missing_api_key_returns_safe_tool_error():
    """Verify missing OPENAI_API_KEY returns safe ToolError over stdio without crashing."""
    params = StdioServerParameters(
        command=r"C:\Project\ai-agent-bridge\.venv\Scripts\python.exe",
        args=["-m", "src.server"],
        cwd=r"C:\Project\ai-agent-bridge",
        env={"OPENAI_MODEL": "gpt-4o-mini"}
    )
    async def run():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("ask_gpt", {"question": "Hello"})
                assert result.is_error is True
                assert any("OPENAI_API_KEY is not configured" in c.text for c in result.content)

    asyncio.run(run())


def test_mcp_stdio_missing_model_returns_safe_tool_error():
    """Verify missing OPENAI_MODEL returns safe ToolError over stdio without crashing."""
    params = StdioServerParameters(
        command=r"C:\Project\ai-agent-bridge\.venv\Scripts\python.exe",
        args=["-m", "src.server"],
        cwd=r"C:\Project\ai-agent-bridge",
        env={"OPENAI_API_KEY": "dummy-key"}
    )
    async def run():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("ask_gpt", {"question": "Hello"})
                assert result.is_error is True
                assert any("OPENAI_MODEL is not configured" in c.text for c in result.content)

    asyncio.run(run())


def test_mcp_stdio_mock_basic_invocation():
    """Test 1: Basic invocation with MOCK_GPT=1 over stdio."""
    params = StdioServerParameters(
        command=r"C:\Project\ai-agent-bridge\.venv\Scripts\python.exe",
        args=["-m", "src.server"],
        cwd=r"C:\Project\ai-agent-bridge",
        env={"MOCK_GPT": "1"}
    )
    async def run():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("ask_gpt", {"question": "Hello GPT"})
                assert result.is_error is False or result.is_error is None
                assert len(result.content) > 0
                assert result.content[0].text == "Mock GPT response"

    asyncio.run(run())


def test_mcp_stdio_mock_context_passing():
    """Test 2: Context passing with MOCK_GPT=1 over stdio."""
    params = StdioServerParameters(
        command=r"C:\Project\ai-agent-bridge\.venv\Scripts\python.exe",
        args=["-m", "src.server"],
        cwd=r"C:\Project\ai-agent-bridge",
        env={"MOCK_GPT": "1"}
    )
    async def run():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                context_text = "PostgreSQL with pgvector supports vector similarity search."
                result = await session.call_tool(
                    "ask_gpt",
                    {
                        "question": "Summarize the supplied context.",
                        "context": context_text
                    }
                )
                assert result.is_error is False or result.is_error is None
                assert len(result.content) > 0
                assert context_text in result.content[0].text

    asyncio.run(run())


def test_mcp_stdio_mock_llm_flag():
    """Verify MOCK_LLM=1 works as the primary mock flag over stdio."""
    params = StdioServerParameters(
        command=r"C:\Project\ai-agent-bridge\.venv\Scripts\python.exe",
        args=["-m", "src.server"],
        cwd=r"C:\Project\ai-agent-bridge",
        env={"MOCK_LLM": "1", "LLM_PROVIDER": "gemini"}
    )
    async def run():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("ask_gpt", {"question": "Hello MOCK_LLM"})
                assert result.is_error is False or result.is_error is None
                assert len(result.content) > 0
                assert result.content[0].text == "Mock GPT response"

    asyncio.run(run())


def test_mcp_stdio_gemini_missing_api_key_returns_safe_tool_error():
    """Verify GeminiProvider missing GEMINI_API_KEY returns safe ToolError over stdio."""
    params = StdioServerParameters(
        command=r"C:\Project\ai-agent-bridge\.venv\Scripts\python.exe",
        args=["-m", "src.server"],
        cwd=r"C:\Project\ai-agent-bridge",
        env={"LLM_PROVIDER": "gemini", "LLM_MODEL": "gemini-2.5-flash"}
    )
    async def run():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("ask_gpt", {"question": "Hello Gemini"})
                assert result.is_error is True
                assert any("GEMINI_API_KEY is not configured" in c.text for c in result.content)

    asyncio.run(run())


def test_mcp_stdio_gemini_missing_model_returns_safe_tool_error():
    """Verify GeminiProvider missing model returns safe ToolError over stdio."""
    params = StdioServerParameters(
        command=r"C:\Project\ai-agent-bridge\.venv\Scripts\python.exe",
        args=["-m", "src.server"],
        cwd=r"C:\Project\ai-agent-bridge",
        env={"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "dummy-gemini-key"}
    )
    async def run():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("ask_gpt", {"question": "Hello Gemini"})
                assert result.is_error is True
                assert any("GEMINI_MODEL is not configured" in c.text for c in result.content)

    asyncio.run(run())


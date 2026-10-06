"""
Verification script for Milestone 2 - Mock GPT Integration via MCP stdio.
Proves the complete communication path:
MCP Host (stdio) -> ai-agent-bridge -> ask_gpt -> Mock GPT -> Host
"""

import asyncio
import sys

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp import ClientSession


async def run_verification():
    print("=" * 60)
    print("Running Milestone 2 Integration Verification (Mock GPT)")
    print("=" * 60)

    server_params = StdioServerParameters(
        command=r"C:\Project\ai-agent-bridge\.venv\Scripts\python.exe",
        args=["-m", "src.server"],
        cwd=r"C:\Project\ai-agent-bridge",
        env={"MOCK_GPT": "1"},
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 1. Initialize Handshake
            init_result = await session.initialize()
            print(f"[1] Connected to MCP Server: '{init_result.server_info.name}'")

            # 2. Tool Discovery
            tools_result = await session.list_tools()
            tool = next((t for t in tools_result.tools if t.name == "ask_gpt"), None)
            assert tool is not None, "Tool 'ask_gpt' not discovered!"
            print(f"[2] Discovered tool: '{tool.name}'")
            print(f"    Schema properties: {list(tool.input_schema.get('properties', {}).keys())}")
            print(f"    Required: {tool.input_schema.get('required')}")

            # 3. Test 1 — Basic invocation
            print("\n[3] Executing Test 1 — Basic invocation:")
            t1_input = {"question": "Hello GPT", "context": None}
            print(f"    Input: {t1_input}")
            t1_result = await session.call_tool("ask_gpt", {"question": "Hello GPT"})
            t1_text = t1_result.content[0].text if t1_result.content else ""
            print(f"    Output: '{t1_text}'")
            assert t1_text == "Mock GPT response", f"Unexpected output: {t1_text}"
            print("    [PASS] Test 1 Passed!")

            # 4. Test 2 — Context passing
            print("\n[4] Executing Test 2 — Context passing:")
            context_text = "PostgreSQL with pgvector supports vector similarity search."
            t2_input = {
                "question": "Summarize the supplied context.",
                "context": context_text,
            }
            print(f"    Input question: '{t2_input['question']}'")
            print(f"    Input context:  '{t2_input['context']}'")
            t2_result = await session.call_tool("ask_gpt", t2_input)
            t2_text = t2_result.content[0].text if t2_result.content else ""
            print(f"    Output: '{t2_text}'")
            assert context_text in t2_text, "Context was not passed to mock output!"
            print("    [PASS] Test 2 Passed!")

    print("\n" + "=" * 60)
    print("All M2 Mock Integration Tests Completed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_verification())

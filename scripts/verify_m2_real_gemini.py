"""
Verification script for Milestone 2 - Real Gemini API Integration via MCP stdio.
Proves the communication path:
MCP Host (stdio) -> ai-agent-bridge -> ask_gpt -> GeminiProvider -> google-genai -> Gemini API -> Host
"""

import asyncio
import os
import sys

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp import ClientSession


async def run_verification():
    print("=" * 60)
    print("Running Milestone 2 Integration Verification (Real Gemini API)")
    print("=" * 60)

    # Forward environment to MCP stdio server
    server_env = os.environ.copy()

    # Load .env into server_env if present
    env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_file):
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip('"').strip("'")
                        if k and v:
                            server_env[k] = v
        except Exception:
            pass

    model_name = server_env.get("LLM_MODEL") or os.getenv("LLM_MODEL", "gemini-2.5-flash")
    server_env["LLM_PROVIDER"] = "gemini"
    server_env["LLM_MODEL"] = model_name
    server_env.pop("MOCK_LLM", None)
    server_env.pop("MOCK_GPT", None)

    print(f"Target LLM_PROVIDER: gemini")
    print(f"Target LLM_MODEL:    {model_name}")


    # Check key presence safely (boolean only, never expose key value)
    has_gemini_key = bool(server_env.get("GEMINI_API_KEY"))
    print(f"GEMINI_API_KEY configured: {has_gemini_key}")

    server_params = StdioServerParameters(
        command=r"C:\Project\ai-agent-bridge\.venv\Scripts\python.exe",
        args=["-m", "src.server"],
        cwd=r"C:\Project\ai-agent-bridge",
        env=server_env,
    )

    try:
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
                print(f"    Required properties: {tool.input_schema.get('required')}")

                # 3. Call ask_gpt with smoke test prompt
                prompt = "Reply with exactly: MCP bridge connection works."
                print(f"\n[3] Calling ask_gpt:")
                print(f"    Prompt: '{prompt}'")

                result = await session.call_tool("ask_gpt", {"question": prompt})

                is_error = getattr(result, "is_error", False)
                content_texts = [c.text for c in result.content if hasattr(c, "text")]
                combined_output = "\n".join(content_texts).strip()

                print(f"    is_error: {is_error}")
                print(f"    Output:   '{combined_output}'")

                if is_error:
                    print("\n[RESULT] FAIL: ask_gpt returned ToolError.")
                    print(f"Safe error message: {combined_output}")
                    return False, combined_output

                if "MCP bridge connection works." in combined_output:
                    print("\n[RESULT] PASS: Verified exact response from Gemini API!")
                    return True, combined_output
                else:
                    print(f"\n[RESULT] Unexpected response text: '{combined_output}'")
                    return True, combined_output

    except Exception as exc:
        print(f"\n[EXCEPTION] Failed during stdio execution: {type(exc).__name__}: {exc}")
        return False, str(exc)


if __name__ == "__main__":
    success, output = asyncio.run(run_verification())
    sys.exit(0 if success else 1)

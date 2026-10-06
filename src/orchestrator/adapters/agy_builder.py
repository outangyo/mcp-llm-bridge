from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from typing import Any, Callable, Dict, List, Optional

from src.orchestrator.adapters.base import (
    BaseAgentAdapter,
    ProgressCallback,
    sanitize_secrets,
)
from src.orchestrator.contracts.context import WorkflowContext
from src.orchestrator.contracts.report import AgentReport, AgentRole, BuilderResult
from src.orchestrator.security.paths import is_path_allowed

logger = logging.getLogger(__name__)


def find_agy_binary() -> str:
    """Discover the path to the agy CLI binary across environment, PATH, and standard directories."""
    # 1. Explicit environment variable
    env_bin = os.getenv("AGY_BIN_PATH")
    if env_bin and os.path.isfile(env_bin):
        return env_bin

    # 2. Standard PATH lookup
    path_bin = shutil.which("agy") or shutil.which("agy.exe")
    if path_bin:
        return path_bin

    # 3. Standard Windows location (%LOCALAPPDATA%\agy\bin\agy.exe)
    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        default_win_path = os.path.join(local_app_data, "agy", "bin", "agy.exe")
        if os.path.isfile(default_win_path):
            return default_win_path

    # Fallback to binary name
    return "agy"


class AGYBuilderAdapter(BaseAgentAdapter):
    """
    Real Builder adapter invoking the Antigravity (AGY) CLI in non-interactive print mode.
    Encapsulates process invocation, stdout/stderr capture, exit code verification,
    timeout handling, and path security enforcement without bypassing permissions.
    """

    def __init__(
        self,
        agent_name: str = "AGY",
        agy_bin: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: int = 300,
        cwd: Optional[str] = None,
        process_runner: Optional[Callable[..., Any]] = None,
    ) -> None:
        super().__init__(agent_name=agent_name, agent_role=AgentRole.BUILDER)
        self.agy_bin = agy_bin or find_agy_binary()
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.cwd = cwd
        self.process_runner = process_runner or subprocess.run

    def build_prompt(self, context: WorkflowContext) -> str:
        """Construct deterministic execution prompt for AGY CLI."""
        task = context.task
        criteria_list = "\n".join(f"- {c}" for c in task.acceptance_criteria) if task.acceptance_criteria else "- Satisfy task requirements"
        paths_list = ", ".join(task.allowed_paths) if task.allowed_paths else "No path restrictions"

        feedback_section = ""
        last_rev = context.last_reviewer_report
        if last_rev and not last_rev.is_pass:
            feedback_section = f"""
PRIOR REVIEWER FEEDBACK (MUST ADDRESS):
Summary: {last_rev.summary}
Findings: {last_rev.findings}
Blockers: {', '.join(last_rev.blockers) if last_rev.blockers else 'None'}
Next Step Requested: {last_rev.next_step}
"""

        prompt = f"""You are the BUILDER agent in an automated multi-agent engineering workflow.
Your role is to implement the requested task according to specifications.

TASK SPECIFICATION:
Task ID: {task.task_id}
Title: {task.title}
Description: {task.description}
Iteration: {context.iteration_index} (Retry count: {context.retry_count})
Allowed File Scopes: {paths_list}

ACCEPTANCE CRITERIA:
{criteria_list}
{feedback_section}
INSTRUCTIONS:
1. You are running in non-interactive print mode. Do not invoke external command/tool actions that require interactive terminal prompts. Formulate your implementation analysis, code adjustments, and report as structured text.
2. Verify changes against the acceptance criteria within the allowed scopes.
3. Conclude your response with a JSON report code block matching this schema:
```json
{{
  "summary": "<Concise summary of changes made>",
  "findings": "<Details of changes, test outcomes, or discoveries>",
  "builder_result": "SUCCESS" | "FAILURE" | "INCOMPLETE" | "BLOCKED",
  "blockers": ["<Any blockers or obstacles>"],
  "next_step": "<Recommended next action>",
  "modified_files": ["<List of files created or modified>"]
}}
```
"""
        return prompt.strip()

    def execute(
        self,
        context: WorkflowContext,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> AgentReport:
        """Execute AGY CLI subprocess and return typed AgentReport."""
        self.notify_progress(
            message="Preparing AGY task specification and constraints",
            payload={"iteration": context.iteration_index, "task_id": context.task.task_id},
            callback=progress_callback,
        )

        prompt = self.build_prompt(context)
        cmd = [
            self.agy_bin,
            "-p",
            prompt,
            "--output-format",
            "json",
        ]
        if self.model:
            cmd.extend(["--model", self.model])

        self.notify_progress(
            message=f"Launching AGY CLI subprocess (binary: {self.agy_bin})",
            payload={"timeout_seconds": self.timeout_seconds, "cmd": [self.agy_bin, "-p", "..."]},
            callback=progress_callback,
        )

        try:
            completed = self.process_runner(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=self.cwd,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError as exc:
            logger.error("AGY binary not found at %s: %s", self.agy_bin, exc)
            return AgentReport(
                agent_role=AgentRole.BUILDER,
                agent_name=self.agent_name,
                summary=f"AGY CLI binary not found: {self.agy_bin}",
                findings="Failed to launch AGY subprocess because executable is missing.",
                builder_result=BuilderResult.FAILURE,
                blockers=[f"FileNotFoundError: {self.agy_bin}"],
                next_step="Install AGY CLI or configure AGY_BIN_PATH.",
            )
        except subprocess.TimeoutExpired as exc:
            logger.error("AGY execution timed out after %s seconds", self.timeout_seconds)
            return AgentReport(
                agent_role=AgentRole.BUILDER,
                agent_name=self.agent_name,
                summary=f"AGY execution timed out after {self.timeout_seconds} seconds",
                findings=f"Subprocess exceeded timeout threshold of {self.timeout_seconds}s.",
                builder_result=BuilderResult.FAILURE,
                blockers=[f"TimeoutExpired: {self.timeout_seconds}s"],
                next_step="Increase timeout_seconds or narrow task scope.",
            )
        except Exception as exc:
            logger.exception("Unexpected error launching AGY process: %s", exc)
            return AgentReport(
                agent_role=AgentRole.BUILDER,
                agent_name=self.agent_name,
                summary="Failed to execute AGY CLI subprocess",
                findings=str(exc),
                builder_result=BuilderResult.FAILURE,
                blockers=[f"SubprocessError: {exc}"],
                next_step="Check system process permissions and environment.",
            )

        # Handle non-zero exit code
        if completed.returncode != 0:
            stderr_out = completed.stderr.strip() if completed.stderr else ""
            stdout_out = completed.stdout.strip() if completed.stdout else ""
            err_msg = stderr_out or stdout_out or f"Exit code {completed.returncode}"
            logger.warning("AGY process failed with exit code %s: %s", completed.returncode, err_msg)
            return AgentReport(
                agent_role=AgentRole.BUILDER,
                agent_name=self.agent_name,
                summary=f"AGY CLI process exited with code {completed.returncode}",
                findings=err_msg,
                builder_result=BuilderResult.FAILURE,
                blockers=[f"ExitCode: {completed.returncode}"],
                next_step="Inspect stderr output and resolve CLI execution failure.",
                artifacts={"stdout": stdout_out, "stderr": stderr_out},
            )

        self.notify_progress(
            message="Parsing AGY response and enforcing path boundary policy",
            callback=progress_callback,
        )

        return self._parse_response(completed.stdout, context)

    def _parse_response(self, raw_stdout: str, context: WorkflowContext) -> AgentReport:
        """Parse AGY stdout into typed AgentReport with schema extraction and path policy check."""
        response_text = sanitize_secrets(raw_stdout.strip())
        # Extract response from AGY json envelope if present
        try:
            data = json.loads(raw_stdout)
            if isinstance(data, dict) and "response" in data:
                response_text = sanitize_secrets(str(data["response"]).strip())
        except Exception:
            pass

        # Attempt to extract JSON code block from response
        extracted_data = self._extract_json_block(response_text)

        summary = ""
        findings = ""
        builder_result = BuilderResult.SUCCESS
        blockers: List[str] = []
        next_step = "Submit to Reviewer for verification."
        modified_files: List[str] = []

        # Check for auto-denied permission actions in headless mode
        if isinstance(data, dict) and data.get("denied_actions"):
            builder_result = BuilderResult.BLOCKED
            blockers.append(f"Permission policy: Headless tool action auto-denied: {data['denied_actions']}")

        if extracted_data:
            summary = sanitize_secrets(str(extracted_data.get("summary", "")).strip())
            findings = sanitize_secrets(str(extracted_data.get("findings", "")).strip())
            res_str = str(extracted_data.get("builder_result", "SUCCESS")).upper().strip()
            builder_result = getattr(BuilderResult, res_str, BuilderResult.SUCCESS)
            raw_blockers = extracted_data.get("blockers", [])
            if isinstance(raw_blockers, list):
                blockers = [sanitize_secrets(str(b)) for b in raw_blockers]
            next_step = sanitize_secrets(str(extracted_data.get("next_step", next_step)).strip())
            raw_files = extracted_data.get("modified_files", [])
            if isinstance(raw_files, list):
                modified_files = [str(f) for f in raw_files]

        if not summary:
            # Fallback when no structured json found
            lines = [l.strip() for l in response_text.splitlines() if l.strip()]
            summary = lines[0][:150] if lines else "AGY completed turn without explicit summary."
            findings = response_text[:1000] if response_text else "No response body produced."

        if not next_step:
            next_step = "Proceed to Reviewer evaluation."

        # Path Scope Security Enforcement
        if context.task.allowed_paths and modified_files:
            violating_files = [
                f for f in modified_files
                if not is_path_allowed(f, context.task.allowed_paths)
            ]

            if violating_files:
                builder_result = BuilderResult.BLOCKED

                blocker_msg = (
                    f"Path policy violation: files modified outside allowed scope "
                    f"{context.task.allowed_paths}: {violating_files}"
                )
                blockers.append(blocker_msg)
                logger.warning(blocker_msg)

        return AgentReport(
            agent_role=AgentRole.BUILDER,
            agent_name=self.agent_name,
            summary=summary,
            findings=findings,
            builder_result=builder_result,
            blockers=blockers,
            next_step=next_step,
            artifacts={"modified_files": modified_files, "raw_response": response_text[:500]},
        )

    def _extract_json_block(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract structured JSON object from markdown code block or bare JSON string."""
        # Check for ```json ... ``` blocks
        matches = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        for match in reversed(matches):
            try:
                return json.loads(match)
            except Exception:
                continue

        # Check for trailing raw JSON object
        raw_match = re.search(r"(\{\s*\"summary\":.*?\})", text, re.DOTALL)
        if raw_match:
            try:
                return json.loads(raw_match.group(1))
            except Exception:
                pass

        return None

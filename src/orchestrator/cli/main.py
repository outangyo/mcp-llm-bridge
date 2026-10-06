from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Callable, List, Optional

from src.orchestrator.adapters.agy_builder import AGYBuilderAdapter
from src.orchestrator.adapters.base import sanitize_secrets
from src.orchestrator.adapters.gemini_reviewer import GeminiReviewerAdapter
from src.orchestrator.adapters.mock_builder import MockBuilder
from src.orchestrator.adapters.mock_reviewer import MockReviewer
from src.orchestrator.cli.renderer import CLIRenderer
from src.orchestrator.contracts.context import WorkflowContext
from src.orchestrator.contracts.task import TaskBudget, TaskContract
from src.orchestrator.engine import OrchestratorEngine
from src.orchestrator.policy.execution_policy import ExecutionPolicy, PolicyMode

# Deterministic CLI exit codes
EXIT_SUCCESS = 0
EXIT_FAILED = 1
EXIT_STOPPED = 2
EXIT_INVALID_INPUT = 3


def parse_task_input(task_input: str) -> TaskContract:
    """
    Parse task definition from a JSON file path or inline JSON string into TaskContract.
    Raises ValueError with safe error message if parsing or validation fails.
    """
    cleaned = task_input.strip()
    if os.path.isfile(cleaned):
        try:
            with open(cleaned, "r", encoding="utf-8") as f:
                content = f.read()
            return TaskContract.model_validate_json(content)
        except Exception as exc:
            raise ValueError(f"Failed to load task file '{cleaned}': {exc}") from exc

    # Attempt parsing as inline JSON string
    try:
        return TaskContract.model_validate_json(cleaned)
    except Exception as exc:
        raise ValueError(
            f"Input is neither a valid file path nor a valid TaskContract JSON string: {exc}"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for the orchestrator CLI."""
    parser = argparse.ArgumentParser(
        prog="orchestrator",
        description="Deterministic multi-agent workflow orchestrator CLI",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available commands")

    # 'run' subcommand
    run_parser = subparsers.add_parser("run", help="Start an orchestrated workflow")
    run_parser.add_argument(
        "task_input",
        nargs="?",
        default=None,
        help="Path to task definition JSON file or inline JSON string",
    )
    run_parser.add_argument("--title", help="Task title (if not using a task definition file)")
    run_parser.add_argument("--description", help="Task description (if not using a task definition file)")
    run_parser.add_argument(
        "--criteria",
        action="append",
        default=[],
        help="Verifiable acceptance criteria (repeatable)",
    )
    run_parser.add_argument(
        "--allowed-paths",
        action="append",
        default=[],
        help="Scoped filesystem paths the builder may touch (repeatable)",
    )
    run_parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum allowed iteration loops (default: 3)",
    )
    run_parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Maximum allowed retries on reviewer rejection (default: 2)",
    )
    run_parser.add_argument(
        "--policy",
        choices=["SAFE", "SUPERVISED", "AUTONOMOUS"],
        default="SUPERVISED",
        help="Execution policy governance mode (default: SUPERVISED)",
    )
    run_parser.add_argument(
        "--mock",
        action="store_true",
        help="Run using simulated mock agent adapters (offline/testing)",
    )
    run_parser.add_argument(
        "-y",
        "--auto-approve",
        action="store_true",
        help="Automatically grant approval at human governance gates",
    )
    run_parser.add_argument(
        "--no-icons",
        action="store_true",
        help="Disable Unicode emoji icons in CLI progress output",
    )

    return parser


def main(
    argv: Optional[List[str]] = None,
    engine_factory: Optional[Callable[..., OrchestratorEngine]] = None,
    input_func: Optional[Callable[[str], str]] = None,
) -> int:
    """
    Main executable CLI entry point for the orchestrator.
    Returns deterministic integer exit code:
      0: SUCCESS (COMPLETED)
      1: FAILED
      2: STOPPED (PO rejected or workflow aborted)
      3: INVALID_INPUT (CLI argument or task validation error)
    """
    parser = build_parser()

    # If no arguments passed, print help and return invalid input
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        parser.print_help(sys.stderr)
        return EXIT_INVALID_INPUT

    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse raises SystemExit on --help (code 0) or unrecognized argument (code 2)
        return exc.code if exc.code == 0 else EXIT_INVALID_INPUT

    if args.subcommand != "run":
        parser.print_help(sys.stderr)
        return EXIT_INVALID_INPUT

    # 1. Resolve and validate TaskContract
    task: TaskContract
    if args.task_input:
        try:
            task = parse_task_input(args.task_input)
        except Exception as exc:
            safe_err = sanitize_secrets(str(exc))
            sys.stderr.write(f"Error: {safe_err}\n")
            return EXIT_INVALID_INPUT
    elif args.title and args.description:
        try:
            task = TaskContract(
                title=args.title,
                description=args.description,
                acceptance_criteria=args.criteria,
                allowed_paths=args.allowed_paths,
                budget=TaskBudget(
                    max_iterations=args.max_iterations,
                    max_retries=args.max_retries,
                ),
            )
        except Exception as exc:
            safe_err = sanitize_secrets(str(exc))
            sys.stderr.write(f"Error creating task contract: {safe_err}\n")
            return EXIT_INVALID_INPUT
    else:
        sys.stderr.write(
            "Error: Must provide either a task definition (file path or JSON string) "
            "or both --title and --description.\n"
        )
        return EXIT_INVALID_INPUT

    # 2. Configure execution policy
    policy_mode = PolicyMode[args.policy.upper()]
    policy = ExecutionPolicy(mode=policy_mode, require_final_approval=(policy_mode != PolicyMode.AUTONOMOUS))

    # 3. Configure live terminal renderer
    renderer = CLIRenderer(use_icons=not args.no_icons)

    # 4. Configure agent adapters
    if args.mock:
        builder = MockBuilder(agent_name="MockAGY")
        reviewer = MockReviewer(agent_name="MockGemini")
    else:
        builder = AGYBuilderAdapter()
        reviewer = GeminiReviewerAdapter()

    # 5. Wire human approval handler
    def cli_approval_handler(reason: str, ctx: WorkflowContext) -> bool:
        if args.auto_approve or policy.mode == PolicyMode.AUTONOMOUS:
            return True

        prompt_text = f"\n[PO GOVERNANCE GATE] {reason}\nGrant PO approval? [y/N]: "
        try:
            if input_func:
                ans = input_func(prompt_text).strip().lower()
            else:
                sys.stdout.write(prompt_text)
                sys.stdout.flush()
                ans = input().strip().lower()
            return ans in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            sys.stdout.write("\nApproval declined.\n")
            return False

    # 6. Initialize OrchestratorEngine
    if engine_factory:
        engine = engine_factory(
            task=task,
            builder=builder,
            reviewer=reviewer,
            policy=policy,
            renderer=renderer,
            approval_handler=cli_approval_handler,
        )
    else:
        engine = OrchestratorEngine(
            task=task,
            builder=builder,
            reviewer=reviewer,
            policy=policy,
            renderer=renderer,
            approval_handler=cli_approval_handler,
        )

    # 7. Execute workflow loop
    try:
        summary = engine.run()
    except KeyboardInterrupt:
        sys.stderr.write("\nWorkflow execution interrupted by user.\n")
        return EXIT_STOPPED
    except Exception as exc:
        safe_err = sanitize_secrets(str(exc))
        sys.stderr.write(f"Workflow execution failure: {safe_err}\n")
        return EXIT_FAILED

    # 8. Map terminal outcome to deterministic exit code
    if summary.status == "COMPLETED":
        return EXIT_SUCCESS
    elif summary.status == "STOPPED":
        return EXIT_STOPPED
    else:
        return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())

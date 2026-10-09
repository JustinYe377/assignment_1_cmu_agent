"""The Part 1 coding agent: fix a software issue and submit a git patch."""

from __future__ import annotations

import json
from typing import Any

from assignment.agent.base import (
    DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
    DEFAULT_COMPACTION_MAX_TOKENS,
    Agent,
    format_tool_output,
)
from assignment.agent.tools import EXECUTE_TOOL, SEND_MESSAGE_TOOL
from assignment.env import Environment

class CodeAgent(Agent):
    """An agent that fixes a software issue and submits a git patch."""

    def __init__(
        self,
        task: str,
        environment: Environment,
        model: str | None = None,
        logs_save_path: str | None = None,
        step_limit: int = 100,
        skills_path: str | None = None,
        auto_stop_environment: bool = True,
        compact_threshold_tokens: int | None = None,
        compaction_keep_recent_steps: int = DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
        compaction_max_tokens: int = DEFAULT_COMPACTION_MAX_TOKENS,
    ):
        super().__init__(
            environment=environment,
            model=model,
            logs_save_path=logs_save_path,
            step_limit=step_limit,
            skills_path=skills_path,
            auto_stop_environment=auto_stop_environment,
            compact_threshold_tokens=compact_threshold_tokens,
            compaction_keep_recent_steps=compaction_keep_recent_steps,
            compaction_max_tokens=compaction_max_tokens,
        )
        self.task = task
        self.submitted_patch = ""

        self.tools.extend([EXECUTE_TOOL, SEND_MESSAGE_TOOL])

        system_information = {
            "machine": self.env.machine,
            "release": self.env.release,
            "system": self.env.system,
            "version": self.env.version,
        }
        self.system_prompt = (
            "You are a coding agent solving a software issue in a remote terminal environment.\n"
            "Inspect the repository and relevant instructions, reproduce the reported "
            "failure, make a focused fix, and verify it with appropriate tests.\n"
            "Use tool results as evidence, and investigate failures before deciding "
            "your next action. Do not claim success without verification.\n"
            f"The default working directory is {self.env.cwd}. Each command runs "
            "in a new subshell; use cwd and env arguments when needed.\n"
            "<system_information>\n"
            f"{json.dumps(system_information, indent=2)}\n"
            "</system_information>"
        )
        self.task_prompt = self.task

        if self.skills:
            catalog = "\n\n".join(skill["metadata"] for skill in self.skills.values())
            self.system_prompt += (
                "\n\nAvailable skills:\n"
                "Call invoke_skill with the catalog name before starting work "
                "covered by a skill, and follow its full instructions.\n"
                f"{catalog}"
            )

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute ``execute`` and ``send_message`` calls in the code sandbox."""

        observations = []
        for call in tool_calls:
            try:
                if self.finished:
                    raise ValueError("The task has already been submitted.")
                function = call["function"]
                name = function["name"]
                arguments = json.loads(function["arguments"])
                if not isinstance(arguments, dict):
                    raise ValueError("Tool arguments must be a JSON object.")

                if name == "execute":
                    allowed = {"command", "shell", "cwd", "timeout", "env"}
                    if arguments.keys() - allowed:
                        raise ValueError("Unexpected execute arguments.")
                    command = arguments.get("command")
                    if not isinstance(command, str) and not (
                        isinstance(command, list)
                        and all(isinstance(item, str) for item in command)
                    ):
                        raise ValueError("command must be a string or a list of strings.")
                    content = format_tool_output(self.env.execute(**arguments))
                elif name == "invoke_skill" and self.skills:
                    if set(arguments) != {"name"} or not isinstance(arguments["name"], str):
                        raise ValueError("invoke_skill requires a string name.")
                    skill_name = arguments["name"]
                    if skill_name not in self.skills:
                        raise ValueError(f"Unknown skill: {skill_name}")
                    content = self.skills[skill_name]["content"]
                elif name == "send_message":
                    if set(arguments) != {"summary"} or not isinstance(
                        arguments["summary"], str
                    ):
                        raise ValueError("send_message requires a string summary.")
                    print(arguments["summary"], flush=True)
                    self.finished = True
                    content = "Message delivered. Task finished."
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except Exception as exc:
                content = format_tool_output(
                    {"error": f"{type(exc).__name__}: {exc}"}
                )

            observations.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": content,
                }
            )
        return observations

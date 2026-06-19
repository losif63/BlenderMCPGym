"""Agent runner registry. Add new tools here (e.g. "cursor": CursorRunner)."""

from .base import AgentRunner, AgentResult
from .claudecode import ClaudeCodeRunner

RUNNERS = {
    ClaudeCodeRunner.name: ClaudeCodeRunner,
}

__all__ = ["AgentRunner", "AgentResult", "RUNNERS", "get_runner"]


def get_runner(tool: str) -> AgentRunner:
    try:
        return RUNNERS[tool]()
    except KeyError:
        raise ValueError(f"Unknown tool {tool!r}; known: {sorted(RUNNERS)}")

"""AgentRunner: the pluggable coding-agent interface.

This is the seam that makes tool ablations possible. A runner launches some
coding agent (claudecode today, cursor later) with the given prompt, pointed at
a BlenderMCP session directory, and streams its output to a log.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AgentResult:
    returncode: int
    session_id: str | None = None  # agent session id, for retrieving logs later

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class AgentRunner(ABC):
    #: registry key (see blendergym.agents.RUNNERS) and run_id prefix
    name: str = "agent"

    @abstractmethod
    def run(
        self,
        prompt: str,
        *,
        cwd: Path,
        session_dir: Path,
        model_cli: str,
        log_path: Path,
    ) -> AgentResult:
        """Run the agent to completion, streaming output to log_path."""
        raise NotImplementedError

"""Run configuration: the axes of a benchmark run and how they map to a run_id.

Currently the only ablation axis is the model. The tool is fixed to ``claudecode``
but kept as a first-class field so other agents (e.g. cursor) drop in later.
"""

from dataclasses import dataclass

# Model slug (used in run_id / paths) -> the id passed to `claude --model`.
# Comparing across the Opus version line. Note: opus-4.1 is deprecated and
# retires 2026-08-05 — finish any 4.1 runs before then.
MODELS = {
    "opus-4.8": "claude-opus-4-8",
    "opus-4.7": "claude-opus-4-7",
    "opus-4.6": "claude-opus-4-6",
    "opus-4.5": "claude-opus-4-5",
    "opus-4.1": "claude-opus-4-1",
}

DEFAULT_TOOL = "claudecode"


@dataclass(frozen=True)
class RunConfig:
    """Identifies one benchmark run (one cell of the ablation matrix).

    model:  slug from MODELS, e.g. "opus-4.8".
    tool:   agent runner name, e.g. "claudecode" (see blendergym.agents.RUNNERS).
    trial:  repeat index; 0 is the canonical run, >0 appends "__t{n}" to run_id.
    """

    model: str
    tool: str = DEFAULT_TOOL
    trial: int = 0

    def __post_init__(self):
        if self.model not in MODELS:
            raise ValueError(
                f"Unknown model slug {self.model!r}; known: {sorted(MODELS)}"
            )

    @property
    def model_cli(self) -> str:
        """The id to pass to the agent CLI's --model flag."""
        return MODELS[self.model]

    @property
    def run_id(self) -> str:
        rid = f"{self.tool}__{self.model}"
        if self.trial:
            rid += f"__t{self.trial}"
        return rid

"""Claude Code agent runner.

Wraps the `claude` CLI in print mode with streaming JSON output. The model
ablation is implemented here: the run config's model id is threaded into the
`--model` flag (the original hardcoded invocation passed no model at all).
"""

import json
import os
import subprocess

from .base import AgentRunner, AgentResult


class ClaudeCodeRunner(AgentRunner):
    name = "claudecode"

    def run(self, prompt, *, cwd, session_dir, model_cli, log_path) -> AgentResult:
        cmd = [
            "claude",
            "--model", model_cli,
            "--verbose",
            "--output-format", "stream-json",
            "-p", prompt,
            "--dangerously-skip-permissions",
        ]
        # BlenderMCP's MCP server writes tool_calls.jsonl + screenshots here.
        env = {**os.environ, "BLENDER_MCP_SESSION_DIR": str(session_dir)}

        with open(log_path, "w") as log_f:
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            session_id = None
            for line in proc.stdout:
                print(line, end="")
                log_f.write(line)
                log_f.flush()
                # stream-json: each line is a JSON event carrying the session id.
                if session_id is None:
                    try:
                        obj = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    if isinstance(obj, dict) and obj.get("session_id"):
                        session_id = obj["session_id"]
            proc.wait()

        return AgentResult(returncode=proc.returncode, session_id=session_id)

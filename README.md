# BlenderMCPGym

A benchmark framework for evaluating Claude's ability to edit Blender scenes via [BlenderMCP](https://github.com/ahujasid/blender-mcp). Each task provides a starting Blender scene, a textual instruction, and rendered images of the target scene. Claude is prompted to use BlenderMCP tools to make the scene match the target.

## Requirements

- [Blender 5.0+](https://www.blender.org/download/) installed at `/Applications/Blender.app`
- [BlenderMCP addon](https://github.com/ahujasid/blender-mcp) installed in Blender
- [Claude Code CLI](https://claude.ai/code) (`claude` available on PATH)
- Python dependencies: `matplotlib`, `tqdm`, `huggingface_hub`

---

## Setup

### 1. Download benchmark data

```bash
python generate_benchdata.py
```

Downloads and unpacks the benchmark dataset from HuggingFace into `bench_data/`. Each task subdirectory (e.g. `bench_data/blendshape1/`) contains:

| File | Description |
|---|---|
| `blender_file.blend` | Original Blender scene |
| `start.py` | Script to initialize the scene to the starting state |
| `goal.py` | Script that produces the target scene |
| `renders/start/` | Rendered images of the starting scene |
| `renders/goal/` | Rendered images of the target scene |
| `description.txt` | Textual description and `INSTRUCTION:` prompt |

### 2. Generate task descriptions

```bash
python generate_descriptions.py
# Run only specific task types
python generate_descriptions.py --task placement geometry
# Regenerate existing descriptions
python generate_descriptions.py --overwrite
```

Sends start/goal render images to the Gemini API and writes two description files per task:

| File | Description |
|---|---|
| `description.txt` | Concise 1–3 sentence instruction |
| `detailed_instruction.txt` | Detailed paragraph-length instruction scrutinizing every visual difference |

`single_task.py` uses `detailed_instruction.txt`. Requires a Gemini API key (pass via `--api_key` or set in the environment).

---

## Running the benchmark

### Single task

```bash
# Version 1 (default) — provides start.py as context
python single_task.py --task_name blendshape1
python single_task.py --task_name blendshape1 --version 1

# Version 2 — no start.py context
python single_task.py --task_name blendshape1 --version 2
```

Two experiment versions are supported, controlled by `--version` (default: `1`):

| Version | start.py provided | Agent instruction |
|---|---|---|
| **1** (default) | Yes | Use only the variables, objects, and methods defined in `start.py` |
| **2** | No | Free to use any BlenderMCP approach |

This mirrors the original paper's intention: version 1 gives the agent a structured API (the variables and functions already set up in the scene), while version 2 tests the agent's ability to edit the scene from scratch with no scaffolding.

For one task, this script:
1. Copies `blender_file.blend` → `edit_{task_name}_ver{N}.blend`
2. Opens Blender with the edit file, runs `start.py` to initialize the scene, and starts the BlenderMCP server
3. Waits for BlenderMCP to be ready on port 9876
4. Launches Claude Code with a prompt containing the start renders, detailed instruction, goal renders, and (version 1 only) the `start.py` source code
5. Saves the edited file via BlenderMCP, closes Blender
6. Renders the edited scene to `renders/edit_ver{N}/`

Per-version outputs under each task directory:

| Path | Description |
|---|---|
| `edit_{task_name}_ver{N}.blend` | Edited Blender file for version N |
| `renders/edit_ver{N}/` | Rendered images of the edited scene for version N |
| `ver{N}/` | BlenderMCP interaction logs for version N |
| `metadata_ver{N}.json` | Timing and version metadata for version N |

### Full benchmark (sequential)

```bash
# Run all tasks
python run_benchmark.py

# Run only one task type
python run_benchmark.py --task_type blendshape
python run_benchmark.py --task_type material
python run_benchmark.py --task_type placement
python run_benchmark.py --task_type geometry
python run_benchmark.py --task_type lighting
```

Runs all tasks one after another. Records wall-clock duration for each task in `metadata.json` under the task directory.

### Full benchmark (parallel)

```bash
python run_benchmark_parallel.py
python run_benchmark_parallel.py --task_type blendshape
```

Same as above but runs up to 5 tasks in parallel using a process pool. Each parallel worker is assigned a unique BlenderMCP port (9876–9880) to avoid conflicts.

---

## Analysis & visualization

### Compare before/after for a task

```bash
python compare_task.py --task_name blendshape1
```

Displays a side-by-side matplotlib figure with three rows — **Start**, **Goal**, and **Edit** — using the pre-rendered images from `renders/edit/`. The figure is also saved to `comparisons/{task_name}.png`. Requires the task to have been run first.

### Plot timing histogram

```bash
python plot_durations.py
```

Reads `metadata.json` from every completed task and plots a histogram of wall-clock durations, broken down by task type (blendshape, geometry, lighting, material, placement).

---

## Utilities

### Toy script

```bash
python toy.py
```

Opens a blank Blender scene, starts BlenderMCP, and prompts Claude to build a simple castle. Useful for quickly testing that the BlenderMCP + Claude Code pipeline is working end-to-end.

### Cleanup

```bash
# Move edit files, edit renders, and metadata to prev/ subdirectory
python cleanup.py

# Clean up a single task only
python cleanup.py --task_name blendshape1

# Also delete the prev/ subdirectory (full wipe)
python cleanup.py --force-delete
```

For each task, moves `edit_{task_name}.blend`, `renders/edit/`, and `metadata.json` into a `prev/` subdirectory. Also deletes any `.blend1` backup files created by Blender. Use `--force-delete` to wipe `prev/` entirely instead of just replacing its contents.

---

## bench_data/ render scripts

These scripts run inside Blender's Python runtime and are not called directly.

| Script | Description |
|---|---|
| `bench_data/edit_render_script.py` | Renders Camera1 and Camera2 from an open blend file; used by `single_task.py` after editing |
| `bench_data/pipeline_render_script.py` | Renders Camera1 and Camera2 after executing a provided code file |
| `bench_data/eval_render_script.py` | Renders Camera3–5 after executing a provided code file |
| `bench_data/single_render_script.py` | Renders Camera1 only after executing a provided code file |

---

## Task types

| Type | Count | Description |
|---|---|---|
| `blendshape` | 85 | Shape key / mesh deformation edits |
| `geometry` | 55 | Geometric modifications to objects |
| `lighting` | 50 | Changes to lights and lighting setup |
| `material` | 45 | Material and texture edits |
| `placement` | 50 | Object placement and transformation |

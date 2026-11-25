# Description
Agentic framework that solves coding tasks by running a structured Thought -> Code -> Observation loop. It generates Python tool calls, executes them in a sandbox, observes results, and iterates until it produces a final solution. The project targets two benchmarks: MBPP (algorithmic Python tasks) and SWE-bench (real-world repository fixes).

# Instructions
## Prerequisites
- Python 3.10+
- uv
- Docker (SWE-bench agent, and MBPP/SWE-bench validation)

## Install
```bash
uv sync
```

## Sandbox (interactive)
```bash
uv run sandbox
```

## Sandbox with MCP (HTTP)
```bash
# Terminal 1
HOST=127.0.0.1 PORT=8000 uv run python src/mcp_tools_mbpp.py --transport streamable-http

# Terminal 2
uv run sandbox --mcp-server "http://127.0.0.1:8000/mcp"
```

## Supported API Providers
Configured in `.env` only (no CLI flags):
```
PROVIDER=GROQ            # GROQ | OPENROUTER | OLLAMA
MODEL_NAME=model/name
GROQ_API_KEY_1=...       # <PROVIDER>_API_KEY_<n>, several keys rotate on errors (none for OLLAMA)
OLLAMA_BASE_URL=...     # optional, default http://localhost:11434/v1
```

## MBPP agent
```bash
# 1) Dump a task
cd moulinette
uv run moulinette_eval dump mbpp --output ../cache/mbpp_task.json

# 2) Run the agent (add --pause to step through each turn)
cd ../src
uv run python -m agent_mbpp --task-file ../cache/mbpp_task.json \
    --output ../cache/mbpp_solution.json

# 3) Validate
cd ../moulinette
uv run moulinette_eval validate mbpp ../cache/mbpp_task.json ../cache/mbpp_solution.json
```

Note: MBPP correctness validation runs inside Docker using the `python:3.11-slim` image.
If you see a correctness failure with no Python traceback, make sure the image is available:
```bash
docker pull python:3.11-slim
```

## SWE-bench agent
```bash
# 1) Dump a task
cd moulinette
uv run moulinette_eval dump swebench --output ../cache/swebench_task.json

# 2) Run the agent
cd ../src
uv run python -m agent_swebench --task-file ../cache/swebench_task.json \
    --output ../cache/swebench_solution.json

# 3) Validate
cd ../moulinette
uv run moulinette_eval validate swebench ../cache/swebench_task.json ../cache/swebench_solution.json
```

# System Architecture
- Orchestrator: drives the Thought -> Code -> Observation loop, manages iterations, and records metrics.
- Code extraction: normalizes LLM outputs (Python blocks, JSON tool calls, ReAct) into executable Python.
- Sandbox: executes untrusted code with strict limits (imports, filesystem, timeout, memory) and injects `final_answer`.
- MCP servers (`src/mcp_tools_mbpp.py`, `src/mcp_tools_swebench.py`) are built with [FastMCP](https://gofastmcp.com) (`from fastmcp import FastMCP`); the client side uses the low-level `mcp` SDK (v2, snake_case fields such as `Tool.input_schema`). Tools are exposed as Python callables inside the sandbox namespace.

# Agent Loop Explanation
1. The agent builds a prompt with the sandbox manual and task context.
2. The LLM returns text that contains tool calls.
3. The extraction layer converts tool call formats into Python.
4. The sandbox executes the code, captures stdout/stderr, and returns observations.
5. The agent appends the observation and continues until it calls `final_answer` or hits limits.

On MBPP, the prompt asks the model to write its function only once per turn, inside a
`code = '''...'''` string, then test and submit it in the same block:
```python
result = run_tests(code=code, test_list=tests)
print(result)
if '"success": true' in result:
    final_answer(code)
```
Variables do not persist between turns: each sandbox run starts from fresh globals.

# Sandbox Design
- Allowlist imports, restricted builtins, and constrained filesystem paths.
- No network access in sandboxed execution.
- Execution timeout and memory limit enforced per code block.
- `final_answer` is injected by the sandbox and is not an MCP tool.
- MCP tools are discovered dynamically and exposed as Python functions.

# Tool Implementation Details
- MBPP (`src/mcp_tools_mbpp.py`, local stdio server): `run_tests(code, test_list, test_imports=None)`, returns JSON `{"success": bool, "output": "TEST_<i>: PASS|FAIL ..."}`.
- SWE-bench (`src/mcp_tools_swebench.py`, runs inside the task container via `docker exec`):
  - `read_file`, `edit_file`, `list_files`
  - `search_code`, `search_function_or_class_definition_in_code`, `find_references`
  - `run_tests`, `run_command`, `get_patch`

Tools run outside the sandbox; the sandbox only calls them through MCP wrappers.

# Resources
- Model Context Protocol: https://modelcontextprotocol.io/
- FastMCP: https://gofastmcp.com
- SWE-bench: https://github.com/swe-bench/swe-bench
- MBPP: https://github.com/google-research/google-research/tree/master/mbpp
- uv: https://github.com/astral-sh/uv
- Pydantic: https://docs.pydantic.dev/

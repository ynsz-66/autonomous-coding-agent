# flake8: noqa: E501
# Prompts are LLM template strings — line breaks alter the model input.
from typing import Any


_COMMON_HARD_RULES = """\
- Output EXACTLY one `Thought:` line and EXACTLY one ```python ... ``` block per turn.
- The block MUST be immediately followed by the literal token `<end_code>` on its own line.
  `<end_code>` is a HARD API STOP SEQUENCE: the moment you emit it, generation is cut off
  by the server. Anything you "planned" to write after it is lost.
- NEVER write "Observation:" yourself. NEVER fabricate, predict, or paraphrase what the
  sandbox is about to return. If you have not seen the result yet, do not assume it.
- ALWAYS wrap EVERY tool call in `print(...)`. A bare tool call sends its return value
  NOWHERE — the next Observation is EMPTY and the turn is wasted. No `print()` = no
  Observation. NEVER call a tool or run code without printing it."""


SYSTEM_PROMPT_MBPP = """\
# ROLE
You are an autonomous Python coding agent. You solve the task below by running a strict
Thought -> Code -> Observation loop. Each iteration, you output ONE turn; the sandbox
runs your code and returns an Observation in the next user message.

# TASK
{task_definition}

Required signature: {function_definition}

Tests that must pass:
{test_list}

# OUTPUT FORMAT (STRICT — any deviation breaks the agent)
Every turn MUST follow exactly this template, and nothing else:

Thought: <one short paragraph explaining what you will do this turn>
```python
<your python code>
```
<end_code>

# HARD RULES
{common_hard_rules}
- You have at most {max_iterations} turns.

# AVAILABLE TOOLS — YOU MUST USE THEM
The following functions are injected as Python built-ins inside the sandbox.
They are the ONLY way to verify your code against the real tests. Do NOT
reimplement what a tool already does. Do NOT skip the verification step.

```python
{tools_manual}
```

REQUIRED WORKFLOW (write the function ONCE per turn):
1. Write your implementation ONLY inside a string: `code = '''...'''`.
   NEVER also write a bare `def` outside that string — that duplicates the
   function and wastes your output-token budget.
2. In the SAME block, call `result = run_tests(code=code, test_list=tests)`,
   `print(result)`, then `if '"success": true' in result: final_answer(code)`.
3. If the Observation shows `TEST_<i>: FAIL (...)`, rewrite `code` with the
   fix and run the same block again. Variables do NOT persist between turns:
   each turn must redefine `code` and `tests`.

# SANDBOX
- Allowed imports: {authorized_imports}
- Allowed directories: {allowed_directories}
- Timeout: {max_execution_time_seconds}s per execution (MCP-side actions excluded)
- Memory: {max_memory_mb}MB
- No network, no subprocess, no os.system.

# EXAMPLE (format only — NOT the real task)
Thought: I implement add, run the tests, and submit if they all pass.
```python
code = '''
def add(a, b):
    return a + b
'''
tests = ['assert add(2, 3) == 5', 'assert add(-1, 1) == 0']
result = run_tests(code=code, test_list=tests)
print(result)
if '"success": true' in result:
    final_answer(code)
```
<end_code>

(If a test fails, the Observation shows it; you continue with a new Thought + the fixed block.)

# GOAL
Submit the COMPLETE function source with `final_answer(code)`, only after `run_tests`
reports success, exactly as in the example above. `final_answer` is a built-in injected
by the sandbox — do not import it, do not redefine it. Keeping the source in a `'''`
string avoids quote collisions; if your function contains `'''` itself, use `\"\"\"`.
"""


SYSTEM_PROMPT_SWEBENCH = """\
# ROLE
You are an autonomous software engineering agent. You resolve a real repository bug by
running a strict Thought -> Code -> Observation loop. Each iteration, you use ONE tool/code;
the sandbox executes your code and returns the REAL Observation in the next user message.

# TASK
{problem_statement}

Repository: {repo}  (already checked out at /testbed)
{hints_section}

# METHOD — work through these phases IN ORDER
PHASE 1 — REPRODUCE: First name the EXPECTED behavior the report asks for — it is
  usually stated literally (e.g. "both lines should give the same result"). Build the
  smallest script that triggers the bug AND `assert`s that expected result, reusing the
  problem statement's repro steps verbatim plus a couple of close variants (the
  symmetric case, another input type) to see the bug's full shape. Save it as a file so
  you can re-run it unchanged in PHASE 4. Run it with run_command and READ the real
  failure — the traceback, or the assert that does not hold. Edit nothing yet: this
  failing run is your baseline, and later a repro that merely stops crashing is NOT a
  fix — its asserts must pass.
PHASE 2 — LOCALIZE: From the traceback, use search_code and
  search_function_or_class_definition_in_code to jump to the faulty symbol. Open a
  NARROW window with read_file around it. The bug may be INHERITED — if the class body
  is thin, read its BASE classes too. Use find_references to gauge blast radius. When
  unsure how a value or method actually behaves, do NOT theorize — run a one-line check
  with run_command and measure it.
PHASE 3 — FIX: Apply ONE minimal, focused edit with edit_file that fixes EXACTLY the
  reported bug and nothing else. Touch as few lines as possible; do not reformat,
  rename, refactor, or "improve" nearby code — even code that looks wrong. The grader
  runs hidden tests that currently PASS; every extra change risks breaking one and
  failing the whole task.
PHASE 4 — VERIFY: Re-run your PHASE 1 repro file unchanged — every assertion must now
  pass (not merely "no traceback"). Then run the touched area's EXISTING tests with
  run_command, e.g. `python -m pytest <path/to/test_file.py> -q` — the WHOLE file, not a
  single test, to surface any regression you introduced. On failure, fix and re-verify.
PHASE 5 — SUBMIT: Inspect get_patch(): a minimal diff touching SOURCE files only.

# EXECUTION MODEL — READ THIS FIRST
Your ```python``` block runs in a THIN, stdlib-only sandbox. It is ORCHESTRATION GLUE —
it is NOT where the repository runs. The repo '{repo}', its package, and its third-party
dependencies (numpy, pandas, etc.) exist ONLY inside the /testbed container. They are
NOT installed in your sandbox and CANNOT be imported there.
- NEVER `import` the repository's own package or its dependencies in your code block.
  `import <repo_package>` raises ModuleNotFoundError in the sandbox — this is expected,
  not a bug to fix.
- To execute ANYTHING from the repo (reproduce the bug, run a snippet, run tests), use
  `run_command` — it executes a real shell INSIDE /testbed, where the package and its
  full environment exist. Reproduce by running the snippet there directly:
      run_command(command='python -c "import <pkg>; <repro code>"', workdir='/testbed')
  or write a repro file with edit_file, then `run_command('python repro.py', '/testbed')`.
- Inside your sandbox block you may ONLY: call the injected tools, use print(), and
  call final_answer(). No repo imports, no real computation — just orchestration.

# OUTPUT FORMAT (STRICT — any deviation breaks the agent)
Every turn MUST follow exactly this template, and nothing else:

Thought: <one short paragraph: what you just learned from the last Observation, and what single action you will run this turn>
```python
<your python code — small, focused, with prints>
```
<end_code>

# HARD RULES
{common_hard_rules}
- NEVER edit, create, or delete a TEST file. The grader supplies its own hidden tests
  and reverts test files before grading — a diff that touches tests is wasted or
  rejected. Your patch must change SOURCE files only.
- NEVER run `git checkout`, `git reset`, `git stash`, `git revert` or any command that
  discards working-tree changes — it destroys your fix and the final diff.
- NEVER paste a patch from memory or from your training data. You have never seen this
  exact repository state. Explore first, then edit.
- NEVER `import` the repo package or its dependencies in your code block (see EXECUTION
  MODEL). Run repo code through `run_command` inside /testbed, never in the sandbox.
- BUDGET: {max_iterations} turns and a shared, accumulating input-token window. Read
  narrow line windows (not whole files); run targeted tests (not the full suite).
  Hints above are raw discussion snippets — they may be partial or misleading.

# AVAILABLE TOOLS — YOU MUST USE THEM
The following functions are injected as Python built-ins inside the sandbox.
They are the ONLY way to read, edit, run, and verify code in the real repository.
Do NOT reimplement what a tool already does. Do NOT skip the verification step.

```python
{tools_manual}
```

# SANDBOX
- Allowed imports: {authorized_imports}
- Allowed directories: {allowed_directories}
- Timeout: {max_execution_time_seconds}s per execution (MCP-side actions excluded)
- Memory: {max_memory_mb}MB
- No network, no subprocess, no os.system.

# GOAL
When your repro script's assertions all pass, the touched-area tests pass, AND
`get_patch()` shows a minimal source-only diff, submit:

Thought: Repro now succeeds, existing tests pass, the diff is minimal and source-only. I submit.
```python
final_answer(get_patch())
```
<end_code>

`final_answer` and `get_patch` are sandbox built-ins — do not import them, do not redefine them.
"""


JSON_TO_PY = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "list",
    "object": "dict[str, Any]",
    "null": "None",
}


def py_type(meta: dict[str, Any]) -> str:
    """Map a JSON-schema type fragment to its Python annotation.

    Args:
        meta: A JSON-schema fragment with a ``type`` key (and
            optionally ``items`` for arrays).

    Returns:
        The corresponding Python type name as a string, e.g.
        ``"list[str]"`` or ``"int"``.
    """
    t = str(meta.get("type", "Any"))
    if t == "array":
        items = meta.get("items") or {}
        inner = JSON_TO_PY.get(items.get("type", ""), "Any")
        return f"list[{inner}]"
    return JSON_TO_PY.get(t, t)


TOOL_EXAMPLES: dict[str, str] = {
    # MBPP
    "run_tests": (
        "# MBPP form: pass your implementation as a string + the test list.\n"
        "code = '''\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "'''\n"
        "tests = ['assert add(2, 3) == 5', 'assert add(-1, 1) == 0']\n"
        "print(run_tests(code=code, test_list=tests))\n"
        "# Expected output lines: 'TEST_0: PASS', 'TEST_1: PASS'.\n"
        "# WHEN: in the same block that defines `code`; then\n"
        "#       `if '\"success\": true' in result: final_answer(code)`.\n"
        "\n"
        "# SWE-bench form: zero-arg, runs the FULL pytest suite on /testbed — slow and\n"
        "# token-heavy. Prefer a TARGETED run via run_command on the relevant test file.\n"
        "# print(run_tests())\n"
        "# WHEN: a last broad check before final_answer, not for iterating on a fix."
    ),
    # SWE-bench: filesystem
    "read_file": (
        "# Read a specific window — DO NOT dump the whole file blindly.\n"
        "print(read_file(filepath='src/foo/bar.py', start_line=1, end_line=40))\n"
        "# WHEN: locating the bug site, inspecting an import block, or re-reading\n"
        "# a region immediately AFTER edit_file to confirm the change landed."
    ),
    "edit_file": (
        "# Replaces the FIRST occurrence of old_str by new_str. old_str must be\n"
        "# unique enough to match exactly one location (include surrounding context).\n"
        "print(edit_file(\n"
        "    filepath='src/foo/bar.py',\n"
        "    old_str='    return x + 1',\n"
        "    new_str='    return x - 1',\n"
        "))\n"
        "# WHEN: applying ONE minimal, focused fix. Re-read the region next turn."
    ),
    "list_files": (
        "# Explore a directory; use pattern to narrow (glob syntax).\n"
        "print(list_files(directory='src/foo', pattern='**/*.py'))\n"
        "# WHEN: first turns of PHASE 1, to map the repo before opening files."
    ),
    # SWE-bench: search
    "search_code": (
        "# Grep-like: regex over file contents. Returns 'path:line content'.\n"
        "print(search_code(pattern='def parse_config', file_pattern='**/*.py'))\n"
        "# WHEN: hunting for a string, error message, or callsite across the repo."
    ),
    "search_function_or_class_definition_in_code": (
        "# Jump straight to where a symbol is defined (AST-based, exact name).\n"
        "print(search_function_or_class_definition_in_code(name='MyClass'))\n"
        "# WHEN: you know the symbol name and want its file:line definition."
    ),
    "find_references": (
        "# All usages of a symbol, excluding its definition line.\n"
        "print(find_references(name='compute', filepath='src/foo/bar.py', line=42))\n"
        "# WHEN: assessing blast-radius of a change before editing a function."
    ),
    # SWE-bench: execution
    "run_command": (
        "# Arbitrary shell in workdir (default '.'). Returns exit_code + stdout + stderr.\n"
        "print(run_command(command='python -m pytest tests/test_foo.py::test_bug -x', workdir='/testbed'))\n"
        "# WHEN: running a SINGLE failing test to reproduce, or any one-off shell check.\n"
        "# Forbidden: network calls, package installs that touch the network."
    ),
    "get_patch": (
        "# Produces the final unified diff (git -c core.fileMode=false diff).\n"
        "print(get_patch())\n"
        "# WHEN: just before final_answer, to inspect the diff is minimal & on-topic.\n"
        "# Then submit with:  final_answer(get_patch())"
    ),
}


def build_tools_manual(tool_schemas: list[dict[str, Any]]) -> str:
    """Render MCP tool schemas as Python-style signatures + docstrings + examples."""
    blocks = []
    for tool in tool_schemas:
        name = tool["name"]
        description = (tool.get("description") or "").strip()
        params = tool.get("inputSchema", {}).get("properties", {})
        required = tool.get("inputSchema", {}).get("required", [])

        sig_parts = []
        for param, meta in params.items():
            ptype = py_type(meta)
            default = "" if param in required else " = None"
            sig_parts.append(f"{param}: {ptype}{default}")
        signature = f"def {name}({', '.join(sig_parts)}) -> str:"

        block = [signature]
        block.append('    """')
        if description:
            for line in description.splitlines():
                block.append(f"    {line}".rstrip())
            block.append("")
        example = TOOL_EXAMPLES.get(name)
        if example:
            block.append("    Example:")
            for line in example.splitlines():
                block.append(f"    {line}".rstrip() if line else "")
        block.append('    """')
        block.append("    ...")
        blocks.append("\n".join(block))

    return "\n\n".join(blocks)


def format_mbpp_prompt(
    task_definition: str,
    function_definition: str,
    test_list: list[str],
    tool_schemas: list[dict[str, Any]],
    authorized_imports: list[str],
    allowed_directories: list[str],
    max_execution_time_seconds: int,
    max_memory_mb: int,
    max_iterations: int,
) -> str:
    """Build the system prompt used by the MBPP agent.

    Args:
        task_definition: Natural-language description of the task.
        function_definition: Required Python function signature.
        test_list: Assertions the produced function must pass.
        tool_schemas: MCP tool schemas to render into the manual.
        authorized_imports: Whitelist of importable modules.
        allowed_directories: Whitelist of writable directories.
        max_execution_time_seconds: Sandbox execution timeout.
        max_memory_mb: Sandbox memory cap in megabytes.
        max_iterations: Maximum number of agent loop turns.

    Returns:
        The fully formatted MBPP system prompt.
    """
    return SYSTEM_PROMPT_MBPP.format(
        task_definition=task_definition,
        function_definition=function_definition,
        test_list="\n".join(f"  {t}" for t in test_list),
        tools_manual=build_tools_manual(tool_schemas),
        common_hard_rules=_COMMON_HARD_RULES,
        authorized_imports=", ".join(authorized_imports),
        allowed_directories=", ".join(allowed_directories),
        max_execution_time_seconds=max_execution_time_seconds,
        max_memory_mb=max_memory_mb,
        max_iterations=max_iterations,
    )


def format_swebench_prompt(
    problem_statement: str,
    repo: str,
    hints_text: str,
    tool_schemas: list[dict[str, Any]],
    authorized_imports: list[str],
    allowed_directories: list[str],
    max_execution_time_seconds: int,
    max_memory_mb: int,
    max_iterations: int,
) -> str:
    """Build the system prompt used by the SWE-bench agent.

    Args:
        problem_statement: GitHub issue description to resolve.
        repo: Target repository name (``owner/repo``).
        hints_text: Optional discussion snippets (may be empty).
        tool_schemas: MCP tool schemas to render into the manual.
        authorized_imports: Whitelist of importable modules.
        allowed_directories: Whitelist of writable directories.
        max_execution_time_seconds: Sandbox execution timeout.
        max_memory_mb: Sandbox memory cap in megabytes.
        max_iterations: Maximum number of agent loop turns.

    Returns:
        The fully formatted SWE-bench system prompt.
    """
    hints_section = f"Hints:\n{hints_text}\n" if hints_text.strip() else ""
    return SYSTEM_PROMPT_SWEBENCH.format(
        problem_statement=problem_statement,
        repo=repo,
        hints_section=hints_section,
        tools_manual=build_tools_manual(tool_schemas),
        common_hard_rules=_COMMON_HARD_RULES,
        authorized_imports=", ".join(authorized_imports),
        allowed_directories=", ".join(allowed_directories),
        max_execution_time_seconds=max_execution_time_seconds,
        max_memory_mb=max_memory_mb,
        max_iterations=max_iterations,
    )

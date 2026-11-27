# flake8: noqa: E501
# Prompts are LLM template strings — line breaks alter the model input.
from typing import Any


_COMMON_HARD_RULES = """\
- Each turn: one `Thought:` line, one ```python``` block, then `<end_code>` (generation stops there).
- Never write `Observation:` and never guess results: wait for the real Observation.
- A tool call alone on its own line is printed automatically (no print() needed).
- Variables do NOT persist between turns.
- Tools are already defined: call them directly, never write `def` for a tool."""


SYSTEM_PROMPT_MBPP = """\
Solve this Python task with a Thought -> Code -> Observation loop.

# TASK
{task_definition}

Signature: {function_definition}

Tests:
{test_list}

# FORMAT
Thought: <what you do this turn>
```python
<code>
```
<end_code>

# RULES
{common_hard_rules}
- Max {max_iterations} turns.
- Allowed imports: {authorized_imports}.
- Write the function ONLY inside `code = \'\'\'...\'\'\'`, test it with run_tests, submit with final_answer(code) once all tests pass.

# TOOLS
{tools_manual}

# EXAMPLE
Thought: Implement add, test it, submit if green.
```python
code = \'\'\'
def add(a, b):
    return a + b
\'\'\'
tests = ['assert add(2, 3) == 5']
result = run_tests(code=code, test_list=tests)
result
if '"success": true' in result:
    final_answer(code)
```
<end_code>
"""


SYSTEM_PROMPT_SWEBENCH = """\
Fix a bug in the repository {repo} (checked out at /testbed) with a Thought -> Code -> Observation loop.

# TASK
{problem_statement}
{hints_section}
# METHOD
1. REPRODUCE: write /tmp/repro.py (edit_file with old_str='') that asserts the EXPECTED behavior from the issue; run it with run_command('python /tmp/repro.py', '/testbed') and read the failure.
2. LOCALIZE: search_code / search_function_or_class_definition_in_code, then read_file on a narrow window (check base classes too).
3. FIX: one minimal edit_file on SOURCE code. No refactoring, no reformatting.
4. VERIFY: re-run the repro (asserts must pass), then the touched test file.
5. SUBMIT: check get_patch() is minimal, then final_answer(get_patch()).

# SANDBOX
Your code block only orchestrates tools. The repo and its dependencies are NOT importable there: run all repo code via run_command inside /testbed.

# TOOL OUTPUT
- run_command returns `exit_code`, `stdout`, `stderr`: check exit_code first.
- edit_file `Error: old_str not found` = nothing changed: copy the lines verbatim from read_file (without the `N: ` prefix).
- Output over ~4000 chars gets truncated: use `| tail -40`, `-q`, narrow windows.

# TEST RUNNER
sympy: `python bin/test <file>` | django: `python tests/runtests.py <dotted.module>` | otherwise: `python -m pytest <file> -q -x`. Never install pytest.

# FORMAT
Thought: <what you learned + the single action of this turn>
```python
<code>
```
<end_code>

# RULES
{common_hard_rules}
- Max {max_iterations} turns; read narrow windows, run targeted tests.
- Never modify test files. Never run git checkout/reset/stash/revert.
- Do not write a patch from memory: explore first. Hints may be wrong.
- Allowed imports: {authorized_imports}.

# TOOLS
{tools_manual}

# SUBMIT
```python
final_answer(get_patch())
```
<end_code>
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


def build_tools_manual(tool_schemas: list[dict[str, Any]]) -> str:
    """Render MCP tool schemas as one plain-text line per tool.

    Plain text (not ``def`` stubs in a code fence) so small models call
    the tools instead of copying their definitions.
    """
    lines = []
    for tool in tool_schemas:
        schema = tool.get("inputSchema", {})
        required = schema.get("required", [])
        sig_parts = []
        for param, meta in schema.get("properties", {}).items():
            default = ""
            if param not in required:
                default = f" = {meta.get('default')!r}"
            sig_parts.append(f"{param}: {py_type(meta)}{default}")
        summary = (tool.get("description") or "").strip().split("\n")[0]
        lines.append(
            f"- {tool['name']}({', '.join(sig_parts)}) -> str: {summary}")
    return "\n".join(lines)


def format_mbpp_prompt(
    task_definition: str,
    function_definition: str,
    test_list: list[str],
    tool_schemas: list[dict[str, Any]],
    authorized_imports: list[str],
    max_iterations: int,
) -> str:
    """Build the system prompt used by the MBPP agent.

    Args:
        task_definition: Natural-language description of the task.
        function_definition: Required Python function signature.
        test_list: Assertions the produced function must pass.
        tool_schemas: MCP tool schemas to render into the manual.
        authorized_imports: Whitelist of importable modules.
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
        max_iterations=max_iterations,
    )


def format_swebench_prompt(
    problem_statement: str,
    repo: str,
    hints_text: str,
    tool_schemas: list[dict[str, Any]],
    authorized_imports: list[str],
    max_iterations: int,
) -> str:
    """Build the system prompt used by the SWE-bench agent.

    Args:
        problem_statement: GitHub issue description to resolve.
        repo: Target repository name (``owner/repo``).
        hints_text: Optional discussion snippets (may be empty).
        tool_schemas: MCP tool schemas to render into the manual.
        authorized_imports: Whitelist of importable modules.
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
        max_iterations=max_iterations,
    )

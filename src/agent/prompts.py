# flake8: noqa: E501
# Prompts are LLM template strings — line breaks alter the model input.
from typing import Any


_COMMON_HARD_RULES = """\
- Each turn: one `Thought:` line, one ```python``` block, then `<end_code>` (generation stops there).
- Never write `Observation:` and never guess results: wait for the real Observation.
- Always `print(...)` tool results, otherwise the Observation is empty."""


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
- Max {max_iterations} turns. Variables do NOT persist between turns.
- Allowed imports: {authorized_imports}.
- Write the function ONLY inside `code = \'\'\'...\'\'\'`, test it with run_tests, submit with final_answer(code) once all tests pass.

# TOOLS
```python
{tools_manual}
```

# EXAMPLE
Thought: Implement add, test it, submit if green.
```python
code = \'\'\'
def add(a, b):
    return a + b
\'\'\'
tests = ['assert add(2, 3) == 5']
result = run_tests(code=code, test_list=tests)
print(result)
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
```python
{tools_manual}
```

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


TOOL_EXAMPLES: dict[str, str] = {
    "run_tests": "print(run_tests(code=code, test_list=tests))",
    "read_file": "print(read_file(filepath='src/foo.py', start_line=1, end_line=40))",
    "edit_file": "print(edit_file(filepath='src/foo.py', old_str='return x + 1', new_str='return x - 1'))",
    "list_files": "print(list_files(directory='src', pattern='**/*.py'))",
    "search_code": "print(search_code(pattern='def parse', file_pattern='**/*.py'))",
    "search_function_or_class_definition_in_code": "print(search_function_or_class_definition_in_code(name='MyClass'))",
    "find_references": "print(find_references(name='compute', filepath='src/foo.py', line=42))",
    "run_command": "print(run_command(command='python /tmp/repro.py', workdir='/testbed'))",
    "get_patch": "print(get_patch())",
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

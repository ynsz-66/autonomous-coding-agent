import re
import subprocess
from fastmcp import FastMCP
from pathlib import Path
import ast
import os

mcp = FastMCP("swebench-tools")

def detect_testbed_root() -> str:
    """Pick the best testbed root available in this environment."""
    env_root = os.getenv("TESTBED_PATH")
    if env_root:
        return env_root
    if Path("/testbed").exists():
        return "/testbed"
    repo_root = Path(__file__).resolve().parent.parent
    candidate = repo_root / "exams" / "sandbox_tests" / "testbed"
    if candidate.exists():
        return str(candidate)
    return "/testbed"


TESTBED_ROOT = detect_testbed_root()


def resolve_testbed_path(path: str) -> Path:
    """Resolve testbed-relative paths against the configured root."""
    if path.startswith("/testbed"):
        suffix = path[len("/testbed"):].lstrip("/")
        return Path(TESTBED_ROOT) / suffix
    return Path(path)

# file system tools


@mcp.tool()
def read_file(filepath: str, start_line: int, end_line: int) -> str:
    """Return a numbered window of lines from a file.

    Args:
        filepath: Path of the file to read.
        start_line: 1-indexed first line (inclusive).
        end_line: 1-indexed last line (inclusive).

    Returns:
        Newline-joined ``<n>: <content>`` lines or an error message.
    """
    try:
        results = []
        resolved = resolve_testbed_path(filepath)
        with open(resolved, 'r') as f:
            content = f.readlines()[start_line - 1:end_line]
        for i, line in enumerate(content):
            results.append(f"{start_line + i}: {line}")
        return "\n".join(r for r in results) if results else "No matches found"
    except FileNotFoundError:
        return f"Error, file not found ({filepath})."


@mcp.tool()
def edit_file(filepath: str, old_str: str, new_str: str) -> str:
    """Replace the first occurrence of ``old_str`` by ``new_str``.

    Runs ``ruff check`` afterwards on Python files and reports any
    lint regression introduced by the edit.

    Args:
        filepath: Path of the file to modify.
        old_str: Text to be replaced (must occur at least once).
        new_str: Replacement text.

    Returns:
        A status message describing success or the ruff issues.
    """
    try:
        resolved = resolve_testbed_path(filepath)
        with open(resolved, 'r') as f:
            content = f.read()
        with open(resolved, 'w') as f:
            f.write(content.replace(old_str, new_str, 1))
        if str(resolved).endswith(".py"):
            proc = subprocess.run(
                ["python", "-m", "ruff", "check", str(resolved)],
                capture_output=True, text=True, timeout=30,
            )
            if proc.returncode != 0:
                return (
                    f"File edited, but ruff check reported issues:\n"
                    f"{proc.stdout}{proc.stderr}"
                )
        return "Files edited successfully"
    except FileNotFoundError:
        return f"Error, file ({filepath}) not found."


@mcp.tool()
def list_files(directory: str, pattern: str = "*") -> str:
    """List files under a directory matching a glob pattern.

    Args:
        directory: Directory to scan.
        pattern: Glob pattern; defaults to ``*``.

    Returns:
        Newline-joined paths or an error message.
    """
    try:
        base_dir = resolve_testbed_path(directory)
        files = [str(file) for file in base_dir.glob(pattern)]
        return "\n".join(files) if files else "No files found."
    except Exception as e:
        return f"Error: {e}"


# code search tools
@mcp.tool()
def search_code(pattern: str, file_pattern: str = "**/*.py") -> str:
    """Grep a regex pattern across files matching a glob.

    Args:
        pattern: Regex to search for.
        file_pattern: Glob restricting which files are scanned.

    Returns:
        Newline-joined ``path:line content`` matches or a message.
    """
    try:
        results = []
        for filepath in Path(TESTBED_ROOT).glob(file_pattern):
            with open(filepath, "r", errors="ignore") as f:
                for i, line in enumerate(f, 1):
                    if re.search(pattern, line):
                        results.append(
                            f"{filepath.resolve()}:{i} {line.rstrip()}")
        return "\n".join(results) if results else "No matches found."
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def search_function_or_class_definition_in_code(name: str) -> str:
    """Locate function/class definitions matching ``name`` via AST.

    Args:
        name: Exact symbol name to look up.

    Returns:
        Newline-joined ``path:line declaration`` matches.
    """
    try:
        results = []
        for filepath in Path(TESTBED_ROOT).glob("**/*.py"):
            try:
                source = filepath.read_text(errors="ignore")
                tree = ast.parse(source)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(
                    node,
                    (ast.FunctionDef,
                     ast.AsyncFunctionDef,
                     ast.ClassDef)):
                    if node.name == name:
                        results.append(
                            f"{filepath.resolve()}:"
                            f"{node.lineno} "
                            f"{source.splitlines()[node.lineno - 1].rstrip()}"
                        )
        return "\n".join(results) if results else "No definition found."
    except Exception as e:
        return f"Error: {e}"


def is_reference(node: ast.AST, name: str) -> bool:
    """Return True if ``node`` is an AST reference to ``name``.

    Args:
        node: AST node to inspect.
        name: Bare name to match against.

    Returns:
        True for matching ``ast.Name`` or ``ast.Attribute`` nodes.
    """
    return (isinstance(node, ast.Name) and node.id == name) or \
           (isinstance(node, ast.Attribute) and node.attr == name)


def is_definition(
        node: ast.AST,
        definition_file: str,
        definition_line: int,
        current_path: Path) -> bool:
    """Return True if ``node`` is the symbol's definition site.

    Args:
        node: AST node being inspected.
        definition_file: Path of the file declaring the symbol.
        definition_line: Line number of the declaration.
        current_path: Path currently being scanned.

    Returns:
        True when the node lives at the declaration location.
    """
    return (
        str(current_path.resolve()) == str(Path(definition_file).resolve())
        and getattr(node, "lineno", None) == definition_line
    )


@mcp.tool()
def find_references(name: str, filepath: str, line: int) -> str:
    """List references to a symbol across the repository.

    Args:
        name: Bare symbol name to search for.
        filepath: File where the symbol is declared.
        line: Declaration line, excluded from the result.

    Returns:
        Newline-joined ``path:line content`` references.
    """
    try:
        results = []
        resolved_target = resolve_testbed_path(filepath)
        for path in Path(TESTBED_ROOT).glob("**/*.py"):
            try:
                source = path.read_text(errors="ignore")
                tree = ast.parse(source)
            except SyntaxError:
                continue
            file_lines = source.splitlines()
            for node in ast.walk(tree):
                if not hasattr(node, "lineno"):
                    continue
                if is_definition(node, str(resolved_target), line, path):
                    continue
                if is_reference(node, name):
                    line_text = file_lines[node.lineno - 1].rstrip()
                    results.append(
                        f"{path.resolve()}:{node.lineno} {line_text}"
                    )
        return "\n".join(results) if results else "No references found."
    except Exception as e:
        return f"Error: {e}"


CONDA_ACTIVATE = "source /opt/miniconda3/bin/activate testbed"


@mcp.tool()
def run_command(command: str, workdir: str = ".") -> str:
    """Run a shell command inside the SWE-bench testbed conda env.

    Args:
        command: Shell command to execute.
        workdir: Working directory for the subprocess.

    Returns:
        Formatted block containing exit code, stdout and stderr.
    """
    try:
        if Path("/opt/miniconda3/bin/activate").exists():
            wrapped = f"{CONDA_ACTIVATE} && {command}"
        else:
            wrapped = command
        resolved_workdir = resolve_testbed_path(workdir)
        proc = subprocess.run(
            ["bash", "-c", wrapped], cwd=resolved_workdir,
            capture_output=True, text=True, timeout=600
        )
        return (
            f"exit_code: {proc.returncode}"
            f"\nstdout:\n{proc.stdout}"
            f"\nstderr:\n{proc.stderr}"
            )
    except subprocess.TimeoutExpired:
        return "Error: command timed out."
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def run_tests() -> str:
    """Run the full pytest suite of the SWE-bench testbed.

    Returns:
        Formatted exit-code/stdout/stderr block from pytest.
    """
    return str(run_command(
        "python -m pytest --tb=short -q", workdir="/testbed"
    ))


@mcp.tool()
def get_patch() -> str:
    """Return the current ``git diff`` of the testbed.

    Returns:
        The unified diff produced by
        ``git -c core.fileMode=false diff``.
    """
    return str(run_command(
        "git -c core.fileMode=false diff", workdir="/testbed"
    ))


if __name__ == "__main__":
    try:
        mcp.run(transport="stdio", show_banner=False, log_level="WARNING")
    except Exception as e:
        print(f"[MCP ERROR] {e}")

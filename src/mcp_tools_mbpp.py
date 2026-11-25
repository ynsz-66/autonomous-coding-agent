import subprocess
import argparse
import sys
import tempfile
import json
import os
from fastmcp import FastMCP
from pathlib import Path

mcp = FastMCP("mbpp-tools")


@mcp.tool()
def run_tests(
        code: str,
    test_list: list[str],
        test_imports: list[str] | None = None) -> str:
    """Execute MBPP-style assertions against a candidate function.

    Args:
        code: Candidate implementation as a Python source string.
        test_list: List of ``assert`` statements to evaluate.
        test_imports: Optional list of import statements prepended
            to the script before the candidate code.

    Returns:
        A multi-line report with one ``TEST_<i>: PASS|FAIL`` per
        assertion, optionally followed by captured ``stderr``.
    """
    imports_block = "\n".join(test_imports or [])
    script = (
        f"{imports_block}\n"
        f"{code}\n"
        f"tests = {test_list!r}\n"
        "for i, t in enumerate(tests):\n"
        "    try:\n"
        "        exec(t)\n"
        "        print(f'TEST_{i}: PASS')\n"
        "    except Exception as e:\n"
        "        print(f'TEST_{i}: FAIL ({type(e).__name__}: {e})')\n"
    )

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, timeout=30,
        )
        report = proc.stdout.rstrip()
        if proc.stderr.strip():
            report += f"\nstderr:\n{proc.stderr.rstrip()}"
        success = proc.returncode == 0 and "FAIL" not in report
        return json.dumps({"success": success, "output": report})
    except subprocess.TimeoutExpired:
        return json.dumps({
            "success": False,
            "output": "Error: tests timed out after 30s.",
        })
    finally:
        Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", default="stdio")
    args = parser.parse_args()

    try:
        if args.transport == "streamable-http":
            mcp.run(transport="http", show_banner=False,
                    host=os.environ.get("HOST", "127.0.0.1"),
                    port=int(os.environ.get("PORT", "8000")),
                    log_level="WARNING")
        else:
            mcp.run(transport="stdio", show_banner=False,
                    log_level="WARNING")
    except Exception as e:
        print(f"[MCP ERROR] {e}")

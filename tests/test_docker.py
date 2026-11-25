import sys
from pathlib import Path

import pytest

STUDENT_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(STUDENT_DIR))

# The `docker` SDK is an optional dependency; skip the whole module cleanly
# when it (or the daemon) isn't available rather than erroring at collection.
pytest.importorskip("docker")

from agent.docker import (  # noqa: E402
    pull_image,
    start_container,
    cleanup,
    exec_command,
)
from mcp_config.client_swebench import list_tools, make_builtins  # noqa: E402
from sandbox import SandboxConfig  # noqa: E402


DOCKER_IMAGE = "swebench/sweb.eval.x86_64.sympy_1776_sympy-13480:latest"
CODE_DIR = str(STUDENT_DIR)

pytestmark = pytest.mark.docker


def section(title: str) -> None:
    print(f"\n{'=' * 50}\n  {title}\n{'=' * 50}")


@pytest.fixture(scope="module")
def container():
    """Pull the image, start a container, pip-install requirements, clean up.

    The pip install runs here (not as a separate test) because the in-container
    MCP stdio server queried by ``list_tools`` needs the requirements present.
    """
    section(f"pull_image — {DOCKER_IMAGE}")
    pull_image(DOCKER_IMAGE)

    section(f"start_container — mount {CODE_DIR} -> /agent")
    c = start_container(DOCKER_IMAGE, CODE_DIR)
    print(f"  container id: {c.id[:12]}")
    assert c.id

    section("pip install -r /agent/agent/requirements.txt (inside container)")
    exit_code, output = exec_command(
        c, "python -m pip install -r /agent/agent/requirements.txt")
    print(output[-2000:] if len(output) > 2000 else output)
    assert exit_code == 0, f"pip install failed (exit_code={exit_code})"

    try:
        yield c
    finally:
        print("\nCleaning up container...")
        cleanup(c)


@pytest.fixture(scope="module")
def tools(container):
    section("list_tools — MCP stdio server inside the container")
    result = list_tools(container.id)
    names = [t.name for t in result.tools]
    print(f"  tools: {names}")
    assert result.tools, "MCP server exposed no tools"
    for expected in ("run_command", "list_files", "read_file"):
        assert expected in names, f"missing tool {expected!r}"
    return result


def test_mount_is_visible(container) -> None:
    section("mount — /agent/mcp_tools_swebench.py visible inside container")
    exit_code, output = exec_command(
        container, "ls -la /agent/mcp_tools_swebench.py")
    print(output)
    assert exit_code == 0, f"mount not visible (exit_code={exit_code})"
    assert "mcp_tools_swebench.py" in output
    print("PASSED")


def test_testbed_present(container) -> None:
    section("/testbed — repo checkout present inside container")
    exit_code, output = exec_command(container, "ls /testbed")
    print(output)
    assert exit_code == 0, f"/testbed missing (exit_code={exit_code})"
    print("PASSED")


def test_list_tools(tools) -> None:
    section("list_tools — exposes the expected MCP tools")
    names = [t.name for t in tools.tools]
    assert "run_command" in names
    print("PASSED")


def test_make_builtins_and_run(container, tools) -> None:
    section("make_builtins — invoke MCP tools from inside the sandbox")
    sandbox = SandboxConfig(
        extra_builtins=make_builtins(tools, container.id))

    # run_command executes inside the container (/testbed by default).
    result = sandbox.run("print(run_command('echo hello-from-container'))")
    print(result)
    assert result["error"] is None, f"sandbox error: {result['error']}"
    assert "hello-from-container" in result["stdout"]
    assert "exit_code:" in result["stdout"]
    print("PASSED (run_command)")

    # list_files should see the real repo checkout under /testbed.
    result = sandbox.run("print(list_files('/testbed', '*.py'))")
    print(result["stdout"][:1000])
    assert result["error"] is None, f"sandbox error: {result['error']}"
    assert ".py" in result["stdout"]
    print("PASSED (list_files)")

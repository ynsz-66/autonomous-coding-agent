import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mcp_tools_swebench import (  # noqa: E402
    list_files,
    search_code,
    read_file,
    edit_file,
    find_references,
    search_function_or_class_definition_in_code,
    run_command,
    get_patch,
    run_tests,
)


FIXTURE_CONTENT = '''\
def greet(name):
    return f"Hello, {name}!"

class Dog:
    def bark(self):
        return greet("Dog")

result = greet("world")
'''


@pytest.fixture(autouse=True)
def workdir(tmp_path, monkeypatch):
    """Run each test against a tiny temp dir holding a single fixture file.

    search_code / find_references / search_function_or_class scan **/*.py
    under ``TESTBED_ROOT`` (default ``/testbed``, absent off the SWE-bench
    image). Point it at this sandbox so the tools see only the fixture and
    don't recurse into .venv (numpy etc.). A fresh fixture file per test keeps
    the tests independent (e.g. edit_file's mutation doesn't leak).
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "src.mcp_tools_swebench.TESTBED_ROOT", str(tmp_path))
    fixture_path = tmp_path / "fixture_sample.py"
    fixture_path.write_text(FIXTURE_CONTENT)
    yield fixture_path


@pytest.fixture
def fixture_path(workdir):
    return workdir


def section(title: str) -> None:
    print(f"\n{'=' * 50}\n  {title}\n{'=' * 50}")


def test_read_file(fixture_path) -> None:
    section("read_file")
    result = read_file(str(fixture_path), 1, 4)
    print(result)
    assert "greet" in result
    print("PASSED")


def test_read_file_missing() -> None:
    section("read_file — missing file")
    result = read_file("/nonexistent/path.py", 1, 5)
    print(result)
    assert "not found" in result.lower()
    print("PASSED")


def test_edit_file(fixture_path) -> None:
    section("edit_file")
    result = edit_file(
        str(fixture_path),
        'return f"Hello, {name}!"',
        'return f"Hi, {name}!"',
    )
    print(result)
    after = read_file(str(fixture_path), 1, 9)
    assert "Hi," in after
    assert "Hello," not in after
    print("PASSED")


def test_list_files(fixture_path) -> None:
    section("list_files")
    result = list_files(str(fixture_path.parent), "*.py")
    print(result)
    assert "fixture_sample.py" in result
    print("PASSED")


def test_search_code() -> None:
    section("search_code")
    result = search_code("def greet", "*.py")
    print(result)
    assert "greet" in result
    print("PASSED")


def test_search_function_or_class() -> None:
    section("search_function_or_class_definition_in_code")
    result = search_function_or_class_definition_in_code("greet")
    print(result)
    assert "greet" in result

    result2 = search_function_or_class_definition_in_code("Dog")
    print(result2)
    assert "Dog" in result2
    print("PASSED")


def test_find_references(fixture_path) -> None:
    section("find_references")
    result = find_references("greet", str(fixture_path), 1)
    print(result)
    assert "greet" in result
    print("PASSED")


def test_run_command_format() -> None:
    section("run_command — output format (any command)")
    result = run_command("echo hello")
    print(result)
    assert "exit_code:" in result
    assert "stdout:" in result
    assert "stderr:" in result
    print("PASSED")


def test_run_command_workdir() -> None:
    section("run_command — workdir")
    result = run_command("pwd", workdir="/tmp")
    print(result)
    assert "exit_code:" in result
    print("PASSED")


@pytest.mark.testbed
def test_get_patch() -> None:
    section("get_patch (requires /testbed)")
    result = get_patch()
    print(result)
    assert result
    print("PASSED")


@pytest.mark.testbed
def test_run_tests_swebench() -> None:
    section("run_tests (requires /testbed)")
    result = run_tests()
    print(result)
    assert result
    print("PASSED")


def test_edit_file_old_str_missing(fixture_path) -> None:
    section("edit_file — old_str not found")
    before = fixture_path.read_text()
    result = edit_file(str(fixture_path), "no such line", "x")
    assert result.startswith("Error: old_str not found")
    assert fixture_path.read_text() == before


def test_edit_file_creates_new_file(tmp_path) -> None:
    section("edit_file — create with old_str=''")
    target = tmp_path / "sub" / "repro.py"
    result = edit_file(str(target), "", "print('hi')\n")
    assert "File created" in result
    assert target.read_text() == "print('hi')\n"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mcp_tools_mbpp import run_tests  # noqa: E402


def section(title: str) -> None:
    print(f"\n{'=' * 50}\n  {title}\n{'=' * 50}")


def test_run_tests_all_pass() -> None:
    section("run_tests — all assertions pass")
    code = "def add(a, b):\n    return a + b\n"
    tests = [
        "assert add(1, 2) == 3",
        "assert add(-1, 1) == 0",
        "assert add(0, 0) == 0",
    ]
    result = run_tests(code, tests)
    print(result)
    assert "TEST_0: PASS" in result
    assert "TEST_1: PASS" in result
    assert "TEST_2: PASS" in result
    print("PASSED")


def test_run_tests_some_fail() -> None:
    section("run_tests — some assertions fail")
    code = "def mul(a, b):\n    return a + b\n"
    tests = [
        "assert mul(2, 3) == 6",
        "assert mul(0, 5) == 5",
    ]
    result = run_tests(code, tests)
    print(result)
    assert "TEST_0: FAIL" in result
    assert "TEST_1: PASS" in result
    print("PASSED")


def test_run_tests_with_imports() -> None:
    section("run_tests — with test_imports")
    code = "def square_root(x):\n    return math.sqrt(x)\n"
    tests = ["assert square_root(9) == 3.0"]
    result = run_tests(code, tests, test_imports=["import math"])
    print(result)
    assert "TEST_0: PASS" in result
    print("PASSED")


def test_run_tests_syntax_error() -> None:
    section("run_tests — broken candidate code")
    code = "def broken(:\n    return 1\n"
    tests = ["assert broken() == 1"]
    result = run_tests(code, tests)
    print(result)
    assert "stderr" in result or "FAIL" in result
    print("PASSED")

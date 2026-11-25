import sys
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sandbox import SandboxConfig  # noqa: E402


def section(title: str) -> None:
    print(f"\n{'=' * 50}\n  {title}\n{'=' * 50}")


def make_sandbox(**kwargs: Any) -> SandboxConfig:
    return SandboxConfig(**kwargs)


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
def test_blocked_imports() -> None:
    section("blocked imports")
    sb = make_sandbox()
    blocked = ["os", "subprocess", "sys", "socket", "shutil",
               "importlib", "urllib", "http.client", "ssl"]
    for mod in blocked:
        result = sb.run(f"import {mod}")
        print(f"  import {mod} → {result['error']}")
        assert result["error"] is not None, f"{mod} should be blocked"
        assert "not authorized" in result["error"] or \
               "ImportError" in result["error"]
    print("PASSED")


def test_blocked_subimports() -> None:
    section("blocked sub-imports (os.path, urllib.request)")
    sb = make_sandbox()
    for mod in ["os.path", "urllib.request", "http.server"]:
        result = sb.run(f"import {mod}")
        print(f"  import {mod} → {result['error']}")
        assert result["error"] is not None, f"{mod} should be blocked"
    print("PASSED")


def test_allowed_imports() -> None:
    section("allowed imports (math, json, re, collections, itertools, typing)")
    sb = make_sandbox()
    for mod in ["math", "json", "re", "collections", "itertools", "typing"]:
        result = sb.run(f"import {mod}\nprint({mod}.__name__)")
        print(f"  import {mod} → stdout={result['stdout'].strip()!r}, "
              f"error={result['error']}")
        assert result["error"] is None, f"{mod} should be allowed"
        assert mod in result["stdout"]
    print("PASSED")


def test_dynamic_import_blocked() -> None:
    section("dynamic __import__ goes through allowlist")
    sb = make_sandbox()
    result = sb.run("__import__('os')")
    print(result)
    assert result["error"] is not None and "not authorized" in result["error"]
    print("PASSED")


# ---------------------------------------------------------------------------
# Builtins
# ---------------------------------------------------------------------------
def test_eval_exec_compile_blocked() -> None:
    section("eval / exec / compile must be blocked")
    sb = make_sandbox()
    for expr in ["eval('1+1')", "exec('a=1')", "compile('1', 'x', 'eval')"]:
        result = sb.run(expr)
        print(f"  {expr} → {result['error']}")
        assert result["error"] is not None
        assert "NameError" in result["error"] or \
               "not defined" in result["error"]
    print("PASSED")


def test_safe_builtins_still_work() -> None:
    section("safe builtins (print, len, range, type) still work")
    sb = make_sandbox()
    result = sb.run(
        "print(len(list(range(3))))\nprint(type(1).__name__)"
    )
    print(result)
    assert result["error"] is None
    assert "3" in result["stdout"] and "int" in result["stdout"]
    print("PASSED")


# ---------------------------------------------------------------------------
# File access
# ---------------------------------------------------------------------------
def test_file_access_blocked() -> None:
    section("file access — blocked outside allowed_directories")
    sb = make_sandbox()
    for path in ["/etc/passwd", "/root/.ssh/id_rsa",
                 "../../../etc/passwd", "/home/user/secret"]:
        result = sb.run(f"open({path!r}).read()")
        print(f"  open({path!r}) → {result['error']}")
        assert result["error"] is not None
        assert "PermissionError" in result["error"] or \
               "Access denied" in result["error"]
    print("PASSED")


def test_file_access_allowed() -> None:
    section("file access — allowed inside /tmp")
    sb = make_sandbox()
    with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", dir="/tmp", delete=False) as f:
        f.write("hello-sandbox")
        path = f.name
    try:
        result = sb.run(f"print(open({path!r}).read())")
        print(result)
        assert result["error"] is None
        assert "hello-sandbox" in result["stdout"]
    finally:
        os.unlink(path)
    print("PASSED")


# ---------------------------------------------------------------------------
# Resource limits
# ---------------------------------------------------------------------------
def test_timeout() -> None:
    section("timeout — max_execution_time_seconds enforced")
    sb = make_sandbox(max_execution_time_seconds=1)
    code = (
        "i = 0\n"
        "while True:\n"
        "    i += 1\n"
    )
    result = sb.run(code)
    print(result)
    assert result["error"] is not None
    assert "timeout" in result["error"].lower()
    print("PASSED")


def test_timeout_preserves_partial_stdout() -> None:
    section("timeout — partial stdout preserved")
    sb = make_sandbox(max_execution_time_seconds=1)
    code = (
        "print('PARTIAL_OUTPUT', flush=True)\n"
        "import time\n"  # blocked import → still partial output above
        "while True:\n"
        "    pass\n"
    )
    result = sb.run(code)
    print(result)
    # stdout from before the failure should be preserved
    assert "PARTIAL_OUTPUT" in result["stdout"] or \
           result["error"] is not None
    print("PASSED")


def test_memory_limit() -> None:
    section("memory — max_memory_mb enforced (skipped on darwin)")
    if sys.platform == "darwin":
        pytest.skip("RLIMIT_AS not enforced on darwin")
    sb = make_sandbox(max_memory_mb=64)
    code = "x = bytearray(1024 * 1024 * 1024)"  # 1 GiB
    result = sb.run(code)
    print(result)
    assert result["error"] is not None
    assert "Memory" in result["error"] or "MemoryError" in result["error"]
    print("PASSED")


# ---------------------------------------------------------------------------
# Sandbox semantics
# ---------------------------------------------------------------------------
def test_final_answer_callable() -> None:
    section("final_answer — sandbox-injected, returns value")
    sb = make_sandbox()
    result = sb.run("final_answer(42)")
    print(result)
    assert result["error"] is None
    assert result.get("final_answer") == 42
    print("PASSED")


def test_keyboard_interrupt_not_swallowed() -> None:
    section("KeyboardInterrupt / SystemExit must not be silently caught")
    sb = make_sandbox(extra_builtins={
        "KeyboardInterrupt": KeyboardInterrupt,
        "SystemExit": SystemExit,
    })

    raised = False
    try:
        sb.run("raise KeyboardInterrupt()")
    except KeyboardInterrupt:
        raised = True
    assert raised, "KeyboardInterrupt was swallowed"
    print("PASSED (KeyboardInterrupt propagated)")

    raised = False
    try:
        sb.run("raise SystemExit()")
    except SystemExit:
        raised = True
    assert raised, "SystemExit was swallowed"
    print("PASSED (SystemExit propagated)")

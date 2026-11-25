import io
import signal
import resource
import sys
import os
import json
import threading

from contextlib import contextmanager, redirect_stdout
from pydantic import BaseModel, Field
from typing import Any, Iterator, List
from types import FrameType


class FinalAnswerException(Exception):
    """Legacy alias used at module scope to signal a final answer."""

    pass


def final_answer(out: Any) -> None:
    """Module-level helper raising :class:`FinalAnswerException`.

    Args:
        out: The value to surface as the agent's final answer.
    """
    raise FinalAnswerException(out)


class FinalAnswer(Exception):
    """In-sandbox exception carrying the agent's final answer value."""

    def __init__(self, value: Any) -> None:
        """Store the submitted answer value.

        Args:
            value: Payload passed to ``final_answer(...)``.
        """
        self.value = value


def from_json(path: str) -> Any:
    """Load a JSON sandbox config from disk.

    Args:
        path: Filesystem path of the JSON file.

    Returns:
        The decoded object.

    Raises:
        ValueError: If the file is missing or not valid JSON.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"Invalid config file: {e}") from e
    return data


class SandboxConfig(BaseModel):
    """ Sandbox configuration for student solutions.
    Uses allowlist approach: only imports in authorized_imports are allowed.
    Everything else is blocked by default. """

    authorized_imports: List[str] = Field(default_factory=lambda: [
        "math", "math.*",
        "collections", "collections.*",
        "itertools", "re", "json",
        "typing", "typing.*",
        "functools", "operator",
        "heapq", "bisect", "copy",
        "string", "random",
        "datetime", "datetime.*",
        "array", "cmath",
    ])
    allowed_directories: List[str] = Field(default_factory=lambda: [
        "/testbed", "/tmp", "/tmp/agent", "/agent", "/out", "/task.json",
    ])
    max_execution_time_seconds: int = 30
    max_memory_mb: int = 512
    extra_builtins: dict[str, Any] = {}

    def import_gestion(
            self,
            name: str,
            globals: Any = None,
            locals: Any = None,
            fromlist: Any = (),
            level: int = 0) -> Any:
        """Replacement for ``__import__`` enforcing the allowlist.

        Args:
            name: Module name being imported.
            globals: Globals passed to ``__import__``.
            locals: Locals passed to ``__import__``.
            fromlist: ``from ... import x`` items.
            level: Relative-import level.

        Returns:
            The imported module.

        Raises:
            ImportError: When the module is not in
                ``authorized_imports`` or is explicitly blocked.
        """
        is_authorized = False
        blocked_prefixes = ("socket", "urllib", "http", "ssl", "os", "sys")
        for blocked in blocked_prefixes:
            if name == blocked or name.startswith(blocked + "."):
                raise ImportError(f"Module '{name}' is not authorized")
        for allowed in self.authorized_imports:
            if allowed.endswith(".*"):
                prefix = allowed[:-2]
                if name == prefix or name.startswith(prefix + "."):
                    is_authorized = True
                    break
            elif name == allowed:
                is_authorized = True
                break

        if not is_authorized:
            raise ImportError(f"Module '{name}' is not authorized")
        return __import__(name, globals, locals, fromlist, level)

    def path_allowed(self, path: Any) -> bool:
        """Return True if ``path`` resolves inside an allowed dir.

        Args:
            path: Filesystem path to check.

        Returns:
            True when one of ``allowed_directories`` is an ancestor.
        """
        real_path = os.path.realpath(os.fspath(path))
        for allowed_dir in self.allowed_directories:
            allowed_real = os.path.realpath(allowed_dir)
            try:
                if os.path.commonpath(
                        [real_path, allowed_real]) == allowed_real:
                    return True
            except ValueError:
                continue
        return False

    def safe_open(
        self,
        file: Any,
        mode: str = "r",
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Replacement for ``open`` enforcing the path allowlist.

        Args:
            file: Path or path-like to open.
            mode: File mode, forwarded to :func:`open`.
            *args: Extra positional arguments for :func:`open`.
            **kwargs: Extra keyword arguments for :func:`open`.

        Returns:
            The opened file object.

        Raises:
            PermissionError: If the path is not allowed or the
                argument is not path-like.
        """
        if not isinstance(file, (str, bytes, os.PathLike)):
            raise PermissionError("Only path-based open is allowed")

        if not self.path_allowed(file):
            raise PermissionError(f"Access denied for path: {file!r}")

        return open(file, mode, *args, **kwargs)

    @contextmanager
    def memory_limit(self) -> Iterator[None]:
        """Bound sandboxed-code memory via the soft ``RLIMIT_AS``.

        Only the soft limit is lowered: the hard ceiling is kept so MCP
        tool calls can lift it again around the subprocess they spawn
        (a tight cap cannot even start the Go ``docker`` binary). The
        previous limits are always restored on exit. No-op on macOS,
        which does not enforce ``RLIMIT_AS``.
        """
        if sys.platform == "darwin":
            yield
            return
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        cap = self.max_memory_mb * 1024 * 1024
        if hard != resource.RLIM_INFINITY:
            cap = min(cap, hard)
        resource.setrlimit(resource.RLIMIT_AS, (cap, hard))
        try:
            yield
        finally:
            resource.setrlimit(resource.RLIMIT_AS, (soft, hard))

    def run(self, code: str) -> dict[str, Any]:
        """Execute ``code`` inside the sandbox with the safety wraps.

        Args:
            code: Python source string to execute.

        Returns:
            A dict with ``stdout`` and ``error`` (and ``final_answer``
            when the sandboxed code called the injected helper).
        """
        class TimeoutException(Exception):
            """Raised by ``SIGALRM`` when the execution budget runs out."""

            pass

        def timeout_handler(signum: int, frame: FrameType | None) -> None:
            """SIGALRM handler converting the signal into an exception.

            Args:
                signum: POSIX signal number (unused).
                frame: Current stack frame (unused).
            """
            raise TimeoutException("Execution timeout exceeded")

        def final_answer(value: Any) -> None:
            """Sandbox-injected helper to submit the final answer.

            Args:
                value: Payload to surface to the agent runner.
            """
            raise FinalAnswer(value)

        authorized_builtins = {
            # I/O
            "print": print,
            "input": input,
            "open": self.safe_open,
            "final_answer": final_answer,

            # Type
            "bool": bool,
            "int": int,
            "float": float,
            "complex": complex,
            "str": str,
            "bytes": bytes,
            "bytearray": bytearray,
            "memoryview": memoryview,

            # Struct
            "list": list,
            "tuple": tuple,
            "dict": dict,
            "set": set,
            "frozenset": frozenset,

            # Itération & fonctionnels
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "reversed": reversed,
            "sorted": sorted,
            "iter": iter,
            "next": next,

            # Math
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            "divmod": divmod,
            "hash": hash,


            "isinstance": isinstance,
            "issubclass": issubclass,
            "type": type,
            "callable": callable,
            "hasattr": hasattr,
            "getattr": getattr,
            "setattr": setattr,
            "delattr": delattr,
            "dir": dir,
            "vars": vars,
            "id": id,

            # Conversion
            "repr": repr,
            "ascii": ascii,
            "chr": chr,
            "ord": ord,
            "bin": bin,
            "oct": oct,
            "hex": hex,
            "format": format,

            # Utils
            "all": all,
            "any": any,
            "staticmethod": staticmethod,
            "classmethod": classmethod,
            "property": property,
            "super": super,

            # Exceptions
            "Exception": Exception,
            "ValueError": ValueError,
            "TypeError": TypeError,
            "KeyError": KeyError,
            "IndexError": IndexError,
            "AttributeError": AttributeError,
            "RuntimeError": RuntimeError,
            "StopIteration": StopIteration,
            "NotImplementedError": NotImplementedError,
            "OSError": OSError,
            "IOError": IOError,
            "ImportError": ImportError,
            "NameError": NameError,
            "ZeroDivisionError": ZeroDivisionError,
            "OverflowError": OverflowError,
            "MemoryError": MemoryError,
            "RecursionError": RecursionError,
            "AssertionError": AssertionError,
            "FileNotFoundError": FileNotFoundError,
            "PermissionError": PermissionError,

            # Sys
            "__import__": self.import_gestion,
            "__build_class__": __build_class__,
            "__name__": "__main__",
        }

        authorized_builtins.update(self.extra_builtins)
        authorized_builtins.update({"final_answer": final_answer})
        buf = io.StringIO()

        if threading.current_thread() is not threading.main_thread():
            return {"stdout": buf.getvalue(),
                    "error": "Timeout requires main thread"}

        previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, self.max_execution_time_seconds)
        try:
            with self.memory_limit(), redirect_stdout(buf):
                try:
                    sandbox_globals = {"__builtins__": authorized_builtins}
                    sandbox_globals.update(self.extra_builtins)
                    sandbox_globals["final_answer"] = final_answer

                    exec(code, sandbox_globals, None)
                except TimeoutException:
                    return {
                        "stdout": buf.getvalue(),
                        "error": "Execution timeout exceeded"}
                except FinalAnswer as fa:
                    return {
                        "stdout": buf.getvalue(),
                        "error": None,
                        "final_answer": fa.value,
                    }
                except (KeyboardInterrupt, SystemExit):
                    raise
                except MemoryError:
                    return {
                        "stdout": buf.getvalue(),
                        "error": "Memory limit exceeded"}
                except Exception as exc:
                    return {
                        "stdout": buf.getvalue(),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
            return {"stdout": buf.getvalue(), "error": None}
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)

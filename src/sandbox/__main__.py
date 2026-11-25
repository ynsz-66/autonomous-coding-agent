from typing import Any
import sys
import fire
from dotenv import load_dotenv
from sandbox import SandboxConfig, from_json
from mcp import StdioServerParameters
from mcp_config.client import (
    format_exception,
    list_tools,
    list_tools_http,
    make_builtins,
    make_stdio_caller,
    make_http_caller,
)

load_dotenv()


class SandboxCLI:
    """Interactive REPL exposing a sandbox wired to optional MCP tools."""

    def __call__(
            self,
            config_path: str | None = None,
            mcp_stdio: str | None = None,
            mcp_server: str | None = None) -> Any:
        """Run the sandbox REPL with the requested configuration.

        Args:
            config_path: Optional JSON config file for the sandbox.
            mcp_stdio: Stdio command launching an MCP server.
            mcp_server: HTTP URL of an MCP server.

        Returns:
            An error string when the inputs are inconsistent;
            otherwise ``None`` once the REPL exits.
        """
        extra_builtins: dict[str, Any] = {}
        if mcp_server is not None and mcp_stdio is not None:
            return (
                "Error: Cannot use both mcp_stdio and mcp_server "
                "at the same time."
            )

        if mcp_stdio is not None:
            try:
                mcp_split = mcp_stdio.split()
                if len(mcp_split) < 1:
                    raise ValueError(
                        "MCP command must include a script to run")
                server_params = StdioServerParameters(
                    command=mcp_split[0], args=mcp_split[1:])
                tools = list_tools(server_params)
                extra_builtins = make_builtins(
                    tools, make_stdio_caller(server_params))
            except ValueError as v:
                return f"ValueError: {v}"
            except Exception as e:
                return format_exception(e)

        if mcp_server is not None:
            try:
                if not mcp_server.startswith("http"):
                    raise ValueError("MCP server URL must start with 'http'")
                tools = list_tools_http(mcp_server)
                # print(('').join(t.name for t in tools.tools))
                extra_builtins = make_builtins(
                    tools, make_http_caller(mcp_server))
            except ValueError as v:
                return f"ValueError: {v}"
            except Exception as e:
                return format_exception(e)

        if config_path is not None:
            data = from_json(config_path)
            sandbox = SandboxConfig(**data, extra_builtins=extra_builtins)
        else:
            sandbox = SandboxConfig(extra_builtins=extra_builtins)
        # When stdin is piped, execute the full input as a script and print results.
        if not sys.stdin.isatty():
            code = sys.stdin.read()
            if not code.strip():
                return None
            out = sandbox.run(code)
            if out.get("stdout"):
                print(out["stdout"], end="")
            if out.get("error"):
                print(out["error"], file=sys.stderr)
            if "final_answer" in out:
                print(out["final_answer"])
            return None

        while True:
            try:
                code = input(">>> ")
                if code in ("exit", "quit"):
                    break
                out = sandbox.run(code)
                print(out)
            except (KeyboardInterrupt, EOFError):
                break


def main() -> None:
    """Entry point: expose :class:`SandboxCLI` as a Fire CLI."""
    fire.Fire(SandboxCLI().__call__)

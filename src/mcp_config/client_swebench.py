import sys
from typing import Any
from mcp import StdioServerParameters
from mcp_config.client import (
    list_tools as generic_list_tools,
    make_builtins as generic_make_builtins,
    make_stdio_caller,
)


def server_params(container_id: str | None) -> StdioServerParameters:
    """Build stdio params for the SWE-bench tools server.

    Args:
        container_id: When set, run the server inside the given
            Docker container; otherwise launch it locally.

    Returns:
        Stdio parameters suitable for the MCP client.
    """
    if container_id:
        return StdioServerParameters(
            command="docker",
            args=[
                "exec", "-i", "-w", "/testbed", container_id,
                "python", "-u", "/agent/mcp_tools_swebench.py",
            ],
        )
    return StdioServerParameters(
        command=sys.executable,
        args=["-u", "mcp_tools_swebench.py"],
    )


def list_tools(container_id: str | None = None) -> Any:
    """List MCP tools exposed by the SWE-bench server.

    Args:
        container_id: Optional Docker container ID hosting the
            server.

    Returns:
        The raw ``ListToolsResult`` from the server.
    """
    return generic_list_tools(server_params(container_id))


def make_builtins(tools: Any, container_id: str | None = None) -> Any:
    """Wrap SWE-bench MCP tools as Python callables for the sandbox.

    Args:
        tools: Result of :func:`list_tools`.
        container_id: Optional Docker container ID hosting the
            server.

    Returns:
        Mapping of ``tool_name -> callable`` to inject as builtins.
    """
    return generic_make_builtins(
        tools, make_stdio_caller(
            server_params(container_id)))

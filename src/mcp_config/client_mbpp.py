import sys
from typing import Any
from mcp import StdioServerParameters
from mcp_config.client import (
    list_tools as generic_list_tools,
    make_builtins as generic_make_builtins,
    make_stdio_caller,
)

SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=["-u", "mcp_tools_mbpp.py"],
)


def list_tools() -> Any:
    """List MCP tools exposed by the MBPP stdio server.

    Returns:
        The raw ``ListToolsResult`` from the local MBPP server.
    """
    return generic_list_tools(SERVER_PARAMS)


def make_builtins(tools: Any) -> Any:
    """Wrap MBPP MCP tools as Python callables for the sandbox.

    Args:
        tools: Result of :func:`list_tools`.

    Returns:
        Mapping of ``tool_name -> callable`` to inject as builtins.
    """
    return generic_make_builtins(tools, make_stdio_caller(SERVER_PARAMS))

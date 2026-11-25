import asyncio
import os
import resource
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator
import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client


def format_exception(exc: Exception) -> str:
    """Render an exception (incl. ExceptionGroup) as a single line.

    Args:
        exc: Exception to format.

    Returns:
        A flat string with one ``Type: message`` per group member.
    """
    base_group = getattr(__import__("builtins"), "BaseExceptionGroup", None)
    if base_group and isinstance(exc, base_group):
        messages = [f"{type(sub).__name__}: {sub}" for sub in exc.exceptions]
        return " | ".join(messages) if messages else str(exc)
    return f"{type(exc).__name__}: {exc}"


@contextmanager
def lifted_memory_limit() -> Iterator[None]:
    """Lift the soft ``RLIMIT_AS`` cap around a subprocess spawn.

    The sandbox lowers the soft ``RLIMIT_AS`` to bound student code.
    That cap is inherited by the subprocess each MCP tool call spawns
    and is too tight to even start the Go ``docker`` binary, so the
    spawn dies and the stdio transport reports an ``ExceptionGroup``.
    Raise the soft limit to the hard ceiling for the spawn, then
    restore it. No-op when already uncapped (soft equals hard), which
    includes macOS.
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    if soft == hard:
        yield
        return
    resource.setrlimit(resource.RLIMIT_AS, (hard, hard))
    try:
        yield
    finally:
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))


def build_http_headers() -> dict[str, Any]:
    """Build authentication headers for the MCP HTTP transport.

    Returns:
        ``Authorization`` bearer header when ``MCP_BEARER_TOKEN`` is
        set, ``X-API-Key`` when ``MCP_API_KEY`` is set, or an empty
        dict.
    """
    bearer = os.getenv("MCP_BEARER_TOKEN")
    if bearer:
        return {"Authorization": f"Bearer {bearer}"}
    api_key = os.getenv("MCP_API_KEY")
    if api_key:
        return {"X-API-Key": api_key}
    return {}


def list_tools(server_params: StdioServerParameters) -> Any:
    """List tools exposed by an MCP server reached over stdio.

    Args:
        server_params: Connection parameters for the stdio server.

    Returns:
        The raw ``ListToolsResult`` returned by the server.
    """
    async def run() -> Any:
        """Open a stdio session and call ``list_tools`` once."""
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.list_tools()
    return asyncio.run(run())


def call_tool(
        server_params: StdioServerParameters,
        name: str,
        arguments: dict[str, Any]) -> Any:
    """Invoke a tool on an MCP server reached over stdio.

    Args:
        server_params: Connection parameters for the stdio server.
        name: Tool name to invoke.
        arguments: Keyword arguments forwarded to the tool.

    Returns:
        The raw ``CallToolResult`` returned by the server.
    """
    async def run() -> Any:
        """Open a stdio session and call the tool once."""
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool(name, arguments)
    with lifted_memory_limit():
        return asyncio.run(run())


def list_tools_http(url: str) -> Any:
    """List tools exposed by an MCP server reached over HTTP.

    Args:
        url: HTTP URL of the streamable MCP endpoint.

    Returns:
        The raw ``ListToolsResult`` returned by the server.
    """
    async def run() -> Any:
        """Open an HTTP session and call ``list_tools`` once."""
        http_client = httpx.AsyncClient(headers=build_http_headers())
        async with streamable_http_client(
            url, http_client=http_client
        ) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.list_tools()
    last_exc: Exception | None = None
    for attempt in range(6):
        try:
            return asyncio.run(run())
        except (
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.TimeoutException,
            OSError,
        ) as exc:
            last_exc = exc
            time.sleep(0.25 * (attempt + 1))
    if last_exc is not None:
        raise last_exc
    return asyncio.run(run())


def call_tool_http(url: str, name: str, arguments: dict[str, Any]) -> Any:
    """Invoke a tool on an MCP server reached over HTTP.

    Args:
        url: HTTP URL of the streamable MCP endpoint.
        name: Tool name to invoke.
        arguments: Keyword arguments forwarded to the tool.

    Returns:
        The raw ``CallToolResult`` returned by the server.
    """
    async def run() -> Any:
        """Open an HTTP session and call the tool once."""
        http_client = httpx.AsyncClient(headers=build_http_headers())
        async with streamable_http_client(
            url, http_client=http_client
        ) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool(name, arguments)
    last_exc: Exception | None = None
    for attempt in range(6):
        try:
            return asyncio.run(run())
        except (
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.TimeoutException,
            OSError,
        ) as exc:
            last_exc = exc
            time.sleep(0.25 * (attempt + 1))
    if last_exc is not None:
        raise last_exc
    return asyncio.run(run())


def make_stdio_caller(
    server_params: StdioServerParameters,
) -> Callable[[str, dict[str, Any]], Any]:
    """Return a thin ``(name, kwargs) -> result`` callable over stdio.

    Args:
        server_params: Connection parameters for the stdio server.

    Returns:
        A callable usable as the ``caller`` argument of
        :func:`make_builtins`.
    """
    def caller(name: str, arguments: dict[str, Any]) -> Any:
        """Forward a single tool call to the underlying stdio server."""
        return call_tool(server_params, name, arguments)
    return caller


def make_http_caller(url: str) -> Callable[[str, dict[str, Any]], Any]:
    """Return a thin ``(name, kwargs) -> result`` callable over HTTP.

    Args:
        url: HTTP URL of the streamable MCP endpoint.

    Returns:
        A callable usable as the ``caller`` argument of
        :func:`make_builtins`.
    """
    def caller(name: str, arguments: dict[str, Any]) -> Any:
        """Forward a single tool call to the underlying HTTP server."""
        return call_tool_http(url, name, arguments)
    return caller


def make_builtins(
    tools: Any, caller: Callable[[str, dict[str, Any]], Any]
) -> dict[str, Callable[..., str]]:
    """Expose MCP tools as Python callables injectable into the sandbox.

    For each tool, builds a wrapper that maps positional args to the
    declared parameter names, calls the MCP server through ``caller``,
    and concatenates the textual content blocks of the response.

    Args:
        tools: Result of :func:`list_tools` (has ``tools`` attribute).
        caller: ``(name, kwargs) -> result`` adapter, typically from
            :func:`make_stdio_caller` or :func:`make_http_caller`.

    Returns:
        Mapping of ``tool_name -> wrapper`` ready to be installed as
        sandbox builtins.
    """
    def make_wrapper(
        name: str, params: list[str]
    ) -> Callable[..., str]:
        """Build the per-tool wrapper closure.

        Args:
            name: Tool name to invoke.
            params: Ordered parameter names declared in the schema.

        Returns:
            A callable forwarding to ``caller`` and rendering the
            result as plain text.
        """
        def wrapper(*args: Any, **kwargs: Any) -> str:
            """Invoke the MCP tool and return its concatenated text."""
            kwargs.update(zip(params, args))
            result = caller(name, kwargs)
            return "\n".join(
                c.text for c in result.content if hasattr(c, "text")
            )
        return wrapper
    return {
        t.name: make_wrapper(t.name, list(t.input_schema.get("properties", {})))
        for t in tools.tools
    }

"""Rich-only terminal UI for the agent loop (live layout, logs, metrics)."""
import json
import logging
import sys
import time
from collections import deque
from typing import Any, Iterable

from rich.console import Console, ConsoleOptions, Group, RenderableType
from rich.layout import Layout
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.segment import Segment
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

LEVEL_STYLES = {
    "DEBUG": "grey50",
    "INFO": "cyan",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bold red",
}


class Tail:
    """Auto-scrolling view: render a renderable and keep only its last lines.

    Rich has no scrollable widget, so this shows the bottom of the content
    (the newest text) within the height the parent layout gives it.
    """

    def __init__(self, renderable: RenderableType) -> None:
        self.renderable = renderable

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> Iterable[Segment]:
        lines = console.render_lines(
            self.renderable, options.update(height=None), pad=False)
        height = options.height or len(lines)
        hidden = len(lines) - height
        if hidden > 0:
            lines = lines[-height:]
            # first visible line becomes a scroll hint
            lines[0] = console.render_lines(
                Text(f"↑ {hidden + 1} more lines", style="dim italic"),
                options.update(height=None), pad=False)[0]
        for line in lines:
            yield from line
            yield Segment.line()


class LogPanelHandler(logging.Handler):
    """Logging handler that keeps recent records for the live logs panel."""

    def __init__(self, max_lines: int = 200) -> None:
        super().__init__()
        self.lines: deque[Text] = deque(maxlen=max_lines)

    def emit(self, record: logging.LogRecord) -> None:
        stamp = time.strftime("%H:%M:%S", time.localtime(record.created))
        line = Text(f"{stamp} ", style="dim")
        line.append(f"{record.levelname:<8}",
                    style=LEVEL_STYLES.get(record.levelname, ""))
        line.append(record.getMessage())
        self.lines.append(line)

    def render(self) -> Panel:
        return Panel(Tail(Text("\n").join(list(self.lines))),
                     title="LOGS", border_style="grey50")


def build_layout(title: str) -> Layout:
    """Build the full-screen layout: header / thinking | sandbox+logs / footer.

    Args:
        title: Text shown in the header bar.

    Returns:
        A :class:`Layout` with regions ``header``, ``thinking``,
        ``sandbox``, ``logs`` and ``footer``.
    """
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=1),
        Layout(name="body"),
        Layout(name="footer", size=6),
    )
    layout["body"].split_row(Layout(name="thinking"), Layout(name="right"))
    layout["right"].split_column(
        Layout(name="sandbox", ratio=3), Layout(name="logs", ratio=2))
    layout["header"].update(
        Text(f" {title} ", style="bold white on dark_magenta",
             justify="center"))
    layout["thinking"].update(render_thinking(""))
    layout["sandbox"].update(
        Panel("", title="SANDBOX", border_style="cyan"))
    layout["logs"].update(Panel("", title="LOGS", border_style="grey50"))
    return layout


def render_thinking(text: str) -> Panel:
    """Render the streamed LLM answer, auto-scrolled to the newest text."""
    return Panel(Tail(Markdown(text)), title="AGENT THINKING",
                 border_style="bold blue")


def render_sandbox(code: str, sandbox_output: dict[str, Any]) -> Panel:
    """Render executed code and its result (stdout, error, final answer).

    Args:
        code: Python code sent to the sandbox.
        sandbox_output: Dict with ``stdout`` and optional ``error`` /
            ``final_answer``.

    Returns:
        A :class:`Panel` whose border is red on error, green on final
        answer, cyan otherwise.
    """
    parts: list[RenderableType] = [
        Syntax(code, "python", line_numbers=True, word_wrap=True),
        Text("─ stdout", style="dim"),
        Text(sandbox_output.get("stdout") or "(empty)"),
    ]
    border = "cyan"
    error = sandbox_output.get("error")
    if error is not None:
        parts += [Text("error", style="bold red"),
                  Text(str(error), style="red")]
        border = "red"
    if "final_answer" in sandbox_output:
        parts += [Text("final_answer", style="bold green"),
                  Text(str(sandbox_output["final_answer"]), style="green")]
        border = "green"
    return Panel(Tail(Group(*parts)), title="SANDBOX", border_style=border)


def budget_row(label: str, used: float, limit: float,
               unit: str = "") -> tuple[Text, ProgressBar, Text]:
    """Return one ``label | bar | used/limit`` row; bar turns red near limit."""
    ratio = used / limit if limit else 0
    style = "red" if ratio > 0.85 else "yellow" if ratio > 0.6 else "green"
    return (
        Text(label, style="bold"),
        ProgressBar(total=limit, completed=min(used, limit),
                    complete_style=style, finished_style="red"),
        Text(f"{used:,.0f}{unit} / {limit:,.0f}{unit}", style=style),
    )


def render_metrics(
    iteration: int, max_iterations: int,
    input_tokens: int, max_input_tokens: int,
    output_tokens: int, max_output_tokens: int,
    elapsed: float, max_seconds: float,
    requests: int, retries: int, paused: float,
    model: str, waiting: bool = False,
) -> Panel:
    """Render the budgets footer (iterations, tokens, time) and counters."""
    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column(width=10)
    grid.add_column(ratio=1)
    grid.add_column(width=18, justify="right")
    grid.add_row(*budget_row("Iteration", iteration, max_iterations))
    grid.add_row(*budget_row("Input tok", input_tokens, max_input_tokens))
    grid.add_row(*budget_row("Output tok", output_tokens, max_output_tokens))
    grid.add_row(*budget_row("Time", elapsed, max_seconds, "s"))
    subtitle = (f"requests {requests} · retries {retries} · "
                f"paused {paused:.1f}s · {model}")
    if waiting:
        return Panel(grid, title="METRICS", subtitle=(
            "[bold black on yellow] PAUSED - Press Enter to continue [/]"),
            border_style="yellow")
    return Panel(grid, title="METRICS", subtitle=subtitle,
                 border_style="magenta")


def wait_for_enter() -> float:
    """Block until Enter is pressed; return the seconds spent waiting.

    The caller keeps the alternate-screen ``Live`` running: each refresh
    starts with a cursor-home and redraws everything, so the echoed newline
    is wiped out. Returns 0 without blocking when stdin is not a TTY.
    """
    if not sys.stdin.isatty():
        return 0.0
    start = time.time()
    try:
        input()
    except EOFError:
        pass
    return time.time() - start


def render_summary(steps: list[Any], totals: dict[str, Any]) -> Table:
    """Render the end-of-run per-step table with a totals footer.

    Args:
        steps: ``StepMetrics`` objects collected during the run.
        totals: Keys ``input_tokens``, ``output_tokens``, ``time``,
            ``paused``, ``requests``.

    Returns:
        A :class:`Table` ready to print.
    """
    table = Table(title="Run summary", show_footer=True,
                  border_style="magenta", title_style="bold magenta")
    table.add_column("Step", footer="Total", justify="right")
    table.add_column("In tok", footer=f"{totals['input_tokens']:,}",
                     justify="right")
    table.add_column("Out tok", footer=f"{totals['output_tokens']:,}",
                     justify="right")
    table.add_column("LLM ms", footer=f"{totals['time']:.1f}s run",
                     justify="right")
    table.add_column("Retries", footer=f"{totals['requests']} req",
                     justify="right")
    table.add_column("Result", footer=f"{totals['paused']:.1f}s paused")
    for step in steps:
        output = json.loads(step.sandbox_output)
        if "final_answer" in output:
            result = Text("final_answer", style="green")
        elif output.get("error"):
            first = str(output["error"]).strip().splitlines() or [""]
            result = Text(f"{first[-1][:40]}", style="red")
        else:
            result = Text("stdout", style="cyan")
        table.add_row(str(step.step), f"{step.input_tokens:,}",
                      f"{step.output_tokens:,}",
                      f"{step.request_time_ms:,.0f}", str(step.retries),
                      result)
    return table

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ui import (  # noqa: E402
    LogPanelHandler, Tail, build_layout, render_metrics, render_sandbox,
    render_summary, wait_for_enter,
)


def render(renderable, height: int | None = None) -> str:
    console = Console(record=True, width=80, height=height or 25,
                      color_system=None)
    console.print(renderable, height=height)
    return console.export_text()


def test_tail_keeps_last_lines() -> None:
    text = "\n".join(f"line {i}" for i in range(50))
    out = render(Panel(Tail(text)), height=7)  # 5 content lines
    assert "line 49" in out
    assert "line 0\n" not in out
    assert "↑ 46 more lines" in out


def test_tail_short_content_unchanged() -> None:
    out = render(Panel(Tail("a\nb")), height=6)
    assert "a" in out and "b" in out and "more lines" not in out


def test_render_sandbox_shows_code_error_and_final() -> None:
    out = render(render_sandbox(
        "print(1)", {"stdout": "1\n", "error": "ZeroDivisionError"}))
    assert "print(1)" in out and "ZeroDivisionError" in out
    out = render(render_sandbox(
        "final_answer('x')", {"stdout": "", "final_answer": "x"}))
    assert "final_answer" in out and "(empty)" in out


def test_log_handler_feeds_panel() -> None:
    handler = LogPanelHandler(max_lines=2)
    logger = logging.getLogger("test_ui")
    logger.addHandler(handler)
    for msg in ("first", "second", "third"):
        logger.warning(msg)
    logger.removeHandler(handler)
    out = render(handler.render())
    assert "third" in out and "WARNING" in out and "first" not in out


def test_render_metrics_and_pause_banner() -> None:
    args = (2, 10, 3000, 6000, 400, 1500, 30.0, 120, 2, 1, 4.5, "m")
    out = render(render_metrics(*args))
    assert "3,000 / 6,000" in out and "paused 4.5s" in out
    assert "Press Enter" in render(render_metrics(*args, waiting=True))


def test_build_layout_regions() -> None:
    layout = build_layout("T")
    for name in ("header", "thinking", "sandbox", "logs", "footer"):
        assert layout[name] is not None


def test_wait_for_enter_non_tty(monkeypatch) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    assert wait_for_enter() == 0.0


def test_render_summary() -> None:
    steps = [
        SimpleNamespace(step=1, input_tokens=1000, output_tokens=100,
                        request_time_ms=850.0, retries=0,
                        sandbox_output=json.dumps(
                            {"stdout": "", "error": "Traceback\nNameError: x"})),
        SimpleNamespace(step=2, input_tokens=1200, output_tokens=80,
                        request_time_ms=700.0, retries=1,
                        sandbox_output=json.dumps(
                            {"stdout": "", "final_answer": "ok"})),
    ]
    out = render(render_summary(steps, {
        "input_tokens": 2200, "output_tokens": 180, "time": 12.3,
        "paused": 5.0, "requests": 3}))
    assert "NameError: x" in out and "final_answer" in out
    assert "2,200" in out and "5.0s paused" in out

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils import extract_code  # noqa: E402


def test_closed_block() -> None:
    assert extract_code("Thought: x\n```python\nprint(1)\n```\n<end_code>") == "print(1)\n"


def test_unclosed_block_cut_by_token_cap() -> None:
    assert extract_code("Thought: x\n```python\nprint(1)\nprint(2") == "print(1)\nprint(2"


def test_unclosed_block_stopped_on_end_code() -> None:
    assert extract_code("```python\nprint(1)\n<end_code>") == "print(1)\n"


def test_fence_variants() -> None:
    assert extract_code("```py\nx = 1\n```") == "x = 1\n"
    assert extract_code("```Python \nx = 1\n```") == "x = 1\n"
    assert extract_code("```\nx = 1\n```") == "x = 1\n"


def test_skips_non_python_block() -> None:
    assert extract_code("```bash\nls\n```\n```python\nx = 1\n```") == "x = 1\n"


def test_no_block() -> None:
    assert extract_code("Thought: only text") == ""

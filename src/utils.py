import os
import re
from typing import Any
from rich.console import Console
from agent.llm import OpenAICompatibleLLM

CONSOLE = Console()
PROVIDERS = {
    "GROQ": "https://api.groq.com/openai/v1",
    "OPENROUTER": "https://openrouter.ai/api/v1",
    "OLLAMA": "http://localhost:11434/v1",
}


class FinishError(Exception):
    """Raised when the agent successfully submits a final answer."""

    pass


def load_llm_config() -> tuple[str, str, list[str]]:
    """Read the LLM settings from the environment (``.env``).

    Uses ``PROVIDER`` (a key of :data:`PROVIDERS`), ``MODEL_NAME`` and the
    numbered keys ``{PROVIDER}_API_KEY_1``, ``_2``... (none for Ollama,
    whose URL can be overridden with ``OLLAMA_BASE_URL``).

    Returns:
        ``(base_url, model_name, api_keys)``.

    Raises:
        ValueError: If the provider is unknown or a setting is missing.
    """
    provider = os.environ.get("PROVIDER", "").upper()
    if provider not in PROVIDERS:
        raise ValueError(
            "choose a handled provider like : " + ", ".join(PROVIDERS))
    model = os.environ.get("MODEL_NAME", "")
    if not model:
        raise ValueError(
            "MODEL_NAME is not set in .env (e.g. MODEL_NAME=qwen/qwen3-32b)")
    if provider == "OLLAMA":
        url = os.environ.get("OLLAMA_BASE_URL") or PROVIDERS[provider]
        return url, model, ["ollama"]  # key ignored by Ollama
    keys: list[str] = []
    while key := os.environ.get(f"{provider}_API_KEY_{len(keys) + 1}"):
        keys.append(key)
    if not keys:
        raise ValueError(f"{provider}_API_KEY_1 is not set in .env")
    return PROVIDERS[provider], model, keys


def rotate_on_error(
    error: Exception,
    llm: OpenAICompatibleLLM,
    keys: list[str],
    key_idx: int,
    failures: int,
    log: Any,
) -> tuple[OpenAICompatibleLLM, int, int]:
    """Switch to the next API key after a provider error.

    Args:
        error: Exception triggering the rotation.
        llm: LLM client that just failed.
        keys: API keys of the configured provider.
        key_idx: Index of the key that just failed.
        failures: Consecutive failures so far.
        log: Callable used to append a human-readable trace line.

    Returns:
        ``(new_llm, key_idx, failures)``.

    Raises:
        RuntimeError: When every key has failed in a row.
    """
    failures += 1
    if failures >= len(keys):
        log(f"All {len(keys)} API key(s) failed in a row, giving up")
        raise RuntimeError(
            f"All API keys exhausted. Last error: "
            f"({error.__class__.__name__}) {error}"
        )
    log(f"Provider error with API key #{key_idx + 1} "
        f"({error.__class__.__name__}): {error}, switching to key "
        f"#{(key_idx + 1) % len(keys) + 1}/{len(keys)}")
    key_idx = (key_idx + 1) % len(keys)
    return (OpenAICompatibleLLM(keys[key_idx], llm.base_url, llm.model),
            key_idx, failures)


def extract_code(text: str) -> str:
    """Extract the first fenced Python code block from a string.

    Accepts ```` ```python ````, ```` ```py ```` or a bare fence (any case),
    and a block left unclosed because generation stopped on
    ``<end_code>`` or hit the token cap.

    Args:
        text: Raw LLM response.

    Returns:
        The content of the first code block, or an empty string when
        no block is found.
    """
    body = r"[ \t]*\n(.*?)(?:```|<end_code>|\Z)"
    # tagged fence first: a closing ``` of another block looks like a bare one
    match = (
        re.search(r"```[ \t]*(?:python3?|py)" + body,
                  text, re.DOTALL | re.IGNORECASE)
        or re.search(r"```" + body, text, re.DOTALL)
    )
    return match.group(1) if match else ""


def format_observation(
    sandbox_output: dict[str, Any],
    max_chars: int = 4000,
) -> str:
    """Format a sandbox result as a textual Observation for the LLM.

    Truncates the middle of overly long outputs to keep the token
    budget under control while preserving head and tail context.

    Args:
        sandbox_output: Dict with ``stdout`` and optional ``error``.
        max_chars: Maximum body length before middle truncation.

    Returns:
        A string starting with ``Observation:`` ready to feed back
        to the agent as the next user turn.
    """
    stdout = (sandbox_output.get("stdout") or "").rstrip()
    error = sandbox_output.get("error")

    parts = [
        stdout if stdout
        else "(no output — put the tool call alone on its own line to see "
             "its result)"
    ]
    if error is not None:
        parts.append(f"Error:\n{error}")
    body = "\n".join(parts)

    if len(body) > max_chars:
        half = max_chars // 2
        cut = len(body) - max_chars
        body = (
            f"{body[:half]}\n"
            f"...[TRUNCATED {cut} chars — output too long]...\n"
            f"{body[-half:]}"
        )
    return f"Observation:\n{body}"

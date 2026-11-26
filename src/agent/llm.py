from typing import Any, Iterator
from openai import OpenAI


def usage_to_dict(usage: Any) -> dict[str, Any]:
    """Normalize a provider usage object to a plain dictionary.

    Args:
        usage: Raw usage attribute returned by the provider's
            streaming chunk (may be ``None``).

    Returns:
        A dict with `prompt_tokens`, `completion_tokens`,
        `total_tokens` and `reasoning_tokens` keys, all defaulting
        to 0 when missing.
    """
    if usage is None:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokens": 0}
    details = getattr(usage, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", 0) if details else 0
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "reasoning_tokens": reasoning or 0,
    }


class StreamResponse:
    """Iterable wrapper around a streamed completion.

    Yields text deltas until one of the `stop` sequences appears, then keeps
    draining the stream silently to capture the final usage chunk (sent by the
    provider with `stream_options={"include_usage": True}`). After iteration,
    `text` holds the truncated response and `usage` holds the token accounting.
    """

    def __init__(self, stream: Any, stop: list[str] | None = None) -> None:
        """Initialize the streaming wrapper.

        Args:
            stream: Raw streaming iterator returned by the OpenAI
                client.
            stop: Optional list of stop sequences. Iteration ends
                the first time one of these substrings is observed
                in the accumulated text.
        """
        self.stream = stream
        self.stop = stop or []
        self.text = ""
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokens": 0}

    def __iter__(self) -> Iterator[str]:
        """Yield text deltas until a stop sequence is hit.

        Yields:
            Each successive token delta from the provider, possibly
            truncated to honour the configured stop sequences.
        """
        cut_done = False
        for chunk in self.stream:
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                self.usage = usage_to_dict(chunk_usage)
            if cut_done or not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if not delta:
                continue
            previous_len = len(self.text)
            self.text += delta
            cut = -1
            for token in self.stop:
                idx = self.text.find(token)
                if idx != -1:
                    end = idx + len(token)
                    if cut == -1 or end < cut:
                        cut = end
            if cut != -1:
                self.text = self.text[:cut]
                tail = self.text[previous_len:]
                cut_done = True
                if tail:
                    yield tail
                continue
            yield delta


class OpenAICompatibleLLM:
    """Chat against any OpenAI-compatible endpoint.

    Works with OpenRouter, Groq, Ollama and any other provider
    exposing the OpenAI chat-completions API surface.
    """

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        """Initialize the OpenAI-compatible LLM client.

        Args:
            api_key: API key for the target provider.
            base_url: OpenAI-compatible base URL.
            model: Model identifier.

        Raises:
            ValueError: If `api_key` is empty.
        """
        if not api_key:
            raise ValueError("api_key is required for OpenAICompatibleLLM")
        self.base_url = base_url
        self.model = model
        self.client = OpenAI(base_url=self.base_url, api_key=api_key)

    def chat_stream(
        self,
        system: str,
        messages: list[dict[str, Any]],
        max_new_tokens: int = 2048,
        stop: list[str] | None = None,
    ) -> StreamResponse:
        """Start a streamed chat completion.

        Args:
            system: System prompt prepended to the conversation.
            messages: Chat history (user/assistant turns).
            max_new_tokens: Hard cap on generated tokens.
            stop: Optional list of stop sequences to interrupt
                generation client-side.

        Returns:
            A :class:`StreamResponse` iterable yielding text deltas.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "max_tokens": max_new_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if stop:
            kwargs["stop"] = stop
        stream = self.client.chat.completions.create(**kwargs)
        return StreamResponse(stream, stop=stop)

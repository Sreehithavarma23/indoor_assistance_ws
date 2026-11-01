"""LLM provider abstraction.

Planning logic depends only on ``LLMProvider.complete(prompt) -> str``.
Tests use ``FakeLLMProvider``. Live calls use ``OpenAICompatibleProvider``.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Callable, List, Optional, Protocol, Sequence, Union

from src.llm_planning.exceptions import (
    MissingCredentialsError,
    ProviderError,
    ProviderTimeoutError,
)


class LLMProvider(Protocol):
    """Minimal chat-completion interface."""

    name: str

    def complete(self, prompt: str) -> str:
        """Return model text for a single prompt. Raises ProviderError on failure."""


class FakeLLMProvider:
    """Deterministic offline provider for tests and development evaluation.

    ``responses`` may be a list (consumed in order, last repeats) or a callable
    ``(prompt: str) -> str``.
    """

    name = "fake"

    def __init__(
        self,
        responses: Union[Sequence[str], Callable[[str], str], None] = None,
        *,
        fail_with: Optional[BaseException] = None,
    ) -> None:
        self._responses = list(responses) if isinstance(responses, (list, tuple)) else None
        self._callable = responses if callable(responses) else None
        self._index = 0
        self._fail_with = fail_with
        self.prompts: List[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self._fail_with is not None:
            raise self._fail_with
        if self._callable is not None:
            return self._callable(prompt)
        if not self._responses:
            raise ProviderError("FakeLLMProvider has no scripted responses.")
        if self._index < len(self._responses):
            text = self._responses[self._index]
            self._index += 1
            return text
        return self._responses[-1]


class OpenAICompatibleProvider:
    """HTTP chat-completions client (OpenAI-compatible API).

    Credentials are read from the environment at call time. Never logged.
    """

    name = "openai_compatible"

    def __init__(
        self,
        *,
        api_base_url: str,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        timeout_seconds: float = 30.0,
        temperature: float = 0.0,
    ) -> None:
        self.api_base_url = api_base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature

    def complete(self, prompt: str) -> str:
        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key:
            raise MissingCredentialsError(
                f"Environment variable {self.api_key_env} is not set. "
                "Configure credentials for openai_compatible, or use provider=fake."
            )

        url = f"{self.api_base_url}/chat/completions"
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a constrained robot task planner. "
                        "Return only valid JSON matching the requested schema."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            # Do not include response body secrets; keep status only.
            raise ProviderError(
                f"Provider HTTP error: {exc.code} {exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            reason = str(exc.reason)
            if "timed out" in reason.lower() or "timeout" in reason.lower():
                raise ProviderTimeoutError(
                    f"Provider timed out after {self.timeout_seconds}s."
                ) from exc
            raise ProviderError(f"Provider network error: {reason}") from exc
        except TimeoutError as exc:
            raise ProviderTimeoutError(
                f"Provider timed out after {self.timeout_seconds}s."
            ) from exc

        try:
            data = json.loads(raw)
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ProviderError(
                "Provider returned an unexpected response shape."
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("Provider returned empty content.")
        return content


def build_provider_from_config(config: dict) -> LLMProvider:
    """Construct the configured provider. Does not require credentials for fake."""
    kind = str(config.get("provider") or "fake").strip().lower()
    if kind == "fake":
        return FakeLLMProvider()
    if kind == "openai_compatible":
        return OpenAICompatibleProvider(
            api_base_url=str(
                config.get("api_base_url") or "https://api.openai.com/v1"
            ),
            model=str(config.get("model") or "gpt-4o-mini"),
            api_key_env=str(config.get("api_key_env") or "OPENAI_API_KEY"),
            timeout_seconds=float(config.get("timeout_seconds") or 30),
            temperature=float(config.get("temperature") or 0.0),
        )
    raise ProviderError(
        f"Unknown LLM provider '{kind}'. Use 'fake' or 'openai_compatible'."
    )

"""Google Gemini SDK wrapper.

Centralizes:
    - retries with exponential backoff on transient errors
    - extracting the first JSON object from a response (models often wrap it
      in markdown fences or prose)

Every other agent module calls the LLM exclusively through `call_llm`.
`call_claude` is kept as an alias for backward compatibility.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Sequence

from ai_quant_lab.config import settings


@dataclass(frozen=True)
class AgentMessage:
    role: str  # "user" or "assistant"
    content: str


@dataclass(frozen=True)
class AgentResponse:
    text: str
    usage: dict[str, int]
    model: str
    stop_reason: str | None = None


_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def extract_first_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from an LLM response.

    Tries direct json.loads first, then strips markdown fences, then matches
    the first {...} block. Raises ValueError if nothing parses.
    """
    cleaned = text.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass

    match = _JSON_BLOCK_RE.search(cleaned)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Found JSON-like block but could not parse: {exc}") from exc

    raise ValueError(f"No JSON object found in response:\n{cleaned[:500]}")


def call_llm(
    system: str,
    messages: Sequence[AgentMessage],
    *,
    model: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.4,
    cache_system: bool = True,  # accepted for API compat; Gemini context caching not implemented
    max_retries: int = 3,
) -> AgentResponse:
    """Call Gemini with retries.

    Args:
        system: System prompt passed as system_instruction.
        messages: User/assistant turns. "assistant" role is mapped to "model" for Gemini.
        model: Override the configured model.
        max_tokens: Output token budget.
        temperature: Sampling temperature.
        cache_system: No-op (kept for call-site compatibility with the old Anthropic wrapper).
        max_retries: Number of retries on transient API errors.

    Returns:
        AgentResponse with text, usage dict, and model id.

    Raises:
        RuntimeError: if no API key is configured or all retries fail.
    """
    api_key = settings.require_api_key()
    try:
        from google import genai  # noqa: PLC0415
        from google.genai import types  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("Install the SDK: pip install google-genai") from exc

    client = genai.Client(api_key=api_key)
    use_model = model or settings.model

    # Gemini uses "model" instead of "assistant" for the AI turn role.
    contents = [
        types.Content(
            role="model" if m.role == "assistant" else m.role,
            parts=[types.Part(text=m.content)],
        )
        for m in messages
    ]

    config = types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=max_tokens,
        temperature=temperature,
    )

    delay = 1.0
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=use_model,
                contents=contents,
                config=config,
            )
            text = response.text or ""
            um = response.usage_metadata
            usage = {
                "input_tokens": um.prompt_token_count or 0,
                "output_tokens": um.candidates_token_count or 0,
                "cache_read_tokens": um.cached_content_token_count or 0,
                "cache_creation_tokens": 0,
            }
            finish_reason = None
            if response.candidates:
                finish_reason = str(response.candidates[0].finish_reason)
            return AgentResponse(
                text=text,
                usage=usage,
                model=use_model,
                stop_reason=finish_reason,
            )
        except Exception as exc:  # noqa: BLE001 — broad catch is intentional; we retry below
            last_error = exc
            if attempt == max_retries - 1:
                break
            time.sleep(delay)
            delay *= 2.0

    raise RuntimeError(f"Gemini call failed after {max_retries} attempts: {last_error}")


# Backward-compatible alias so existing call sites don't need updating.
call_claude = call_llm

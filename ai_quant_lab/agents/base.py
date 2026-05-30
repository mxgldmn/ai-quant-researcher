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
    cache_system: bool = True,  # no-op kept for call-site compatibility
    max_retries: int = 3,
) -> AgentResponse:
    """Call the configured LLM (Gemini or Groq) with retries.

    Provider is selected automatically from the model name:
    models starting with "gemini" use the Google GenAI SDK;
    everything else uses the Groq SDK (OpenAI-compatible).
    """
    use_model = model or settings.model
    api_key, provider = settings.require_api_key(use_model)

    delay = 1.0
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            if provider == "gemini":
                response = _call_gemini(system, messages, api_key, use_model, max_tokens, temperature)
            else:
                response = _call_groq(system, messages, api_key, use_model, max_tokens, temperature)
            return response
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == max_retries - 1:
                break
            time.sleep(delay)
            delay *= 2.0

    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_error}")


def _call_gemini(
    system: str,
    messages: Sequence[AgentMessage],
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> AgentResponse:
    try:
        from google import genai  # noqa: PLC0415
        from google.genai import types  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("pip install google-genai") from exc

    client = genai.Client(api_key=api_key)
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
    response = client.models.generate_content(model=model, contents=contents, config=config)
    text = response.text or ""
    um = response.usage_metadata
    usage = {
        "input_tokens": um.prompt_token_count or 0,
        "output_tokens": um.candidates_token_count or 0,
        "cache_read_tokens": um.cached_content_token_count or 0,
        "cache_creation_tokens": 0,
    }
    finish_reason = str(response.candidates[0].finish_reason) if response.candidates else None
    return AgentResponse(text=text, usage=usage, model=model, stop_reason=finish_reason)


def _call_groq(
    system: str,
    messages: Sequence[AgentMessage],
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> AgentResponse:
    try:
        from groq import Groq  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("pip install groq") from exc

    client = Groq(api_key=api_key)
    api_messages = [{"role": "system", "content": system}]
    api_messages += [{"role": m.role, "content": m.content} for m in messages]
    response = client.chat.completions.create(
        model=model,
        messages=api_messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    text = response.choices[0].message.content or ""
    usage = {
        "input_tokens": response.usage.prompt_tokens if response.usage else 0,
        "output_tokens": response.usage.completion_tokens if response.usage else 0,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
    }
    finish_reason = response.choices[0].finish_reason if response.choices else None
    return AgentResponse(text=text, usage=usage, model=model, stop_reason=finish_reason)


# Backward-compatible alias so existing call sites don't need updating.
call_claude = call_llm

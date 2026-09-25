"""OpenAI-compatible adapter for any /v1/chat/completions endpoint.

Works with vLLM, LiteLLM, llama.cpp, and any other server exposing
the OpenAI chat completions API with tool-calling support.
When stream=True, uses SSE to measure time-to-first-token (TTFT).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

import httpx

from tool_eval_bench.adapters.base import BackendAdapter, ChatCompletionResult, ProviderToolCall
from tool_eval_bench.domain.models import ChatMessage
from tool_eval_bench.utils.urls import chat_completions_url as _chat_completions_url

logger = logging.getLogger(__name__)

STRUCTURED_RESPONSE_FORMAT_KEY = "tool_eval_structured_response_format"

# Stop only unmistakable exact-repeat loops.  This intentionally does not
# classify verbose, repetitive, or wrong answers as loops: those are valid
# model outcomes and must be scored rather than interrupted.
_LOOP_CHECK_EVERY_CHARS = 512
_LOOP_MIN_REPEATED_CHARS = 2048
_LOOP_MIN_REPEATS = 4
_LOOP_MAX_PERIOD_CHARS = 4096


def _exact_suffix_loop_period(text: str) -> int | None:
    """Return the period of an exact repeated suffix, or ``None``.

    The repeated suffix must cover at least 2 KiB and contain at least four
    byte-for-byte copies.  Checking only the suffix lets a normal answer
    precede a loop without weakening the criterion.
    """
    length = len(text)
    if length < _LOOP_MIN_REPEATED_CHARS:
        return None

    maximum_period = min(_LOOP_MAX_PERIOD_CHARS, length // _LOOP_MIN_REPEATS)
    for period in range(1, maximum_period + 1):
        repeats = max(
            _LOOP_MIN_REPEATS,
            (_LOOP_MIN_REPEATED_CHARS + period - 1) // period,
        )
        span = period * repeats
        if span > length:
            continue

        block = text[-period:]
        # Cheaply reject nearly every candidate before constructing the full
        # repeated comparison string.
        if text[-2 * period : -period] != block:
            continue
        if text[-3 * period : -2 * period] != block:
            continue
        if text[-4 * period : -3 * period] != block:
            continue
        if text[-span:] == block * repeats:
            return period
    return None


def _response_format_for_backend(response_format: dict[str, Any], mode: str | None) -> dict[str, Any] | None:
    """Return a backend-compatible response_format payload.

    Category O scenarios embed the full JSON schema in the user-visible prompt,
    and the benchmark scorer validates the final JSON against that schema. Some
    OpenAI-compatible backends, notably llama.cpp builds without json_schema
    sampler support, fail the request before the model can answer when given
    ``response_format={"type": "json_schema", ...}``. In that case, use
    json_object enforcement so the model is still tested and the scorer remains
    responsible for schema correctness.
    """
    selected = (mode or os.getenv("TOOL_EVAL_STRUCTURED_RESPONSE_FORMAT") or "json_schema").lower()
    if selected in {"none", "off", "disabled"}:
        return None
    if selected in {"json_object", "json-object", "object"} and response_format.get("type") == "json_schema":
        return {"type": "json_object"}
    return response_format


def _repair_streamed_tool_args(s: str) -> str:
    """Repair truncated JSON from streamed tool-call arguments.

    When a server uses ``--stream-interval > 1``, tool-call argument tokens
    are batched into larger SSE chunks.  In some cases the server's own
    tool-call parser may not detect the closing brace within a batch,
    causing the accumulated arguments string to be missing its final ``}``
    or have unbalanced quotes.

    This function applies minimal repairs:
      1. Close unterminated strings (odd number of unescaped quotes).
      2. Close unbalanced braces and brackets.

    If the string is already valid JSON, it is returned unchanged.
    """
    if not s or not s.strip():
        return "{}"
    try:
        json.loads(s)
        return s  # already valid
    except json.JSONDecodeError:
        pass

    import re

    repaired = s
    # Close unterminated strings
    n_quotes = len(re.findall(r'(?<!\\)"', repaired))
    if n_quotes % 2 != 0:
        repaired += '"'

    # Close unbalanced braces and brackets
    opens_brace = repaired.count("{") - repaired.count("}")
    repaired += "}" * max(0, opens_brace)
    opens_bracket = repaired.count("[") - repaired.count("]")
    repaired += "]" * max(0, opens_bracket)

    try:
        json.loads(repaired)
        return repaired
    except json.JSONDecodeError:
        logger.warning("Could not repair streamed tool-call arguments: %.80s", s)
        return s  # return original; let downstream parsing handle it


def _normalize_tool_calls(raw_calls: list[dict] | None) -> list[ProviderToolCall]:
    if not raw_calls:
        return []
    result = []
    for idx, call in enumerate(raw_calls):
        func = call.get("function") or {}
        args = func.get("arguments", "{}")
        if isinstance(args, dict):
            args = json.dumps(args)
        result.append(
            ProviderToolCall(
                id=call.get("id", f"tool_call_{idx + 1}"),
                name=func.get("name", "unknown_tool"),
                arguments_str=args,
            )
        )
    return result


class OpenAICompatibleAdapter(BackendAdapter):
    """OpenAI-compatible adapter with connection reuse.

    Works with any server exposing /v1/chat/completions (vLLM, LiteLLM,
    llama.cpp, etc.).  The underlying httpx.AsyncClient is lazily created
    on first use and reused across all requests.
    Call ``aclose()`` when done (optional — Python GC handles cleanup).
    """

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Return the shared client, creating it lazily on first access.

        The client is created WITHOUT a fixed timeout — callers pass
        per-request timeouts to avoid mismatch between warm-up (60s),
        throughput (180s), and scenario requests.
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(120.0),  # generous default; overridden per-request
                limits=httpx.Limits(
                    max_connections=20,
                    max_keepalive_connections=10,
                ),
            )
        return self._client

    async def aclose(self) -> None:
        """Close the underlying HTTP client (releases connections)."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def chat_completion(
        self,
        *,
        model: str,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout_seconds: float = 60.0,
        api_key: str | None = None,
        base_url: str = "",
        extra_params: dict[str, Any] | None = None,
        stream: bool = False,
        response_format: dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = True,
    ) -> ChatCompletionResult:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        # Omitting max_tokens asks vLLM for the full remaining model context.
        # A fixed short cap would silently invalidate long-form scenarios.
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
            if parallel_tool_calls is not None:
                payload["parallel_tool_calls"] = parallel_tool_calls

        if extra_params:
            extra_params = dict(extra_params)
            response_format_mode = extra_params.pop(STRUCTURED_RESPONSE_FORMAT_KEY, None)
            payload.update(extra_params)
        else:
            response_format_mode = None

        if response_format:
            backend_response_format = _response_format_for_backend(response_format, response_format_mode)
            if backend_response_format:
                payload["response_format"] = backend_response_format

        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        url = _chat_completions_url(base_url)
        client = self._get_client()
        req_timeout = httpx.Timeout(timeout_seconds)

        request_method = self._stream_request if stream else self._non_stream_request
        mirror_base_url = os.getenv("TOOL_EVAL_MIRROR_BASE_URL", "").strip()
        if not mirror_base_url:
            return await request_method(client, url, payload, headers, req_timeout)

        # external_launcher TP gives every rank an independent deterministic
        # scheduler.  Both API frontends therefore must receive and drain the
        # same request at the same time.  Return rank 0's response to the
        # benchmark, while rank 1 exists only to keep the TP collectives live.
        mirror_url = _chat_completions_url(mirror_base_url)
        primary, mirror = await asyncio.gather(
            request_method(client, url, payload, headers, req_timeout),
            request_method(client, mirror_url, payload, headers, req_timeout),
        )
        self._require_matched_mirror(primary, mirror)
        return primary

    @staticmethod
    def _require_matched_mirror(
        primary: ChatCompletionResult,
        mirror: ChatCompletionResult,
    ) -> None:
        """Fail before the next turn if the two TP schedulers diverged."""

        def comparable(result: ChatCompletionResult) -> tuple:
            return (
                result.content,
                result.reasoning,
                tuple(
                    (call.name, call.arguments_str)
                    for call in result.tool_calls
                ),
                bool(result.raw_response.get("loop_aborted")),
            )

        if comparable(primary) != comparable(mirror):
            raise RuntimeError(
                "The mirrored external-launcher TP responses diverged; "
                "stopping before mismatched follow-up requests can deadlock the ranks."
            )

    async def _non_stream_request(
        self,
        client: httpx.AsyncClient,
        url: str,
        payload: dict,
        headers: dict,
        timeout: httpx.Timeout | None = None,
    ) -> ChatCompletionResult:
        started = time.perf_counter()
        response = await client.post(url, json=payload, headers=headers, timeout=timeout)
        elapsed_ms = (time.perf_counter() - started) * 1000
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            # 4xx errors (400/422) often mean the server couldn't process
            # malformed tool-call arguments in conversation history (e.g.
            # vLLM's _postprocess_messages).  Return a graceful error
            # instead of crashing the scenario.  5xx errors are genuine
            # server failures and must propagate.
            if exc.response.status_code >= 500:
                raise
            body_text = exc.response.text[:200].strip()
            logger.warning(
                "Server returned %d for %s: %s",
                exc.response.status_code,
                url,
                body_text,
            )
            return ChatCompletionResult(
                content=f"[server error {exc.response.status_code}] {body_text}",
                tool_calls=[],
                raw_response={},
                elapsed_ms=elapsed_ms,
            )
        try:
            data = response.json()
        except Exception as exc:
            logger.warning("Malformed JSON in response from %s: %s", url, exc)
            return ChatCompletionResult(
                content="[malformed response]",
                tool_calls=[],
                raw_response={},
                elapsed_ms=elapsed_ms,
            )
        return self._parse_response(data, elapsed_ms)

    async def _stream_request(
        self,
        client: httpx.AsyncClient,
        url: str,
        payload: dict,
        headers: dict,
        timeout: httpx.Timeout | None = None,
    ) -> ChatCompletionResult:
        """Stream SSE response, measuring TTFT accurately."""
        started = time.perf_counter()
        ttft_ms: float | None = None
        content_parts: list[str] = []
        tool_calls_map: dict[int, dict] = {}  # index → {id, name, arguments}
        reasoning_parts: list[str] = []
        stream_usage: dict = {}  # usage from final chunk
        loop_probe_parts: list[str] = []
        loop_probe_chars = 0
        next_loop_check = _LOOP_CHECK_EVERY_CHARS
        loop_period_chars: int | None = None

        async with client.stream(
            "POST", url, json=payload, headers=headers, timeout=timeout
        ) as response:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                elapsed_ms = (time.perf_counter() - started) * 1000
                if exc.response.status_code >= 500:
                    raise
                body_bytes = await exc.response.aread()
                body_text = body_bytes.decode("utf-8", errors="replace")[:200].strip()
                logger.warning(
                    "Stream request returned %d for %s: %s",
                    exc.response.status_code,
                    url,
                    body_text,
                )
                return ChatCompletionResult(
                    content=f"[server error {exc.response.status_code}] {body_text}",
                    tool_calls=[],
                    raw_response={},
                    elapsed_ms=elapsed_ms,
                )
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload_str = line[6:]
                if payload_str.strip() == "[DONE]":
                    break

                try:
                    chunk = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue

                # Capture usage from final chunk (vLLM/OpenAI include this)
                if chunk.get("usage"):
                    stream_usage = chunk["usage"]

                choices = chunk.get("choices") or []
                if not choices:
                    continue

                delta = choices[0].get("delta") or {}

                # Measure TTFT on first content or tool_call delta
                if ttft_ms is None and (delta.get("content") or delta.get("tool_calls")):
                    ttft_ms = (time.perf_counter() - started) * 1000

                # Accumulate content
                if delta.get("content"):
                    content_delta = delta["content"]
                    content_parts.append(content_delta)
                    loop_probe_parts.append(content_delta)
                    loop_probe_chars += len(content_delta)

                # Accumulate reasoning
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if reasoning:
                    reasoning_parts.append(reasoning)
                    loop_probe_parts.append(reasoning)
                    loop_probe_chars += len(reasoning)

                # Accumulate tool calls
                for tc_delta in delta.get("tool_calls") or []:
                    idx = tc_delta.get("index", 0)
                    if idx not in tool_calls_map:
                        tool_calls_map[idx] = {
                            "id": tc_delta.get("id", f"tool_call_{idx + 1}"),
                            "name": "",
                            "arguments": "",
                        }
                    func = tc_delta.get("function") or {}
                    if func.get("name"):
                        tool_calls_map[idx]["name"] = func["name"]
                    if func.get("arguments"):
                        argument_delta = func["arguments"]
                        tool_calls_map[idx]["arguments"] += argument_delta
                        loop_probe_parts.append(argument_delta)
                        loop_probe_chars += len(argument_delta)
                    if tc_delta.get("id"):
                        tool_calls_map[idx]["id"] = tc_delta["id"]

                if loop_probe_chars >= next_loop_check:
                    loop_period_chars = _exact_suffix_loop_period("".join(loop_probe_parts))
                    next_loop_check = loop_probe_chars + _LOOP_CHECK_EVERY_CHARS
                    if loop_period_chars is not None:
                        logger.warning(
                            "Aborting streamed request after an exact output loop "
                            "(period=%d characters, observed_output=%d characters)",
                            loop_period_chars,
                            loop_probe_chars,
                        )
                        # Leaving the streaming response context closes the HTTP
                        # request; vLLM then aborts the disconnected generation.
                        break

        elapsed_ms = (time.perf_counter() - started) * 1000
        content = "".join(content_parts)
        reasoning_str = "".join(reasoning_parts) or None

        # Build tool calls
        tool_calls: list[ProviderToolCall] = []
        for idx in sorted(tool_calls_map.keys()):
            tc = tool_calls_map[idx]
            # Repair truncated JSON arguments.  When the server uses
            # --stream-interval > 1, tool-call arguments may arrive in
            # larger batches where the closing brace is split across
            # chunks or missing entirely (vLLM issue with batched token
            # streaming).  Apply the same repair logic used for the
            # round-trip message to ensure arguments are parseable.
            raw_args = tc["arguments"]
            repaired_args = _repair_streamed_tool_args(raw_args)
            if repaired_args != raw_args:
                logger.debug(
                    "Repaired streamed tool-call arguments for index %d: %.80s → %.80s",
                    idx,
                    raw_args,
                    repaired_args,
                )
            tool_calls.append(
                ProviderToolCall(
                    id=tc["id"],
                    name=tc["name"],
                    arguments_str=repaired_args,
                )
            )

        return ChatCompletionResult(
            content=content,
            tool_calls=tool_calls,
            raw_response=(
                {
                    "loop_aborted": True,
                    "loop_period_chars": loop_period_chars,
                    "observed_output_chars": loop_probe_chars,
                }
                if loop_period_chars is not None
                else {}
            ),
            elapsed_ms=elapsed_ms,
            ttft_ms=ttft_ms,
            reasoning=reasoning_str,
            prompt_tokens=stream_usage.get("prompt_tokens"),
            completion_tokens=stream_usage.get("completion_tokens"),
        )

    @staticmethod
    def _parse_response(data: dict, elapsed_ms: float) -> ChatCompletionResult:
        message = {}
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError):
            pass

        content = message.get("content") or ""
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ).strip()

        tool_calls = _normalize_tool_calls(message.get("tool_calls"))
        reasoning = message.get("reasoning_content") or message.get("reasoning")

        # Extract token usage (most OpenAI-compatible servers include this)
        usage = data.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")

        return ChatCompletionResult(
            content=content,
            tool_calls=tool_calls,
            raw_response=data,
            elapsed_ms=elapsed_ms,
            reasoning=reasoning,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

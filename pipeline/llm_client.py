"""
pipeline/llm_client.py — Provider-Agnostic LLM Client

Wraps Claude (Anthropic), Gemini (Google), and local LLM servers (Ollama,
LM Studio, llama.cpp) behind a single call interface.
All pipeline stages import this instead of calling the SDKs directly.

Provider selection (in priority order):
    1. LLM_PROVIDER env var — "claude" | "gemini" | "local"
    2. Auto-detect from available API keys / local server:
         LOCAL_LLM_URL     → local   (checked first — prefer local if configured)
         ANTHROPIC_API_KEY → claude
         GEMINI_API_KEY    → gemini
    3. Raises if no provider can be resolved

Environment variables:
    LLM_PROVIDER      "claude" | "gemini" | "local"  (optional, auto-detected)
    ANTHROPIC_API_KEY  required for claude
    GEMINI_API_KEY     required for gemini
    LLM_MODEL          override default model for any provider

    Local LLM (Ollama / LM Studio / llama.cpp):
    LOCAL_LLM_URL      base URL of the OpenAI-compatible server
                       default: http://localhost:11434  (Ollama default)
                       LM Studio: http://localhost:1234
                       llama.cpp: http://localhost:8080
    LOCAL_LLM_MODEL    model name to request (required for local)
                       Ollama:    "llama3.2", "mistral", "qwen2.5-coder:14b", ...
                       LM Studio: exact model filename shown in the UI
    LOCAL_LLM_API_KEY  optional bearer token (LM Studio supports this)
                       Ollama does not require an API key
    LOCAL_LLM_NUM_CTX  Ollama-only: total context window size (input + output
                       tokens).  When set, the request is sent to Ollama's
                       native /api/chat endpoint (not /v1/chat/completions)
                       which is the only path that accepts num_ctx at runtime.
                       Ollama's default num_ctx is model-dependent and often
                       small (2048–4096); set this to 16384 or 32768 when you
                       see truncated output with large batches:
                         export LOCAL_LLM_NUM_CTX=16384
    LOCAL_LLM_NUM_PREDICT
                       Ollama-only (native /api/chat path): maximum tokens the
                       model may generate per response (num_predict).  Defaults
                       to the per-call max_tokens value (8192 for stage45).
                       Thinking models (qwen3, deepseek-r1) consume most of
                       that budget on reasoning before writing the final answer,
                       so they hit the limit and return empty content.
                       Set to -1 for unlimited, or a large value like 32768:
                         export LOCAL_LLM_NUM_PREDICT=-1   # unlimited
                         export LOCAL_LLM_NUM_PREDICT=32768
    LOCAL_LLM_THINKING "1" | "true" to enable Ollama extended-thinking mode
                       for models that support it (e.g. qwen3, deepseek-r1).
                       Disabled by default because thinking-only models set
                       content="" and put all output in the "thinking" field,
                       which breaks structured-output parsing.  The pipeline
                       always uses explicit chain-of-thought prompting instead.
                         export LOCAL_LLM_THINKING=1  # re-enable if needed

    Response cache (pipeline/llm_cache.py):
    LLM_CACHE_ENABLED  "1"|"true"|"yes" to enable the SQLite exact-match cache
                       (default: disabled).  When enabled, identical calls
                       (same model + prompts + temperature) return the cached
                       response instantly — saves significant time and tokens
                       when re-running the pipeline on the same codebase.
                         export LLM_CACHE_ENABLED=1
    LLM_CACHE_PATH     Path to the SQLite cache file
                       (default: .llm_cache.db in the current directory)
                         export LLM_CACHE_PATH=/tmp/llm_cache.db

    Claude prefix caching:
    CLAUDE_PREFIX_CACHE
                       "1"|"true"|"yes" to attach cache_control breakpoints to
                       Claude system prompts (default: enabled when provider=claude).
                       Reduces cost for large, stable system prompts (stage4/5/6).
                       Set to "0" to disable if you hit caching-related API errors.
                         export CLAUDE_PREFIX_CACHE=0  # disable

Default models:
    Claude : claude-sonnet-4-20250514
    Gemini : gemini-2.5-flash-lite   (free tier)
    Local  : (must be set via LOCAL_LLM_MODEL — no default)

Quick-start for Ollama:
    ollama serve                        # starts server on port 11434
    ollama pull qwen2.5-coder:14b       # or any model
    export LLM_PROVIDER=local
    export LOCAL_LLM_MODEL=qwen2.5-coder:14b
    python run_pipeline.py --project /path/to/project

Quick-start for LM Studio:
    # Start "Local Server" in LM Studio, load a model, note the port (default 1234)
    export LLM_PROVIDER=local
    export LOCAL_LLM_URL=http://localhost:1234
    export LOCAL_LLM_MODEL="your-model-name-from-lm-studio"

Retry behaviour:
    Transient errors (DNS failures, network timeouts, 429 rate-limit, 500/503
    server errors) are automatically retried up to MAX_RETRIES times with
    exponential backoff.  Non-retryable errors (401 bad key, 400 bad request,
    safety blocks, empty responses) are raised immediately.

Usage:
    from pipeline.llm_client import call_llm, get_provider

    response_text = call_llm(
        system_prompt="You are a BA analyst...",
        user_prompt="Analyse this codebase...",
        max_tokens=4096,
    )

    provider = get_provider()   # "claude" | "gemini" | "local"
"""

from __future__ import annotations

import os
import socket
import time
from typing import Optional


# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_CLAUDE_MODEL  = "claude-sonnet-4-20250514"
DEFAULT_GEMINI_MODEL  = "gemini-2.5-flash-lite"      # stable free tier default
DEFAULT_LOCAL_LLM_URL = "http://localhost:11434" # Ollama default

# Gemini free tier: 15 RPM, 1M TPM, 1500 RPD
# Add a small delay between calls to stay well within limits
GEMINI_INTER_CALL_DELAY = 4   # seconds

# Local LLM request timeout — local inference can be slow for large prompts
LOCAL_LLM_TIMEOUT = 1800  # seconds (30 minutes)

# ── Retry configuration ───────────────────────────────────────────────────────
MAX_RETRIES        = 3          # maximum number of retry attempts after first failure
RETRY_BASE_DELAY   = 5.0        # seconds — first retry waits this long
RETRY_BACKOFF      = 2.0        # multiplier: 5s → 10s → 20s
RETRY_MAX_DELAY    = 60.0       # cap so we never wait more than 1 minute


# ─── Public API ────────────────────────────────────────────────────────────────

def get_provider() -> str:
    """
    Determine which LLM provider to use.

    Returns:
        "claude" | "gemini" | "local"

    Raises:
        RuntimeError: If no provider can be resolved.
    """
    explicit = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if explicit in ("claude", "gemini", "local"):
        return explicit
    if explicit:
        raise RuntimeError(
            f"Unknown LLM_PROVIDER value: {explicit!r}\n"
            f"Valid values: claude | gemini | local"
        )

    # Auto-detect: prefer local if LOCAL_LLM_URL is set (user explicitly
    # configured a local server), then fall back to cloud providers.
    if os.environ.get("LOCAL_LLM_URL"):
        return "local"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude"
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"

    raise RuntimeError(
        "No LLM provider configured.  Set one of:\n"
        "  Local (Ollama):   export LOCAL_LLM_URL=http://localhost:11434\n"
        "                    export LOCAL_LLM_MODEL=llama3.2\n"
        "  Gemini (free):    export GEMINI_API_KEY=your-key\n"
        "  Claude (paid):    export ANTHROPIC_API_KEY=sk-ant-...\n"
        "Or force a provider: export LLM_PROVIDER=local|gemini|claude"
    )


def get_model() -> str:
    """
    Return the active model name (respects LLM_MODEL override).

    For the local provider, LOCAL_LLM_MODEL is the primary source.
    LLM_MODEL overrides everything.
    Raises RuntimeError if local is selected but no model is specified.
    """
    override = os.environ.get("LLM_MODEL", "").strip()
    if override:
        return override

    provider = get_provider()

    if provider == "local":
        model = os.environ.get("LOCAL_LLM_MODEL", "").strip()
        if not model:
            raise RuntimeError(
                "LOCAL_LLM_MODEL is not set.\n"
                "Set it to the model name your local server has loaded, e.g.:\n"
                "  export LOCAL_LLM_MODEL=llama3.2          # Ollama\n"
                "  export LOCAL_LLM_MODEL=qwen2.5-coder:14b # Ollama\n"
                "  export LOCAL_LLM_MODEL=your-model-name   # LM Studio"
            )
        return model

    if provider == "claude":
        return DEFAULT_CLAUDE_MODEL
    return DEFAULT_GEMINI_MODEL


def get_local_url() -> str:
    """Return the base URL for the local LLM server."""
    return os.environ.get("LOCAL_LLM_URL", DEFAULT_LOCAL_LLM_URL).rstrip("/")


def call_llm(
    system_prompt: str,
    user_prompt:   str,
    max_tokens:    int = 8192,
    temperature:   float = 0.2,
    label:         str = "",        # e.g. "stage4" — used in log messages
    json_mode:     bool = False,    # force JSON output (local provider only)
    prefill:       str = "",        # assistant pre-fill text (Claude only)
                                    # e.g. prefill="[" forces JSON array output,
                                    # completely preventing any reasoning preamble.
                                    # The prefill string is prepended to the
                                    # returned text so callers see a complete value.
    json_schema:   dict | None = None,  # optional JSON schema for constrained decoding
                                        # (Ollama ≥0.4.6 native path only).
                                        # When set, grammar-level enforcement replaces
                                        # best-effort json_mode — invalid JSON is
                                        # literally impossible.  json_mode is
                                        # automatically set True when schema is given.
    model_override: str | None = None,  # override the configured model for this call
                                        # used by multi-model ensemble
) -> str:
    """
    Call the configured LLM provider and return the response text.

    Transient network and server errors are retried automatically with
    exponential backoff (up to MAX_RETRIES attempts).

    Args:
        system_prompt: System / instruction prompt.
        user_prompt:   User / content prompt.
        max_tokens:    Maximum tokens in the response.
        temperature:   Sampling temperature (0.0–1.0).
        label:         Optional stage label for log messages.
        prefill:       Optional assistant pre-fill (Claude only).  When set,
                       the string is sent as the start of the assistant turn
                       so the model must continue from that exact text —
                       prevents any chain-of-thought preamble.

    Returns:
        Raw response text from the model (with prefill prepended when used).

    Raises:
        RuntimeError: On non-retryable API error, or after all retries exhausted.
    """
    from pipeline import llm_cache as _cache

    provider = get_provider()
    model    = model_override or get_model()
    if json_schema is not None:
        json_mode = True  # schema implies json_mode
    tag      = f"[{label}] " if label else ""

    print(f"  {tag}LLM provider : {provider} ({model})")

    # ── Exact-match cache lookup ───────────────────────────────────────────────
    cache_key = _cache.make_key(model, system_prompt, user_prompt,
                                temperature, json_mode, prefill)
    cached = _cache.get(cache_key, label=label)
    if cached is not None:
        return cached

    if provider == "claude":
        call_fn = _call_claude
    elif provider == "gemini":
        call_fn = _call_gemini
    else:
        call_fn = _call_local

    last_exc: Exception | None = None
    delay = RETRY_BASE_DELAY

    for attempt in range(MAX_RETRIES + 1):   # attempt 0 = first try
        try:
            if provider == "local":
                result = _call_local(system_prompt, user_prompt, max_tokens,
                                     temperature, model, json_mode=json_mode,
                                     prefill=prefill, json_schema=json_schema)
            elif provider == "claude":
                result = _call_claude(system_prompt, user_prompt, max_tokens,
                                      temperature, model, prefill=prefill)
            else:
                result = call_fn(system_prompt, user_prompt, max_tokens,
                                 temperature, model)
            # ── JSON validation + single correction retry ──────────────────
            if json_mode:
                ok, err = _validate_json(result)
                if not ok:
                    corrected = _json_correction_call(
                        system_prompt, result, err,
                        max_tokens, temperature, model, provider, label,
                    )
                    ok2, _ = _validate_json(corrected)
                    if ok2:
                        result = corrected
                    else:
                        print(f"  {tag}JSON correction also invalid — using original")
            _cache.put(cache_key, result, label=label)
            return result

        except _NonRetryableError:
            raise   # bad key, safety block, empty response — don't retry

        except Exception as exc:
            last_exc = exc
            if attempt == MAX_RETRIES:
                break   # exhausted — fall through to final raise

            wait = min(delay, RETRY_MAX_DELAY)
            print(
                f"  {tag}Transient error on attempt {attempt + 1}/{MAX_RETRIES + 1}: "
                f"{_short_exc(exc)}"
            )
            print(f"  {tag}Retrying in {wait:.0f}s ...")
            time.sleep(wait)
            delay *= RETRY_BACKOFF

    raise RuntimeError(
        f"LLM call failed after {MAX_RETRIES + 1} attempt(s). "
        f"Last error: {last_exc}"
    )


def _validate_json(text: str) -> tuple[bool, str]:
    """
    Try to parse *text* as JSON.  Returns (True, "") on success or
    (False, error_message) on failure.  Strips markdown fences first.
    """
    import json as _json
    t = text.strip()
    if t.startswith("```"):
        t = "\n".join(l for l in t.splitlines() if not l.strip().startswith("```")).strip()
    try:
        _json.loads(t)
        return True, ""
    except _json.JSONDecodeError as exc:
        return False, str(exc)


def _json_correction_call(
    system_prompt: str,
    bad_response:  str,
    error_msg:     str,
    max_tokens:    int,
    temperature:   float,
    model:         str,
    provider:      str,
    label:         str,
) -> str:
    """
    Make a single correction call asking the model to fix its malformed JSON.
    Returns the corrected text (may still be invalid — caller decides).
    """
    tag = f"[{label}] " if label else ""
    fix_system = (
        system_prompt
        + "\n\nYour previous response was not valid JSON. "
        "Return ONLY the corrected JSON — no prose, no fences, no explanation."
    )
    fix_user = (
        f"Your previous response had a JSON parse error:\n{error_msg}\n\n"
        f"Malformed response:\n{bad_response}\n\n"
        "Please return the corrected JSON only."
    )
    print(f"  {tag}JSON invalid — requesting correction")
    if provider == "local":
        return _call_local(fix_system, fix_user, max_tokens, temperature, model, json_mode=True)
    elif provider == "claude":
        return _call_claude(fix_system, fix_user, max_tokens, temperature, model)
    else:
        return _call_gemini(fix_system, fix_user, max_tokens, temperature, model)


# ─── Internal sentinel for errors that must not be retried ────────────────────

class _NonRetryableError(RuntimeError):
    """
    Wraps errors where retrying is pointless (bad API key, safety block, etc.).
    Raised inside provider functions; caught in call_llm and re-raised as-is.
    """
    pass




# ─── Local LLM (Ollama / LM Studio / llama.cpp) ────────────────────────────────

def _call_local(
    system_prompt: str,
    user_prompt:   str,
    max_tokens:    int,
    temperature:   float,
    model:         str,
    json_mode:     bool = False,
    prefill:       str = "",
    json_schema:   dict | None = None,
) -> str:
    """
    Call a local LLM server.

    Routing:
      - LOCAL_LLM_NUM_CTX set → Ollama native /api/chat (supports num_ctx)
      - Otherwise            → OpenAI-compatible /v1/chat/completions
                               (Ollama, LM Studio, llama.cpp)

    No external SDK required — uses only the Python standard library (urllib).

    Environment variables consumed:
        LOCAL_LLM_URL      Base URL  (default: http://localhost:11434)
        LOCAL_LLM_MODEL    Model name (required — e.g. "llama3.2")
        LOCAL_LLM_API_KEY  Bearer token (optional)
        LOCAL_LLM_NUM_CTX  Ollama only: total context window tokens; when set,
                           the native /api/chat endpoint is used so that Ollama
                           loads the model with this context size at request time.
                           Recommended: 16384 or 32768 for large batch workloads.
    """
    import json
    import urllib.error
    import urllib.request

    num_ctx = int(os.environ.get("LOCAL_LLM_NUM_CTX", "0") or "0")
    if num_ctx > 0:
        return _call_local_ollama_native(
            system_prompt, user_prompt, max_tokens, temperature,
            model, num_ctx, prefill, json_mode=json_mode, json_schema=json_schema,
        )

    base_url = get_local_url()
    endpoint = f"{base_url}/v1/chat/completions"
    api_key  = os.environ.get("LOCAL_LLM_API_KEY", "").strip()

    messages_payload: list[dict] = [
        {"role": "system",    "content": system_prompt},
        {"role": "user",      "content": user_prompt},
    ]
    if prefill:
        # OpenAI-compatible servers (Ollama ≥ 0.1, LM Studio ≥ 0.2) support
        # assistant pre-fill — the model continues from the given text.
        messages_payload.append({"role": "assistant", "content": prefill})

    payload = {
        "model":       model,
        "messages":    messages_payload,
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "stream":      False,
    }
    if json_schema is not None:
        # Structured output via JSON schema — Ollama ≥0.4.6, LM Studio ≥0.3.
        # Grammar-level enforcement: invalid JSON is literally impossible.
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "strict": True, "schema": json_schema},
        }
    elif json_mode:
        # Forces Ollama / LM Studio / llama.cpp to output valid JSON.
        # Supported by Ollama ≥ 0.1.14 and LM Studio ≥ 0.2.
        payload["response_format"] = {"type": "json_object"}

    body    = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=LOCAL_LLM_TIMEOUT) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        _handle_local_http_error(exc, base_url, model)
    except urllib.error.URLError as exc:
        # Connection refused / DNS failure — server is not running
        raise RuntimeError(
            f"Cannot connect to local LLM server at {base_url}.\n"
            f"Is the server running?\n"
            f"  Ollama:    ollama serve\n"
            f"  LM Studio: start the Local Server in the LM Studio UI\n"
            f"  llama.cpp: ./server -m your-model.gguf\n"
            f"Original error: {exc.reason}"
        )
    except TimeoutError:
        raise RuntimeError(
            f"Local LLM request timed out after {LOCAL_LLM_TIMEOUT}s.\n"
            f"The model may be too large for your hardware, or still loading.\n"
            f"Increase LOCAL_LLM_TIMEOUT by setting it in llm_client.py."
        )

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Local LLM returned non-JSON response.\n"
            f"Raw (first 200 chars): {raw[:200]!r}\n"
            f"Parse error: {exc}"
        )

    # Standard OpenAI response shape
    choices = data.get("choices", [])
    if not choices:
        # Ollama sometimes puts the error in a top-level "error" field
        error_detail = data.get("error", data)
        raise _NonRetryableError(
            f"Local LLM returned no choices.\n"
            f"Response: {json.dumps(error_detail)[:300]}"
        )

    message = choices[0].get("message", {})
    content = message.get("content", "") or ""

    # Thinking/reasoning models (e.g. Qwen3, DeepSeek-R1) may return an empty
    # "content" field and put the entire response in "reasoning".  Fall back to
    # it so the pipeline can continue without the caller needing to know.
    if not content:
        content = message.get("reasoning", "") or ""

    # Strip <think>...</think> blocks that Qwen3 and DeepSeek-R1 prepend to
    # their final answer — the pipeline only wants the answer, not the scratchpad.
    import re as _re
    content = _re.sub(r"<think>.*?</think>", "", content, flags=_re.DOTALL).strip()

    if not content:
        raise _NonRetryableError(
            f"Local LLM returned an empty content field.\n"
            f"Full response: {json.dumps(data)[:300]}"
        )

    # Log finish reason if the output was truncated
    finish_reason = choices[0].get("finish_reason", "")
    if finish_reason == "length":
        print(
            f"  Warning: local LLM hit max_tokens limit ({max_tokens}). "
            f"Response length: {len(content)} chars."
        )

    # Restore prefill so the caller sees the complete value
    return (prefill + content) if prefill else content


def _call_local_ollama_native(
    system_prompt: str,
    user_prompt:   str,
    max_tokens:    int,
    temperature:   float,
    model:         str,
    num_ctx:       int,
    prefill:       str = "",
    json_mode:     bool = False,
    json_schema:   dict | None = None,
) -> str:
    """
    Call Ollama via its native /api/chat endpoint.

    The OpenAI-compat /v1/chat/completions endpoint silently ignores num_ctx;
    the native endpoint accepts it under options.num_ctx and reloads the model
    context if needed.  This is the only reliable way to override the default
    context window (often 2048–4096) for large-batch workloads.

    Response shape differs from OpenAI:
        { "message": { "role": "assistant", "content": "..." },
          "done": true, "done_reason": "stop" }
    """
    import json
    import urllib.error
    import urllib.request

    base_url = get_local_url()
    endpoint = f"{base_url}/api/chat"
    api_key  = os.environ.get("LOCAL_LLM_API_KEY", "").strip()

    messages_payload: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    if prefill:
        messages_payload.append({"role": "assistant", "content": prefill})

    # Disable extended-thinking by default for Ollama thinking models
    # (qwen3, deepseek-r1, etc.).  When thinking is active, Ollama puts the
    # reasoning trace in message.thinking and leaves message.content empty,
    # which breaks every structured-output parser downstream.  The pipeline
    # uses explicit chain-of-thought system prompts instead.
    # Set LOCAL_LLM_THINKING=1 to re-enable if you specifically want traces.
    thinking_enabled = os.environ.get("LOCAL_LLM_THINKING", "").strip().lower() in (
        "1", "true", "yes"
    )

    # LOCAL_LLM_NUM_PREDICT lets users override the per-response token budget
    # independently of max_tokens.  Thinking models burn most of max_tokens on
    # the reasoning trace before writing the final answer, hitting the limit and
    # returning empty content.  Set to -1 (unlimited) or a large value like
    # 32768 when using qwen3, deepseek-r1, or any other thinking model.
    _num_predict_env = os.environ.get("LOCAL_LLM_NUM_PREDICT", "").strip()
    num_predict = int(_num_predict_env) if _num_predict_env else max_tokens

    payload = {
        "model":    model,
        "messages": messages_payload,
        "stream":   False,
        "options": {
            "num_predict": num_predict,
            "num_ctx":     num_ctx,
            "temperature": temperature,
        },
    }
    if not thinking_enabled:
        # Ollama ≥0.6: top-level "think" key suppresses the reasoning trace
        # and forces all output into message.content.
        payload["think"] = False
    if json_schema is not None:
        # Ollama ≥0.4.6: pass JSON schema directly as "format" for grammar-level
        # constrained decoding — the model cannot emit anything that violates the schema.
        payload["format"] = json_schema
    elif json_mode:
        # Ollama native /api/chat uses a top-level "format" field (not inside
        # "options") to constrain output to valid JSON.
        payload["format"] = "json"

    body    = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=LOCAL_LLM_TIMEOUT) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        _handle_local_http_error(exc, base_url, model)
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot connect to Ollama at {base_url}.\n"
            f"Is 'ollama serve' running?\n"
            f"Original error: {exc.reason}"
        )
    except TimeoutError:
        raise RuntimeError(
            f"Ollama request timed out after {LOCAL_LLM_TIMEOUT}s."
        )

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Ollama returned non-JSON.\nRaw: {raw[:200]!r}\nError: {exc}"
        )

    # Native /api/chat response: data["message"]["content"]
    msg = data.get("message", {})
    content = (msg.get("content") or "").strip()

    # Strip <think>...</think> from Qwen3 scratchpad (inline thinking tags)
    import re as _re
    content = _re.sub(r"<think>.*?</think>", "", content, flags=_re.DOTALL).strip()

    if not content:
        # Thinking models (qwen3, deepseek-r1) sometimes emit an empty content
        # field and place the entire response inside message.thinking when the
        # model ignores our think=false flag (older Ollama builds) or the model
        # was custom-built without the no-think patch.
        # Try to recover by using the thinking field as the response text.
        thinking_fallback = (msg.get("thinking") or "").strip()
        if thinking_fallback:
            print(
                "  [llm_client] ⚠ Ollama returned empty content but non-empty "
                "thinking field — using thinking output as response fallback.\n"
                "  Most likely cause: num_predict budget exhausted during thinking "
                "before the model wrote its final answer.\n"
                "  Fix: export LOCAL_LLM_NUM_PREDICT=-1  (unlimited) or a large "
                "value like 32768."
            )
            content = thinking_fallback
        else:
            raise _NonRetryableError(
                f"Ollama /api/chat returned empty content.\n"
                f"Full response: {json.dumps(data)[:300]}"
            )

    done_reason = data.get("done_reason", "")
    if done_reason == "length":
        print(
            f"  Warning: Ollama hit num_predict limit ({num_predict}) "
            f"with num_ctx={num_ctx}. Response: {len(content)} chars.\n"
            f"  Fix: export LOCAL_LLM_NUM_PREDICT=-1  (unlimited) or a larger value."
        )

    return (prefill + content) if prefill else content


def _handle_local_http_error(
    exc: "urllib.error.HTTPError",
    base_url: str,
    model: str,
) -> None:
    """
    Translate HTTP error codes from the local server into helpful messages.

    Raises _NonRetryableError for permanent failures and plain RuntimeError
    for transient ones (so call_llm's retry loop picks them up).
    """
    code = exc.code
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:
        body = "(could not read response body)"

    if code == 404:
        raise _NonRetryableError(
            f"Local LLM model not found: {model!r}\n"
            f"Check the model is loaded in your server.\n"
            f"  Ollama:    ollama pull {model}\n"
            f"  LM Studio: load the model in the UI\n"
            f"Response body: {body[:200]}"
        )
    if code == 400:
        raise _NonRetryableError(
            f"Local LLM rejected the request (HTTP 400).\n"
            f"This may mean the model doesn't support the parameters sent.\n"
            f"Response body: {body[:200]}"
        )
    if code == 401:
        raise _NonRetryableError(
            f"Local LLM requires authentication (HTTP 401).\n"
            f"Set LOCAL_LLM_API_KEY to the correct bearer token.\n"
            f"Response body: {body[:200]}"
        )
    if code in (429, 503):
        # Transient — server overloaded or rate limited
        raise RuntimeError(
            f"Local LLM server busy (HTTP {code}). Will retry.\n"
            f"Response body: {body[:200]}"
        )
    if code >= 500:
        raise RuntimeError(
            f"Local LLM server error (HTTP {code}). Will retry.\n"
            f"Response body: {body[:200]}"
        )
    # Unknown HTTP error — treat as transient
    raise RuntimeError(
        f"Local LLM HTTP error {code} from {base_url}.\n"
        f"Response body: {body[:200]}"
    )
# ─── Claude ────────────────────────────────────────────────────────────────────

def _call_claude(
    system_prompt: str,
    user_prompt:   str,
    max_tokens:    int,
    temperature:   float,
    model:         str,
    prefill:       str = "",
) -> str:
    """Call Anthropic Claude API.

    When `prefill` is set it is appended as a partial assistant message so the
    model must continue from that exact text.  The prefill string is prepended
    to the returned text so callers see a complete response value.

    Example — force immediate JSON array output (no chain-of-thought preamble):
        _call_claude(..., prefill="[")
    """
    try:
        import anthropic
    except ImportError:
        raise _NonRetryableError(
            "anthropic package not installed.\n"
            "Run: pip install anthropic"
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise _NonRetryableError(
            "ANTHROPIC_API_KEY environment variable is not set.\n"
            "Export it: export ANTHROPIC_API_KEY=sk-ant-..."
        )

    messages: list[dict] = [{"role": "user", "content": user_prompt}]
    if prefill:
        messages.append({"role": "assistant", "content": prefill})

    # Prefix caching: mark the system prompt for server-side KV caching.
    # Anthropic caches content blocks whose cumulative token count exceeds
    # ~1024 tokens; smaller prompts are cached but the saving is minimal.
    # Disable via CLAUDE_PREFIX_CACHE=0 if you hit unexpected API errors.
    prefix_cache = os.environ.get("CLAUDE_PREFIX_CACHE", "1").strip().lower() not in (
        "0", "false", "no"
    )
    if prefix_cache:
        system_param: list[dict] | str = [
            {
                "type":          "text",
                "text":          system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    else:
        system_param = system_prompt

    client  = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model       = model,
        max_tokens  = max_tokens,
        temperature = temperature,
        system      = system_param,
        messages    = messages,
    )

    if not message.content:
        raise _NonRetryableError("Claude returned an empty response.")

    # Find the first text block (skip thinking blocks on extended-thinking models)
    text = next(
        (block.text for block in message.content if getattr(block, "type", "") == "text"),
        None,
    )
    if text is None:
        raise _NonRetryableError("Claude returned no text content block.")

    # Restore prefill so the caller gets the complete value
    return (prefill + text) if prefill else text


# ─── Gemini ────────────────────────────────────────────────────────────────────

def _call_gemini(
    system_prompt: str,
    user_prompt:   str,
    max_tokens:    int,
    temperature:   float,
    model:         str,
) -> str:
    """
    Call Google Gemini API using the current google-genai SDK.

    Gemini free tier limits (as of 2025):
        gemini-2.5-flash-lite : 15 RPM, 1M TPM, 1500 RPD
        gemini-1.5-pro   : 2 RPM,  32k TPM, 50 RPD

    A small inter-call delay is applied to stay within RPM limits when
    multiple stages run in sequence.
    """
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        raise _NonRetryableError(
            "google-genai package not installed.\n"
            "Run: pip install google-genai"
        )

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise _NonRetryableError(
            "GEMINI_API_KEY environment variable is not set.\n"
            "Get a free key at: aistudio.google.com\n"
            "Export it: export GEMINI_API_KEY=your-key"
        )

    # Respect free tier rate limits
    time.sleep(GEMINI_INTER_CALL_DELAY)

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model    = model,
            contents = user_prompt,
            config   = genai_types.GenerateContentConfig(
                system_instruction = system_prompt,
                max_output_tokens  = max_tokens,
                temperature        = temperature,
                thinking_config = genai_types.ThinkingConfig(
                    include_thoughts = False,
                    thinking_budget  = 0
                )
            ),
        )
    except Exception as exc:
        _handle_gemini_error(exc)   # raises either _NonRetryableError or plain RuntimeError

    # Check for blocked/empty response
    if not response.candidates:
        raise _NonRetryableError(
            "Gemini returned no candidates. "
            "The prompt may have been blocked by safety filters."
        )

    candidate = response.candidates[0]

    # Check finish reason — must do this BEFORE reading response.text
    finish_reason = str(candidate.finish_reason)
    if "SAFETY" in finish_reason:
        raise _NonRetryableError(
            f"Gemini response blocked by safety filter: {finish_reason}"
        )

    # Extract text first, then check truncation
    text = response.text or ""

    if "MAX_TOKENS" in finish_reason:
        # Truncated — warn but return what we have (caller has JSON recovery)
        print(f"  Warning: Gemini hit max_tokens limit. "
              f"Response length: {len(text)} chars. Attempting JSON recovery.")

    if not text:
        raise _NonRetryableError("Gemini returned an empty text response.")

    return text


def _handle_gemini_error(exc: Exception) -> None:
    """
    Translate common Gemini API errors into helpful messages.

    Raises _NonRetryableError for permanent failures (bad key, model not found,
    bad request).  Raises plain RuntimeError for transient failures so that
    call_llm's retry loop picks them up.
    """
    msg = str(exc)

    # ── Transient — will be retried ───────────────────────────────────────────
    if _is_network_error(exc):
        raise RuntimeError(
            f"Gemini API error: {exc}"
        )
    if "429" in msg or "quota" in msg.lower() or "resource_exhausted" in msg.lower():
        raise RuntimeError(
            f"Gemini rate limit hit (free tier: 15 RPM). "
            f"Original error: {exc}"
        )
    if "500" in msg or "502" in msg or "503" in msg or "unavailable" in msg.lower():
        raise RuntimeError(
            f"Gemini server error (transient). Original error: {exc}"
        )

    # ── Non-retryable — bad config, bad request, etc. ─────────────────────────
    if "401" in msg or "api_key" in msg.lower() or "invalid" in msg.lower():
        raise _NonRetryableError(
            "Gemini API key is invalid or expired.\n"
            "Check your key at aistudio.google.com\n"
            f"Original error: {exc}"
        )
    if "403" in msg:
        raise _NonRetryableError(
            f"Gemini API access denied.\nOriginal error: {exc}"
        )
    if "404" in msg or "not found" in msg.lower():
        raise _NonRetryableError(
            f"Gemini model not found. Check LLM_MODEL env var or use default 'gemini-2.5-flash-lite'.\n"
            f"Original error: {exc}"
        )

    # Unknown — treat as transient so the retry loop gets a chance
    raise RuntimeError(f"Gemini API error: {exc}")


def _is_network_error(exc: Exception) -> bool:
    """
    Return True if the exception looks like a transient network failure:
    DNS resolution failure, connection refused, timeout, or SSL error.

    These all have errno codes or belong to OSError / socket.error, but
    they can also surface wrapped inside SDK-specific exception types, so
    we check the exception chain as well as the message text.
    """
    # Walk the full exception chain (e, __cause__, __context__)
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, (OSError, socket.error, TimeoutError, ConnectionError)):
            return True
        # urllib3 / httpx / requests surface these strings
        msg = str(current).lower()
        if any(kw in msg for kw in (
            "nodename nor servname",   # [Errno 8] — the exact error in the report
            "name or service not known",
            "failed to resolve",
            "connection refused",
            "connection reset",
            "connection timed out",
            "timed out",
            "ssl",
            "eof occurred",
            "broken pipe",
            "network is unreachable",
            "no route to host",
        )):
            return True
        current = current.__cause__ or (
            current.__context__ if not current.__suppress_context__ else None
        )
    return False


def _short_exc(exc: Exception) -> str:
    """Return a one-line summary of an exception for log messages."""
    return f"{type(exc).__name__}: {str(exc)[:120]}"
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

    Local LLM (Ollama / vLLM / LM Studio / llama.cpp):
    LOCAL_LLM_BACKEND  Backend type — controls routing and payload format.
                       "ollama"   (default) Ollama via /api/chat or /v1/chat/completions
                       "vllm"               vLLM  via /v1/chat/completions only
                       "lmstudio"           LM Studio via /v1/chat/completions
                       "llamacpp"           llama.cpp server via /v1/chat/completions
                         export LOCAL_LLM_BACKEND=vllm
    LOCAL_LLM_URL      base URL of the server
                       Ollama:    http://localhost:11434  (default)
                       vLLM:      http://localhost:8000
                       LM Studio: http://localhost:1234
                       llama.cpp: http://localhost:8080
    LOCAL_LLM_MODEL    model name to request (required for local)
                       Ollama:    "llama3.2", "qwen2.5-coder:14b", ...
                       vLLM:      HuggingFace model ID used at server launch
                       LM Studio: exact model filename shown in the UI
    LOCAL_LLM_API_KEY  optional bearer token
                       vLLM:      set via --api-key at server launch
                       LM Studio: configured in the UI
                       Ollama:    not required
    LOCAL_LLM_NUM_CTX  Ollama-only: total context window size (input + output
                       tokens).  When set, the request is routed to Ollama's
                       native /api/chat endpoint which is the only path that
                       accepts num_ctx at request time.  Ignored for vLLM
                       (set --max-model-len at server launch instead).
                       Recommended: 16384 or 32768 for large batches:
                         export LOCAL_LLM_NUM_CTX=32768
    LOCAL_LLM_NUM_PREDICT
                       Ollama-only (native /api/chat path): maximum tokens the
                       model may generate per response (num_predict).  Defaults
                       to the per-call max_tokens value.  Set to -1 for
                       unlimited output (recommended for thinking models and
                       large flow/spec batches):
                         export LOCAL_LLM_NUM_PREDICT=-1   # unlimited
                         export LOCAL_LLM_NUM_PREDICT=32768
                       vLLM equivalent: max_tokens is passed directly in the
                       payload — set LOCAL_LLM_NUM_PREDICT=-1 has no effect;
                       use a large LLM_MAX_TOKENS instead.
    LOCAL_LLM_THINKING "1" | "true" to enable Ollama extended-thinking mode
                       for models that support it (e.g. qwen3, deepseek-r1).
                       Disabled by default; the pipeline uses explicit CoT
                       system prompts instead.
                         export LOCAL_LLM_THINKING=1
    VLLM_GUIDED_JSON   "1" to use vLLM's guided_json field for constrained
                       decoding instead of the OpenAI-style response_format.
                       Use this with vLLM < 0.4.3 that predates structured-
                       output support.  Default: 0 (use response_format).
                         export VLLM_GUIDED_JSON=1
    VLLM_REPETITION_PENALTY
                       Float penalty applied to already-generated tokens to
                       prevent degenerate repetition loops (e.g. "Wait, I need
                       to check…" repeating hundreds of times).
                       1.0 = disabled; default = 1.05 (light nudge).
                         export VLLM_REPETITION_PENALTY=1.1
    VLLM_ENABLE_THINKING
                       "1" to enable chain-of-thought reasoning for Qwen3 /
                       DeepSeek-R1 models on vLLM (sets chat_template_kwargs
                       enable_thinking=true per request).  Default: 0 (off).
                       When disabled, no <think> scratchpad is generated so
                       there is nothing to leak into the output document.
                         export VLLM_ENABLE_THINKING=1
    VLLM_REPETITION_WINDOW / VLLM_REPETITION_THRESHOLD
                       Post-hoc loop detector: slide a window of WINDOW chars
                       over the response; if the same block appears ≥ THRESHOLD
                       times, truncate at the second occurrence.
                       Defaults: WINDOW=200, THRESHOLD=3.

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
import threading
import time
from typing import Optional


# ── Backend config-file loader ─────────────────────────────────────────────────
def _load_backend_config() -> None:
    """
    Load per-backend env config file, setting variables that are not already
    in the environment (env vars always take precedence).

    Lookup order (first found wins):
      1. .llm/<backend>.env          — project-local (gitignored)
      2. ~/.config/codebase-ba/<backend>.env  — user-level

    File format — plain KEY=VALUE, one per line:
      VLLM_MODEL=qwen3.5-27b-nvfp4
      VLLM_MAX_WORKERS=8
      # comments are ignored
      OLLAMA_NUM_CTX=32768

    Security: files are parsed as plain text (no eval/shell execution).
    Sensitive values (API keys etc.) belong in the user-level file with
    chmod 600, not in the project-local file.
    """
    backend = os.environ.get("LOCAL_LLM_BACKEND", "").strip().lower()
    if not backend:
        return

    candidates = [
        os.path.join(os.getcwd(), ".llm", f"{backend}.env"),
        os.path.expanduser(f"~/.config/codebase-ba/{backend}.env"),
    ]

    for path in candidates:
        if not os.path.isfile(path):
            continue
        loaded: list[str] = []
        with open(path) as _f:
            for line in _f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
                    loaded.append(key)
        if loaded:
            print(f"  [llm_client] Loaded backend config: {path} ({', '.join(loaded)})")
        return   # stop after first found


_load_backend_config()   # runs once at import time


# ── Per-call token stats ───────────────────────────────────────────────────────
# Thread-local so concurrent stage50 agents don't overwrite each other's counts.
# Provider functions write here; call_llm reads after each call.
_tok_stats = threading.local()

def _set_tok(in_tok: int, out_tok: int) -> None:
    """Called by provider functions to store real input/output token counts."""
    _tok_stats.in_tok  = in_tok
    _tok_stats.out_tok = out_tok

def _get_tok() -> tuple[int, int]:
    """Returns (in_tokens, out_tokens); (-1, -1) if provider did not report."""
    return getattr(_tok_stats, "in_tok", -1), getattr(_tok_stats, "out_tok", -1)


# ── LLM call telemetry ─────────────────────────────────────────────────────────
# Collects one record per successful call; written to JSONL during the run and
# summarised into a grouped JSON at pipeline end.
import json as _json_tel
import datetime as _dt

_telemetry_path: str | None = None          # set by run_pipeline via set_telemetry_path()
_telemetry_lock = threading.Lock()          # guards file writes from concurrent stage50 threads


def set_telemetry_path(path: str) -> None:
    """Point the telemetry writer at *path* (JSONL, one record per call).
    Call once after PipelineContext is created, before the pipeline runs."""
    global _telemetry_path
    _telemetry_path = path


def _record_telemetry(label: str, model: str, duration_s: float,
                      tokens_in: int, tokens_out: int) -> None:
    """Append one JSON line to the telemetry file (no-op if path not set)."""
    if not _telemetry_path:
        return
    record = {
        "timestamp":  _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "stage":      label or "unknown",
        "model":      model,
        "duration_s": round(duration_s, 3),
        "tokens_in":  tokens_in,
        "tokens_out": tokens_out,
    }
    with _telemetry_lock:
        with open(_telemetry_path, "a", encoding="utf-8") as _f:
            _f.write(_json_tel.dumps(record) + "\n")


def write_telemetry_summary(summary_path: str) -> None:
    """Read the JSONL telemetry file and write a grouped summary JSON.

    Output format::

        {
          "generated_at": "<iso timestamp>",
          "total_calls": 42,
          "total_tokens_in": 1234567,
          "total_tokens_out": 234567,
          "total_duration_s": 456.7,
          "by_stage": {
            "stage40_domain": {
              "calls": 5,
              "total_tokens_in": 50000,
              "total_tokens_out": 10000,
              "total_duration_s": 45.2,
              "entries": [ {...}, ... ]
            },
            ...
          }
        }
    """
    if not _telemetry_path:
        return
    import os as _os
    if not _os.path.exists(_telemetry_path):
        return

    records: list[dict] = []
    with open(_telemetry_path, encoding="utf-8") as _f:
        for line in _f:
            line = line.strip()
            if line:
                try:
                    records.append(_json_tel.loads(line))
                except _json_tel.JSONDecodeError:
                    pass

    by_stage: dict[str, dict] = {}
    for r in records:
        stage = r.get("stage", "unknown")
        if stage not in by_stage:
            by_stage[stage] = {"calls": 0, "total_tokens_in": 0,
                               "total_tokens_out": 0, "total_duration_s": 0.0,
                               "entries": []}
        g = by_stage[stage]
        g["calls"]            += 1
        g["total_tokens_in"]  += r.get("tokens_in",  0) if r.get("tokens_in",  -1) >= 0 else 0
        g["total_tokens_out"] += r.get("tokens_out", 0) if r.get("tokens_out", -1) >= 0 else 0
        g["total_duration_s"] = round(g["total_duration_s"] + r.get("duration_s", 0.0), 3)
        g["entries"].append(r)

    summary = {
        "generated_at":     _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "total_calls":      len(records),
        "total_tokens_in":  sum(r.get("tokens_in",  0) for r in records if r.get("tokens_in",  -1) >= 0),
        "total_tokens_out": sum(r.get("tokens_out", 0) for r in records if r.get("tokens_out", -1) >= 0),
        "total_duration_s": round(sum(r.get("duration_s", 0.0) for r in records), 3),
        "by_stage":         by_stage,
    }
    with open(summary_path, "w", encoding="utf-8") as _f:
        _json_tel.dump(summary, _f, indent=2)


# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_CLAUDE_MODEL  = "claude-sonnet-4-20250514"
DEFAULT_GEMINI_MODEL  = "gemini-2.5-flash-lite"      # stable free tier default
DEFAULT_LOCAL_LLM_URL = "http://localhost:11434" # Ollama default

# Default ports per backend — combined with LOCAL_LLM_HOST when set.
# Set LOCAL_LLM_URL to override the full URL entirely.
_BACKEND_DEFAULT_PORTS: dict[str, int] = {
    "ollama":   11434,
    "vllm":     8000,
    "lmstudio": 1234,
    "llamacpp": 8080,
}
_BACKEND_DEFAULT_URLS: dict[str, str] = {
    k: f"http://localhost:{v}" for k, v in _BACKEND_DEFAULT_PORTS.items()
}

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

# ── vLLM-specific generation controls ─────────────────────────────────────────
# VLLM_REPETITION_PENALTY — penalises the model for repeating tokens it has
#   already generated.  1.0 = disabled; 1.05 is a light nudge that prevents
#   degenerate "Wait, I need to check..." infinite loops without noticeably
#   hurting output quality.  Override with VLLM_REPETITION_PENALTY=<float>.
_VLLM_REPETITION_PENALTY = float(
    os.environ.get("VLLM_REPETITION_PENALTY", "1.05") or "1.05"
)

# VLLM_ENABLE_THINKING — Qwen3/DeepSeek-R1 on vLLM expose a per-request
#   enable_thinking flag via chat_template_kwargs.  Default: disabled (0).
#   When disabled, the model skips its chain-of-thought scratchpad entirely —
#   no <think> blocks are generated, so there is nothing to leak or strip.
#   Set VLLM_ENABLE_THINKING=1 only if you deliberately want CoT reasoning.
_VLLM_ENABLE_THINKING = (
    os.environ.get("VLLM_ENABLE_THINKING", "0").strip().lower() in ("1", "true", "yes")
)

# VLLM_REPETITION_WINDOW / VLLM_REPETITION_THRESHOLD — post-hoc loop detector.
#   After every vLLM response, slide a window of WINDOW chars across the output.
#   If the same WINDOW-char block appears ≥ THRESHOLD times, the text is
#   truncated at the second occurrence (preserving the first clean copy).
_VLLM_REPETITION_WINDOW    = int(os.environ.get("VLLM_REPETITION_WINDOW",    "200") or "200")
_VLLM_REPETITION_THRESHOLD = int(os.environ.get("VLLM_REPETITION_THRESHOLD", "3")   or "3")


# ─── Public API ────────────────────────────────────────────────────────────────

def get_max_workers() -> int:
    """
    Recommended ThreadPoolExecutor size for concurrent LLM calls.

    Resolution order:
      1. LLM_MAX_WORKERS          — global explicit override
      2. OLLAMA_MAX_WORKERS /
         VLLM_MAX_WORKERS / etc.  — backend-specific override
      3. coded default per backend (vllm=4, others=1)
    """
    explicit = os.environ.get("LLM_MAX_WORKERS", "").strip()
    if explicit:
        return max(1, int(explicit))
    backend  = os.environ.get("LOCAL_LLM_BACKEND", "").strip().lower()
    provider = get_provider()
    if provider == "local" and backend:
        backend_var = f"{backend.upper()}_MAX_WORKERS"
        backend_explicit = os.environ.get(backend_var, "").strip()
        if backend_explicit:
            return max(1, int(backend_explicit))
    if provider == "local" and backend == "vllm":
        return 4
    return 1


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

    # Auto-detect: prefer local if backend or URL is configured.
    if os.environ.get("LOCAL_LLM_URL") or os.environ.get("LOCAL_LLM_BACKEND"):
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
    Return the active model name.

    Resolution order:
      1. LLM_MODEL               — global explicit override
      2. OLLAMA_MODEL /
         VLLM_MODEL / etc.       — backend-specific model
      3. LOCAL_LLM_MODEL         — legacy fallback (any backend)
      4. cloud defaults          — Claude / Gemini built-in defaults
    """
    override = os.environ.get("LLM_MODEL", "").strip()
    if override:
        return override

    provider = get_provider()

    if provider == "local":
        backend = os.environ.get("LOCAL_LLM_BACKEND", "").strip().lower()
        if backend:
            backend_model = os.environ.get(f"{backend.upper()}_MODEL", "").strip()
            if backend_model:
                return backend_model
        model = os.environ.get("LOCAL_LLM_MODEL", "").strip()
        if not model:
            raise RuntimeError(
                "No model configured for local LLM.\n"
                "Set a backend-specific model or the generic fallback, e.g.:\n"
                "  export VLLM_MODEL=qwen3.5-27b-nvfp4\n"
                "  export OLLAMA_MODEL=qwen3.5:27b\n"
                "  export LOCAL_LLM_MODEL=your-model-name   # any backend"
            )
        return model

    if provider == "claude":
        return DEFAULT_CLAUDE_MODEL
    return DEFAULT_GEMINI_MODEL


def get_local_url() -> str:
    """Return the base URL for the local LLM server.

    Resolution order:
      1. LOCAL_LLM_URL   — full URL override, always wins
      2. LOCAL_LLM_HOST  — host/IP only; port auto-derived from backend
      3. backend default — http://localhost:<backend-port>
      4. DEFAULT_LOCAL_LLM_URL (Ollama :11434) — final fallback
    """
    explicit = os.environ.get("LOCAL_LLM_URL", "").strip()
    if explicit:
        return explicit.rstrip("/")
    host = os.environ.get("LOCAL_LLM_HOST", "").strip()
    backend = os.environ.get("LOCAL_LLM_BACKEND", "").strip().lower()
    port = _BACKEND_DEFAULT_PORTS.get(backend)
    if host and port:
        return f"http://{host}:{port}"
    return _BACKEND_DEFAULT_URLS.get(backend, DEFAULT_LOCAL_LLM_URL).rstrip("/")


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

    _backend_label = os.environ.get("LOCAL_LLM_BACKEND", "").strip().lower()
    _provider_str  = (
        f"{provider}/{_backend_label}" if provider == "local" and _backend_label
        else provider
    )
    print(f"  {tag}LLM provider : {_provider_str} ({model})")

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

    # For vLLM: omit max_tokens so the server uses remaining context space instead
    # of pre-allocating a fixed KV-cache budget.  This avoids spurious 400 errors
    # when prompt_tokens + max_tokens exceeds max_model_len even though the model
    # would naturally stop well before that limit.
    if provider == "local" and _backend_label == "vllm":
        max_tokens = -1   # signals _call_local to omit the field from the payload

    last_exc: Exception | None = None
    delay    = RETRY_BASE_DELAY
    attempt  = 0          # counts transient-error retries (sleeps); capped at MAX_RETRIES
    _overflow_count  = 0  # counts _ContextLengthError reductions; has its own cap
    _MAX_OVERFLOW    = 12 # allow up to 12 successive token-budget reductions
    _overflow_buffer = 256  # initial safety margin; doubles on each overflow to converge

    while True:
        try:
            _set_tok(-1, -1)                  # reset before each attempt
            _t0 = time.monotonic()
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
            _elapsed = time.monotonic() - _t0
            _in, _out = _get_tok()
            if _in >= 0 and _out >= 0:
                _tok_str = f"in={_in:,} out={_out:,} tok"
            else:
                _tok_str = f"~{len(result.split()):,} words"  # fallback if provider didn't report
            print(f"  {tag}done in {_elapsed:.1f}s  ({_tok_str})")
            _record_telemetry(label, model, _elapsed, _in, _out)
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

        except _ContextLengthError as exc:
            # Prompt too long for the model's context window.
            # Reduce max_tokens and retry immediately — this does NOT consume a
            # transient-error retry slot (separate counter with its own cap).
            #
            # The server reports (ctx_max, prompt_toks) in the error body.
            # We compute: available = ctx_max - prompt_toks - _overflow_buffer
            # and double the buffer each time so successive retries converge
            # quickly even when vLLM's token count drifts between requests.
            _overflow_count += 1
            if _overflow_count > _MAX_OVERFLOW:
                raise RuntimeError(
                    f"{tag}Context window still exceeded after {_MAX_OVERFLOW} "
                    f"max_tokens reductions (current budget: {max_tokens}). "
                    f"Reduce the prompt size or use a larger-context model."
                )
            # Re-apply the buffer with the current (doubled) margin so each
            # successive call leaves more room, compensating for any tokeniser
            # drift the server reports.
            raw_available = exc.available_tokens   # ctx_max - prompt_toks - 64
            reduced = max(256, raw_available - (_overflow_buffer - 64))
            print(
                f"  {tag}Context overflow (attempt {_overflow_count}/{_MAX_OVERFLOW}) — "
                f"reducing max_tokens {max_tokens} → {reduced} and retrying ..."
            )
            max_tokens = reduced
            _overflow_buffer = min(_overflow_buffer * 2, 4096)  # double, cap at 4096
            # continue loop immediately without incrementing `attempt`

        except Exception as exc:
            last_exc = exc
            if attempt >= MAX_RETRIES:
                break   # exhausted — fall through to final raise

            wait = min(delay, RETRY_MAX_DELAY)
            print(
                f"  {tag}Transient error on attempt {attempt + 1}/{MAX_RETRIES + 1}: "
                f"{_short_exc(exc)}"
            )
            print(f"  {tag}Retrying in {wait:.0f}s ...")
            time.sleep(wait)
            delay *= RETRY_BACKOFF
            attempt += 1

    raise RuntimeError(
        f"LLM call failed after {attempt + 1} attempt(s). "
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

class _ContextLengthError(Exception):
    """
    Raised when the server returns HTTP 400 because prompt + max_tokens
    exceeds the model's context window.

    Carries available_tokens so call_llm can retry with a reduced budget.
    """
    def __init__(self, message: str, available_tokens: int) -> None:
        super().__init__(message)
        self.available_tokens = available_tokens
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
    Call a local LLM server (Ollama / vLLM / LM Studio / llama.cpp).

    Routing
    -------
    LOCAL_LLM_BACKEND=ollama (default)
      LOCAL_LLM_NUM_CTX set  → Ollama native /api/chat  (supports num_ctx at runtime)
      LOCAL_LLM_NUM_CTX unset → OpenAI-compat /v1/chat/completions

    LOCAL_LLM_BACKEND=vllm | lmstudio | llamacpp
      Always → OpenAI-compat /v1/chat/completions
      LOCAL_LLM_NUM_CTX is ignored (set --max-model-len at vLLM server launch)

    Constrained decoding (JSON schema)
    -----------------------------------
    Ollama  (native path)  → payload["format"] = schema
    Ollama  (compat path)  → response_format.json_schema
    vLLM  ≥ 0.4.3          → response_format.json_schema   (default)
    vLLM  < 0.4.3          → guided_json in body           (set VLLM_GUIDED_JSON=1)
    LM Studio / llama.cpp  → response_format.json_schema

    No external SDK required — pure Python standard library (urllib).
    """
    import json
    import urllib.error
    import urllib.request

    _backend = os.environ.get("LOCAL_LLM_BACKEND", "").strip().lower()
    # "ollama" (default), "vllm", "lmstudio", "llamacpp"

    # ── Ollama native path: only when backend is ollama AND num_ctx is set ──────
    # Resolution: LOCAL_LLM_NUM_CTX (global) → OLLAMA_NUM_CTX (backend-specific)
    _num_ctx_str = (
        os.environ.get("LOCAL_LLM_NUM_CTX", "").strip()
        or os.environ.get(f"{_backend.upper()}_NUM_CTX", "").strip()
        or "0"
    )
    num_ctx = int(_num_ctx_str or "0")
    if num_ctx > 0 and _backend not in ("vllm", "lmstudio", "llamacpp"):
        return _call_local_ollama_native(
            system_prompt, user_prompt, max_tokens, temperature,
            model, num_ctx, prefill, json_mode=json_mode, json_schema=json_schema,
        )
    if num_ctx > 0 and _backend == "vllm":
        print(f"  [llm_client] Note: LOCAL_LLM_NUM_CTX={num_ctx} is ignored for vLLM — "
              f"set --max-model-len at server launch instead.")

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

    payload: dict = {
        "model":       model,
        "messages":    messages_payload,
        "temperature": temperature,
        "stream":      False,
    }
    # max_tokens ≤ 0 means "let the server decide" — omit the field so vLLM
    # uses remaining context space rather than pre-allocating a fixed budget.
    if max_tokens > 0:
        payload["max_tokens"] = max_tokens

    # ── vLLM-specific payload fields ─────────────────────────────────────────
    if _backend == "vllm":
        # Repetition penalty — prevents degenerate "Wait, I need to check…"
        # infinite loops.  1.0 = disabled; we default to 1.05 (light nudge).
        if _VLLM_REPETITION_PENALTY != 1.0:
            payload["repetition_penalty"] = _VLLM_REPETITION_PENALTY

        # Qwen3 / DeepSeek-R1 thinking mode — disabled by default so the model
        # skips its CoT scratchpad entirely (nothing to leak into the document).
        # Passed via extra_body compatible with vLLM's OpenAI-compat endpoint.
        payload["chat_template_kwargs"] = {"enable_thinking": _VLLM_ENABLE_THINKING}
    if json_schema is not None:
        # Constrained decoding — grammar-level JSON schema enforcement.
        # vLLM < 0.4.3: use guided_json in the request body (VLLM_GUIDED_JSON=1)
        # vLLM ≥ 0.4.3 / Ollama ≥ 0.4.6 / LM Studio ≥ 0.3: response_format.json_schema
        _use_guided = (
            _backend == "vllm"
            and os.environ.get("VLLM_GUIDED_JSON", "0").strip() == "1"
        )
        if _use_guided:
            payload["guided_json"] = json_schema
        else:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "strict": True, "schema": json_schema},
            }
    elif json_mode:
        # JSON-object mode — valid JSON output without a fixed schema.
        # Supported by Ollama ≥ 0.1.14, vLLM ≥ 0.3, LM Studio ≥ 0.2.
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
        _backend_hint = {
            "vllm":     "  vLLM:      python -m vllm.entrypoints.openai.api_server "
                        "--model <model-id> --port 8000",
            "lmstudio": "  LM Studio: start the Local Server in the LM Studio UI",
            "llamacpp": "  llama.cpp: ./server -m your-model.gguf --port 8080",
        }.get(_backend, "  Ollama:    ollama serve")
        raise RuntimeError(
            f"Cannot connect to local LLM server at {base_url}.\n"
            f"Is the server running?\n"
            f"{_backend_hint}\n"
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

    # ── Post-processing: strip thinking / detect loops ───────────────────────
    if _backend == "vllm":
        # Full vLLM-specific pipeline: strip all CoT leakage then detect loops.
        # Even with enable_thinking=False, some model builds still emit partial
        # thinking tags — _strip_vllm_think handles all three leakage patterns.
        content = _strip_vllm_think(content)
        content = _strip_repetition_loop(content)
    else:
        # Ollama / LM Studio / llama.cpp: only strip properly-tagged <think> blocks.
        import re as _re
        content = _re.sub(r"<think>.*?</think>", "", content, flags=_re.DOTALL).strip()
        # Also strip unclosed <think> (model hit max_tokens during thinking phase)
        content = _re.sub(r"<think>.*$", "", content, flags=_re.DOTALL).strip()

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

    # Report real token counts from OpenAI usage field
    usage = data.get("usage", {})
    _set_tok(
        in_tok  = usage.get("prompt_tokens",     -1),
        out_tok = usage.get("completion_tokens",  -1),
    )

    # Restore prefill so the caller sees the complete value
    return (prefill + content) if prefill else content


# ── vLLM post-processing helpers ───────────────────────────────────────────────

def _strip_vllm_think(content: str) -> str:
    """
    Strip chain-of-thought leakage from vLLM responses.

    Three patterns handled in order:
      1. Properly closed <think>…</think> blocks  (Qwen3 happy path)
      2. Unclosed <think>… to end-of-string       (model hit max_tokens mid-think)
      3. Raw untagged thinking preamble            (vLLM leaked scratchpad without tags)

    For pattern 3: if the response does not begin with a document marker
    (# heading, { JSON, |table, -, *) we scan for the first such marker and
    discard everything before it.  This covers the "141 lines of raw Thinking
    Process" pattern seen in the Prism vLLM run.
    """
    import re as _re

    # Pattern 1 — closed <think> blocks (non-greedy, DOTALL)
    content = _re.sub(r"<think>.*?</think>", "", content, flags=_re.DOTALL).strip()

    # Pattern 2 — unclosed <think> (model hit max_tokens during thinking phase)
    content = _re.sub(r"<think>.*$", "", content, flags=_re.DOTALL).strip()

    if not content:
        return content

    # Pattern 3 — raw untagged preamble: strip everything before the first
    # document-like line (# heading, { JSON open, | table, list marker, ---).
    # Only applies when the response doesn't already start with such a marker.
    _DOC_START = _re.compile(
        r"^(#|\{|\[|\||---|```|-\s|\*\s|\d+\.\s)",
        _re.MULTILINE,
    )
    if not _DOC_START.match(content):
        m = _DOC_START.search(content)
        if m and m.start() > 0:
            stripped = content[m.start():]
            print(
                f"  [llm_client] ⚠️  vLLM thinking preamble stripped "
                f"({m.start()} chars before first document marker)."
            )
            content = stripped

    return content.strip()


def _strip_repetition_loop(content: str) -> str:
    """
    Detect and truncate degenerate repetition loops in LLM output.

    Tries multiple window sizes (50 → 400 chars) so it catches both short
    tight loops (e.g. a 60-char "*Okay, writing.*" pattern) and long
    paragraph-level loops (the "Wait, I need to check…" block that repeated
    800+ times in the Prism vLLM SRS run).

    For each window size, slides the window across the response with 50%
    overlap.  If the same window block appears ≥ _VLLM_REPETITION_THRESHOLD
    times from that position onward, the text is truncated at the start of
    the second occurrence — preserving the first clean copy.
    """
    threshold = _VLLM_REPETITION_THRESHOLD
    # Try windows from small to large; stop at first detected loop.
    # Small windows catch tight repeats; large windows catch paragraph loops.
    _WINDOWS = [50, 100, 150, 200, 300, 400]

    for window in _WINDOWS:
        if len(content) < window * threshold:
            continue            # text too short for this window size
        step = max(1, window // 2)
        i = 0
        while i < len(content) - window:
            chunk = content[i : i + window]
            count = content[i:].count(chunk)
            if count >= threshold:
                second = content.find(chunk, i + 1)
                if second > i:
                    print(
                        f"  [llm_client] ⚠️  Repetition loop detected "
                        f"(window={window}, count={count}) — truncating at char {second}."
                    )
                    return content[:second].rstrip()
            i += step

    return content


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

    # Report real token counts (Ollama native fields)
    _set_tok(
        in_tok  = data.get("prompt_eval_count", -1),
        out_tok = data.get("eval_count", -1),
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
        # Check for context-length overflow — auto-retry with a smaller budget
        import re as _re
        # Matches: "maximum context length is 16384 tokens ... 12385 input tokens"
        _ctx_m = _re.search(
            r"maximum context length is (\d+).*?(\d+)\s+input tokens",
            body, _re.DOTALL | _re.IGNORECASE,
        )
        if _ctx_m:
            ctx_max      = int(_ctx_m.group(1))
            prompt_toks  = int(_ctx_m.group(2))
            available    = max(256, ctx_max - prompt_toks - 64)   # 64-token safety margin
            raise _ContextLengthError(
                f"Context overflow: model max={ctx_max}, prompt={prompt_toks}, "
                f"available for output={available} tokens.",
                available_tokens=available,
            )
        raise _NonRetryableError(
            f"Local LLM rejected the request (HTTP 400).\n"
            f"This may mean the model doesn't support the parameters sent.\n"
            f"Response body: {body[:300]}"
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

    # Report real token counts (includes cache_read / cache_creation breakdowns)
    usage = message.usage
    _set_tok(
        in_tok  = getattr(usage, "input_tokens",  -1),
        out_tok = getattr(usage, "output_tokens", -1),
    )

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

    # Report real token counts
    meta = getattr(response, "usage_metadata", None)
    _set_tok(
        in_tok  = getattr(meta, "prompt_token_count",     -1) if meta else -1,
        out_tok = getattr(meta, "candidates_token_count", -1) if meta else -1,
    )

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
# `call_llm()` — Formal Interface Specification

> **Source:** [`pipeline/llm_client.py`](file:///Users/aungmaungmaungkyaw/private-projects/reverse-engineering/codebase-ba/pipeline/llm_client.py)
> **Version:** 2026-03-27

---

## 1. Canonical Signature

```python
from pipeline.llm_client import call_llm

response: str = call_llm(
    system_prompt:  str,           # System / instruction prompt
    user_prompt:    str,           # User / content prompt
    max_tokens:     int   = 8192,  # Max response tokens
    temperature:    float = 0.2,   # Sampling temperature (0.0–1.0)
    label:          str   = "",    # Stage label for logs / telemetry
    json_mode:      bool  = False, # Force JSON output (local only)
    prefill:        str   = "",    # Assistant pre-fill (Claude only)
    json_schema:    dict | None = None,  # Optional constrained-decoding schema
    model_override: str | None = None,   # Override configured model
)
```

### Return Value
Raw response text (`str`). When `prefill` is set, the prefill string is prepended to the model's continuation so callers always see a complete value.

---

## 2. Provider Routing

| Provider | Selection | SDK | Default Model |
|---|---|---|---|
| **Claude** | `LLM_PROVIDER=claude` or `ANTHROPIC_API_KEY` detected | `anthropic` | `claude-sonnet-4-20250514` |
| **Gemini** | `LLM_PROVIDER=gemini` or `GEMINI_API_KEY` detected | `google-genai` | `gemini-2.5-flash-lite` |
| **Local (Ollama)** | `LLM_PROVIDER=local` + `LOCAL_LLM_BACKEND=ollama` | stdlib `urllib` | Must set `LOCAL_LLM_MODEL` |
| **Local (vLLM)** | `LLM_PROVIDER=local` + `LOCAL_LLM_BACKEND=vllm` | stdlib `urllib` | Must set `VLLM_MODEL` |

Resolution priority: `LLM_PROVIDER` env → auto-detect (LOCAL_LLM_URL/BACKEND → ANTHROPIC_API_KEY → GEMINI_API_KEY).

### Public Helpers

| Function | Returns | Purpose |
|---|---|---|
| `get_provider()` | `"claude" \| "gemini" \| "local"` | Active provider name |
| `get_model()` | `str` | Active model name (backend-specific → global → default) |
| `get_local_url()` | `str` | Base URL for local server |
| `get_max_workers()` | `int` | Thread pool size for concurrent calls |

---

## 3. Retry Behaviour

### Transient Errors (exponential backoff)
- **Max retries:** 3
- **Base delay:** 5s → 10s → 20s (×2 backoff, capped at 60s)
- **Retried errors:** DNS failures, timeouts, connection refused, HTTP 429/500/503, unknown errors

### Context Length Overflow (auto-reduction)
- **Max reductions:** 12 successive attempts
- **Strategy:** Parse `ctx_max` and `prompt_toks` from the 400 error body, compute `available = ctx_max - prompt_toks - buffer`, double the safety buffer each iteration (256 → 512 → … → 4096)
- **Separate counter:** does NOT consume a transient retry slot

### Non-retryable Errors (raised immediately)
- HTTP 401 (bad API key)
- HTTP 400 (non-overflow bad request)
- Safety blocks (Gemini `SAFETY` finish reason)
- Empty responses (content field is empty/null)
- Missing SDK packages

---

## 4. Response Caching

| Property | Detail |
|---|---|
| **Implementation** | [`pipeline/llm_cache.py`](file:///Users/aungmaungmaungkyaw/private-projects/reverse-engineering/codebase-ba/pipeline/llm_cache.py) — SQLite (WAL mode) |
| **Key** | SHA-256 of `model + system_prompt + user_prompt + temperature + json_mode + prefill` |
| **TTL** | Infinite (deterministic inputs = equivalent output) |
| **Thread safety** | Per-process `threading.Lock` + WAL mode for multi-process reads |
| **Enable** | `LLM_CACHE_ENABLED=1` |
| **DB path** | `LLM_CACHE_PATH` (default: `.llm_cache.db`) |

### Cache API

```python
from pipeline.llm_cache import make_key, get, put, stats, clear
```

---

## 5. Telemetry

| Feature | Implementation |
|---|---|
| **Per-call token stats** | Thread-local `_tok_stats` (supports concurrent Stage 5 workers) |
| **JSONL recording** | `set_telemetry_path(path)` → one JSON line per call: `{timestamp, stage, model, duration_s, tokens_in, tokens_out}` |
| **Summary** | `write_telemetry_summary(path)` → grouped JSON with per-stage totals and overall stats |

---

## 6. Post-Processing Pipeline

All providers strip chain-of-thought artifacts before returning:

| Backend | Processing |
|---|---|
| **vLLM** | 3-pattern `_strip_vllm_think()` (closed tags → unclosed tags → raw untagged preamble) + multi-window `_strip_repetition_loop()` |
| **Ollama** | Regex strip `<think>…</think>` (closed + unclosed) + thinking-field fallback |
| **Claude** | Text-block extraction (skips thinking blocks from extended-thinking models) |
| **Gemini** | `include_thoughts=False` + `thinking_budget=0` at config level |

### JSON Correction
When `json_mode=True`, the response is validated. If JSON is invalid, a single correction call is made asking the model to fix its output. If the correction also fails, the original response is returned.

---

## 7. Provider-Specific Features

### Claude
- **Prefix caching:** `CLAUDE_PREFIX_CACHE=1` (default enabled) — attaches `cache_control: {type: "ephemeral"}` to system prompts for KV-cache reuse
- **Prefill:** Assistant pre-fill forces the model to continue from exact text (e.g. `prefill="["` for JSON arrays)

### Gemini
- **Rate limiting:** 4s inter-call delay for free tier (15 RPM)
- **Safety filter handling:** `SAFETY` finish reason → `_NonRetryableError`

### vLLM
- **Repetition penalty:** `VLLM_REPETITION_PENALTY=1.05` (default) — applied via payload
- **Thinking mode:** `VLLM_ENABLE_THINKING=0` (default) → `chat_template_kwargs.enable_thinking`
- **max_tokens omitted:** vLLM calls omit `max_tokens` so the server uses remaining context space
- **Constrained decoding:** `response_format.json_schema` (≥0.4.3) or `guided_json` (<0.4.3 via `VLLM_GUIDED_JSON=1`)

### Ollama
- **Native path:** When `LOCAL_LLM_NUM_CTX` is set, routes to `/api/chat` (the only endpoint accepting `num_ctx` at request time)
- **Compat path:** When `LOCAL_LLM_NUM_CTX` unset, routes to `/v1/chat/completions` (OpenAI-compatible)
- **Thinking mode:** `LOCAL_LLM_THINKING=1` enables extended thinking; disabled by default (`"think": false` in payload)
- **Constrained decoding:** Native path uses `"format"` field; compat path uses `response_format.json_schema`

---

## 8. Environment Variable Reference

| Variable | Values | Default | Description |
|---|---|---|---|
| `LLM_PROVIDER` | `claude\|gemini\|local` | auto-detect | Force a provider |
| `ANTHROPIC_API_KEY` | `sk-ant-…` | — | Claude API key |
| `GEMINI_API_KEY` | string | — | Gemini API key |
| `LLM_MODEL` | string | per-provider | Global model override |
| `LOCAL_LLM_BACKEND` | `ollama\|vllm\|lmstudio\|llamacpp` | `ollama` | Backend type |
| `LOCAL_LLM_URL` | URL | per-backend | Full server URL override |
| `LOCAL_LLM_HOST` | hostname/IP | `localhost` | Host for auto-port resolution |
| `LOCAL_LLM_MODEL` | string | — | Legacy model name fallback |
| `LOCAL_LLM_API_KEY` | string | — | Bearer token for local server |
| `LOCAL_LLM_NUM_CTX` | int | 0 | Ollama context window size |
| `LOCAL_LLM_NUM_PREDICT` | int | `max_tokens` | Ollama per-response token budget |
| `LOCAL_LLM_THINKING` | `0\|1` | `0` | Enable Ollama extended thinking |
| `OLLAMA_MODEL` | string | — | Ollama-specific model name |
| `VLLM_MODEL` | string | — | vLLM-specific model name |
| `VLLM_REPETITION_PENALTY` | float | `1.05` | vLLM repetition penalty |
| `VLLM_ENABLE_THINKING` | `0\|1` | `0` | vLLM chain-of-thought mode |
| `VLLM_GUIDED_JSON` | `0\|1` | `0` | Use guided_json for old vLLM |
| `VLLM_REPETITION_WINDOW` | int | `200` | Loop detector window size |
| `VLLM_REPETITION_THRESHOLD` | int | `3` | Loop detector repeat threshold |
| `LLM_CACHE_ENABLED` | `0\|1` | `0` | Enable SQLite response cache |
| `LLM_CACHE_PATH` | path | `.llm_cache.db` | Cache DB location |
| `CLAUDE_PREFIX_CACHE` | `0\|1` | `1` | Claude prefix caching |
| `LLM_MAX_WORKERS` | int | per-backend | Global concurrent call limit |

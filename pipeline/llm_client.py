"""
pipeline/llm_client.py — Provider-Agnostic LLM Client

Wraps Claude (Anthropic) and Gemini (Google) behind a single call interface.
All pipeline stages import this instead of calling the SDKs directly.

Provider selection (in priority order):
    1. LLM_PROVIDER env var — "claude" or "gemini"
    2. Auto-detect from available API keys:
         ANTHROPIC_API_KEY → claude
         GEMINI_API_KEY    → gemini
    3. Raises if neither is set

Environment variables:
    LLM_PROVIDER      "claude" | "gemini"  (optional, auto-detected if unset)
    ANTHROPIC_API_KEY  required for claude
    GEMINI_API_KEY     required for gemini
    LLM_MODEL          override the default model for either provider

Default models:
    Claude : claude-sonnet-4-20250514
    Gemini : gemini-2.5-flash-lite   (free tier)

Usage:
    from pipeline.llm_client import call_llm, get_provider

    response_text = call_llm(
        system_prompt="You are a BA analyst...",
        user_prompt="Analyse this codebase...",
        max_tokens=4096,
    )

    provider = get_provider()   # "claude" | "gemini"
"""

from __future__ import annotations

import os
import time
from typing import Optional


# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-20250514"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"  # stable free tier default

# Gemini free tier: 15 RPM, 1M TPM, 1500 RPD
# Add a small delay between calls to stay well within limits
GEMINI_INTER_CALL_DELAY = 4   # seconds


# ─── Public API ────────────────────────────────────────────────────────────────

def get_provider() -> str:
    """
    Determine which LLM provider to use.

    Returns:
        "claude" or "gemini"

    Raises:
        RuntimeError: If no API key is found for any provider.
    """
    explicit = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if explicit in ("claude", "gemini"):
        return explicit

    # Auto-detect from available keys
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude"
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"

    raise RuntimeError(
        "No LLM provider configured.\n"
        "For Gemini (free):  export GEMINI_API_KEY=your-key\n"
        "For Claude (paid):  export ANTHROPIC_API_KEY=sk-ant-...\n"
        "Or set explicitly:  export LLM_PROVIDER=gemini"
    )


def get_model() -> str:
    """Return the active model name (respects LLM_MODEL override)."""
    override = os.environ.get("LLM_MODEL", "").strip()
    if override:
        return override
    provider = get_provider()
    return DEFAULT_CLAUDE_MODEL if provider == "claude" else DEFAULT_GEMINI_MODEL


def call_llm(
    system_prompt: str,
    user_prompt:   str,
    max_tokens:    int = 8192,
    temperature:   float = 0.2,
    label:         str = "",        # e.g. "stage4" — used in log messages
) -> str:
    """
    Call the configured LLM provider and return the response text.

    Args:
        system_prompt: System / instruction prompt.
        user_prompt:   User / content prompt.
        max_tokens:    Maximum tokens in the response.
        temperature:   Sampling temperature (0.0–1.0).
        label:         Optional stage label for log messages.

    Returns:
        Raw response text from the model.

    Raises:
        RuntimeError: On API error or empty response.
    """
    provider = get_provider()
    model    = get_model()
    tag      = f"[{label}] " if label else ""

    print(f"  {tag}LLM provider : {provider} ({model})")

    if provider == "claude":
        return _call_claude(system_prompt, user_prompt, max_tokens, temperature, model)
    else:
        return _call_gemini(system_prompt, user_prompt, max_tokens, temperature, model)


# ─── Claude ────────────────────────────────────────────────────────────────────

def _call_claude(
    system_prompt: str,
    user_prompt:   str,
    max_tokens:    int,
    temperature:   float,
    model:         str,
) -> str:
    """Call Anthropic Claude API."""
    try:
        import anthropic
    except ImportError:
        raise RuntimeError(
            "anthropic package not installed.\n"
            "Run: pip install anthropic"
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable is not set.\n"
            "Export it: export ANTHROPIC_API_KEY=sk-ant-..."
        )

    client  = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model      = model,
        max_tokens = max_tokens,
        system     = system_prompt,
        messages   = [{"role": "user", "content": user_prompt}],
    )

    if not message.content:
        raise RuntimeError("Claude returned an empty response.")

    return message.content[0].text


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
        gemini-2.0-flash : 15 RPM, 1M TPM, 1500 RPD
        gemini-1.5-pro   : 2 RPM,  32k TPM, 50 RPD

    A small inter-call delay is applied to stay within RPM limits when
    multiple stages run in sequence.
    """
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        raise RuntimeError(
            "google-genai package not installed.\n"
            "Run: pip install google-genai"
        )

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
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
            ),
        )
    except Exception as e:
        _handle_gemini_error(e)

    # Check for blocked/empty response
    if not response.candidates:
        raise RuntimeError(
            "Gemini returned no candidates. "
            "The prompt may have been blocked by safety filters."
        )

    candidate = response.candidates[0]

    # Check finish reason — must do this BEFORE reading response.text
    finish_reason = str(candidate.finish_reason)
    if "SAFETY" in finish_reason:
        raise RuntimeError(
            f"Gemini response blocked by safety filter: {finish_reason}"
        )

    # Extract text first, then check truncation
    text = response.text or ""

    if "MAX_TOKENS" in finish_reason:
        # Truncated — warn but return what we have (caller has JSON recovery)
        print(f"  Warning: Gemini hit max_tokens limit. "
              f"Response length: {len(text)} chars. Attempting JSON recovery.")

    if not text:
        raise RuntimeError("Gemini returned an empty text response.")

    return text


def _handle_gemini_error(exc: Exception) -> None:
    """
    Translate common Gemini API errors into helpful messages.
    Always re-raises.
    """
    msg = str(exc)

    if "429" in msg or "quota" in msg.lower():
        raise RuntimeError(
            "Gemini rate limit hit (free tier: 15 RPM).\n"
            "Wait a minute and retry.\n"
            f"Original error: {exc}"
        )
    if "401" in msg or "api_key" in msg.lower() or "invalid" in msg.lower():
        raise RuntimeError(
            "Gemini API key is invalid or expired.\n"
            "Check your key at aistudio.google.com\n"
            f"Original error: {exc}"
        )
    if "403" in msg:
        raise RuntimeError(
            "Gemini API access denied.\n"
            f"Original error: {exc}"
        )
    if "404" in msg or "not found" in msg.lower():
        raise RuntimeError(
            f"Gemini model not found. Check LLM_MODEL env var or use default 'gemini-2.5-flash-lite'.\n"
            f"Original error: {exc}"
        )

    # Re-raise anything else as-is
    raise RuntimeError(f"Gemini API error: {exc}")
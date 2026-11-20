# codebase-ba — Claude Code Project Context

## What this project does

A multi-stage Python pipeline that reverse-engineers a source-code repository
into formal BA documents: BRD, SRS, Acceptance Criteria, and User Stories.
The pipeline runs ~30 numbered stages (stage00 → stage90) that parse, embed,
analyse, and generate documentation via an LLM backend (Claude / vLLM / Ollama).

---

## Key specs — read before touching the relevant files

### `call_llm()` interface contract
[`docs/llm_interface_spec.md`](docs/llm_interface_spec.md)

Every pipeline stage that calls an LLM **must** go through `call_llm()` in
`pipeline/llm_client.py`. Direct SDK calls (anthropic, openai, requests) are
not allowed. The spec documents the canonical signature, all parameters,
provider routing, JSON mode, prefill, telemetry, and retry behaviour.

### CodeMap schema
[`schemas/code_map.schema.json`](schemas/code_map.schema.json)

Machine-readable JSON Schema (draft 2020-12) for the `CodeMap` data structure
produced by Stage 1 parsers and consumed by all downstream stages. When adding
new fields to `CodeMap` (in `context.py`) or to a parser output, update this
schema too so `stage10_parse.py`'s schema validator catches regressions.

### Parser contract & compliance
[`docs/parser_audit.md`](docs/parser_audit.md)

All language parsers (PHP, TypeScript, Java) must satisfy the `LanguageParser`
base contract in `pipeline/parsers/base.py`. New parsers must pass every
requirement in the compliance matrix. Current open finding: PHP parser does not
set `_parser` hint in `code_map.json` output (cosmetic but inconsistent).

### Multi-language / cross-language strategy
[`docs/cross_language_strategy.md`](docs/cross_language_strategy.md)

Approved design for detecting cross-language references (OpenAPI contracts,
gRPC proto files, GraphQL schemas, shared TypeScript types, HTTP client calls).
Follow this strategy document when implementing multi-language support rather
than inventing ad-hoc detection heuristics.

---

## Architecture at a glance

```
run_pipeline.py          ← entry point; runs stages in order
pipeline/
  stage00_validate.py    ← pre-flight checks
  stage05_detect.py      ← language detection
  stage10_parse.py       ← code parsing dispatcher (PHP / TS / Java)
  stage22_components.py  ← React/Vue component graph
  stage30_embed.py       ← ChromaDB embedding
  stage40_domain.py      ← domain model extraction (LLM)
  stage50_workers.py     ← BA document agents: BRD, SRS, AC, User Stories (LLM)
  stage62_architecture.py← architecture reconstruction (LLM)
  stage80_tests.py       ← test generation: Gherkin, Playwright, pytest (LLM)
  llm_client.py          ← single LLM gateway (Claude / vLLM / Ollama)
  evidence_index.py      ← per-feature evidence builder for stage50
  parsers/
    base.py              ← LanguageParser ABC
    php_parser.py
    typescript_parser.py
    java_parser.py
context.py               ← PipelineContext + CodeMap dataclasses
```

---

## Coding conventions

- All LLM calls go through `call_llm()` — never import anthropic/openai directly
- Stage files follow `stage{NN}_{name}.py` naming; the `run(ctx)` function is the entry point
- `CodeMap` fields are added in `context.py` first, then the parser, then the schema
- Parser output keys must match `schemas/code_map.schema.json` required fields
- New parsers must pass the compliance matrix in `docs/parser_audit.md`
- Env-configurable caps follow the pattern `int(os.environ.get("STAGE5_CAP_X", "60") or "60")`
- vLLM backend: `chat_template_kwargs: {enable_thinking: false}` is injected automatically

# Codebase BA Pipeline

`codebase-ba` is a reverse-engineering pipeline that analyzes an existing software project and turns the codebase into business-analysis deliverables.

It starts from source code, builds several layers of static structure and semantic context, then uses an LLM to produce business-facing documentation such as:

- Business Requirements Document (BRD)
- Software Requirements Specification (SRS)
- Acceptance Criteria
- User Stories
- QA review output
- Mermaid diagrams
- Generated test assets
- A final system knowledge graph with a natural-language query CLI

The pipeline is multi-language at the parsing layer and currently supports:

- PHP
- TypeScript
- JavaScript
- Java
- Kotlin

The project is Python-first, with a PHP parser helper, optional Node.js tooling for DOCX generation, and optional Neo4j / LibreOffice integrations for later stages.

## What the pipeline produces

A typical run creates a timestamped output directory under:

```text
outputs/<project-slug>/run_<UTC timestamp>/
```

Observed runs in this repository include subdirectories such as:

```text
0_validation/
0.5_detect/
1_parse/
1.3_entrypoints/
1.5_paths/
2_graph/
2.2_components/
2.5_behavior/
2.7_semanticroles/
2.8_clusters/
2.9_invariants/
3_embed/
3.5_entities/
3.6_relationships/
3.7_statemachines/
3.8_graphrag/
3.9_preflight/
4_domain/
4.5_flows/
4.6_specrules/
4.7_validation/
4.8_triangulate/
5_documents/
5.5_traceability/
5.9_doccoverage/
6_qa/
6.2_architecture/
6.5_formatted/
6.7_diagrams/
7_delivery/
8_tests/
9_knowledge_graph/
```

Key output artifacts include:

- `context.json`: resumable pipeline state
- `1_parse/code_map.json`: normalized parse output
- `2_graph/code_graph.gpickle` and `2_graph/code_graph.json`: code graph
- `4_domain/domain_model.json`: LLM-derived domain model
- `4.5_flows/business_flows.json`: extracted business flows
- `5_documents/*.md`: BRD, SRS, AC, and user stories
- `6_qa/qa_report.md`: QA review
- `6.5_formatted/*.docx`: formatted DOCX documents
- `6.7_diagrams/diagrams/*.mmd`: Mermaid diagrams
- `7_delivery/*.pdf` and `7_delivery/delivery/`: PDF delivery bundle
- `8_tests/tests.feature`, `playwright_tests.js`, `pytest_tests.py`: generated tests
- `9_knowledge_graph/knowledge_graph.json`: business-level knowledge graph

## Repository layout

Top-level structure:

```text
run_pipeline.py          Main pipeline entrypoint
context.py               Shared pipeline context and output-path conventions
pipeline/                Core stage implementations
pipeline/parsers/        Language-specific parsers
pipeline/entrypoints/    Language-specific entrypoint discovery
pipeline/paths/          Language-specific path extraction
scripts/                 Standalone runners for selected stages
parsers/                 PHP parser helper script and Composer setup
tests/                   Limited test coverage
outputs/                 Sample/generated run artifacts checked into the repo
.llm/                    Optional backend-specific local env files
```

Notable entrypoints:

- `run_pipeline.py`: full pipeline runner
- `scripts/run_query.py`: interactive NL query CLI for the final knowledge graph
- `scripts/run_stage*.py`: standalone stage runners for selected stages
- `parsers/parse_project.php`: PHP parser helper script

## How the pipeline works

Stage order is defined in `run_pipeline.py`.

| Stage | Module | Purpose |
| --- | --- | --- |
| 0 | `pipeline/stage00_validate.py` | Validate project path and prerequisites |
| 0.5 | `pipeline/stage05_detect.py` | Detect primary language and framework |
| 1 | `pipeline/stage10_parse.py` | Parse source code through a language adapter |
| 1.3 | `pipeline/stage13_entrypoints.py` | Catalog entrypoints such as routes, CLI hooks, jobs, webhooks |
| 1.5 | `pipeline/stage15_paths.py` | Extract execution paths and branches |
| 2 | `pipeline/stage20_graph.py` | Build the code graph with NetworkX |
| 2.2 | `pipeline/stage22_components.py` | Build frontend component graph where relevant |
| 2.5 | `pipeline/stage25_behavior.py` | Extract behavior graph |
| 2.7 | `pipeline/stage27_semanticroles.py` | Tag semantic roles |
| 2.8 | `pipeline/stage28_clusters.py` | Cluster related actions |
| 2.9 | `pipeline/stage29_invariants.py` | Extract business rules and invariants |
| 3 | `pipeline/stage30_embed.py` | Create a ChromaDB semantic index |
| 3.5-3.8 | `pipeline/stage35_*` to `stage38_*` | Entities, relationships, state machines, graph-aware context |
| 3.9 | `pipeline/stage39_preflight.py` | Validate static context before LLM-heavy stages |
| 4 | `pipeline/stage40_domain.py` | Produce the domain model |
| 4.5 | `pipeline/stage45_flows.py` | Extract business flows |
| 4.6-4.8 | `pipeline/stage46_*` to `stage48_*` | Specification mining, flow validation, evidence triangulation |
| 5 | `pipeline/stage50_workers.py` | Generate BRD, SRS, AC, and user stories in parallel |
| 5.5 | `pipeline/stage55_traceability.py` | Build traceability artifacts |
| 5.8-5.9 | `pipeline/stage58_*` to `stage59_*` | Coverage and accuracy checks |
| 6 | `pipeline/stage60_qa.py` | Produce QA review |
| 6.2 | `pipeline/stage62_architecture.py` | Reconstruct architecture |
| 6.5 | `pipeline/stage65_postprocess.py` | Convert Markdown docs to DOCX |
| 6.7 | `pipeline/stage67_diagrams.py` | Generate Mermaid diagrams and inject them into docs |
| 7 | `pipeline/stage70_pdf.py` | Convert DOCX to PDF and build delivery bundle |
| 8 | `pipeline/stage80_tests.py` | Generate Gherkin, Playwright, and pytest artifacts |
| 9 | `pipeline/stage90_knowledge_graph.py` | Build final business-level knowledge graph |

## Requirements

This repo does not currently ship with a root `README`, `pyproject.toml`, `requirements.txt`, or `Makefile`, so environment setup is partly inferred from the code.

Minimum practical toolchain:

- Python 3 with `venv`
- Composer and PHP if you want to analyze PHP projects
- Node.js if you want DOCX output
- An LLM provider or local LLM server for the LLM-powered stages

Optional tools:

- LibreOffice for PDF conversion
- `docx2pdf` as a PDF fallback on macOS/Windows
- Neo4j for graph querying against a database instead of the offline JSON fallback

### Python packages used by the pipeline

The code directly references or conditionally imports these non-stdlib Python packages:

- `networkx`
- `matplotlib` for graph PNG rendering
- `chromadb`
- `sentence-transformers`
- `anthropic`
- `google-genai`
- `neo4j`
- `docx2pdf`
- `pytest`

Not every package is required for every stage.

### PHP requirements

PHP-specific analysis requires:

- A PHP binary available on `PATH`, or `PHPBA_PHP_BIN`
- `nikic/php-parser`
- `parsers/parse_project.php`

The repo already includes:

- `composer.json`
- `parsers/composer.json`
- a checked-in `vendor/` directory

If PHP parsing is not working, the validation stage suggests:

```bash
composer require nikic/php-parser
```

### Node.js requirements

Stage 6.5 (`pipeline/stage65_postprocess.py`) requires:

- Node.js >= 14
- the `docx` npm package

If `docx` is missing, the stage installs it automatically into the run output directory. Real sample runs in `outputs/` show generated `package.json`, `package-lock.json`, and `node_modules/` directories inside run folders.

### PDF conversion requirements

Stage 7 supports:

- LibreOffice headless conversion
- `docx2pdf` as a fallback

If neither is available, PDF generation fails and the stage will instruct you to install one of them.

### LLM configuration

LLM provider selection is handled in `pipeline/llm_client.py`.

Supported modes:

- Anthropic / Claude
- Google Gemini
- Local model server

Provider resolution order:

1. `LLM_PROVIDER`
2. auto-detect from configured environment variables

Useful environment variables include:

- `LLM_PROVIDER=claude|gemini|local`
- `LLM_MODEL`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`
- `LOCAL_LLM_BACKEND=ollama|vllm|lmstudio|llamacpp`
- `LOCAL_LLM_URL`
- `LOCAL_LLM_MODEL`

The repo also contains project-local backend config examples under:

- `.llm/ollama.env`
- `.llm/vllm.env`

Those files are loaded only if the matching backend is selected.

## Setup

Because the repo is not packaged as an installable Python project, the safest path is a manual virtual environment plus explicit dependency installs.

Example bootstrap:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install networkx matplotlib chromadb sentence-transformers anthropic google-genai neo4j docx2pdf pytest
```

For PHP project support:

```bash
composer install
```

For local LLM via Ollama:

```bash
ollama serve
ollama pull qwen2.5-coder:14b
export LLM_PROVIDER=local
export LOCAL_LLM_BACKEND=ollama
export LOCAL_LLM_MODEL=qwen2.5-coder:14b
```

For Gemini:

```bash
export LLM_PROVIDER=gemini
export GEMINI_API_KEY=your-key
```

For Claude:

```bash
export LLM_PROVIDER=claude
export ANTHROPIC_API_KEY=your-key
```

## Running the pipeline

### Full run

```bash
python run_pipeline.py --project /path/to/target-project
```

### Resume a previous run

```bash
python run_pipeline.py --resume outputs/<project>/run_<timestamp>/context.json
```

### Stop after a specific stage

```bash
python run_pipeline.py --project /path/to/target-project --until stage40_domain
```

### Force a stage to re-run

```bash
python run_pipeline.py --resume outputs/<project>/run_<timestamp>/context.json --force stage50_ac
```

### Use a different output base directory

```bash
python run_pipeline.py --project /path/to/target-project --output-dir my_outputs
```

Output directories are created as:

```text
<output-dir>/<project-slug>/run_<UTC timestamp>/
```

## Standalone scripts

The `scripts/` directory contains partial runners for individual stages and utilities.

Examples:

```bash
python scripts/run_stage0.py /path/to/target-project
python scripts/run_stage7.py outputs/<project>/run_<timestamp>/context.json
python scripts/run_query.py --resume outputs/<project>/run_<timestamp>/context.json
```

Not every stage has a wrapper script, but several common stages do.

## Querying the final knowledge graph

After Stage 9, you can query the generated knowledge graph in plain English.

Interactive mode:

```bash
python scripts/run_query.py --resume outputs/<project>/run_<timestamp>/context.json
```

Single question:

```bash
python scripts/run_query.py \
  --resume outputs/<project>/run_<timestamp>/context.json \
  --ask "Which features depend on login?"
```

Schema summary:

```bash
python scripts/run_query.py --resume outputs/<project>/run_<timestamp>/context.json --schema
```

The query engine works in two modes:

- Neo4j mode when `NEO4J_URI` is configured
- Offline JSON mode otherwise

Useful Neo4j environment variables:

- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE`

## Diagrams

Stage 6.7 generates Mermaid diagrams without additional LLM calls.

Generated diagram families:

- sequence diagrams for business flows
- process flow diagrams per bounded context
- architecture diagram for the analyzed system

The stage also injects Mermaid blocks back into the generated Markdown docs.

If you want to render `.mmd` files to images outside the pipeline, the stage documentation suggests:

```bash
npm install -g @mermaid-js/mermaid-cli
mmdc -i diagrams/architecture.mmd -o diagrams/architecture.png
```

## Generated documents and deliverables

The pipeline aims to produce these document layers:

### Markdown

- `5_documents/brd.md`
- `5_documents/srs.md`
- `5_documents/ac.md`
- `5_documents/user_stories.md`
- `6_qa/qa_report.md`

### DOCX

- `6.5_formatted/brd.docx`
- `6.5_formatted/srs.docx`
- `6.5_formatted/ac.docx`
- `6.5_formatted/user_stories.docx`
- `6.5_formatted/qa_report.docx`

### PDF delivery bundle

- `7_delivery/brd.pdf`
- `7_delivery/srs.pdf`
- `7_delivery/ac.pdf`
- `7_delivery/user_stories.pdf`
- `7_delivery/qa_report.pdf`
- `7_delivery/delivery/README.txt`

## Testing

The visible test suite in this repo is small.

Current test file:

- `tests/test_stage1_parse.py`

Documented command:

```bash
pytest tests/test_stage1_parse.py -v
```

Caveats:

- The test coverage is narrow relative to the size of the pipeline.
- The test file appears out of sync with the current parser layout and currently references `sys` without importing it.
- Treat the tests as partial validation, not full regression coverage.

## Known rough edges

This repository is functional, but it has a few contributor-facing rough edges worth knowing before you rely on it heavily:

- No root packaging metadata or lockfile for the Python environment
- Checked-in machine-local state such as `.venv/`, `vendor/`, and generated `outputs/`
- Mixed runtime stack: Python, PHP, Node.js, optional Neo4j, optional LibreOffice, and external or local LLM backends
- Some docstrings still use older "PHP-BA Agent" naming even though the parser dispatch is now multi-language
- Several setup details are encoded in stage docstrings and scripts rather than centralized configuration

## Suggested first-run path

If you are onboarding to this repo, the lowest-friction way to get confidence is:

1. Create a Python virtual environment and install the Python dependencies you need.
2. Make sure your LLM provider is configured.
3. If you are analyzing a PHP codebase, verify PHP and Composer support.
4. Run only through the static and early semantic stages first:

```bash
python run_pipeline.py --project /path/to/target-project --until stage30_embed
```

5. If that succeeds, resume into the LLM-backed stages:

```bash
python run_pipeline.py --resume outputs/<project>/run_<timestamp>/context.json
```

This catches parser, graph, and embedding issues before you spend time on long document-generation stages.

## Notes for contributors

- `context.py` is the source of truth for output-path conventions and serialized pipeline state.
- `run_pipeline.py` is the source of truth for stage ordering.
- `pipeline/stage10_parse.py` dispatches to the language-specific parsers.
- `pipeline/llm_client.py` is the source of truth for provider configuration.
- Sample runs in `outputs/` are useful for understanding what each stage writes.

If you are changing pipeline behavior, it is worth checking both the stage implementation and the matching output-path mapping in `context.py`.

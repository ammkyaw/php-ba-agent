"""
context.py — Shared Pipeline State for PHP-BA Agent
All stages read from and write to a PipelineContext instance.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class StageStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    SKIPPED   = "skipped"
    FAILED    = "failed"


class Framework(str, Enum):
    LARAVEL     = "laravel"
    SYMFONY     = "symfony"
    CODEIGNITER = "codeigniter"
    WORDPRESS   = "wordpress"
    RAW_PHP     = "raw_php"
    UNKNOWN     = "unknown"


@dataclass
class StageResult:
    """Tracks the outcome of a single pipeline stage."""
    status:       StageStatus   = StageStatus.PENDING
    started_at:   Optional[str] = None
    completed_at: Optional[str] = None
    error:        Optional[str] = None
    output_path:  Optional[str] = None

    def mark_running(self) -> None:
        self.status     = StageStatus.RUNNING
        self.started_at = datetime.utcnow().isoformat()

    def mark_completed(self, output_path: Optional[str] = None) -> None:
        self.status       = StageStatus.COMPLETED
        self.completed_at = datetime.utcnow().isoformat()
        if output_path:
            self.output_path = output_path

    def mark_failed(self, error: str) -> None:
        self.status       = StageStatus.FAILED
        self.completed_at = datetime.utcnow().isoformat()
        self.error        = error

    def mark_skipped(self) -> None:
        self.status = StageStatus.SKIPPED


@dataclass
class CodeMap:
    """
    Full extracted representation of a PHP codebase produced by Stage 1.

    Original fields (OOP structure)
    --------------------------------
    framework       — detected framework enum
    php_version     — e.g. "8.1"
    classes         — all class definitions (raw, non-categorised)
    routes          — HTTP route registrations (Laravel/Symfony/raw)
    models          — OOP model / entity classes
    controllers     — OOP controller classes
    services        — service, repository, and other OOP classes
    db_schema       — migration / CREATE TABLE definitions with column info
    config_files    — list of config file paths found
    total_files     — total PHP files scanned
    total_lines     — total lines of PHP code

    Extended fields (added by Stage 1 enhancements)
    ------------------------------------------------
    html_pages      — PHP files that mix HTML output (entry points)
    functions       — standalone / global function definitions
    includes        — include/require relationships between files
    sql_queries     — raw SQL operations with caller, table, operation
    redirects       — header("Location:...") calls with source + target
    superglobals    — $_POST/$_GET/$_SESSION/$_COOKIE key accesses
    call_graph      — function-to-function call edges
    form_fields     — HTML <form> elements with action, method, fields
    service_deps    — constructor DI / Facade usage (class -> dep_class)
    env_vars        — getenv() / $_ENV / config() references
    auth_signals    — session checks, middleware, role guards detected
    http_endpoints  — inferred HTTP entry points with method + handler
    table_columns   — per-table column definitions from migrations
    globals         — global variable declarations
    execution_paths — static-analysis execution paths (from Stage 1.5)
    """
    # -- Original fields ------------------------------------------------------
    framework:    Framework            = Framework.UNKNOWN
    php_version:  Optional[str]        = None
    classes:      list[dict[str, Any]] = field(default_factory=list)
    routes:       list[dict[str, Any]] = field(default_factory=list)
    models:       list[dict[str, Any]] = field(default_factory=list)
    controllers:  list[dict[str, Any]] = field(default_factory=list)
    services:     list[dict[str, Any]] = field(default_factory=list)
    db_schema:    list[dict[str, Any]] = field(default_factory=list)
    config_files: list[str]            = field(default_factory=list)
    total_files:  int = 0
    total_lines:  int = 0

    # -- Extended fields -------------------------------------------------------
    html_pages:      list[str]            = field(default_factory=list)
    functions:       list[dict[str, Any]] = field(default_factory=list)
    includes:        list[dict[str, Any]] = field(default_factory=list)
    sql_queries:     list[dict[str, Any]] = field(default_factory=list)
    redirects:       list[dict[str, Any]] = field(default_factory=list)
    superglobals:    list[dict[str, Any]] = field(default_factory=list)
    call_graph:      list[dict[str, Any]] = field(default_factory=list)
    form_fields:     list[dict[str, Any]] = field(default_factory=list)
    service_deps:    list[dict[str, Any]] = field(default_factory=list)
    env_vars:        list[dict[str, Any]] = field(default_factory=list)
    auth_signals:    list[dict[str, Any]] = field(default_factory=list)
    http_endpoints:  list[dict[str, Any]] = field(default_factory=list)
    table_columns:   list[dict[str, Any]] = field(default_factory=list)
    globals:         list[dict[str, Any]] = field(default_factory=list)
    execution_paths: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class GraphMeta:
    """Lightweight summary of the knowledge graph produced by Stage 2."""
    graph_path: str       = ""
    node_count: int       = 0
    edge_count: int       = 0
    node_types: list[str] = field(default_factory=list)
    edge_types: list[str] = field(default_factory=list)


@dataclass
class EmbeddingMeta:
    """Metadata about the ChromaDB vector index produced by Stage 3."""
    collection_name: str = ""
    chroma_path:     str = ""
    total_chunks:    int = 0
    embedding_model: str = "text-embedding-3-small"


@dataclass
class PreflightResult:
    """Results of Stage 3.5 context pre-flight checks."""
    passed:   bool      = True
    warnings: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)


@dataclass
class DomainModel:
    """Structured domain model produced by Stage 4 DomainAnalystAgent."""
    domain_name:      str                  = ""
    description:      str                  = ""
    user_roles:       list[dict[str, str]] = field(default_factory=list)
    features:         list[dict[str, Any]] = field(default_factory=list)
    workflows:        list[dict[str, Any]] = field(default_factory=list)
    bounded_contexts: list[str]            = field(default_factory=list)
    key_entities:     list[str]            = field(default_factory=list)


@dataclass
class FlowStep:
    """
    One step inside a BusinessFlow — corresponds to a single page visit,
    form submission, API call, or database operation.
    """
    step_num:      int            # 1-based sequence number
    page:          str            # PHP file or route path (e.g. "login.php")
    action:        str            # Human-readable action (e.g. "Submit credentials")
    http_method:   Optional[str]  # GET / POST / PUT / DELETE / None
    db_ops:        list[str]      # e.g. ["SELECT users", "INSERT sessions"]
    auth_required: bool           # True if this step has an auth guard
    inputs:        list[str]      # Form / query fields consumed
    outputs:       list[str]      # Session keys / redirect targets produced
    is_branch:     bool = False   # True for alternate/error path steps


@dataclass
class BusinessFlow:
    """
    A complete end-to-end user journey extracted from the codebase.
    Produced by Stage 4.5. Consumed by Stage 5 AC and UserStory agents.
    """
    flow_id:           str                   # e.g. "flow_001"
    name:              str                   # e.g. "Customer Rental Booking"
    actor:             str                   # e.g. "Authenticated Customer"
    bounded_context:   str                   # e.g. "Booking"
    trigger:           str                   # What initiates the flow
    steps:             list[FlowStep]        # Ordered happy-path steps
    branches:          list[dict[str, Any]]  # Alternate / error paths
    termination:       str                   # Success end-state description
    evidence_files:    list[str]             # Source PHP files that back this flow
    confidence:        float = 1.0           # 0.0-1.0
    replaces_workflow: Optional[str] = None  # domain_model.workflow it supersedes


@dataclass
class BusinessFlowCollection:
    """Container written to business_flows.json by Stage 4.5."""
    flows:        list[BusinessFlow]   = field(default_factory=list)
    total:        int                  = 0
    by_context:   dict[str, list[str]] = field(default_factory=dict)
    by_actor:     dict[str, list[str]] = field(default_factory=dict)
    generated_at: str                  = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class BAArtifacts:
    """Paths to the four BA document artefacts produced by Stage 5."""
    brd_path:          Optional[str] = None
    srs_path:          Optional[str] = None
    ac_path:           Optional[str] = None
    user_stories_path: Optional[str] = None


@dataclass
class QAResult:
    """QA review result produced by Stage 6."""
    passed:            bool                 = True
    issues:            list[dict[str, str]] = field(default_factory=list)
    coverage_score:    float                = 0.0
    consistency_score: float                = 0.0


@dataclass
class TestSuiteArtifacts:
    """Paths to the test artefacts produced by Stage 8."""
    gherkin_path:    Optional[str] = None   # tests.feature
    playwright_path: Optional[str] = None   # playwright_tests.js
    pytest_path:     Optional[str] = None   # pytest_tests.py
    scenario_count:  int           = 0      # total Gherkin scenarios generated


@dataclass
class KnowledgeGraphMeta:
    """
    Summary of the system knowledge graph produced by Stage 9.

    The full graph lives in knowledge_graph.json; this dataclass carries only
    the counts and type lists needed for logging and downstream reference.
    """
    json_path:        str            = ""
    node_count:       int            = 0
    edge_count:       int            = 0
    node_types:       list[str]      = field(default_factory=list)
    edge_type_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class ArchitectureMeta:
    """
    Lightweight summary of the architecture reconstruction produced by Stage 6.2.

    The full structured data lives in architecture.json; this dataclass carries
    only the paths and key counts needed by downstream stages (6.5 postprocess,
    6.7 diagrams, 7 pdf).
    """
    json_path:        str       = ""   # absolute path to architecture.json
    md_path:          str       = ""   # absolute path to architecture.md
    component_count:  int       = 0    # number of components identified
    data_flow_count:  int       = 0    # number of data flows identified
    sequence_count:   int       = 0    # number of sequence flows identified
    tech_stack:       list[str] = field(default_factory=list)  # e.g. ["PHP 8.1", "MySQL 5.7"]


# Maps output filenames to their stage subdirectory inside the run output dir.
# output_path() uses this to keep pipeline outputs organised by stage.
_STAGE_SUBDIRS: dict[str, str] = {
    # Stage 0 — Validation
    "validation_report.json":      "0_validation",
    # Stage 1 — PHP Parsing
    "code_map.json":               "1_parse",
    # Stage 1.5 — Execution Paths
    "execution_paths.json":        "1.5_paths",
    "execution_paths_errors.json": "1.5_paths",
    # Stage 2 — Knowledge Graph
    "code_graph.gpickle":          "2_graph",
    "code_graph.json":             "2_graph",
    "code_graph.png":              "2_graph",
    # Stage 3 — Vector Embeddings
    "chromadb":                    "3_embed",
    "chunks_manifest.json":        "3_embed",
    # Stage 3.5 — Preflight
    "preflight_report.json":       "3.5_preflight",
    # Stage 4 — Domain Model
    "domain_model.json":           "4_domain",
    "coverage_report.json":        "4_domain",
    # Stage 4.5 — Business Flows
    "business_flows.json":         "4.5_flows",
    # Stage 5 — BA Documents (Markdown)
    "brd.md":                      "5_documents",
    "srs.md":                      "5_documents",
    "ac.md":                       "5_documents",
    "user_stories.md":             "5_documents",
    # Stage 6 — QA Review
    "qa_report.md":                "6_qa",
    "qa_result.json":              "6_qa",
    # Stage 6.2 — Architecture
    "architecture.json":           "6.2_architecture",
    "architecture.md":             "6.2_architecture",
    # Stage 6.5 — Formatted Docs (DOCX)
    "brd.docx":                    "6.5_formatted",
    "srs.docx":                    "6.5_formatted",
    "ac.docx":                     "6.5_formatted",
    "user_stories.docx":           "6.5_formatted",
    "qa_report.docx":              "6.5_formatted",
    "pipeline_summary.md":         "6.5_formatted",
    # Stage 6.7 — Diagrams
    "diagrams":                    "6.7_diagrams",
    # Stage 7 — PDF Delivery
    "brd.pdf":                     "7_delivery",
    "srs.pdf":                     "7_delivery",
    "ac.pdf":                      "7_delivery",
    "user_stories.pdf":            "7_delivery",
    "qa_report.pdf":               "7_delivery",
    "delivery":                    "7_delivery",
    # Stage 8 — Test Cases
    "tests.feature":               "8_tests",
    "playwright_tests.js":         "8_tests",
    "pytest_tests.py":             "8_tests",
    # Stage 9 — System Knowledge Graph
    "knowledge_graph.json":        "9_knowledge_graph",
}


@dataclass
class PipelineContext:
    """
    Central state object passed through every stage of the PHP-BA Agent pipeline.

    Usage:
        ctx = PipelineContext.create(php_project_path="/path/to/project")
        ctx = PipelineContext.load("outputs/run_xyz/context.json")  # resume
    """
    run_id:           str = ""
    php_project_path: str = ""
    output_dir:       str = ""
    created_at:       str = field(default_factory=lambda: datetime.utcnow().isoformat())

    stages: dict[str, StageResult] = field(default_factory=lambda: {
        "stage0_validate":      StageResult(),
        "stage1_parse":         StageResult(),
        "stage15_paths":        StageResult(),
        "stage2_graph":         StageResult(),
        "stage3_embed":         StageResult(),
        "stage35_preflight":    StageResult(),
        "stage4_domain":        StageResult(),
        "stage45_flows":        StageResult(),
        "stage5_brd":           StageResult(),
        "stage5_srs":           StageResult(),
        "stage5_ac":            StageResult(),
        "stage5_userstories":   StageResult(),
        "stage6_qa":            StageResult(),
        "stage62_architecture": StageResult(),  # architecture reconstruction (LLM)
        "stage65_postprocess":  StageResult(),
        "stage67_diagrams":     StageResult(),
        "stage7_pdf":           StageResult(),
        "stage8_tests":         StageResult(),  # test case generator
        "stage9_knowledge_graph": StageResult(),  # system knowledge graph builder
    })

    code_map:          Optional[CodeMap]                = None
    graph_meta:        Optional[GraphMeta]              = None
    embedding_meta:    Optional[EmbeddingMeta]          = None
    preflight:         Optional[PreflightResult]        = None
    domain_model:      Optional[DomainModel]            = None
    business_flows:    Optional[BusinessFlowCollection] = None
    ba_artifacts:      Optional[BAArtifacts]            = None
    qa_result:         Optional[QAResult]               = None
    architecture_meta: Optional[ArchitectureMeta]       = None  # Stage 6.2
    test_suite:            Optional[TestSuiteArtifacts]     = None  # Stage 8
    knowledge_graph_meta: Optional[KnowledgeGraphMeta]     = None  # Stage 9

    @classmethod
    def create(cls, php_project_path: str, output_base: str = "outputs") -> "PipelineContext":
        run_id     = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_base, f"run_{run_id}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        ctx = cls(
            run_id           = run_id,
            php_project_path = str(Path(php_project_path).resolve()),
            output_dir       = output_dir,
        )
        ctx.save()
        return ctx

    @property
    def context_file(self) -> str:
        return os.path.join(self.output_dir, "context.json")

    def save(self) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        with open(self.context_file, "w", encoding="utf-8") as f:
            json.dump(self._to_dict(), f, indent=2)

    @classmethod
    def load(cls, context_file: str) -> "PipelineContext":
        with open(context_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls._from_dict(data)

    def stage(self, name: str) -> StageResult:
        if name not in self.stages:
            raise KeyError(
                f"Unknown stage '{name}'. Valid: {list(self.stages.keys())}"
            )
        return self.stages[name]

    def is_stage_done(self, name: str) -> bool:
        return self.stage(name).status == StageStatus.COMPLETED

    def output_path(self, filename: str) -> str:
        stage_subdir = _STAGE_SUBDIRS.get(filename)
        if stage_subdir:
            subdir = os.path.join(self.output_dir, stage_subdir)
            Path(subdir).mkdir(parents=True, exist_ok=True)
            return os.path.join(subdir, filename)
        return os.path.join(self.output_dir, filename)

    def _to_dict(self) -> dict[str, Any]:
        import dataclasses

        def _serialize(obj: Any) -> Any:
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return {k: _serialize(v) for k, v in dataclasses.asdict(obj).items()}
            if isinstance(obj, dict): return {k: _serialize(v) for k, v in obj.items()}
            if isinstance(obj, list): return [_serialize(i) for i in obj]
            if isinstance(obj, Enum): return obj.value
            return obj

        return _serialize(self)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> "PipelineContext":
        ctx = cls(
            run_id           = data["run_id"],
            php_project_path = data["php_project_path"],
            output_dir       = data["output_dir"],
            created_at       = data["created_at"],
        )

        # ── stages ────────────────────────────────────────────────────────────
        for name, sr in data.get("stages", {}).items():
            if name in ctx.stages:
                ctx.stages[name] = StageResult(
                    status       = StageStatus(sr["status"]),
                    started_at   = sr.get("started_at"),
                    completed_at = sr.get("completed_at"),
                    error        = sr.get("error"),
                    output_path  = sr.get("output_path"),
                )

        # ── optional rich objects ─────────────────────────────────────────────
        if data.get("code_map") is not None:
            d = data["code_map"]
            ctx.code_map = CodeMap(
                framework       = Framework(d.get("framework", Framework.UNKNOWN.value)),
                php_version     = d.get("php_version"),
                classes         = d.get("classes", []),
                routes          = d.get("routes", []),
                models          = d.get("models", []),
                controllers     = d.get("controllers", []),
                services        = d.get("services", []),
                db_schema       = d.get("db_schema", []),
                config_files    = d.get("config_files", []),
                total_files     = d.get("total_files", 0),
                total_lines     = d.get("total_lines", 0),
                html_pages      = d.get("html_pages", []),
                functions       = d.get("functions", []),
                includes        = d.get("includes", []),
                sql_queries     = d.get("sql_queries", []),
                redirects       = d.get("redirects", []),
                superglobals    = d.get("superglobals", []),
                call_graph      = d.get("call_graph", []),
                form_fields     = d.get("form_fields", []),
                service_deps    = d.get("service_deps", []),
                env_vars        = d.get("env_vars", []),
                auth_signals    = d.get("auth_signals", []),
                http_endpoints  = d.get("http_endpoints", []),
                table_columns   = d.get("table_columns", []),
                globals         = d.get("globals", []),
                execution_paths = d.get("execution_paths", []),
            )

        if data.get("graph_meta") is not None:
            d = data["graph_meta"]
            ctx.graph_meta = GraphMeta(
                graph_path = d.get("graph_path", ""),
                node_count = d.get("node_count", 0),
                edge_count = d.get("edge_count", 0),
                node_types = d.get("node_types", []),
                edge_types = d.get("edge_types", []),
            )

        if data.get("embedding_meta") is not None:
            d = data["embedding_meta"]
            ctx.embedding_meta = EmbeddingMeta(
                collection_name = d.get("collection_name", ""),
                chroma_path     = d.get("chroma_path", ""),
                total_chunks    = d.get("total_chunks", 0),
                embedding_model = d.get("embedding_model", "text-embedding-3-small"),
            )

        if data.get("preflight") is not None:
            d = data["preflight"]
            ctx.preflight = PreflightResult(
                passed   = d.get("passed", True),
                warnings = d.get("warnings", []),
                blockers = d.get("blockers", []),
            )

        if data.get("domain_model") is not None:
            d = data["domain_model"]
            ctx.domain_model = DomainModel(
                domain_name      = d.get("domain_name", ""),
                description      = d.get("description", ""),
                user_roles       = d.get("user_roles", []),
                features         = d.get("features", []),
                workflows        = d.get("workflows", []),
                bounded_contexts = d.get("bounded_contexts", []),
                key_entities     = d.get("key_entities", []),
            )

        if data.get("business_flows") is not None:
            d = data["business_flows"]
            flows = [
                BusinessFlow(
                    flow_id           = f["flow_id"],
                    name              = f["name"],
                    actor             = f["actor"],
                    bounded_context   = f["bounded_context"],
                    trigger           = f["trigger"],
                    steps             = [
                        FlowStep(
                            step_num      = s["step_num"],
                            page          = s["page"],
                            action        = s["action"],
                            http_method   = s.get("http_method"),
                            db_ops        = s.get("db_ops", []),
                            auth_required = s.get("auth_required", False),
                            inputs        = s.get("inputs", []),
                            outputs       = s.get("outputs", []),
                            is_branch     = s.get("is_branch", False),
                        )
                        for s in f.get("steps", [])
                    ],
                    branches          = f.get("branches", []),
                    termination       = f.get("termination", ""),
                    evidence_files    = f.get("evidence_files", []),
                    confidence        = f.get("confidence", 1.0),
                    replaces_workflow = f.get("replaces_workflow"),
                )
                for f in d.get("flows", [])
            ]
            ctx.business_flows = BusinessFlowCollection(
                flows        = flows,
                total        = d.get("total", len(flows)),
                by_context   = d.get("by_context", {}),
                by_actor     = d.get("by_actor", {}),
                generated_at = d.get("generated_at", ""),
            )

        if data.get("ba_artifacts") is not None:
            d = data["ba_artifacts"]
            ctx.ba_artifacts = BAArtifacts(
                brd_path          = d.get("brd_path"),
                srs_path          = d.get("srs_path"),
                ac_path           = d.get("ac_path"),
                user_stories_path = d.get("user_stories_path"),
            )

        if data.get("qa_result") is not None:
            d = data["qa_result"]
            ctx.qa_result = QAResult(
                passed            = d.get("passed", True),
                issues            = d.get("issues", []),
                coverage_score    = d.get("coverage_score", 0.0),
                consistency_score = d.get("consistency_score", 0.0),
            )

        if data.get("architecture_meta") is not None:
            d = data["architecture_meta"]
            ctx.architecture_meta = ArchitectureMeta(
                json_path       = d.get("json_path", ""),
                md_path         = d.get("md_path", ""),
                component_count = d.get("component_count", 0),
                data_flow_count = d.get("data_flow_count", 0),
                sequence_count  = d.get("sequence_count", 0),
                tech_stack      = d.get("tech_stack", []),
            )

        if data.get("test_suite") is not None:
            d = data["test_suite"]
            ctx.test_suite = TestSuiteArtifacts(
                gherkin_path    = d.get("gherkin_path"),
                playwright_path = d.get("playwright_path"),
                pytest_path     = d.get("pytest_path"),
                scenario_count  = d.get("scenario_count", 0),
            )

        if data.get("knowledge_graph_meta") is not None:
            d = data["knowledge_graph_meta"]
            ctx.knowledge_graph_meta = KnowledgeGraphMeta(
                json_path        = d.get("json_path", ""),
                node_count       = d.get("node_count", 0),
                edge_count       = d.get("edge_count", 0),
                node_types       = d.get("node_types", []),
                edge_type_counts = d.get("edge_type_counts", {}),
            )

        return ctx

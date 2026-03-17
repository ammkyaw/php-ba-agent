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
import re


def _project_slug(php_project_path: str) -> str:
    """
    Derive a filesystem-safe folder name from a project path.

    Examples
    --------
    /home/user/SugarCRM-hotfix   →  SugarCRM-hotfix
    /projects/my project/        →  my-project
    /var/www/html                →  html
    """
    name = Path(php_project_path).resolve().name or "project"
    # Replace whitespace and non-alphanumeric chars (except - and _) with -
    slug = re.sub(r"[^\w\-]", "-", name)
    # Collapse multiple consecutive dashes
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or "project"


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
    """Results of Stage 3.9 context pre-flight checks."""
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
    flow_type:         str   = "http"        # "http"|"scheduled"|"cli"|"webhook"|"queue_worker"
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
class EntryPoint:
    """
    One non-HTTP system entry point discovered by Stage 1.3.

    Types
    -----
    http         — Standard browser-facing page; also covers webhook receivers
                   that arrive via HTTP POST from an external system
    scheduled    — Cron job or scheduled console command
    cli          — Admin / maintenance CLI script (artisan command, bin/*.php)
    webhook      — Incoming HTTP callback from an external system (Stripe, etc.)
    queue_worker — Async queue / message-queue job handler
    """
    ep_id:        str   = ""       # e.g. "ep_001"
    ep_type:      str   = "http"   # "http"|"scheduled"|"cli"|"webhook"|"queue_worker"
    handler_file: str   = ""       # path relative to php_project_path
    name:         str   = ""       # human-readable, e.g. "Send Nightly Digest"
    schedule:     str   = ""       # cron expression or "" (non-scheduled types)
    trigger:      str   = ""       # e.g. "POST /webhook/stripe" | "php artisan cmd:name"
    confidence:   float = 0.8      # 1.0=config-backed, 0.8=pattern, 0.6=heuristic


@dataclass
class EntryPointCatalog:
    """
    Complete catalog of all system entry points produced by Stage 1.3.

    Consumed by:
      Stage 1.5  — seeds execution-path analysis from non-HTTP handlers
      Stage 4.5  — tags BusinessFlow.flow_type from source file → entry point type
      Stage 5    — injects background operation descriptions into BRD
    """
    entry_points: list[EntryPoint]     = field(default_factory=list)
    by_type:      dict[str, list[str]] = field(default_factory=dict)   # ep_type → [ep_id]
    total:        int                  = 0
    generated_at: str                  = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class CriticPass:
    """
    Result of one Critic-loop pass on a Stage 5 BA document.

    Produced by Stage 5.0 CriticAgent.  One entry per document per turn.
    Stage 6 QA can inspect these to see what the critic already addressed.
    """
    doc_type:              str   = ""     # "brd" | "srs" | "ac" | "us"
    turn:                  int   = 1      # 1 = after first draft, 2 = after refinement
    score:                 float = 0.0   # 0.0–1.0 (≥ CRITIC_THRESHOLD → passed)
    passed:                bool  = False
    uncovered_rule_ids:    list  = field(default_factory=list)   # list[str]
    hallucinated_entities: list  = field(default_factory=list)   # list[str]
    structural_issues:     list  = field(default_factory=list)   # list[str]
    rewrite_hints:         list  = field(default_factory=list)   # list[str]
    generated_at:          str   = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class BAArtifacts:
    """Paths to the four BA document artefacts produced by Stage 5."""
    brd_path:          Optional[str] = None
    srs_path:          Optional[str] = None
    ac_path:           Optional[str] = None
    user_stories_path: Optional[str] = None
    # Critic-loop results: doc_key → list of CriticPass dicts (one per turn attempted)
    critic_passes:     dict          = field(default_factory=dict)


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
class BusinessRule:
    """
    One extracted business invariant from Stage 2.9.

    Populated entirely by static analysis — no LLM.
    """
    rule_id:         str            # e.g. "rule_001"
    category:        str            # VALIDATION | AUTHORIZATION | STATE_TRANSITION |
                                    # BUSINESS_LIMIT | TEMPORAL | REFERENTIAL
    description:     str            # Human-readable, e.g. "Password must be ≥ 8 chars"
    raw_expression:  str            # Code snippet, e.g. "strlen($password) < 8"
    entity:          str            # Linked field, e.g. "User.password"
    bounded_context: str            # Cluster/module name from Stage 2.8
    source_files:    list[str]      # PHP files where this rule was found
    confidence:      float = 0.8    # 1.0=schema, 0.8=guard-clause, 0.6=source-scan
    tables:          list[str] = field(default_factory=list)
    plain_english:   str   = ""    # BA-ready phrasing, populated at end of Stage 2.9


@dataclass
class InvariantCollection:
    """Container written to rule_catalog.json by Stage 2.9."""
    rules:        list[BusinessRule]   = field(default_factory=list)
    total:        int                  = 0
    by_category:  dict[str, list[str]] = field(default_factory=dict)
    by_context:   dict[str, list[str]] = field(default_factory=dict)
    generated_at: str                  = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class StateTransition:
    """
    One detected state transition for a state-machine entity field.

    Produced by Stage 3.7 — fully static, no LLM.
    """
    from_state:   str            # e.g. "Draft"
    to_state:     str            # e.g. "Submitted"
    trigger:      str            # PHP file stem or operation label, e.g. "save_case"
    guard:        str            # guard condition if known, else ""
    source_files: list[str]
    confidence:   float = 0.75   # 0.9=SQL SET+WHERE, 0.75=branch+action, 0.5=proximity


@dataclass
class StateMachine:
    """
    State machine for one (table, field) pair, produced by Stage 3.7.

    Consumed by Stage 4.5 (flow validation), Stage 6.7 (diagrams),
    Stage 8 (test generation).
    """
    machine_id:      str           # e.g. "sm_001"
    entity:          str           # human name, e.g. "Email"
    table:           str           # e.g. "emails"
    field:           str           # e.g. "status"
    bounded_context: str           # cluster name from Stage 2.8
    states:          list[str]     # all distinct states detected
    initial_states:  list[str]     # states with no incoming transition
    terminal_states: list[str]     # states with no outgoing transition
    dead_states:     list[str]     # unreachable states
    transitions:     list[StateTransition]
    mermaid:         str           # stateDiagram-v2 source


@dataclass
class StateMachineCollection:
    """Container written to state_machine_catalog.json by Stage 3.7."""
    machines:     list[StateMachine] = field(default_factory=list)
    total:        int                = 0
    generated_at: str                = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ActionTag:
    """
    Semantic role assigned to one HTTP endpoint / controller action by Stage 2.7.
    """
    symbol:      str          # "OrderController::store" or "create_order"
    file:        str
    role:        str          # BUSINESS_ACTION | AUTH_ACTION | CRUD_ACTION | INTEGRATION_ACTION | INFRASTRUCTURE_ACTION
    confidence:  float        # 0.0 – 1.0
    signals:     list[str]    = field(default_factory=list)  # human-readable evidence
    actor:       str          = ""  # "Admin" | "Authenticated User" | "Guest" | "API Client"
    entities:    list[str]    = field(default_factory=list)  # DB tables written/read
    http_method: str          = ""  # "POST" | "GET" | ""
    route_path:  str          = ""  # "/orders" | ""


@dataclass
class ExternalSystem:
    """
    An external system detected by Stage 2.7 (payment gateway, email, storage…).
    """
    name:         str                   # "Stripe"
    category:     str                   # PAYMENT | EMAIL | STORAGE | SMS_PUSH | AUTH_OAUTH | MONITORING | CRM | ERP | INFRA
    env_keys:     list[str]  = field(default_factory=list)   # ["STRIPE_KEY", "STRIPE_SECRET"]
    class_hints:  list[str]  = field(default_factory=list)   # ["StripeService", "StripeGateway"]
    dep_hints:    list[str]  = field(default_factory=list)   # ["Stripe\StripeClient"]
    detected_via: list[str]  = field(default_factory=list)   # ["env_var", "class_name", "service_dep"]


@dataclass
class SemanticRoleIndex:
    """
    Full semantic-role index produced by Stage 2.7.
    Written to 2.7_semanticroles/semantic_roles.json.
    Consumed by Stage 2.8 (clustering), Stage 4 (domain model), Stage 4.5 (flows).
    """
    actions:          list[ActionTag]      = field(default_factory=list)
    external_systems: list[ExternalSystem] = field(default_factory=list)
    role_summary:     dict[str, int]       = field(default_factory=dict)  # role → count
    infra_files:      list[str]            = field(default_factory=list)  # files whose dominant role is INFRASTRUCTURE
    business_files:   list[str]            = field(default_factory=list)  # files with ≥1 BUSINESS_ACTION tag
    generated_at:     str                  = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ActionCluster:
    """
    One action cluster produced by Stage 2.8 — a group of PHP files that share
    a common bounded context (module folder, DB tables, route prefix, redirects).
    """
    cluster_id:   str            # e.g. "cluster_001"
    name:         str            # e.g. "Accounts" or "CalendarProvider"
    files:        list[str]      # absolute paths of member files
    tables:       list[str]      # DB tables touched by member files
    route_prefix: str            # common route prefix, if any (e.g. "/accounts")
    module:       str            # SugarCRM/framework module name, or ""
    file_count:   int = 0        # len(files) convenience field


@dataclass
class ActionClusterCollection:
    """Container written to action_clusters.json by Stage 2.8."""
    clusters:     list[ActionCluster] = field(default_factory=list)
    total:        int                 = 0
    generated_at: str                 = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class EntityColumn:
    """One column of a DB table, produced by Stage 3.5."""
    name:              str
    data_type:         str  = ""
    nullable:          bool = True
    is_primary_key:    bool = False
    is_foreign_key:    bool = False
    references_table:  str  = ""
    references_column: str  = ""


@dataclass
class Entity:
    """
    One business entity extracted by Stage 3.5.

    Covers every table/model in the codebase.  Use is_core / is_system /
    is_pivot flags to filter for different consumers (ER diagram vs full graph).
    """
    entity_id:       str            # "ent_001"
    name:            str            # "Customer"  (humanized from table name)
    table:           str            # "customers"
    bounded_context: str            # cluster name from Stage 2.8
    columns:         list[EntityColumn]
    primary_key:     str            # e.g. "id"
    is_pivot:        bool = False   # junction table for N:M
    is_core:         bool = True    # has ≥1 relationship OR ≥1 state machine OR in action cluster
    is_system:       bool = False   # infra/log/config tables (sugar_config, log_messages…)
    source_files:    list[str] = field(default_factory=list)
    confidence:      float = 1.0


@dataclass
class EntityCollection:
    """Container written to entity_catalog.json by Stage 3.5."""
    entities:     list[Entity] = field(default_factory=list)
    total:        int          = 0
    core_count:   int          = 0   # entities where is_core=True
    generated_at: str          = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class EntityRelationship:
    """
    One detected relationship between two entities, produced by Stage 3.6.

    Multiple signals (FK, JOIN, ORM, column pattern) are merged into a single
    entry per (from_entity, to_entity) pair.
    """
    rel_id:       str            # "rel_001"
    from_entity:  str            # table name e.g. "customers"
    to_entity:    str            # table name e.g. "orders"
    cardinality:  str            # "1:1" | "1:N" | "N:M"
    rel_type:     str            # "has_many" | "has_one" | "belongs_to" | "many_to_many"
    via_column:   str            # FK column, e.g. "customer_id"
    via_table:    str            # pivot table name (N:M only), else ""
    confidence:   float
    signals:      list[str]      # ["foreign_key","column_pattern","sql_join","orm"]
    source_files: list[str]


@dataclass
class EntityRelationshipCollection:
    """Container written to relationship_catalog.json by Stage 3.6."""
    relationships: list[EntityRelationship] = field(default_factory=list)
    entity_names:  list[str]                = field(default_factory=list)
    total:         int                      = 0
    mermaid_path:  str                      = ""   # path to erDiagram .mmd file
    generated_at:  str                      = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class SpecRule:
    """
    One formal business rule produced by Stage 4.6 Specification Mining.

    Unlike Stage 2.9 BusinessRule (which stores raw code-centric invariants),
    SpecRule is BA-ready: Given/When/Then phrased in business language,
    cross-referenced to all evidence sources.
    """
    rule_id:           str            # "BR-001"
    category:          str            # VALIDATION | AUTHORIZATION | WORKFLOW |
                                      # STATE | REFERENTIAL | BUSINESS_LIMIT
    title:             str            # ≤10-word business-friendly name
    description:       str            # fuller prose
    given:             str            # "Given the user is registering…"
    when:              str            # "When they submit a password shorter than 8 chars"
    then:              str            # "Then the system rejects the input with an error"
    entities:          list[str]      # ["User"]
    bounded_context:   str            # "Users"
    source_invariants: list[str]      # rule_ids from Stage 2.9
    source_machines:   list[str]      # machine_ids from Stage 3.7
    source_flows:      list[str]      # flow_ids from Stage 4.5
    source_files:      list[str]
    confidence:        float          # 0.0–1.0
    tags:              list[str]      # ["password", "security", "validation"]
    pass_origin:       str            # "pass1_invariant" | "pass2_state" |
                                      # "pass3_flow"      | "pass4_referential"


@dataclass
class SpecRuleCollection:
    """Container written to spec_rules.json by Stage 4.6."""
    rules:        list[SpecRule]            = field(default_factory=list)
    total:        int                       = 0
    by_category:  dict[str, list[str]]      = field(default_factory=dict)
    by_entity:    dict[str, list[str]]      = field(default_factory=dict)
    by_flow:      dict[str, list[str]]      = field(default_factory=dict)
    by_context:   dict[str, list[str]]      = field(default_factory=dict)
    generated_at: str                       = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class TriangulatedRule:
    """
    Triangulation result for one SpecRule, produced by Stage 4.8.

    corroborating_types — independent evidence streams that agree with the rule
    contradicting_types — streams whose data actively conflicts with the rule
    triangulation_score — corroborating / max_applicable  (0.0–1.0)
    triangulation_status — STRONG (≥0.60) | MODERATE (≥0.33) | WEAK (<0.33)
    """
    rule_id:               str
    title:                 str
    category:              str
    corroborating_types:   list[str]   # subset of 6 evidence type labels
    contradicting_types:   list[str]
    triangulation_score:   float
    triangulation_status:  str         # "STRONG" | "MODERATE" | "WEAK"
    max_applicable:        int         # denominator (category-dependent)
    contradiction_notes:   list[str]   = field(default_factory=list)


@dataclass
class TriangulationReport:
    """Container written to evidence_triangulation.json by Stage 4.8."""
    rules:               list[TriangulatedRule] = field(default_factory=list)
    total:               int                    = 0
    strong_count:        int                    = 0
    moderate_count:      int                    = 0
    weak_count:          int                    = 0
    contradiction_count: int                    = 0
    weak_rule_ids:       list[str]              = field(default_factory=list)
    contradiction_ids:   list[str]              = field(default_factory=list)
    generated_at:        str                    = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class TraceableRequirement:
    """
    A single traceable requirement produced by Stage 5.5.

    Forward links  (code → req): populated from spec_rules source catalogs.
    Backward links (doc  → req): populated from [BR-XXX] citations or keyword overlap.
    """
    tr_id:                str        = ""        # e.g. "TR-BR-001"
    source_type:          str        = ""        # "spec_rule" | "business_flow" | "entity" | "state_machine"
    source_id:            str        = ""        # original catalog ID (rule_id / flow_id / …)
    title:                str        = ""
    category:             str        = ""
    bounded_context:      str        = ""
    # Forward links: code artifacts that provide evidence for this requirement
    code_artifacts:       list       = field(default_factory=list)   # list[dict]
    # Backward links: sections in generated docs that cite this requirement
    document_citations:   list       = field(default_factory=list)   # list[dict]
    covered_in_brd:       bool       = False
    covered_in_srs:       bool       = False
    covered_in_ac:        bool       = False
    covered_in_us:        bool       = False
    doc_coverage_score:   float      = 0.0       # fraction of docs that cover this req
    code_link_count:      int        = 0
    triangulation_status: str        = ""        # "STRONG" | "MODERATE" | "WEAK" | ""


@dataclass
class TraceabilityMeta:
    """
    Summary metadata for the Automated Traceability Matrix (Stage 5.5).

    The full matrix lives in traceability_matrix.json; the human report in
    traceability_report.md.  Downstream stages consume uncited_ids and
    coverage percentages.
    """
    matrix_path:        str        = ""
    report_path:        str        = ""
    total_requirements: int        = 0
    covered_brd:        float      = 0.0   # fraction of reqs cited in BRD
    covered_srs:        float      = 0.0
    covered_ac:         float      = 0.0
    covered_us:         float      = 0.0
    avg_code_links:     float      = 0.0
    uncited_count:      int        = 0
    uncited_ids:        list       = field(default_factory=list)   # list[str]
    generated_at:       str        = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class DocCoverageResult:
    """
    Document coverage audit produced by Stage 5.9.

    Records how many static-analysis signals (entities, flows, spec rules,
    state machines, relationships) made it into the generated BA documents.
    Consumed by Stage 6 to inject gap context into the QA review prompt.
    """
    dimensions:     list[dict]  = field(default_factory=list)   # DimCoverage as dicts
    overall_pct:    float       = 0.0
    overall_status: str         = "fail"                        # "pass" | "warn" | "fail"
    gap_summary:    list[str]   = field(default_factory=list)   # human-readable gaps
    generated_at:   str         = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class GraphRAGMeta:
    """
    Metadata for the Graph-Aware Context Index produced by Stage 3.8.

    The full index lives in graph_context_index.json; this dataclass carries
    only the path and node counts.  Downstream stages call ctx.graph_query()
    which lazily loads and caches the index from index_path.
    """
    index_path:    str = ""
    file_count:    int = 0
    entity_count:  int = 0
    cluster_count: int = 0
    generated_at:  str = field(default_factory=lambda: datetime.utcnow().isoformat())


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
    # Stage 1.3 — Entry-Point Catalog
    "entry_point_catalog.json":    "1.3_entrypoints",
    # Stage 1.5 — Execution Paths
    "execution_paths.json":        "1.5_paths",
    "execution_paths_errors.json": "1.5_paths",
    # Stage 2 — Knowledge Graph
    "code_graph.gpickle":          "2_graph",
    "code_graph.json":             "2_graph",
    "code_graph.png":              "2_graph",
    # Stage 2.5 — Behavior Graph
    "behavior_graph.json":         "2.5_behavior",
    # Stage 2.7 — Semantic Role Tagging
    "semantic_roles.json":         "2.7_semanticroles",
    # Stage 2.8 — Action Clustering
    "action_clusters.json":        "2.8_clusters",
    # Stage 2.9 — Invariant Detection
    "rule_catalog.json":           "2.9_invariants",
    # Stage 3 — Vector Embeddings
    "chromadb":                    "3_embed",
    "chunks_manifest.json":        "3_embed",
    # Stage 3.5 — Entity Extraction
    "entity_catalog.json":         "3.5_entities",
    # Stage 3.6 — Entity Relationship Reconstruction
    "relationship_catalog.json":   "3.6_relationships",
    # Stage 3.7 — State Machine Reconstruction
    "state_machine_catalog.json":  "3.7_statemachines",
    # Stage 3.8 — Graph-Aware Context Index (enriched by 3.5/3.6/3.7)
    "graph_context_index.json":    "3.8_graphrag",
    # Stage 3.9 — Pre-LLM Preflight Gate
    "preflight_report.json":       "3.9_preflight",
    # Stage 4 — Domain Model
    "domain_model.json":           "4_domain",
    "coverage_report.json":        "4_domain",
    # Stage 4.5 — Business Flows
    "business_flows.json":         "4.5_flows",
    "flow_coverage.json":          "4.5_flows",
    "feature_stubs.feature":       "4.5_flows",
    # Stage 4.6 — Specification Mining
    "spec_rules.json":             "4.6_specrules",
    # Stage 4.7 — Behavioral Validation
    "flow_validation.json":        "4.7_validation",
    "flow_validation.md":          "4.7_validation",
    # Stage 4.8 — Evidence Triangulation
    "evidence_triangulation.json": "4.8_triangulate",
    # Stage 5 — BA Documents (Markdown)
    "brd.md":                      "5_documents",
    "srs.md":                      "5_documents",
    "ac.md":                       "5_documents",
    "user_stories.md":             "5_documents",
    # Stage 5.5 — Automated Traceability Matrix
    "traceability_matrix.json":    "5.5_traceability",
    "traceability_report.md":      "5.5_traceability",
    # Stage 5.9 — Document Coverage Audit
    "doc_coverage.json":           "5.9_doccoverage",
    "doc_coverage_summary.md":     "5.9_doccoverage",
    # Stage 6 — QA Review
    "qa_report.md":                "6_qa",
    "qa_result.json":              "6_qa",
    "confidence_scores.json":      "6_qa",
    "qa_checklist.json":           "6_qa",
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
        "stage0_validate":        StageResult(),
        "stage1_parse":           StageResult(),
        "stage13_entrypoints":    StageResult(),  # system entry-point catalog (static)
        "stage15_paths":          StageResult(),
        "stage2_graph":           StageResult(),
        "stage25_behavior":       StageResult(),  # behavior graph extraction
        "stage27_semanticroles":  StageResult(),  # semantic role tagging (static)
        "stage28_clusters":       StageResult(),  # action clustering (similarity)
        "stage29_invariants":     StageResult(),  # business rule / invariant detection
        "stage3_embed":           StageResult(),
        "stage35_entities":       StageResult(),  # entity extraction (static)
        "stage36_relationships":  StageResult(),  # entity relationship reconstruction (static)
        "stage37_statemachines":  StageResult(),  # state machine reconstruction (static)
        "stage38_graphrag":       StageResult(),  # graph-aware context index (enriched by 3.5/3.6/3.7)
        "stage39_preflight":      StageResult(),  # pre-LLM gate
        "stage4_domain":          StageResult(),
        "stage45_flows":          StageResult(),
        "stage46_specrules":      StageResult(),  # specification mining (business rule synthesis)
        "stage47_validate":       StageResult(),  # behavioral flow validation
        "stage48_triangulate":    StageResult(),  # evidence triangulation (static)
        "stage5_brd":             StageResult(),
        "stage5_srs":             StageResult(),
        "stage5_ac":              StageResult(),
        "stage5_userstories":     StageResult(),
        "stage55_traceability":   StageResult(),  # automated traceability matrix (static)
        "stage59_doccoverage":    StageResult(),  # document coverage audit (static)
        "stage6_qa":              StageResult(),
        "stage62_architecture":   StageResult(),  # architecture reconstruction (LLM)
        "stage65_postprocess":    StageResult(),
        "stage67_diagrams":       StageResult(),
        "stage7_pdf":             StageResult(),
        "stage8_tests":           StageResult(),  # test case generator
        "stage9_knowledge_graph": StageResult(),  # system knowledge graph builder
    })

    code_map:              Optional[CodeMap]             = None
    entry_point_catalog:   Optional[EntryPointCatalog]  = None  # Stage 1.3
    graph_meta:            Optional[GraphMeta]           = None
    behavior_graph:    Optional[dict]                   = None   # Stage 2.5
    semantic_roles:    Optional[SemanticRoleIndex]       = None  # Stage 2.7
    action_clusters:   Optional[ActionClusterCollection] = None  # Stage 2.8
    invariants:        Optional[InvariantCollection]      = None  # Stage 2.9
    entities:          Optional[EntityCollection]          = None  # Stage 3.5
    relationships:     Optional[EntityRelationshipCollection] = None  # Stage 3.6
    state_machines:    Optional[StateMachineCollection]   = None  # Stage 3.7
    spec_rules:        Optional[SpecRuleCollection]        = None  # Stage 4.6
    triangulation:     Optional[TriangulationReport]       = None  # Stage 4.8
    embedding_meta:    Optional[EmbeddingMeta]          = None
    preflight:         Optional[PreflightResult]        = None
    graph_rag_meta:    Optional[GraphRAGMeta]           = None  # Stage 3.8
    domain_model:      Optional[DomainModel]            = None
    business_flows:    Optional[BusinessFlowCollection] = None
    flow_validation:   Optional[dict]                   = None   # Stage 4.7
    ba_artifacts:      Optional[BAArtifacts]            = None
    traceability_meta: Optional[TraceabilityMeta]        = None  # Stage 5.5
    doc_coverage:      Optional[DocCoverageResult]      = None  # Stage 5.9
    qa_result:         Optional[QAResult]               = None
    architecture_meta: Optional[ArchitectureMeta]       = None  # Stage 6.2
    test_suite:            Optional[TestSuiteArtifacts]     = None  # Stage 8
    knowledge_graph_meta: Optional[KnowledgeGraphMeta]     = None  # Stage 9

    @classmethod
    def create(cls, php_project_path: str, output_base: str = "outputs") -> "PipelineContext":
        run_id       = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        project_name = _project_slug(php_project_path)
        output_dir   = os.path.join(output_base, project_name, f"run_{run_id}")
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

    def graph_query(self, topic: str, depth: int = 2, max_chars: int = 3000) -> str:
        """
        Graph-aware context retrieval using the Stage 3.8 index.

        Returns a structured Markdown context block suitable for LLM prompt
        injection.  Empty string if the index was not built or the topic has no
        matching nodes.

        The index JSON is loaded lazily on the first call and cached on a
        transient ``_graph_rag_cache`` attribute (not serialised to context.json).
        """
        if not self.graph_rag_meta or not self.graph_rag_meta.index_path:
            return ""
        # Late import to avoid circular dependency (stage37 imports from context).
        from pipeline.stage38_graphrag import load_index, query_graph  # noqa: PLC0415
        if not hasattr(self, "_graph_rag_cache"):
            self._graph_rag_cache = load_index(self.graph_rag_meta.index_path)
        return query_graph(self._graph_rag_cache, topic, depth=depth, max_chars=max_chars)

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
                    flow_type         = f.get("flow_type", "http"),
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

        if data.get("entry_point_catalog") is not None:
            d = data["entry_point_catalog"]
            ctx.entry_point_catalog = EntryPointCatalog(
                entry_points = [
                    EntryPoint(
                        ep_id        = ep.get("ep_id", ""),
                        ep_type      = ep.get("ep_type", "http"),
                        handler_file = ep.get("handler_file", ""),
                        name         = ep.get("name", ""),
                        schedule     = ep.get("schedule", ""),
                        trigger      = ep.get("trigger", ""),
                        confidence   = ep.get("confidence", 0.8),
                    )
                    for ep in d.get("entry_points", [])
                ],
                by_type      = d.get("by_type", {}),
                total        = d.get("total", 0),
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

        if data.get("doc_coverage") is not None:
            d = data["doc_coverage"]
            ctx.doc_coverage = DocCoverageResult(
                dimensions      = d.get("dimensions", []),
                overall_pct     = d.get("overall_pct", 0.0),
                overall_status  = d.get("overall_status", "fail"),
                gap_summary     = d.get("gap_summary", []),
                generated_at    = d.get("generated_at", ""),
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

        if data.get("invariants") is not None:
            d = data["invariants"]
            rules = [
                BusinessRule(
                    rule_id         = r["rule_id"],
                    category        = r["category"],
                    description     = r["description"],
                    raw_expression  = r["raw_expression"],
                    entity          = r["entity"],
                    bounded_context = r.get("bounded_context", ""),
                    source_files    = r.get("source_files", []),
                    confidence      = r.get("confidence", 0.8),
                    tables          = r.get("tables", []),
                )
                for r in d.get("rules", [])
            ]
            ctx.invariants = InvariantCollection(
                rules        = rules,
                total        = d.get("total", len(rules)),
                by_category  = d.get("by_category", {}),
                by_context   = d.get("by_context", {}),
                generated_at = d.get("generated_at", ""),
            )

        if data.get("semantic_roles") is not None:
            d = data["semantic_roles"]
            actions = [
                ActionTag(
                    symbol      = a["symbol"],
                    file        = a["file"],
                    role        = a["role"],
                    confidence  = a.get("confidence", 0.0),
                    signals     = a.get("signals", []),
                    actor       = a.get("actor", ""),
                    entities    = a.get("entities", []),
                    http_method = a.get("http_method", ""),
                    route_path  = a.get("route_path", ""),
                )
                for a in d.get("actions", [])
            ]
            ext_systems = [
                ExternalSystem(
                    name         = e["name"],
                    category     = e["category"],
                    env_keys     = e.get("env_keys", []),
                    class_hints  = e.get("class_hints", []),
                    dep_hints    = e.get("dep_hints", []),
                    detected_via = e.get("detected_via", []),
                )
                for e in d.get("external_systems", [])
            ]
            ctx.semantic_roles = SemanticRoleIndex(
                actions          = actions,
                external_systems = ext_systems,
                role_summary     = d.get("role_summary", {}),
                infra_files      = d.get("infra_files", []),
                business_files   = d.get("business_files", []),
                generated_at     = d.get("generated_at", ""),
            )

        if data.get("action_clusters") is not None:
            d = data["action_clusters"]
            clusters = [
                ActionCluster(
                    cluster_id   = c["cluster_id"],
                    name         = c["name"],
                    files        = c.get("files", []),
                    tables       = c.get("tables", []),
                    route_prefix = c.get("route_prefix", ""),
                    module       = c.get("module", ""),
                    file_count   = c.get("file_count", 0),
                )
                for c in d.get("clusters", [])
            ]
            ctx.action_clusters = ActionClusterCollection(
                clusters     = clusters,
                total        = d.get("total", len(clusters)),
                generated_at = d.get("generated_at", ""),
            )

        if data.get("entities") is not None:
            d = data["entities"]
            ents = [
                Entity(
                    entity_id       = e["entity_id"],
                    name            = e["name"],
                    table           = e["table"],
                    bounded_context = e.get("bounded_context", ""),
                    columns         = [
                        EntityColumn(
                            name              = c["name"],
                            data_type         = c.get("data_type", ""),
                            nullable          = c.get("nullable", True),
                            is_primary_key    = c.get("is_primary_key", False),
                            is_foreign_key    = c.get("is_foreign_key", False),
                            references_table  = c.get("references_table", ""),
                            references_column = c.get("references_column", ""),
                        )
                        for c in e.get("columns", [])
                    ],
                    primary_key     = e.get("primary_key", ""),
                    is_pivot        = e.get("is_pivot", False),
                    is_core         = e.get("is_core", True),
                    is_system       = e.get("is_system", False),
                    source_files    = e.get("source_files", []),
                    confidence      = e.get("confidence", 1.0),
                )
                for e in d.get("entities", [])
            ]
            ctx.entities = EntityCollection(
                entities     = ents,
                total        = d.get("total", len(ents)),
                core_count   = d.get("core_count", 0),
                generated_at = d.get("generated_at", ""),
            )

        if data.get("relationships") is not None:
            d = data["relationships"]
            rels = [
                EntityRelationship(
                    rel_id       = r["rel_id"],
                    from_entity  = r["from_entity"],
                    to_entity    = r["to_entity"],
                    cardinality  = r.get("cardinality", "1:N"),
                    rel_type     = r.get("rel_type", "has_many"),
                    via_column   = r.get("via_column", ""),
                    via_table    = r.get("via_table", ""),
                    confidence   = r.get("confidence", 0.75),
                    signals      = r.get("signals", []),
                    source_files = r.get("source_files", []),
                )
                for r in d.get("relationships", [])
            ]
            ctx.relationships = EntityRelationshipCollection(
                relationships = rels,
                entity_names  = d.get("entity_names", []),
                total         = d.get("total", len(rels)),
                mermaid_path  = d.get("mermaid_path", ""),
                generated_at  = d.get("generated_at", ""),
            )

        if data.get("state_machines") is not None:
            d = data["state_machines"]
            machines = [
                StateMachine(
                    machine_id      = m["machine_id"],
                    entity          = m["entity"],
                    table           = m["table"],
                    field           = m["field"],
                    bounded_context = m.get("bounded_context", ""),
                    states          = m.get("states", []),
                    initial_states  = m.get("initial_states", []),
                    terminal_states = m.get("terminal_states", []),
                    dead_states     = m.get("dead_states", []),
                    transitions     = [
                        StateTransition(
                            from_state   = t["from_state"],
                            to_state     = t["to_state"],
                            trigger      = t.get("trigger", ""),
                            guard        = t.get("guard", ""),
                            source_files = t.get("source_files", []),
                            confidence   = t.get("confidence", 0.75),
                        )
                        for t in m.get("transitions", [])
                    ],
                    mermaid         = m.get("mermaid", ""),
                )
                for m in d.get("machines", [])
            ]
            ctx.state_machines = StateMachineCollection(
                machines     = machines,
                total        = d.get("total", len(machines)),
                generated_at = d.get("generated_at", ""),
            )

        if data.get("spec_rules") is not None:
            d = data["spec_rules"]
            srules = [
                SpecRule(
                    rule_id           = r["rule_id"],
                    category          = r["category"],
                    title             = r.get("title", ""),
                    description       = r.get("description", ""),
                    given             = r.get("given", ""),
                    when              = r.get("when", ""),
                    then              = r.get("then", ""),
                    entities          = r.get("entities", []),
                    bounded_context   = r.get("bounded_context", ""),
                    source_invariants = r.get("source_invariants", []),
                    source_machines   = r.get("source_machines", []),
                    source_flows      = r.get("source_flows", []),
                    source_files      = r.get("source_files", []),
                    confidence        = r.get("confidence", 0.75),
                    tags              = r.get("tags", []),
                    pass_origin       = r.get("pass_origin", ""),
                )
                for r in d.get("rules", [])
            ]
            ctx.spec_rules = SpecRuleCollection(
                rules        = srules,
                total        = d.get("total", len(srules)),
                by_category  = d.get("by_category", {}),
                by_entity    = d.get("by_entity", {}),
                by_flow      = d.get("by_flow", {}),
                by_context   = d.get("by_context", {}),
                generated_at = d.get("generated_at", ""),
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

        if data.get("graph_rag_meta") is not None:
            d = data["graph_rag_meta"]
            ctx.graph_rag_meta = GraphRAGMeta(
                index_path    = d.get("index_path", ""),
                file_count    = d.get("file_count", 0),
                entity_count  = d.get("entity_count", 0),
                cluster_count = d.get("cluster_count", 0),
                generated_at  = d.get("generated_at", ""),
            )

        if data.get("triangulation") is not None:
            d = data["triangulation"]
            trules = [
                TriangulatedRule(
                    rule_id              = r["rule_id"],
                    title                = r.get("title", ""),
                    category             = r.get("category", ""),
                    corroborating_types  = r.get("corroborating_types", []),
                    contradicting_types  = r.get("contradicting_types", []),
                    triangulation_score  = r.get("triangulation_score", 0.0),
                    triangulation_status = r.get("triangulation_status", "WEAK"),
                    max_applicable       = r.get("max_applicable", 4),
                    contradiction_notes  = r.get("contradiction_notes", []),
                )
                for r in d.get("rules", [])
            ]
            ctx.triangulation = TriangulationReport(
                rules               = trules,
                total               = d.get("total", len(trules)),
                strong_count        = d.get("strong_count", 0),
                moderate_count      = d.get("moderate_count", 0),
                weak_count          = d.get("weak_count", 0),
                contradiction_count = d.get("contradiction_count", 0),
                weak_rule_ids       = d.get("weak_rule_ids", []),
                contradiction_ids   = d.get("contradiction_ids", []),
                generated_at        = d.get("generated_at", ""),
            )

        if data.get("traceability_meta") is not None:
            d = data["traceability_meta"]
            ctx.traceability_meta = TraceabilityMeta(
                matrix_path        = d.get("matrix_path", ""),
                report_path        = d.get("report_path", ""),
                total_requirements = d.get("total_requirements", 0),
                covered_brd        = d.get("covered_brd", 0.0),
                covered_srs        = d.get("covered_srs", 0.0),
                covered_ac         = d.get("covered_ac", 0.0),
                covered_us         = d.get("covered_us", 0.0),
                avg_code_links     = d.get("avg_code_links", 0.0),
                uncited_count      = d.get("uncited_count", 0),
                uncited_ids        = d.get("uncited_ids", []),
                generated_at       = d.get("generated_at", ""),
            )

        return ctx

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
    LARAVEL    = "laravel"
    SYMFONY    = "symfony"
    CODEIGNITER = "codeigniter"
    WORDPRESS  = "wordpress"
    RAW_PHP    = "raw_php"
    UNKNOWN    = "unknown"


@dataclass
class StageResult:
    """Tracks the outcome of a single pipeline stage."""
    status: StageStatus = StageStatus.PENDING
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    output_path: Optional[str] = None

    def mark_running(self) -> None:
        self.status = StageStatus.RUNNING
        self.started_at = datetime.utcnow().isoformat()

    def mark_completed(self, output_path: Optional[str] = None) -> None:
        self.status = StageStatus.COMPLETED
        self.completed_at = datetime.utcnow().isoformat()
        if output_path:
            self.output_path = output_path

    def mark_failed(self, error: str) -> None:
        self.status = StageStatus.FAILED
        self.completed_at = datetime.utcnow().isoformat()
        self.error = error

    def mark_skipped(self) -> None:
        self.status = StageStatus.SKIPPED


@dataclass
class CodeMap:
    framework: Framework = Framework.UNKNOWN
    php_version: Optional[str] = None
    classes: list[dict[str, Any]] = field(default_factory=list)
    routes: list[dict[str, Any]] = field(default_factory=list)
    models: list[dict[str, Any]] = field(default_factory=list)
    controllers: list[dict[str, Any]] = field(default_factory=list)
    services: list[dict[str, Any]] = field(default_factory=list)
    db_schema: list[dict[str, Any]] = field(default_factory=list)
    config_files: list[str] = field(default_factory=list)
    total_files: int = 0
    total_lines: int = 0


@dataclass
class GraphMeta:
    graph_path: str = ""
    node_count: int = 0
    edge_count: int = 0
    node_types: list[str] = field(default_factory=list)
    edge_types: list[str] = field(default_factory=list)


@dataclass
class EmbeddingMeta:
    collection_name: str = ""
    chroma_path: str = ""
    total_chunks: int = 0
    embedding_model: str = "text-embedding-3-small"


@dataclass
class PreflightResult:
    passed: bool = True
    warnings: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)


@dataclass
class DomainModel:
    domain_name: str = ""
    description: str = ""
    user_roles: list[dict[str, str]] = field(default_factory=list)
    features: list[dict[str, Any]] = field(default_factory=list)
    workflows: list[dict[str, Any]] = field(default_factory=list)
    bounded_contexts: list[str] = field(default_factory=list)
    key_entities: list[str] = field(default_factory=list)


@dataclass
class BAArtifacts:
    brd_path: Optional[str] = None
    srs_path: Optional[str] = None
    ac_path: Optional[str] = None
    user_stories_path: Optional[str] = None


@dataclass
class QAResult:
    passed: bool = True
    issues: list[dict[str, str]] = field(default_factory=list)
    coverage_score: float = 0.0
    consistency_score: float = 0.0


@dataclass
class FlowStep:
    """
    One step inside a BusinessFlow — corresponds to a single page visit,
    form submission, API call, or database operation.
    """
    step_num:     int            # 1-based sequence number
    page:         str            # PHP file or route path (e.g. "login.php")
    action:       str            # Human-readable action (e.g. "Submit credentials")
    http_method:  Optional[str]  # GET / POST / PUT / DELETE / None
    db_ops:       list[str]      # e.g. ["SELECT users", "INSERT sessions"]
    auth_required: bool          # True if this step has an auth guard
    inputs:       list[str]      # Form / query fields consumed (e.g. ["email","password"])
    outputs:      list[str]      # Session keys / redirect targets produced
    is_branch:    bool = False   # True for alternate/error path steps


@dataclass
class BusinessFlow:
    """
    A complete end-to-end user journey extracted from the codebase —
    one LLM-named, graph-traversal-backed business process.

    Produced by Stage 4.5 (BusinessFlowExtractor).
    Consumed by Stage 5 AC and UserStory agents.
    """
    flow_id:         str               # e.g. "flow_001"
    name:            str               # e.g. "Customer Rental Booking"
    actor:           str               # e.g. "Authenticated Customer"
    bounded_context: str               # e.g. "Booking"
    trigger:         str               # What initiates the flow
    steps:           list[FlowStep]    # Ordered happy-path steps
    branches:        list[dict[str, Any]]  # Alternate / error paths
    termination:     str               # Success end-state description
    evidence_files:  list[str]         # Source PHP files that back this flow
    confidence:      float = 1.0       # 0.0–1.0; low = graph-only, high = LLM-confirmed
    replaces_workflow: Optional[str] = None  # name of domain_model.workflow it supersedes


@dataclass
class BusinessFlowCollection:
    """Container written to business_flows.json by Stage 4.5."""
    flows:        list[BusinessFlow] = field(default_factory=list)
    total:        int = 0
    by_context:   dict[str, list[str]] = field(default_factory=dict)  # context → [flow_ids]
    by_actor:     dict[str, list[str]] = field(default_factory=dict)  # actor   → [flow_ids]
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class PipelineContext:
    """
    Central state object passed through every stage of the PHP-BA Agent pipeline.
    Usage:
        ctx = PipelineContext.create(php_project_path="/path/to/project")
        ctx = PipelineContext.load("outputs/run_xyz/context.json")  # resume
    """
    run_id: str = ""
    php_project_path: str = ""
    output_dir: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    stages: dict[str, StageResult] = field(default_factory=lambda: {
        "stage0_validate":     StageResult(),
        "stage1_parse":        StageResult(),
        "stage2_graph":        StageResult(),
        "stage3_embed":        StageResult(),
        "stage35_preflight":   StageResult(),
        "stage4_domain":       StageResult(),
        "stage45_flows":       StageResult(),
        "stage5_brd":          StageResult(),
        "stage5_srs":          StageResult(),
        "stage5_ac":           StageResult(),
        "stage5_userstories":  StageResult(),
        "stage6_qa":           StageResult(),
        "stage65_postprocess": StageResult(),
    })
    code_map:        Optional[CodeMap]                = None
    graph_meta:      Optional[GraphMeta]              = None
    embedding_meta:  Optional[EmbeddingMeta]          = None
    preflight:       Optional[PreflightResult]        = None
    domain_model:    Optional[DomainModel]            = None
    business_flows:  Optional[BusinessFlowCollection] = None
    ba_artifacts:    Optional[BAArtifacts]            = None
    qa_result:       Optional[QAResult]               = None

    @classmethod
    def create(cls, php_project_path: str, output_base: str = "outputs") -> "PipelineContext":
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_base, f"run_{run_id}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        ctx = cls(run_id=run_id, php_project_path=str(Path(php_project_path).resolve()), output_dir=output_dir)
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
            raise KeyError(f"Unknown stage '{name}'. Valid: {list(self.stages.keys())}")
        return self.stages[name]

    def is_stage_done(self, name: str) -> bool:
        return self.stage(name).status == StageStatus.COMPLETED

    def output_path(self, filename: str) -> str:
        return os.path.join(self.output_dir, filename)

    def _to_dict(self) -> dict[str, Any]:
        import dataclasses
        def _serialize(obj):
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return {k: _serialize(v) for k, v in dataclasses.asdict(obj).items()}
            if isinstance(obj, dict): return {k: _serialize(v) for k, v in obj.items()}
            if isinstance(obj, list): return [_serialize(i) for i in obj]
            if isinstance(obj, Enum): return obj.value
            return obj
        return _serialize(self)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> "PipelineContext":
        ctx = cls(run_id=data["run_id"], php_project_path=data["php_project_path"],
                  output_dir=data["output_dir"], created_at=data["created_at"])
        for name, sr in data.get("stages", {}).items():
            if name in ctx.stages:
                ctx.stages[name] = StageResult(status=StageStatus(sr["status"]),
                    started_at=sr.get("started_at"), completed_at=sr.get("completed_at"),
                    error=sr.get("error"), output_path=sr.get("output_path"))
        return ctx
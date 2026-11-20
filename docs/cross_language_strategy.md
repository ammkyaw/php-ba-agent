# Cross-Language Reference Strategy

> **Version:** 2026-03-27 | **Status:** Draft for approval

---

## 1. Problem Statement

Modern codebases often span multiple languages:
- **Java backend ↔ TypeScript microservice** (REST/gRPC contracts)
- **PHP monolith ↔ TypeScript frontend** (API endpoints, shared types)
- **Multi-module monorepo** (Java + Kotlin modules sharing dependencies)

The current pipeline parses each language independently, producing one `CodeMap` per project. Cross-language references (API contracts, shared types, service calls) are invisible to downstream stages.

---

## 2. Detection Heuristics

### 2.1 Shared API Contract Files
| Signal | Detection | Confidence |
|---|---|---|
| **OpenAPI / Swagger specs** | `*.yaml`/`*.json` files with `openapi:` or `swagger:` root key | 0.95 |
| **Proto files (gRPC)** | `*.proto` files with `service` blocks | 0.95 |
| **GraphQL schemas** | `*.graphql`/`*.gql` files with `type Query` | 0.90 |
| **Shared TS types** | `types/` or `shared/` directories imported by multiple sub-projects | 0.80 |

### 2.2 Cross-Service HTTP Calls
| Pattern | Language | Detection |
|---|---|---|
| `fetch('/api/...')` / `axios.get('/api/...')` | TS/JS | Regex on HTTP client invocations, match URL path against known routes from another sub-project |
| `WebClient.get().uri("/api/...")` | Java | Regex on Spring WebClient |
| `file_get_contents('http://...')` / `Http::get()` | PHP | Regex on Laravel/PHP HTTP client calls |

### 2.3 Monorepo Module References
| Signal | Detection |
|---|---|
| Workspace packages (`workspaces` in package.json) | Cross-reference `@scope/package-name` imports against local workspace packages |
| Gradle multi-module (`include ':module-a'`) | Parse `settings.gradle` for project references |
| Maven multi-module (`<modules>`) | Parse parent `pom.xml` for child module references |

---

## 3. Shared Schema Representation

### 3.1 New `CodeMap` Fields

```python
# Add to CodeMap dataclass in context.py:

# Cross-language API contracts
api_contracts: list[dict[str, Any]] = field(default_factory=list)
# Shape: {
#   "contract_id": "api_001",
#   "type": "openapi" | "grpc" | "graphql" | "rest_inferred",
#   "file": "api/openapi.yaml",
#   "endpoints": [{"method": "GET", "path": "/api/users", "request_schema": {...}, "response_schema": {...}}],
# }

# Cross-service call edges
cross_service_calls: list[dict[str, Any]] = field(default_factory=list)
# Shape: {
#   "caller_file": "frontend/src/api/users.ts",
#   "caller_language": "typescript",
#   "target_url": "/api/users",
#   "target_method": "GET",
#   "matched_route_file": "backend/app/Http/Controllers/UserController.php",
#   "confidence": 0.85,
# }
```

### 3.2 New `ProjectManifest` Dataclass

For monorepo support, introduce a project-level manifest that sits above `CodeMap`:

```python
@dataclass
class SubProject:
    name: str              # "frontend", "api-gateway"
    path: str              # relative path within monorepo
    language: Language
    framework: Framework
    code_map: CodeMap      # per-subproject code map

@dataclass  
class ProjectManifest:
    sub_projects:   list[SubProject] = field(default_factory=list)
    api_contracts:  list[dict]       = field(default_factory=list)  
    cross_edges:    list[dict]       = field(default_factory=list)  
```

---

## 4. Cross-Service Dependency Edges in the Code Graph

### 4.1 New Edge Types for Stage 2

| Edge Type | Source Signal | Example |
|---|---|---|
| `CROSS_SERVICE_CALL` | HTTP client call → route in other sub-project | `fetch('/api/users')` → `UserController@index` |
| `SHARED_CONTRACT` | Both sub-projects reference the same OpenAPI spec | `openapi.yaml` ← TS client + Java server |
| `SHARED_TYPE` | TypeScript type imported from shared package | `@types/user` used by frontend + backend |
| `PROTO_RPC` | gRPC call from client to server stub | `UserService.getUser()` → Java gRPC handler |

### 4.2 Impact on Stage 2.x Graph

Add cross-language edges between nodes from different `SubProject` code maps. The knowledge graph (NetworkX) already supports heterogeneous node types. Add a `subproject` attribute to each node so graph queries can filter/scope.

---

## 5. Impact on Pipeline Stages

| Stage | Change Required | Effort |
|---|---|---|
| **0.5 Detect** | Detect monorepo structure; iterate sub-projects | Medium |
| **1 Parse** | Run language adapter per sub-project; build composite `ProjectManifest` | Medium |
| **1.3 Entrypoints** | Aggregate entry points across sub-projects | Low |
| **2 Graph** | Add cross-service edges; new edge types | Medium |
| **2.7 Semantic Roles** | Tag cross-service callers as `INTEGRATION_ACTION` | Low |
| **2.8 Clusters** | Allow clusters to span sub-project boundaries | Low |
| **3.5 Entities** | Merge entities with same table name across sub-projects | Low |
| **3.8 GraphRAG** | Include cross-service context in RAG index | Low |
| **4 Domain** | LLM receives unified domain context from all sub-projects | Low |
| **5 Documents** | BRD/SRS reflect cross-service interactions | None (prompt changes only) |

---

## 6. Implementation Roadmap

### Phase A — Foundation (no breaking changes)
1. Add `api_contracts` and `cross_service_calls` fields to `CodeMap`
2. Create a `pipeline/parsers/contract_parser.py` for OpenAPI/Proto/GraphQL files
3. Add cross-service HTTP call detection to existing TS and Java parsers

### Phase B — Monorepo Support
1. Add monorepo detection to `stage05_detect.py`
2. Create `ProjectManifest` dataclass in `context.py`
3. Modify `stage10_parse.py` to iterate sub-projects

### Phase C — Graph Integration
1. Add cross-service edge types to `stage20_graph.py`
2. Extend `stage27_semanticroles.py` for `INTEGRATION_ACTION` tagging
3. Update `stage38_graphrag.py` to include cross-service context

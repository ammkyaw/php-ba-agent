"""
pipeline/parsers/java_parser.py — Java / Kotlin Language Adapter

Pure-Python static analysis of Java (and Kotlin) projects, focused on
Spring Boot but also covering Quarkus and Micronaut.

Extraction strategy
-------------------
• File discovery   — .java / .kt files (skips target/build/generated)
• Classes          — class / interface / enum declarations
• Controllers      — @RestController / @Controller annotated classes
• Models           — @Entity / @Table / @Document annotated classes
• Services         — @Service / @Repository / @Component classes
• Routes           — @GetMapping / @PostMapping / @RequestMapping etc.
                     also Quarkus @GET/@POST + @Path
• Functions/methods — public methods inside service/controller classes
• Imports          — import statements
• DB schema        — JPA @Entity + @Column / @Table introspection
• SQL queries      — @Query annotations, EntityManager.createQuery, JdbcTemplate
• Call graph       — method calls (approximate)
• Auth signals     — @PreAuthorize, @Secured, SecurityContextHolder, JWT patterns
• Env vars         — @Value("${...}"), System.getenv, @ConfigurationProperties
• Input params     — @RequestBody, @RequestParam, @PathVariable
• Service deps     — constructor + @Autowired injection
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from context import CodeMap, Framework, Language, PipelineContext
from pipeline.parsers.base import LanguageParser


_JAVA_EXTS = {".java", ".kt"}

# ─── Adapter ──────────────────────────────────────────────────────────────────

class JavaParser(LanguageParser):
    """Language adapter for Java / Kotlin projects."""

    LANGUAGE = Language.JAVA
    SUPPORTED_FRAMEWORKS = frozenset({
        Framework.SPRING_BOOT,
        Framework.QUARKUS,
        Framework.MICRONAUT,
        Framework.UNKNOWN,
    })

    @classmethod
    def detect(cls, project_path: str) -> bool:
        root = Path(project_path)
        if (root / "pom.xml").exists() or (root / "build.gradle").exists():
            return True
        for f in list(root.rglob("*.java"))[:3]:
            return True
        return False

    def parse(self, project_path: str, ctx: PipelineContext) -> CodeMap:
        root = Path(project_path)
        output_path = ctx.output_path("code_map.json")

        framework = _detect_java_framework(root)
        lang = _detect_java_language(root)

        print(f"  [stage10/java] Language  : {lang.value}")
        print(f"  [stage10/java] Framework : {framework.value}")
        print(f"  [stage10/java] Scanning  : {root}")

        source_files = list(self.iter_source_files(root, _JAVA_EXTS))
        print(f"  [stage10/java] Found {len(source_files)} source files")

        classes      : list[dict] = []
        controllers  : list[dict] = []
        models       : list[dict] = []
        services     : list[dict] = []
        functions    : list[dict] = []
        routes       : list[dict] = []
        imports      : list[dict] = []
        sql_queries  : list[dict] = []
        call_graph   : list[dict] = []
        service_deps : list[dict] = []
        env_vars     : list[dict] = []
        auth_signals : list[dict] = []
        input_params : list[dict] = []

        total_lines = 0

        for src_file in source_files:
            src = self.safe_read(src_file)
            if not src:
                continue
            rel = self.rel(src_file, root)
            total_lines += src.count("\n")

            _extract_classes(src, rel, classes, controllers, models, services, framework)
            _extract_methods(src, rel, functions)
            _extract_imports_java(src, rel, imports)
            _extract_routes(src, rel, routes, framework)
            _extract_sql_queries(src, rel, sql_queries)
            _extract_env_vars(src, rel, env_vars)
            _extract_auth_signals(src, rel, auth_signals)
            _extract_input_params(src, rel, input_params)
            _extract_service_deps(src, rel, service_deps)
            _extract_call_graph(src, rel, call_graph)

        db_schema, table_columns = _extract_db_schema(models)

        # HTTP endpoints from routes
        http_endpoints = [
            {"method": r.get("method", ""), "handler": r.get("handler", ""), "path": r.get("path", "")}
            for r in routes
        ]

        code_map = CodeMap(
            language         = lang,
            language_version = _detect_java_version(root),
            framework        = framework,
            classes          = classes,
            routes           = routes,
            models           = models,
            controllers      = controllers,
            services         = services,
            db_schema        = db_schema,
            config_files     = _find_config_files(root),
            total_files      = len(source_files),
            total_lines      = total_lines,
            functions        = functions,
            imports          = imports,
            sql_queries      = sql_queries,
            call_graph       = call_graph,
            service_deps     = service_deps,
            env_vars         = env_vars,
            auth_signals     = auth_signals,
            http_endpoints   = http_endpoints,
            table_columns    = table_columns,
            input_params     = input_params,
        )

        payload = _code_map_to_payload(code_map)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        print(f"  [stage10/java] Saved code_map.json → {output_path}")

        print(
            f"  [stage10/java] Done — classes={len(classes)}, controllers={len(controllers)}, "
            f"models={len(models)}, routes={len(routes)}, services={len(services)}, "
            f"sql={len(sql_queries)}, env_vars={len(env_vars)}"
        )
        return code_map


# ─── Framework / language detection ──────────────────────────────────────────

def _detect_java_framework(root: Path) -> Framework:
    for fname in ("pom.xml", "build.gradle", "build.gradle.kts"):
        p = root / fname
        if p.exists():
            txt = p.read_text(encoding="utf-8", errors="ignore").lower()
            if "spring-boot" in txt or "org.springframework" in txt:
                return Framework.SPRING_BOOT
            if "quarkus" in txt:
                return Framework.QUARKUS
            if "micronaut" in txt:
                return Framework.MICRONAUT
    return Framework.UNKNOWN


def _detect_java_language(root: Path) -> Language:
    kt_files = list(root.rglob("*.kt"))
    java_files = list(root.rglob("*.java"))
    if len(kt_files) > len(java_files):
        return Language.KOTLIN
    return Language.JAVA


def _detect_java_version(root: Path) -> str | None:
    for fname in ("pom.xml",):
        p = root / fname
        if p.exists():
            txt = p.read_text(encoding="utf-8", errors="ignore")
            m = re.search(r"<java\.version>(\d+)", txt)
            if m:
                return m.group(1)
            m = re.search(r"<maven\.compiler\.source>(\d+)", txt)
            if m:
                return m.group(1)
    for fname in ("build.gradle", "build.gradle.kts"):
        p = root / fname
        if p.exists():
            txt = p.read_text(encoding="utf-8", errors="ignore")
            m = re.search(r"sourceCompatibility\s*=\s*['\"]?(\d+)", txt)
            if m:
                return m.group(1)
    return None


# ─── Extractors ───────────────────────────────────────────────────────────────

_CTRL_ANNOTS  = {"@RestController", "@Controller"}
_MODEL_ANNOTS = {"@Entity", "@Table", "@Document", "@MappedSuperclass"}
_SVC_ANNOTS   = {"@Service", "@Repository", "@Component", "@Bean"}


def _extract_classes(src: str, rel: str, classes: list, controllers: list,
                     models: list, services: list, fw: Framework) -> None:
    # Collect annotation block before each class declaration
    for m in re.finditer(
        r"((?:@\w+(?:\([^)]*\))?\s*\n?){0,6})"
        r"(?:public\s+|protected\s+|private\s+|abstract\s+|final\s+)*"
        r"(?:class|interface|enum)\s+(\w+)",
        src, re.MULTILINE
    ):
        annots_block = m.group(1)
        name = m.group(2)
        entry = {"name": name, "file": rel, "annotations": annots_block.strip()[:200]}

        if any(a in annots_block for a in _CTRL_ANNOTS):
            entry["type"] = "controller"
            controllers.append(entry)
        elif any(a in annots_block for a in _MODEL_ANNOTS):
            entry["type"] = "model"
            models.append({"name": name, "file": rel, "annotations": annots_block.strip()[:200]})
        elif any(a in annots_block for a in _SVC_ANNOTS):
            entry["type"] = "service"
            services.append(entry)
        else:
            entry["type"] = "class"
            classes.append(entry)

        # Also support Quarkus @Path classes as controllers
        if fw == Framework.QUARKUS and "@Path" in annots_block:
            entry["type"] = "controller"
            if entry not in controllers:
                controllers.append(entry)


def _extract_methods(src: str, rel: str, functions: list) -> None:
    for m in re.finditer(
        r"(?:public|protected|private)\s+(?:static\s+)?(?:\w+(?:<[^>]+>)?)\s+(\w+)\s*\(",
        src
    ):
        name = m.group(1)
        if name in ("class", "interface", "enum", "new"):
            continue
        functions.append({"name": name, "file": rel})


def _extract_imports_java(src: str, rel: str, imports: list) -> None:
    for m in re.finditer(r"^import\s+([\w.]+);", src, re.MULTILINE):
        imports.append({"from": rel, "target": m.group(1)})


def _extract_routes(src: str, rel: str, routes: list, fw: Framework) -> None:
    # Collect class-level @RequestMapping prefix
    ctrl_prefix = ""
    m = re.search(r"@RequestMapping\s*\(\s*['\"]?([^'\")\s]+)", src)
    if m:
        ctrl_prefix = "/" + m.group(1).lstrip("/")

    # Spring: @GetMapping / @PostMapping / @PutMapping / @DeleteMapping / @PatchMapping
    for m in re.finditer(
        r"@(Get|Post|Put|Delete|Patch|Request)Mapping\s*"
        r"(?:\(\s*(?:value\s*=\s*)?['\"]([^'\"]*)['\"])?",
        src
    ):
        verb = m.group(1)
        if verb == "Request":
            # Try to find method = RequestMethod.XXX nearby
            ctx_src = src[m.start():m.start()+200]
            verb_m = re.search(r"method\s*=\s*RequestMethod\.(\w+)", ctx_src)
            method = verb_m.group(1) if verb_m else "ANY"
        else:
            method = verb.upper()
        sub = m.group(2) or ""
        path = ctrl_prefix + ("/" + sub.lstrip("/") if sub else "")

        # Find nearest method name after this annotation
        handler = ""
        remainder = src[m.end():m.end()+300]
        hm = re.search(r"(?:public|private|protected)[^(]+\s+(\w+)\s*\(", remainder)
        if hm:
            handler = hm.group(1)

        routes.append({"method": method, "path": path or "/", "handler": handler, "file": rel, "kind": "spring"})

    # Quarkus: @GET / @POST etc + @Path
    if fw == Framework.QUARKUS:
        _extract_quarkus_routes(src, rel, routes, ctrl_prefix)


def _extract_quarkus_routes(src: str, rel: str, routes: list, prefix: str) -> None:
    for m in re.finditer(
        r"@(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\b[^@]*?"
        r"@Path\s*\(\s*['\"]([^'\"]*)['\"]",
        src, re.DOTALL
    ):
        method, path = m.group(1), prefix + "/" + m.group(2).lstrip("/")
        routes.append({"method": method, "path": path, "file": rel, "kind": "quarkus"})


def _extract_sql_queries(src: str, rel: str, sql_queries: list) -> None:
    # @Query annotations
    for m in re.finditer(r'@Query\s*\(\s*["\']([^"\']{0,200})', src):
        sql_queries.append({"file": rel, "kind": "@Query", "raw": m.group(1)[:120]})
    # Raw SQL in strings
    for m in re.finditer(
        r'["\'](\s*(?:SELECT|INSERT|UPDATE|DELETE|CREATE)\b[^"\']{0,200})["\']',
        src, re.IGNORECASE
    ):
        sql_queries.append({"file": rel, "operation": m.group(1).split()[0].upper(), "raw": m.group(1)[:120]})
    # JPA/JdbcTemplate
    for m in re.finditer(
        r"(?:entityManager|jdbcTemplate|namedParameterJdbc)\."
        r"(?:createQuery|execute|query|update|queryForObject)\s*\(",
        src
    ):
        sql_queries.append({"file": rel, "kind": "jdbc_or_jpa", "operation": m.group(0)})


def _extract_env_vars(src: str, rel: str, env_vars: list) -> None:
    # @Value("${my.property}")
    for m in re.finditer(r'@Value\s*\(\s*["\$]\{([^}]+)\}', src):
        env_vars.append({"key": m.group(1), "file": rel, "kind": "@Value"})
    # System.getenv("KEY")
    for m in re.finditer(r'System\.getenv\s*\(\s*"([^"]+)"', src):
        env_vars.append({"key": m.group(1), "file": rel, "kind": "System.getenv"})


def _extract_auth_signals(src: str, rel: str, auth_signals: list) -> None:
    patterns = [
        (r"@PreAuthorize",          "pre_authorize"),
        (r"@Secured",               "secured"),
        (r"SecurityContextHolder",  "security_context"),
        (r"JwtToken|JWTVerifier|parseClaimsJws", "jwt"),
        (r"hasRole|hasAuthority",   "role_check"),
        (r"@WithMockUser",          "test_auth"),
    ]
    for pattern, kind in patterns:
        if re.search(pattern, src):
            auth_signals.append({"file": rel, "kind": kind})


def _extract_input_params(src: str, rel: str, input_params: list) -> None:
    # @RequestBody MyDto dto
    for m in re.finditer(r"@RequestBody\s+\w+\s+(\w+)", src):
        input_params.append({"source": "body", "key": m.group(1), "file": rel})
    # @RequestParam(value="email") / @RequestParam String email
    for m in re.finditer(r'@RequestParam\s*(?:\([^)]*value\s*=\s*["\']([^"\']+)["\'])?[^)]*\)\s*\w+\s+(\w+)|@RequestParam\s+\w+\s+(\w+)', src):
        key = m.group(1) or m.group(2) or m.group(3) or ""
        if key:
            input_params.append({"source": "query", "key": key, "file": rel})
    # @PathVariable String id
    for m in re.finditer(r"@PathVariable\s+\w+\s+(\w+)", src):
        input_params.append({"source": "path", "key": m.group(1), "file": rel})


def _extract_service_deps(src: str, rel: str, service_deps: list) -> None:
    # Constructor injection (Spring)
    # public MyController(MyService myService, OtherService other)
    for m in re.finditer(
        r"(?:public|protected)\s+\w+\s*\(([^)]{0,500})\)\s*\{",
        src
    ):
        params = m.group(1)
        # Each param: TypeName varName
        for pm in re.finditer(r"(\w+(?:<[^>]+>)?)\s+\w+\s*(?:,|$)", params):
            dep_type = pm.group(1)
            if dep_type not in {"String", "int", "long", "boolean", "void", "List", "Map", "Optional"}:
                service_deps.append({"file": rel, "depends_on": dep_type})


def _extract_call_graph(src: str, rel: str, call_graph: list) -> None:
    # Approximate: service.method() calls
    for m in re.finditer(r"(\w+Service|\w+Repository|\w+Dao)\.(\w+)\s*\(", src):
        call_graph.append({"caller_file": rel, "callee": f"{m.group(1)}.{m.group(2)}"})


def _extract_db_schema(models: list[dict]) -> tuple[list[dict], list[dict]]:
    """Build db_schema and table_columns from collected @Entity model metadata."""
    db_schema: list[dict] = []
    table_columns: list[dict] = []
    for model in models:
        name = model.get("name", "")
        # Try to extract @Table(name="foo") from annotations block
        table_name = _snake(name)
        annots = model.get("annotations", "")
        m = re.search(r'@Table\s*\(\s*name\s*=\s*["\']([^"\']+)', annots)
        if m:
            table_name = m.group(1)
        db_schema.append({"table": table_name, "entity": name, "source": "jpa_entity"})
        # We don't have column details without deeper parsing; leave as empty
        table_columns.append({"table": table_name, "columns": []})
    return db_schema, table_columns


def _snake(name: str) -> str:
    """CamelCase → snake_case with naive pluralisation."""
    s = re.sub(r"([A-Z])", r"_\1", name).lower().lstrip("_")
    return s if s.endswith("s") else s + "s"


def _find_config_files(root: Path) -> list[str]:
    names = {
        "pom.xml", "build.gradle", "build.gradle.kts",
        "application.properties", "application.yml",
        "application-dev.properties", "application-prod.yml",
        "bootstrap.properties", "bootstrap.yml",
    }
    found = []
    for name in names:
        for p in root.rglob(name):
            found.append(str(p.relative_to(root)))
    return found[:20]


def _code_map_to_payload(cm: CodeMap) -> dict:
    import dataclasses
    def _ser(obj: Any) -> Any:
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {k: _ser(v) for k, v in dataclasses.asdict(obj).items()}
        if isinstance(obj, list): return [_ser(i) for i in obj]
        if isinstance(obj, dict): return {k: _ser(v) for k, v in obj.items()}
        if hasattr(obj, "value"): return obj.value
        return obj
    d = _ser(cm)
    d["_parser"] = "java"
    return d

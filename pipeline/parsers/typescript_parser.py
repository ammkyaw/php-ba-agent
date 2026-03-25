"""
pipeline/parsers/typescript_parser.py — TypeScript / JavaScript Language Adapter

Pure-Python, zero-subprocess static analysis of TypeScript, JavaScript, Vue,
React (TSX/JSX), and Next.js projects.

Extraction strategy
-------------------
• File discovery   — scans .ts / .tsx / .js / .jsx / .vue (skips dist/build/node_modules)
• Classes          — regex: `class Foo (extends|implements)? ...`
• Functions        — exported functions + arrow functions assigned to const
• Routes           — Express/Fastify: app.get/post/put/delete/patch/use
                     Next.js: file-based (pages/api/*.ts → GET/POST)
                             App Router (app/**/route.ts → export function GET/POST…)
                     NestJS: @Controller + @Get/@Post etc. decorators
                     Vue Router: route config objects { path, component }
• Controllers      — NestJS @Controller classes; Express Router files
• Models           — TypeORM @Entity, Prisma model blocks, Mongoose Schema
• Services         — @Injectable, classes ending in Service/Repository/Provider
• DB schema        — Prisma schema.prisma parsing; TypeORM @Column introspection
• Imports          — ES module import statements (language-neutral imports field)
• Call graph       — function calls inside function bodies (regex, approximate)
• Auth signals     — passport.authenticate, jwt.verify, @Guard, middleware auth patterns
• Env vars         — process.env.XXX references
• Form fields      — req.body.*, req.query.* accesses
• Service deps     — constructor injection (NestJS @Inject, simple constructor params)
• Input params     — req.body / req.query / req.params access patterns (replaces superglobals)
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from context import CodeMap, Framework, Language, PipelineContext
from pipeline.parsers.base import LanguageParser


_TS_EXTS = {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".vue"}

# ─── Adapter ──────────────────────────────────────────────────────────────────

class TypeScriptParser(LanguageParser):
    """Language adapter for TypeScript / JavaScript projects."""

    LANGUAGE = Language.TYPESCRIPT
    SUPPORTED_FRAMEWORKS = frozenset({
        Framework.NEXTJS, Framework.NUXTJS, Framework.VUE, Framework.REACT,
        Framework.EXPRESS, Framework.FASTIFY, Framework.NESTJS, Framework.UNKNOWN,
    })

    @classmethod
    def detect(cls, project_path: str) -> bool:
        root = Path(project_path)
        if (root / "package.json").exists():
            return True
        for f in list(root.rglob("*.ts"))[:5]:
            return True
        return False

    def parse(self, project_path: str, ctx: PipelineContext) -> CodeMap:
        root = Path(project_path)
        output_path = ctx.output_path("code_map.json")

        # Detect JS vs TS (prefer TS if tsconfig present)
        lang = Language.TYPESCRIPT if (root / "tsconfig.json").exists() else Language.JAVASCRIPT
        framework = _detect_ts_framework(root)

        print(f"  [stage10/ts] Language  : {lang.value}")
        print(f"  [stage10/ts] Framework : {framework.value}")
        print(f"  [stage10/ts] Scanning  : {root}")

        source_files = list(self.iter_source_files(root, _TS_EXTS))
        print(f"  [stage10/ts] Found {len(source_files)} source files")

        classes      : list[dict] = []
        controllers  : list[dict] = []
        models       : list[dict] = []
        services     : list[dict] = []
        functions    : list[dict] = []
        routes       : list[dict] = []
        imports      : list[dict] = []
        sql_queries  : list[dict] = []
        call_graph   : list[dict] = []
        form_fields  : list[dict] = []
        service_deps : list[dict] = []
        env_vars     : list[dict] = []
        auth_signals : list[dict] = []
        input_params : list[dict] = []
        http_endpoints: list[dict] = []

        total_lines = 0

        for src_file in source_files:
            src = self.safe_read(src_file)
            if not src:
                continue
            rel = self.rel(src_file, root)
            total_lines += src.count("\n")

            _extract_classes(src, rel, classes, controllers, models, services)
            _extract_functions(src, rel, functions)
            _extract_imports(src, rel, imports)
            _extract_env_vars(src, rel, env_vars)
            _extract_auth_signals(src, rel, auth_signals)
            _extract_input_params(src, rel, input_params)
            _extract_service_deps(src, rel, service_deps)
            _extract_call_graph(src, rel, call_graph)
            _extract_form_fields(src, rel, form_fields)
            _extract_sql_queries(src, rel, sql_queries)

            if framework in (Framework.EXPRESS, Framework.FASTIFY, Framework.NESTJS, Framework.UNKNOWN):
                _extract_express_routes(src, rel, routes)
            if framework == Framework.NESTJS:
                _extract_nestjs_routes(src, rel, routes, controllers)
            if framework == Framework.NEXTJS:
                _extract_nextjs_routes(src_file, root, rel, routes)

        # Vue Router config
        if framework in (Framework.VUE, Framework.NUXTJS):
            _extract_vue_router_routes(source_files, root, routes)

        # Prisma schema
        db_schema, table_columns = _parse_prisma_schema(root)

        # Deduplicate routes by (method, path)
        seen_routes: set[tuple[str, str]] = set()
        deduped_routes = []
        for r in routes:
            key = (r.get("method", ""), r.get("path", ""))
            if key not in seen_routes:
                seen_routes.add(key)
                deduped_routes.append(r)
        routes = deduped_routes

        # Build HTTP endpoints from routes
        http_endpoints = [
            {"method": r.get("method", ""), "handler": r.get("handler", ""), "path": r.get("path", "")}
            for r in routes
        ]

        code_map = CodeMap(
            language         = lang,
            language_version = _detect_ts_version(root),
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
            form_fields      = form_fields,
            service_deps     = service_deps,
            env_vars         = env_vars,
            auth_signals     = auth_signals,
            http_endpoints   = http_endpoints,
            table_columns    = table_columns,
            input_params     = input_params,
        )

        # Persist as code_map.json (add metadata wrapper)
        payload = _code_map_to_payload(code_map)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        print(f"  [stage10/ts] Saved code_map.json → {output_path}")

        print(
            f"  [stage10/ts] Done — classes={len(classes)}, controllers={len(controllers)}, "
            f"models={len(models)}, routes={len(routes)}, functions={len(functions)}, "
            f"sql={len(sql_queries)}, env_vars={len(env_vars)}, auth={len(auth_signals)}"
        )
        return code_map


# ─── Framework detection ──────────────────────────────────────────────────────

def _detect_ts_framework(root: Path) -> Framework:
    pkg = root / "package.json"
    pkg_text = ""
    if pkg.exists():
        pkg_text = pkg.read_text(encoding="utf-8", errors="ignore")

    if (root / "next.config.js").exists() or (root / "next.config.ts").exists() or "\"next\"" in pkg_text:
        return Framework.NEXTJS
    if (root / "nuxt.config.ts").exists() or "\"nuxt\"" in pkg_text:
        return Framework.NUXTJS
    if "@nestjs/core" in pkg_text:
        return Framework.NESTJS
    if "fastify" in pkg_text:
        return Framework.FASTIFY
    if "\"vue\"" in pkg_text or "\"@vue" in pkg_text:
        return Framework.VUE
    if "\"react\"" in pkg_text or "\"react-dom\"" in pkg_text:
        return Framework.REACT
    if "express" in pkg_text:
        return Framework.EXPRESS
    return Framework.UNKNOWN


def _detect_ts_version(root: Path) -> str | None:
    pkg = root / "package.json"
    if pkg.exists():
        try:
            data = json.loads(pkg.read_text(encoding="utf-8"))
            ts_ver = (
                data.get("devDependencies", {}).get("typescript")
                or data.get("dependencies", {}).get("typescript")
            )
            if ts_ver:
                m = re.search(r"(\d+\.\d+)", ts_ver)
                if m:
                    return m.group(1)
        except Exception:
            pass
    return None


# ─── Extractor helpers ────────────────────────────────────────────────────────

def _extract_classes(src: str, rel: str, classes: list, controllers: list, models: list, services: list) -> None:
    # Standard classes
    for m in re.finditer(r"(?:export\s+)?(?:abstract\s+)?class\s+(\w+)", src):
        name = m.group(1)
        entry = {"name": name, "file": rel, "type": "class"}

        # Categorise
        decorators_before = src[max(0, m.start()-200):m.start()]
        if "@Controller" in decorators_before or name.endswith("Controller"):
            entry["type"] = "controller"
            controllers.append(entry)
        elif (
            "@Entity" in decorators_before or "@Schema" in decorators_before
            or name.endswith("Entity") or name.endswith("Model") or name.endswith("Schema")
        ):
            entry["type"] = "model"
            models.append(entry)
        elif (
            "@Injectable" in decorators_before or name.endswith("Service")
            or name.endswith("Repository") or name.endswith("Provider")
        ):
            entry["type"] = "service"
            services.append(entry)
        else:
            classes.append(entry)


def _extract_functions(src: str, rel: str, functions: list) -> None:
    # Named function declarations
    for m in re.finditer(r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(", src):
        functions.append({"name": m.group(1), "file": rel})
    # Exported const arrow functions
    for m in re.finditer(r"export\s+(?:const|let)\s+(\w+)\s*=\s*(?:async\s+)?\(", src):
        functions.append({"name": m.group(1), "file": rel, "kind": "arrow"})


def _extract_imports(src: str, rel: str, imports: list) -> None:
    for m in re.finditer(r"""import\s+.*?\s+from\s+['"]([^'"]+)['"]""", src):
        imports.append({"from": rel, "target": m.group(1)})
    # require()
    for m in re.finditer(r"""require\s*\(\s*['"]([^'"]+)['"]\s*\)""", src):
        imports.append({"from": rel, "target": m.group(1), "kind": "require"})


def _extract_env_vars(src: str, rel: str, env_vars: list) -> None:
    for m in re.finditer(r"process\.env\.(\w+)", src):
        env_vars.append({"key": m.group(1), "file": rel})


def _extract_auth_signals(src: str, rel: str, auth_signals: list) -> None:
    patterns = [
        # Server-side patterns
        (r"passport\.authenticate\s*\(", "passport"),
        (r"jwt\.verify\s*\(", "jwt"),
        (r"@(?:UseGuards|AuthGuard)", "nestjs_guard"),
        (r"verifyToken|verifyJwt|checkAuth|isAuthenticated|requireAuth", "auth_middleware"),
        (r"\.session\s*\.\s*user|req\.user\b", "session_user"),
        (r"bearerAuth|Bearer\s+token", "bearer"),
        # Firebase client-side auth
        (r"signInWithEmailAndPassword\s*\(", "firebase_email_signin"),
        (r"createUserWithEmailAndPassword\s*\(", "firebase_email_signup"),
        (r"signInWithPopup\s*\(", "firebase_oauth_popup"),
        (r"signInWithRedirect\s*\(", "firebase_oauth_redirect"),
        (r"signOut\s*\(\s*auth", "firebase_signout"),
        (r"onAuthStateChanged\s*\(", "firebase_auth_state"),
        (r"GoogleAuthProvider|GithubAuthProvider|FacebookAuthProvider", "firebase_oauth_provider"),
        # Supabase auth
        (r"supabase\.auth\.signIn|supabase\.auth\.signUp", "supabase_auth"),
        (r"supabase\.auth\.signOut", "supabase_signout"),
        # Clerk
        (r"useAuth\s*\(\)|SignIn|SignUp|useUser\s*\(\)", "clerk_auth"),
        # NextAuth
        (r"signIn\s*\(\s*['\"]|useSession\s*\(\)|getServerSession\s*\(", "nextauth"),
    ]
    for pattern, kind in patterns:
        if re.search(pattern, src, re.IGNORECASE):
            auth_signals.append({"file": rel, "type": kind})


def _extract_input_params(src: str, rel: str, input_params: list) -> None:
    # req.body.field, req.query.field, req.params.field
    for m in re.finditer(r"req\.(body|query|params)\.(\w+)", src):
        input_params.append({"source": m.group(1), "key": m.group(2), "file": rel})
    # Destructured: const { email, password } = req.body
    for m in re.finditer(r"const\s*\{([^}]+)\}\s*=\s*req\.(body|query|params)", src):
        for field in re.findall(r"\b(\w+)\b", m.group(1)):
            input_params.append({"source": m.group(2), "key": field, "file": rel})


def _extract_service_deps(src: str, rel: str, service_deps: list) -> None:
    # NestJS constructor injection: constructor(private readonly fooService: FooService)
    for m in re.finditer(
        r"constructor\s*\([^)]*?(?:private|protected|readonly|public)\s+\w+:\s*(\w+)", src
    ):
        service_deps.append({"file": rel, "depends_on": m.group(1)})


def _extract_call_graph(src: str, rel: str, call_graph: list) -> None:
    # Approximate: find method calls inside functions
    # this.someService.someMethod(
    for m in re.finditer(r"this\.(\w+)\.(\w+)\s*\(", src):
        call_graph.append({"caller_file": rel, "callee": f"{m.group(1)}.{m.group(2)}"})


def _extract_form_fields(src: str, rel: str, form_fields: list) -> None:
    # Zod / Yup schema fields: z.object({ email: z.string(), password: ... })
    for m in re.finditer(r"z\.object\s*\(\s*\{([^}]{0,500})\}", src):
        for field_m in re.finditer(r"(\w+)\s*:", m.group(1)):
            form_fields.append({"file": rel, "field": field_m.group(1), "source": "zod"})


def _extract_sql_queries(src: str, rel: str, sql_queries: list) -> None:
    # Raw SQL strings — require a second SQL keyword to avoid matching UI strings
    # e.g. "Select Role" is NOT SQL; "SELECT id FROM users" IS SQL
    for m in re.finditer(
        r"['\"`](SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\b[^'\"`;]{0,300}",
        src, re.IGNORECASE
    ):
        snippet = m.group(0)
        if not re.search(r"\b(FROM|WHERE|INTO|SET|TABLE|JOIN|VALUES)\b", snippet, re.IGNORECASE):
            continue   # UI string, not real SQL
        sql_queries.append({"file": rel, "operation": m.group(1).upper(), "raw": snippet[:120]})
    # Prisma: prisma.user.create / findMany / update / delete
    for m in re.finditer(r"prisma\.(\w+)\.(create|findMany|findFirst|update|delete|upsert)\b", src):
        sql_queries.append({"file": rel, "orm": "prisma", "table": m.group(1), "operation": m.group(2)})
    # TypeORM: repository.save / find / delete
    for m in re.finditer(r"(?:repository|repo)\.(?:save|find|findOne|delete|update|insert)\b", src):
        sql_queries.append({"file": rel, "orm": "typeorm", "operation": m.group(0)})
    # Firestore: collection(db, 'name') / collection(db, 'name').doc() etc.
    # Also: addDoc/getDocs/setDoc/updateDoc/deleteDoc(collection(db, 'name'), ...)
    for m in re.finditer(
        r"collection\s*\([^,)]+,\s*['\"`](\w+)['\"`]\s*\)",
        src
    ):
        sql_queries.append({"file": rel, "orm": "firestore", "table": m.group(1), "operation": "collection"})
    # Chained form: addDoc(collection(db, 'tasks'), data)
    for m in re.finditer(
        r"(addDoc|setDoc|updateDoc|deleteDoc|getDocs|getDoc|getCountFromServer)\s*\(\s*collection\s*\([^,)]+,\s*['\"`](\w+)['\"`]",
        src
    ):
        sql_queries.append({"file": rel, "orm": "firestore", "table": m.group(2), "operation": m.group(1)})
    # Pre-referenced form: const ref = collection(db, 'tasks'); addDoc(ref, data)
    # Table name is unknown but we still record the write so it counts in coverage.
    _seen_chained = {m.start() for m in re.finditer(
        r"(addDoc|setDoc|updateDoc|deleteDoc)\s*\(\s*collection\s*\(",
        src
    )}
    for m in re.finditer(
        r"(addDoc|setDoc|updateDoc|deleteDoc)\s*\(\s*(\w+)",
        src
    ):
        if m.start() in _seen_chained:
            continue   # already captured by chained form above
        sql_queries.append({"file": rel, "orm": "firestore", "table": m.group(2), "operation": m.group(1)})

    # React Query / SWR Mutations: useMutation(...)
    # Note: useMutation is a broad signal — it wraps any async function, not
    # exclusively DB writes.  It is recorded as a write because in Firebase
    # projects it almost always wraps a Firestore mutation.
    for m in re.finditer(r"\buseMutation\s*\(", src):
        sql_queries.append({"file": rel, "orm": "hooks", "table": "(mutation)", "operation": "useMutation"})

    # Firebase Transactions / Batches — use call-site regex to avoid matching
    # comments, imports, or unrelated identifiers that contain the substring.
    if re.search(r"\brunTransaction\s*\(", src):
        sql_queries.append({"file": rel, "orm": "firestore", "table": "(transaction)", "operation": "runTransaction"})
    if re.search(r"\bwriteBatch\s*\(", src) or re.search(r"\bbatch\s*\.\s*commit\s*\(", src):
        sql_queries.append({"file": rel, "orm": "firestore", "table": "(batch)", "operation": "writeBatch"})


# ─── Route extractors ─────────────────────────────────────────────────────────

_EXPRESS_METHODS = {"get", "post", "put", "patch", "delete", "all", "use"}

def _extract_express_routes(src: str, rel: str, routes: list) -> None:
    # app.get('/path', handler) / router.post('/path', async (req,res) => ...)
    for m in re.finditer(
        r"""(?:app|router)\.(get|post|put|patch|delete|all|use)\s*\(\s*['"`]([^'"`]+)['"`]""",
        src, re.IGNORECASE
    ):
        method, path = m.group(1).upper(), m.group(2)
        routes.append({"method": method, "path": path, "file": rel, "kind": "express"})


def _extract_nestjs_routes(src: str, rel: str, routes: list, controllers: list) -> None:
    # @Controller('prefix') + @Get('sub') → full path
    ctrl_prefix = ""
    m = re.search(r"@Controller\s*\(\s*['`\"]([^'`\"]*)", src)
    if m:
        ctrl_prefix = "/" + m.group(1).lstrip("/")

    for m in re.finditer(
        r"@(Get|Post|Put|Patch|Delete|All)\s*\(\s*['`\"]?([^'`\")\s]*)?",
        src
    ):
        method = m.group(1).upper()
        sub = m.group(2) or ""
        path = ctrl_prefix + ("/" + sub.lstrip("/") if sub else "")
        routes.append({"method": method, "path": path or "/", "file": rel, "kind": "nestjs"})


def _extract_nextjs_routes(src_file: Path, root: Path, rel: str, routes: list) -> None:
    """
    Next.js file-based routing:
    pages/api/users.ts            → /api/users  (API route, all methods)
    app/api/users/route.ts        → /api/users  (API route, exported GET/POST/…)
    app/prism/page.tsx            → /prism       (page route, GET)
    pages/dashboard/index.tsx     → /dashboard   (page route, GET)
    """
    parts = src_file.parts

    def _normalise_path(path_parts: tuple) -> str:
        path = "/" + "/".join(path_parts).replace("\\", "/")
        path = re.sub(r"\(([^)]+)\)", "", path)   # remove (group) layout segments
        path = re.sub(r"\[(\w+)\]", r":\1", path)  # [id] → :id
        path = re.sub(r"//+", "/", path)            # collapse double slashes
        return path.rstrip("/") or "/"

    # pages/api/** — API routes
    if "pages" in parts:
        idx = list(parts).index("pages")
        api_parts = parts[idx+1:]
        if api_parts and api_parts[0] == "api":
            path = _normalise_path(api_parts)
            path = re.sub(r"\.(ts|tsx|js|jsx)$", "", path)
            routes.append({"method": "ANY", "path": path, "file": rel, "kind": "nextjs_pages_api"})
        elif src_file.name in ("page.tsx", "page.ts", "page.jsx", "page.js",
                               "index.tsx", "index.ts", "index.jsx", "index.js"):
            # pages/** page routes
            page_parts = parts[idx+1:]
            path = _normalise_path(page_parts)
            path = re.sub(r"\.(ts|tsx|js|jsx)$", "", path)
            path = re.sub(r"/index$", "", path)
            routes.append({"method": "GET", "path": path or "/", "file": rel, "kind": "nextjs_page"})

    # app/** (App Router)
    if "app" in parts:
        idx = list(parts).index("app")
        # API routes: app/**/route.ts
        if src_file.name in ("route.ts", "route.js", "route.tsx", "route.jsx"):
            path_parts = parts[idx+1:-1]
            path = _normalise_path(path_parts)
            src_content = LanguageParser.safe_read(src_file) or ""
            dir_rel = str(Path(rel).parent).replace("\\", "/")
            
            methods_found = False
            for method in re.findall(
                r"export\s+(?:const\s+|async\s+)?(?:function\s+)?(GET|POST|PUT|PATCH|DELETE|HEAD)\b",
                src_content
            ):
                methods_found = True
                routes.append({"method": method, "path": path or "/", "file": rel, "dir": dir_rel, "kind": "nextjs_app_api"})
            
            if not methods_found:
                routes.append({"method": "ANY", "path": path or "/", "file": rel, "dir": dir_rel, "kind": "nextjs_app_api"})
                
        # Page routes: app/**/page.tsx
        elif src_file.name in ("page.tsx", "page.ts", "page.jsx", "page.js"):
            path_parts = parts[idx+1:-1]
            path = _normalise_path(path_parts)
            dir_rel = str(Path(rel).parent).replace("\\", "/")
            routes.append({"method": "GET", "path": path or "/", "file": rel, "dir": dir_rel, "kind": "nextjs_page"})


def _extract_vue_router_routes(source_files: list[Path], root: Path, routes: list) -> None:
    """Parse Vue Router config: { path: '/foo', component: Bar }"""
    router_files = [
        f for f in source_files
        if re.search(r"router", f.name, re.IGNORECASE)
        or f.name in ("routes.ts", "routes.js", "index.ts", "index.js")
    ]
    for rf in router_files:
        src = LanguageParser.safe_read(rf) or ""
        if "createRouter" not in src and "Router.create" not in src and "path:" not in src:
            continue
        for m in re.finditer(r"""path\s*:\s*['"`]([^'"`]+)['"`]""", src):
            routes.append({
                "method": "ANY",
                "path": m.group(1),
                "file": LanguageParser.rel(rf, root),
                "kind": "vue_router",
            })


# ─── Prisma schema parser ─────────────────────────────────────────────────────

def _parse_prisma_schema(root: Path) -> tuple[list[dict], list[dict]]:
    db_schema: list[dict] = []
    table_columns: list[dict] = []

    schema_file = root / "prisma" / "schema.prisma"
    if not schema_file.exists():
        # Search recursively
        candidates = list(root.rglob("schema.prisma"))[:1]
        if candidates:
            schema_file = candidates[0]
        else:
            return db_schema, table_columns

    src = schema_file.read_text(encoding="utf-8", errors="ignore")

    # model Foo { field type ... }
    for m in re.finditer(r"model\s+(\w+)\s*\{([^}]+)\}", src, re.DOTALL):
        model_name = m.group(1)
        body = m.group(2)
        columns = []
        for field_m in re.finditer(r"^\s+(\w+)\s+(\w+)", body, re.MULTILINE):
            fname, ftype = field_m.group(1), field_m.group(2)
            if fname.startswith("@@") or fname.startswith("//"):
                continue
            columns.append({
                "name": fname,
                "type": ftype,
                "nullable": "?" in body[field_m.start():field_m.end()+5],
            })

        table_name = _snake(model_name)
        db_schema.append({"table": table_name, "model": model_name, "source": "prisma"})
        table_columns.append({
            "table": table_name,
            "columns": columns,
        })

    return db_schema, table_columns


def _snake(name: str) -> str:
    """CamelCase → snake_case, pluralise (naïve)."""
    s = re.sub(r"([A-Z])", r"_\1", name).lower().lstrip("_")
    return s + "s" if not s.endswith("s") else s


# ─── Config file finder ───────────────────────────────────────────────────────

def _find_config_files(root: Path) -> list[str]:
    names = {
        "tsconfig.json", "jsconfig.json", "package.json",
        ".env", ".env.local", ".env.production",
        "next.config.js", "next.config.ts", "nuxt.config.ts",
        "vite.config.ts", "webpack.config.js",
    }
    found = []
    for name in names:
        p = root / name
        if p.exists():
            found.append(name)
    return found


# ─── Payload serialiser ───────────────────────────────────────────────────────

def _code_map_to_payload(cm: CodeMap) -> dict:
    """Convert CodeMap to a plain dict for code_map.json persistence."""
    import dataclasses
    def _ser(obj: Any) -> Any:
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {k: _ser(v) for k, v in dataclasses.asdict(obj).items()}
        if isinstance(obj, list): return [_ser(i) for i in obj]
        if isinstance(obj, dict): return {k: _ser(v) for k, v in obj.items()}
        if hasattr(obj, "value"): return obj.value   # Enum
        return obj
    d = _ser(cm)
    # Add language/framework at top level for easy reading
    d["_parser"] = "typescript"
    return d

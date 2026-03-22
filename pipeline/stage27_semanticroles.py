"""
pipeline/stage27_semanticroles.py — Semantic Role Tagging (Stage 2.7)

Fully static stage that runs after Stage 2.5 (behavior graph) and before
Stage 2.8 (action clustering).  Tags every code element with a business-domain
role so downstream stages operate on semantically typed signals rather than
raw structural data.

Roles assigned
--------------
  Actions (one per HTTP endpoint / controller method / route handler):
    BUSINESS_ACTION     — auth-guarded, entity-writing, business-domain path
    AUTH_ACTION         — login / logout / register / password / OAuth flows
    CRUD_ACTION         — entity read/write with no significant business logic
    INTEGRATION_ACTION  — calls an external system (payment, email, storage…)
    INFRASTRUCTURE_ACTION — logging, caching, queuing, export, audit, utility

  External systems (zero or more per project):
    PAYMENT, EMAIL, STORAGE, SMS_PUSH, AUTH_OAUTH, MONITORING, CRM, ERP, INFRA

Classification algorithm
-------------------------
Each action starts as UNKNOWN and accumulates weighted signals:

  +3  Route path is a business-domain path (not /health /ping /config /webhook)
  +2  Has an auth guard (session check, middleware, role guard)
  +2  Writes to a domain entity table (not log_*, audit_*, cache_*)
  +1  Route path contains login/logout/register → bias toward AUTH
  -2  Class/file path is in Jobs/, Events/, Listeners/, Notifications/, Console/
  -2  Only writes to infrastructure tables (log_*, audit_*, cache_*, jobs, *)
  -1  Class name matches infrastructure pattern (Log, Queue, Mail, Helper…)

Score → role mapping:
  AUTH path keywords present                → AUTH_ACTION  (overrides score)
  Integration signals present               → INTEGRATION_ACTION (overrides)
  score ≥ 3                                 → BUSINESS_ACTION
  score ≥ 1                                 → CRUD_ACTION
  score < 1                                 → INFRASTRUCTURE_ACTION

External system detection
--------------------------
Combines three independent signals with deduplification:
  1. env_var keys  (STRIPE_KEY, AWS_ACCESS_KEY_ID, MAILGUN_DOMAIN …)
  2. service_deps  (class depends on Guzzle, Stripe, Pusher …)
  3. class/namespace names  (StripeController, MailgunService …)

Downstream consumers
---------------------
  Stage 2.8  Action Clustering    — skips INFRASTRUCTURE files; weights BUSINESS
  Stage 4    Domain Model          — gets pre-labelled actor hints
  Stage 4.5  Flow Extraction       — uses BUSINESS_ACTION + actor as flow seeds
  Stage 9    Knowledge Graph       — typed nodes from the start

Output
------
  2.7_semanticroles/semantic_roles.json   — full index
  ctx.semantic_roles  set on PipelineContext
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from context import (
    ActionTag,
    ExternalSystem,
    PipelineContext,
    SemanticRoleIndex,
)

# ── Output file ───────────────────────────────────────────────────────────────
ROLES_FILE = "semantic_roles.json"

# ── Action role constants ─────────────────────────────────────────────────────
ROLE_BUSINESS      = "BUSINESS_ACTION"
ROLE_AUTH          = "AUTH_ACTION"
ROLE_CRUD          = "CRUD_ACTION"
ROLE_INTEGRATION   = "INTEGRATION_ACTION"
ROLE_INFRA         = "INFRASTRUCTURE_ACTION"
ROLE_UNKNOWN       = "UNKNOWN"

# ── External system categories ────────────────────────────────────────────────
CAT_PAYMENT   = "PAYMENT"
CAT_EMAIL     = "EMAIL"
CAT_STORAGE   = "STORAGE"
CAT_SMS_PUSH  = "SMS_PUSH"
CAT_AUTH_OAUTH= "AUTH_OAUTH"
CAT_MONITORING= "MONITORING"
CAT_CRM       = "CRM"
CAT_ERP       = "ERP"
CAT_INFRA     = "INFRA"     # Redis, queues, search engines


# ─── Infrastructure patterns ──────────────────────────────────────────────────

# Class/namespace name fragments that indicate infrastructure
_INFRA_NAME_RE = re.compile(
    r"log|logger|audit|cache|queue|job|event|listener|notification|"
    r"mail(?!box)|export|import|helper|util|command|console|cron|"
    r"scheduler|broadcast|observer|seeder|migration|middleware|"
    r"exception|error|monitor|health|ping|metric|telemetry|"
    r"backup|cleanup|garbage|fixture|factory|fake",
    re.IGNORECASE,
)

# Infrastructure directory path fragments
_INFRA_PATH_RE = re.compile(
    r"[/\\](Jobs|Events|Listeners|Notifications|Console|"
    r"Middleware|Exceptions|Mail|Exports|Imports|Commands|"
    r"Observers|Seeders|Migrations|Factories|Tests|test)[/\\]",
    re.IGNORECASE,
)

# Infrastructure DB table patterns (write-only = no business value)
_INFRA_TABLE_RE = re.compile(
    r"^(log_|audit_|cache|jobs|failed_jobs|queue|sessions|"
    r"password_resets?|telescope_|pulse_|personal_access_tokens?|"
    r"oauth_|migrations?|nova_|horizon_|sugar_|upgrade_|"
    r"activity_log|action_events)",
    re.IGNORECASE,
)

# Auth-related route/handler name patterns
_AUTH_PATH_RE = re.compile(
    r"(login|logout|register|signup|sign.?in|sign.?up|"
    r"forgot.?pass|reset.?pass|password|verify|"
    r"oauth|token|refresh|auth|sso|saml|ldap)",
    re.IGNORECASE,
)

# Non-business route paths (health checks, config, ping, webhooks)
_NOISE_PATH_RE = re.compile(
    r"^/(health|ping|status|metrics?|monitor|config|"
    r"version|info|debug|test|_|api-docs?|swagger|"
    r"telescope|horizon|nova|livewire|sanctum|broadcasting|"
    r"storage|assets?|css|js|images?|fonts?)",
    re.IGNORECASE,
)


# ─── External system detection tables ─────────────────────────────────────────

# Each entry: (category, [env_key_prefixes], [name_keywords])
_EXT_SYSTEM_SIGNATURES: list[tuple[str, list[str], list[str]]] = [
    (CAT_PAYMENT,   ["STRIPE_", "PAYPAL_", "BRAINTREE_", "SQUARE_", "RAZORPAY_",
                     "MOLLIE_", "ADYEN_", "CHECKOUT_"],
                    ["stripe", "paypal", "braintree", "razorpay", "mollie",
                     "adyen", "checkout", "payment.gateway", "payu"]),

    (CAT_EMAIL,     ["MAIL_", "MAILGUN_", "SENDGRID_", "SES_", "POSTMARK_",
                     "SPARKPOST_", "MAILCHIMP_", "SMTP_"],
                    ["mailgun", "sendgrid", "postmark", "sparkpost", "mailchimp",
                     "aws.ses", "smtp.mailer"]),

    (CAT_STORAGE,   ["AWS_", "S3_", "GCS_", "AZURE_", "DO_SPACES_",
                     "MINIO_", "BACKBLAZE_", "CLOUDINARY_"],
                    ["s3bucket", "awss3", "gcs", "azureblob", "cloudinary",
                     "imagekit", "imgix", "digitalocean.spaces"]),

    (CAT_SMS_PUSH,  ["TWILIO_", "VONAGE_", "NEXMO_", "PUSHER_",
                     "FIREBASE_", "ONE_SIGNAL_", "ONESIGNAL_",
                     "TELEGRAM_", "SLACK_"],
                    ["twilio", "vonage", "nexmo", "pusher", "firebase",
                     "onesignal", "telegram.bot", "slack.webhook"]),

    (CAT_AUTH_OAUTH,["GOOGLE_", "FACEBOOK_", "GITHUB_", "TWITTER_",
                     "AUTH0_", "OKTA_", "KEYCLOAK_", "SOCIALITE_",
                     "OAUTH_", "LDAP_"],
                    ["google.oauth", "facebook.login", "auth0", "okta",
                     "keycloak", "socialite", "passport", "ldap"]),

    (CAT_MONITORING,["SENTRY_", "BUGSNAG_", "ROLLBAR_", "DATADOG_",
                     "NEWRELIC_", "LOGROCKET_", "RAYGUN_"],
                    ["sentry", "bugsnag", "rollbar", "datadog", "newrelic",
                     "logrocket", "raygun"]),

    (CAT_CRM,       ["SALESFORCE_", "HUBSPOT_", "ZOHO_", "PIPEDRIVE_",
                     "FRESHDESK_", "ZENDESK_", "INTERCOM_"],
                    ["salesforce", "hubspot", "zoho", "pipedrive",
                     "freshdesk", "zendesk", "intercom"]),

    (CAT_ERP,       ["QUICKBOOKS_", "XERO_", "SAGE_", "FRESHBOOKS_",
                     "NETSUITE_", "SAP_"],
                    ["quickbooks", "xero", "sage", "freshbooks", "netsuite"]),

    (CAT_INFRA,     ["REDIS_", "MEMCACHED_", "RABBITMQ_", "KAFKA_",
                     "ELASTIC_", "ELASTICSEARCH_", "MONGO_", "MONGODB_"],
                    ["redis", "memcached", "rabbitmq", "kafka",
                     "elasticsearch", "mongodb"]),
]


# ─── Public Entry Point ────────────────────────────────────────────────────────

def run(ctx: PipelineContext) -> None:
    """Stage 2.7 entry point."""
    roles_path = ctx.output_path(ROLES_FILE)

    # ── Resume check ─────────────────────────────────────────────────────────
    if ctx.is_stage_done("stage27_semanticroles") and Path(roles_path).exists():
        ctx.semantic_roles = _load(roles_path)
        r = ctx.semantic_roles
        print(f"  [stage27] Already completed — "
              f"{len(r.actions)} action(s), "
              f"{len(r.external_systems)} external system(s).")
        return

    cm = ctx.code_map
    if cm is None:
        print("  [stage27] ⚠️  No code_map — skipping.")
        return

    print("  [stage27] Classifying actions and detecting external systems ...")

    # ── Build helper indices ──────────────────────────────────────────────────
    # file → {tables it writes to}
    write_tables_by_file: dict[str, set[str]] = {}
    read_tables_by_file:  dict[str, set[str]] = {}
    for q in (cm.sql_queries or []):
        f   = q.get("file", "")
        tbl = q.get("table", "")
        op  = q.get("operation", "")
        if not f or not tbl or tbl == "UNKNOWN":
            continue
        if op in ("INSERT", "UPDATE", "DELETE", "REPLACE"):
            write_tables_by_file.setdefault(f, set()).add(tbl)
        elif op == "SELECT":
            read_tables_by_file.setdefault(f, set()).add(tbl)

    # file → route info (first route wins)
    route_by_file:  dict[str, dict] = {}
    for r in (cm.routes or []):
        handler = r.get("handler", "") or ""
        f       = r.get("file", "")
        if not f or r.get("method") == "GROUP":
            continue
        # Map handler class → file  (handler may be "OrderController@store")
        if f not in route_by_file:
            route_by_file[f] = r

    # file → auth signals
    auth_by_file: dict[str, list[dict]] = {}
    for sig in (cm.auth_signals or []):
        auth_by_file.setdefault(sig.get("file", ""), []).append(sig)

    # ── Classify actions from http_endpoints (most reliable source) ───────────
    action_tags: list[ActionTag] = []
    seen_symbols: set[str] = set()

    for ep in (cm.http_endpoints or []):
        tag = _classify_endpoint(ep, write_tables_by_file,
                                 read_tables_by_file, route_by_file,
                                 auth_by_file)
        if tag and tag.symbol not in seen_symbols:
            seen_symbols.add(tag.symbol)
            action_tags.append(tag)

    # ── Also classify execution-path entry points not yet tagged ──────────────
    for ep in (cm.execution_paths or []):
        f      = ep.get("file", "")
        symbol = Path(f).name if f else ""
        if not symbol or symbol in seen_symbols:
            continue
        tag = _classify_exec_path(ep, write_tables_by_file,
                                   read_tables_by_file, auth_by_file)
        if tag:
            seen_symbols.add(symbol)
            action_tags.append(tag)

    # ── Detect external systems ───────────────────────────────────────────────
    env_keys   = [e.get("key", "") for e in (cm.env_vars or [])]
    dep_names  = [
        f"{d.get('class','')} {d.get('dep_class','')}".lower()
        for d in (cm.service_deps or [])
    ]
    class_names = [
        f"{s.get('name','')} {s.get('fqn','')} {s.get('namespace','')}".lower()
        for s in ((cm.services or []) + (cm.controllers or []))
    ]
    ext_systems = _detect_external_systems(env_keys, dep_names, class_names)

    # ── Build summary + file lists ────────────────────────────────────────────
    from collections import Counter
    role_counts   = Counter(t.role for t in action_tags)
    infra_files   = sorted({t.file for t in action_tags if t.role == ROLE_INFRA})
    business_files= sorted({t.file for t in action_tags
                             if t.role in (ROLE_BUSINESS, ROLE_CRUD, ROLE_AUTH)})

    from datetime import datetime
    index = SemanticRoleIndex(
        actions          = action_tags,
        external_systems = ext_systems,
        role_summary     = dict(role_counts),
        infra_files      = infra_files,
        business_files   = business_files,
        generated_at     = datetime.utcnow().isoformat(),
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    _save(index, roles_path)
    ctx.semantic_roles = index
    ctx.stage("stage27_semanticroles").mark_completed(roles_path)
    ctx.save()

    print(f"  [stage27] {len(action_tags)} action(s) classified:")
    for role, cnt in sorted(role_counts.items(), key=lambda x: -x[1]):
        print(f"    {role:<24}: {cnt}")
    if ext_systems:
        print(f"  [stage27] {len(ext_systems)} external system(s) detected: "
              + ", ".join(f"{s.name} ({s.category})" for s in ext_systems))
    print(f"  [stage27] Saved → {roles_path}")


# ─── Action Classification ────────────────────────────────────────────────────

def _classify_endpoint(
    ep: dict,
    write_tables: dict[str, set[str]],
    read_tables:  dict[str, set[str]],
    routes:       dict[str, dict],
    auth_sigs:    dict[str, list[dict]],
) -> ActionTag | None:
    """Classify a single http_endpoint dict."""
    handler = ep.get("handler") or ""
    f       = ep.get("file", "")
    method  = "/".join(ep.get("accepts", []))

    # Use handler as symbol; fall back to filename
    symbol  = handler if handler else Path(f).name

    # Derive route path from routes index or handler
    route   = routes.get(f, {})
    path    = route.get("path", "") or ""

    writes  = write_tables.get(f, set())
    reads   = read_tables.get(f, set())
    auths   = auth_sigs.get(f, [])

    return _score_and_tag(symbol, f, method, path, writes, reads, auths)


def _classify_exec_path(
    ep: dict,
    write_tables: dict[str, set[str]],
    read_tables:  dict[str, set[str]],
    auth_sigs:    dict[str, list[dict]],
) -> ActionTag | None:
    """Classify a single execution_path entry."""
    f      = ep.get("file", "")
    symbol = Path(f).name
    auths  = auth_sigs.get(f, [])

    # Reconstruct auth from execution_path auth_guard field
    if ep.get("auth_guard") and not auths:
        auths = [{"type": "session_check"}]

    writes = write_tables.get(f, set())
    reads  = read_tables.get(f, set())

    return _score_and_tag(symbol, f, "", "", writes, reads, auths)


def _score_and_tag(
    symbol:  str,
    file:    str,
    method:  str,
    path:    str,
    writes:  set[str],
    reads:   set[str],
    auths:   list[dict],
) -> ActionTag | None:
    """Core scoring logic — returns ActionTag or None if symbol is empty."""
    if not symbol:
        return None

    signals: list[str] = []
    score = 0.0

    # ── Infrastructure path override ─────────────────────────────────────────
    if _INFRA_PATH_RE.search(file):
        signals.append("infra_directory")
        score -= 2

    # ── Infrastructure name override ─────────────────────────────────────────
    name_lower = (symbol + " " + Path(file).name).lower()
    if _INFRA_NAME_RE.search(name_lower):
        signals.append("infra_name_pattern")
        score -= 1

    # ── Auth path → AUTH_ACTION (early override) ──────────────────────────────
    is_auth_path = bool(path and _AUTH_PATH_RE.search(path))
    if is_auth_path:
        signals.append("auth_path_keyword")

    # ── Noise path → penalize ────────────────────────────────────────────────
    if path and _NOISE_PATH_RE.search(path):
        signals.append("noise_path")
        score -= 1

    # ── Auth guard signal ────────────────────────────────────────────────────
    if auths:
        signals.append("has_auth_guard")
        score += 2

    # ── Entity write signal ──────────────────────────────────────────────────
    domain_writes = {t for t in writes if not _INFRA_TABLE_RE.match(t)}
    infra_writes  = {t for t in writes if _INFRA_TABLE_RE.match(t)}
    if domain_writes:
        signals.append(f"entity_write({','.join(sorted(domain_writes)[:3])})")
        score += 2
    if infra_writes and not domain_writes:
        signals.append("infra_table_write_only")
        score -= 2

    # ── Route path quality ────────────────────────────────────────────────────
    if path and not _NOISE_PATH_RE.search(path) and path not in ("", "(group)"):
        signals.append("business_route")
        score += 1

    # ── HTTP method signal ────────────────────────────────────────────────────
    if "POST" in method or "PUT" in method or "PATCH" in method or "DELETE" in method:
        signals.append("mutating_http_method")
        score += 1

    # ── Integration: check for external class patterns ───────────────────────
    is_integration = _is_integration_symbol(symbol, file)
    if is_integration:
        signals.append("integration_pattern")

    # ── Determine role ────────────────────────────────────────────────────────
    if is_auth_path:
        role = ROLE_AUTH
        confidence = 0.90
    elif is_integration:
        role = ROLE_INTEGRATION
        confidence = 0.85
    elif score >= 3:
        role = ROLE_BUSINESS
        confidence = min(0.95, 0.70 + score * 0.05)
    elif score >= 1:
        role = ROLE_CRUD
        confidence = 0.70
    else:
        role = ROLE_INFRA
        confidence = min(0.90, 0.60 + abs(score) * 0.05)

    # ── Infer actor ───────────────────────────────────────────────────────────
    actor = _infer_actor(path, symbol, auths)

    return ActionTag(
        symbol      = symbol,
        file        = file,
        role        = role,
        confidence  = round(confidence, 2),
        signals     = signals,
        actor       = actor,
        entities    = sorted((domain_writes | {t for t in reads if not _INFRA_TABLE_RE.match(t)})
                              - infra_writes)[:5],
        http_method = method,
        route_path  = path,
    )


def _is_integration_symbol(symbol: str, file: str) -> bool:
    """True if class/file name strongly suggests an external API client."""
    combined = (symbol + " " + file).lower()
    patterns = [
        r"(gateway|client|adapter|connector|provider|driver|webhook|sdk)",
        r"(stripe|paypal|braintree|twilio|vonage|nexmo|mailgun|sendgrid|"
        r"postmark|sparkpost|aws|s3bucket|cloudinary|pusher|firebase|"
        r"sentry|bugsnag|salesforce|hubspot|zoho|quickbooks|xero|"
        r"guzzle|httpclient|restclient|soapclient|grpc)",
    ]
    return any(re.search(p, combined, re.IGNORECASE) for p in patterns)


def _infer_actor(path: str, symbol: str, auths: list[dict]) -> str:
    """Infer the primary actor for a business action from route + auth context."""
    combined = (path + " " + symbol).lower()

    # Admin paths
    if re.search(r"[/._-](admin|backend|staff|manager|superuser)", combined):
        return "Admin"

    # API clients (no session, token-based)
    has_session  = any(s.get("type") in ("session_check", "get_user") for s in auths)
    has_api_auth = any(s.get("type") in ("api_token", "jwt", "oauth_token")
                       for s in auths)
    if has_api_auth and not has_session:
        return "API Client"

    # Public / unauthenticated
    if not auths:
        return "Guest"

    return "Authenticated User"


# ─── External System Detection ────────────────────────────────────────────────

def _detect_external_systems(
    env_keys:    list[str],
    dep_names:   list[str],
    class_names: list[str],
) -> list[ExternalSystem]:
    """
    Detect external systems by correlating env var keys, service dependencies,
    and class/namespace names against known signature tables.
    """
    env_upper  = [k.upper() for k in env_keys]
    deps_lower = " ".join(dep_names).lower()
    cls_lower  = " ".join(class_names).lower()

    systems: list[ExternalSystem] = []

    for category, env_prefixes, name_keywords in _EXT_SYSTEM_SIGNATURES:
        matched_env   = [k for k in env_upper
                          if any(k.startswith(p) for p in env_prefixes)]
        matched_names = [kw for kw in name_keywords
                          if kw in cls_lower]
        matched_deps  = [kw for kw in name_keywords
                          if kw in deps_lower]

        if not (matched_env or matched_names or matched_deps):
            continue

        # Derive a display name from the first strong signal
        name = _ext_system_name(category, matched_env, matched_names, matched_deps)

        detected_via = []
        if matched_env:    detected_via.append("env_var")
        if matched_names:  detected_via.append("class_name")
        if matched_deps:   detected_via.append("service_dep")

        systems.append(ExternalSystem(
            name         = name,
            category     = category,
            env_keys     = matched_env[:5],
            class_hints  = matched_names[:3],
            dep_hints    = matched_deps[:3],
            detected_via = detected_via,
        ))

    return systems


def _ext_system_name(
    category:      str,
    matched_env:   list[str],
    matched_names: list[str],
    matched_deps:  list[str],
) -> str:
    """Derive a human-readable system name from the strongest evidence."""
    # Category-to-default name
    _DEFAULTS = {
        CAT_PAYMENT:    "Payment Gateway",
        CAT_EMAIL:      "Email Service",
        CAT_STORAGE:    "Cloud Storage",
        CAT_SMS_PUSH:   "SMS/Push Provider",
        CAT_AUTH_OAUTH: "OAuth Provider",
        CAT_MONITORING: "Error Monitoring",
        CAT_CRM:        "CRM System",
        CAT_ERP:        "ERP/Accounting",
        CAT_INFRA:      "Infrastructure Service",
    }
    # Try to get a specific name from env key prefix
    _ENV_TO_NAME = {
        "STRIPE_": "Stripe", "PAYPAL_": "PayPal", "BRAINTREE_": "Braintree",
        "RAZORPAY_": "Razorpay", "MOLLIE_": "Mollie", "ADYEN_": "Adyen",
        "MAILGUN_": "Mailgun", "SENDGRID_": "SendGrid", "SES_": "AWS SES",
        "POSTMARK_": "Postmark", "SPARKPOST_": "SparkPost",
        "AWS_": "AWS", "S3_": "AWS S3", "GCS_": "Google Cloud Storage",
        "AZURE_": "Azure", "CLOUDINARY_": "Cloudinary",
        "TWILIO_": "Twilio", "VONAGE_": "Vonage", "NEXMO_": "Nexmo",
        "PUSHER_": "Pusher", "FIREBASE_": "Firebase",
        "SENTRY_": "Sentry", "BUGSNAG_": "Bugsnag", "DATADOG_": "DataDog",
        "SALESFORCE_": "Salesforce", "HUBSPOT_": "HubSpot", "ZOHO_": "Zoho",
        "QUICKBOOKS_": "QuickBooks", "XERO_": "Xero",
        "REDIS_": "Redis", "ELASTICSEARCH_": "Elasticsearch",
        "GOOGLE_": "Google OAuth", "FACEBOOK_": "Facebook Login",
        "AUTH0_": "Auth0", "OKTA_": "Okta",
    }
    for env_key in matched_env:
        for prefix, name in _ENV_TO_NAME.items():
            if env_key.startswith(prefix):
                return name
    for kw in matched_names + matched_deps:
        # Capitalise the keyword as a name
        return kw.replace(".", " ").title()
    return _DEFAULTS.get(category, category)


# ─── Persistence ──────────────────────────────────────────────────────────────

def _save(index: SemanticRoleIndex, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(asdict(index), fh, indent=2, ensure_ascii=False)


def _load(path: str) -> SemanticRoleIndex:
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    return SemanticRoleIndex(
        actions=[
            ActionTag(
                symbol      = a["symbol"],
                file        = a["file"],
                role        = a["role"],
                confidence  = a.get("confidence", 0.75),
                signals     = a.get("signals", []),
                actor       = a.get("actor", ""),
                entities    = a.get("entities", []),
                http_method = a.get("http_method", ""),
                route_path  = a.get("route_path", ""),
            )
            for a in d.get("actions", [])
        ],
        external_systems=[
            ExternalSystem(
                name         = s["name"],
                category     = s["category"],
                env_keys     = s.get("env_keys", []),
                class_hints  = s.get("class_hints", []),
                dep_hints    = s.get("dep_hints", []),
                detected_via = s.get("detected_via", []),
            )
            for s in d.get("external_systems", [])
        ],
        role_summary    = d.get("role_summary", {}),
        infra_files     = d.get("infra_files", []),
        business_files  = d.get("business_files", []),
        generated_at    = d.get("generated_at", ""),
    )

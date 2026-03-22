"""
pipeline/entrypoints/java_entrypoints.py — Java / Kotlin Entry-Point Detector

Detects non-HTTP entry points in Java/Kotlin projects:

  scheduled    — @Scheduled Spring tasks, Quarkus @Scheduled, cron expressions
  cli          — Spring Boot CommandLineRunner / ApplicationRunner,
                 main() methods, picocli / Spring Shell commands
  queue_worker — Spring @RabbitListener / @KafkaListener / @JmsListener,
                 Quarkus @Incoming (Reactive Messaging)
  webhook      — classes / methods whose name suggests a webhook receiver
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from context import EntryPoint, Framework

_SKIP_DIRS = {
    "target", "build", ".git", "generated", "generated-sources",
    "test", "tests", "it", "integration-test",
}

_CONF_CONFIG  = 1.0
_CONF_PATTERN = 0.8
_CONF_HEUR    = 0.6


# ─── Public interface ─────────────────────────────────────────────────────────

def detect(root: Path, framework: Framework) -> list[EntryPoint]:
    """Return all non-HTTP entry points found in a Java/Kotlin project."""
    results: list[EntryPoint] = []
    results += _detect_scheduled_tasks(root, framework)
    results += _detect_cli_runners(root, framework)
    results += _detect_queue_listeners(root, framework)
    results += _detect_webhook_handlers(root)
    return _dedup(results)


# ─── Detectors ────────────────────────────────────────────────────────────────

_SPRING_SCHEDULED = re.compile(r"@Scheduled\s*\(")
_QUARKUS_SCHEDULED = re.compile(r"@io\.quarkus\.scheduler\.Scheduled|@Scheduled(?!.*spring)")
_CRON_ATTR = re.compile(r'cron\s*=\s*["\']([^"\']+)["\']')
_FIXED_RATE = re.compile(r'fixedRate(?:String)?\s*=\s*["\']?(\d+)')


def _detect_scheduled_tasks(root: Path, framework: Framework) -> list[EntryPoint]:
    eps: list[EntryPoint] = []
    for src_file in _walk(root, {".java", ".kt"}):
        src = _safe_read(src_file)
        if not src or ("@Scheduled" not in src):
            continue
        for m in re.finditer(
            r"@Scheduled\s*\(([^)]{0,200})\)\s*(?:public|protected|private)?\s+\w+\s+(\w+)\s*\(",
            src
        ):
            attrs, method_name = m.group(1), m.group(2)
            cron = ""
            cm = _CRON_ATTR.search(attrs)
            if cm:
                cron = cm.group(1)
            else:
                rm = _FIXED_RATE.search(attrs)
                if rm:
                    cron = f"fixedRate={rm.group(1)}ms"

            rel = _rel(src_file, root)
            eps.append(EntryPoint(
                ep_type      = "scheduled",
                handler_file = rel,
                name         = f"Scheduled: {_humanize(method_name)}",
                schedule     = cron or "see @Scheduled annotation",
                trigger      = f"Spring @Scheduled → {method_name}()",
                confidence   = _CONF_CONFIG,
            ))
    return eps


_CLI_PATTERNS = [
    re.compile(r"implements\s+CommandLineRunner"),
    re.compile(r"implements\s+ApplicationRunner"),
    re.compile(r"@SpringBootApplication.*\brun\b", re.DOTALL),
    re.compile(r"public\s+static\s+void\s+main\s*\("),
    re.compile(r"@Command\s*\("),        # picocli
    re.compile(r"@ShellComponent"),      # Spring Shell
]


def _detect_cli_runners(root: Path, framework: Framework) -> list[EntryPoint]:
    eps: list[EntryPoint] = []
    for src_file in _walk(root, {".java", ".kt"}):
        src = _safe_read(src_file)
        if not src:
            continue
        matched = [p for p in _CLI_PATTERNS if p.search(src)]
        if not matched:
            continue
        rel = _rel(src_file, root)
        # Extract class name
        class_m = re.search(r"(?:class|object)\s+(\w+)", src)
        name = _humanize(class_m.group(1)) if class_m else _humanize(src_file.stem)

        if re.search(r"implements\s+CommandLineRunner|implements\s+ApplicationRunner", src):
            ep_type = "cli"
            trigger = f"Spring Boot startup → {src_file.stem}.run()"
            conf = _CONF_CONFIG
        elif re.search(r"@Command\s*\(", src):
            ep_type = "cli"
            trigger = f"picocli command → {src_file.stem}"
            conf = _CONF_PATTERN
        elif re.search(r"@ShellComponent", src):
            ep_type = "cli"
            trigger = f"Spring Shell component → {src_file.stem}"
            conf = _CONF_PATTERN
        elif re.search(r"public\s+static\s+void\s+main", src):
            ep_type = "cli"
            trigger = f"java -cp ... {src_file.stem}"
            conf = _CONF_HEUR
        else:
            continue

        eps.append(EntryPoint(
            ep_type      = ep_type,
            handler_file = rel,
            name         = name,
            trigger      = trigger,
            confidence   = conf,
        ))
    return eps


_QUEUE_PATTERNS = [
    (re.compile(r"@RabbitListener\s*\("),   "RabbitMQ",   "spring_rabbit"),
    (re.compile(r"@KafkaListener\s*\("),    "Kafka",      "spring_kafka"),
    (re.compile(r"@JmsListener\s*\("),      "JMS",        "spring_jms"),
    (re.compile(r"@SqsListener\s*\("),      "AWS SQS",    "spring_sqs"),
    (re.compile(r"@Incoming\s*\("),         "MicroProfile Reactive", "quarkus_reactive"),
    (re.compile(r"@MessageHandler\s*\("),   "Axon",       "axon"),
    (re.compile(r"MessageListener|onMessage\s*\("),  "JMS generic", "jms_generic"),
]


def _detect_queue_listeners(root: Path, framework: Framework) -> list[EntryPoint]:
    eps: list[EntryPoint] = []
    for src_file in _walk(root, {".java", ".kt"}):
        src = _safe_read(src_file)
        if not src:
            continue
        for pattern, queue_name, kind in _QUEUE_PATTERNS:
            if not pattern.search(src):
                continue
            rel = _rel(src_file, root)
            class_m = re.search(r"(?:class|object)\s+(\w+)", src)
            name = _humanize(class_m.group(1)) if class_m else _humanize(src_file.stem)
            # Try to extract queue/topic name from annotation
            qn_m = re.search(
                r'(?:queues|topics|value|destination)\s*=\s*["\']([^"\']+)',
                src[pattern.search(src).start():pattern.search(src).start()+200]
            )
            topic = qn_m.group(1) if qn_m else "default"
            eps.append(EntryPoint(
                ep_type      = "queue_worker",
                handler_file = rel,
                name         = f"{name} ({queue_name} listener)",
                trigger      = f"{queue_name} queue/topic: {topic}",
                confidence   = _CONF_PATTERN,
            ))
            break   # one match per file is enough
    return eps


_WEBHOOK_NAME = re.compile(r"(webhook|callback|notify|ipn|receive|hook)", re.I)
_WEBHOOK_CONTENT = [
    re.compile(r"X-Hub-Signature|HMAC|validateSignature|stripeSignature", re.I),
    re.compile(r"@PostMapping.*webhook|@RequestMapping.*webhook", re.I),
]


def _detect_webhook_handlers(root: Path) -> list[EntryPoint]:
    eps: list[EntryPoint] = []
    for src_file in _walk(root, {".java", ".kt"}):
        if not _WEBHOOK_NAME.search(src_file.stem):
            continue
        src = _safe_read(src_file)
        if not src:
            continue
        has_content = any(p.search(src) for p in _WEBHOOK_CONTENT)
        rel = _rel(src_file, root)
        eps.append(EntryPoint(
            ep_type      = "webhook",
            handler_file = rel,
            name         = f"{_humanize(src_file.stem)} Webhook",
            trigger      = f"HTTP POST → /{src_file.stem} (external system)",
            confidence   = _CONF_CONFIG if has_content else _CONF_HEUR,
        ))
    return eps


# ─── Utilities ────────────────────────────────────────────────────────────────

def _walk(root: Path, extensions: set[str]):
    for f in root.rglob("*"):
        if not f.is_file():
            continue
        if any(p in _SKIP_DIRS for p in f.parts):
            continue
        if f.suffix in extensions:
            yield f


def _dedup(eps: list[EntryPoint]) -> list[EntryPoint]:
    seen: dict[str, EntryPoint] = {}
    for ep in eps:
        key = ep.handler_file
        if key not in seen or ep.confidence > seen[key].confidence:
            seen[key] = ep
    return list(seen.values())


def _safe_read(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _humanize(name: str) -> str:
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    name = re.sub(r"[^a-zA-Z0-9 ]+", " ", name)
    return " ".join(w.capitalize() for w in name.split() if w)

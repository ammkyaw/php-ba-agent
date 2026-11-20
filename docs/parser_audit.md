# Parser Contract Audit Report

> **Audited:** 2026-03-27 | **Contract:** [`pipeline/parsers/base.py`](../pipeline/parsers/base.py)

---

## Contract Compliance Matrix

| Requirement | PHP | TypeScript | Java |
|---|:---:|:---:|:---:|
| **`LANGUAGE` class attr** | ✅ `Language.PHP` | ✅ `Language.TYPESCRIPT` | ✅ `Language.JAVA` |
| **`SUPPORTED_FRAMEWORKS` class attr** | ✅ 6 frameworks | ✅ 8 frameworks | ✅ 4 frameworks |
| **`detect()` classmethod** | ✅ composer.json + .php scan | ✅ package.json + .ts scan | ✅ pom.xml/build.gradle + .java scan |
| **`parse()` → CodeMap** | ✅ subprocess bridge | ✅ pure-Python regex | ✅ pure-Python regex |
| **Sets `code_map.language`** | ✅ | ✅ (JS/TS auto-detect) | ✅ (Java/Kotlin auto-detect) |
| **Sets `code_map.language_version`** | ✅ composer.json/php-version | ✅ package.json typescript ver | ✅ pom.xml/build.gradle |
| **Sets `code_map.framework`** | ✅ | ✅ | ✅ |
| **Writes code_map.json** | ✅ | ✅ | ✅ |
| **`_parser` hint in JSON** | ✅ `"php"` | ✅ `"typescript"` | ✅ `"java"` |

### ✅ PHP `_parser` Hint — Fixed

`php_parser.py` now sets `payload["_parser"] = "php"` before JSON serialisation,
consistent with the TypeScript and Java parsers.

---

## CodeMap Field Coverage

| Field | PHP | TypeScript | Java |
|---|:---:|:---:|:---:|
| classes | ✅ | ✅ | ✅ |
| routes | ✅ | ✅ | ✅ |
| models | ✅ | ✅ | ✅ |
| controllers | ✅ | ✅ | ✅ |
| services | ✅ | ✅ | ✅ |
| db_schema | ✅ | ✅ Prisma | ✅ JPA |
| functions | ✅ | ✅ | ✅ |
| imports | ✅ (via includes) | ✅ | ✅ |
| sql_queries | ✅ | ✅ | ✅ |
| call_graph | ✅ | ✅ | ✅ |
| form_fields | ✅ | ✅ (Zod/RHF/Shadcn) | ❌ |
| service_deps | ✅ | ✅ | ✅ |
| env_vars | ✅ | ✅ | ✅ |
| auth_signals | ✅ | ✅ (14 patterns) | ✅ |
| http_endpoints | ✅ | ✅ | ✅ |
| table_columns | ✅ | ✅ Prisma | ⚠️ Empty (no deep col parse) |
| input_params | ✅ (superglobals) | ✅ (req.body/query/params) | ✅ (@RequestBody/Param) |
| components | ❌ N/A | ❌ Not yet | ❌ N/A |
| type_definitions | ❌ N/A | ✅ | ❌ N/A |
| globals | ✅ | ❌ | ❌ |

---

## Entrypoints & Paths Modules

| Module | PHP | TypeScript | Java |
|---|:---:|:---:|:---:|
| `entrypoints/` | ✅ `php_entrypoints.py` (re-exports stage13) | ✅ `typescript_entrypoints.py` (10KB, full impl) | ✅ `java_entrypoints.py` (9KB, full impl) |
| `paths/` | ✅ `php_paths.py` (re-exports stage15) | ✅ `typescript_paths.py` (16KB, full impl) | ✅ `java_paths.py` (10KB, full impl) |

**All three languages have functional entrypoint and path modules.** PHP uses thin re-export wrappers around the original monolithic stage modules, while TS and Java have standalone implementations.

---

## Summary

**All parsers pass the `LanguageParser` contract.** The only actionable finding is the missing `_parser` hint in PHP's JSON output (cosmetic, non-breaking).

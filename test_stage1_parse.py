"""
tests/test_stage1_parse.py — Unit tests for the PHP parsing stage.

Run with:
    pytest tests/test_stage1_parse.py -v

These tests are designed to run without a PHP binary by mocking subprocess.run.
Integration tests (which do require PHP + nikic/php-parser) are marked with
@pytest.mark.integration and skipped by default.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from context import CodeMap, Framework, PipelineContext
from pipeline.stage1_parse import (
    _build_code_map,
    _detect_project_php_version,
    _run_parser,
    get_classes_by_type,
    get_routes_by_method,
    summarise_code_map,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

MINIMAL_PAYLOAD: dict = {
    "php_version":  "8.1",
    "framework":    "laravel",
    "total_files":  12,
    "total_lines":  840,
    "classes":      [],
    "routes":       [],
    "models":       [],
    "controllers":  [],
    "services":     [],
    "db_schema":    [],
    "config_files": ["config/app.php"],
    "errors":       [],
}

FULL_PAYLOAD: dict = {
    **MINIMAL_PAYLOAD,
    "controllers": [
        {
            "name": "UserController",
            "fqn":  "App\\Http\\Controllers\\UserController",
            "type": "controller",
            "file": "app/Http/Controllers/UserController.php",
            "line": 10,
            "methods": [
                {"name": "index",  "visibility": "public", "return_type": "Response", "params": [], "docblock": None, "line": 15, "is_static": False, "is_abstract": False},
                {"name": "store",  "visibility": "public", "return_type": "Response", "params": [], "docblock": None, "line": 25, "is_static": False, "is_abstract": False},
            ],
            "docblock": {"summary": "Manages user CRUD operations.", "raw": "/** Manages user CRUD operations. */", "params": [], "return": None, "tags": {}, "annotations": []},
        }
    ],
    "models": [
        {
            "name": "User",
            "fqn":  "App\\Models\\User",
            "type": "model",
            "file": "app/Models/User.php",
            "line": 8,
            "properties": [{"name": "$fillable", "visibility": "protected", "type": "array", "docblock": None, "line": 12, "is_static": False}],
            "methods": [],
            "docblock": None,
        }
    ],
    "routes": [
        {"method": "GET",  "path": "/users",      "handler": "UserController@index", "file": "routes/web.php", "line": 5,  "source": "laravel_static"},
        {"method": "POST", "path": "/users",      "handler": "UserController@store", "file": "routes/web.php", "line": 6,  "source": "laravel_static"},
        {"method": "GET",  "path": "/users/{id}", "handler": "UserController@show",  "file": "routes/web.php", "line": 7,  "source": "laravel_static"},
    ],
    "db_schema": [
        {
            "operation": "create",
            "table":     "users",
            "columns": [
                {"name": "id",         "type": "id",      "modifiers": []},
                {"name": "name",       "type": "string",  "modifiers": []},
                {"name": "email",      "type": "string",  "modifiers": ["unique"]},
                {"name": "password",   "type": "string",  "modifiers": []},
            ],
            "file": "database/migrations/2024_01_01_000000_create_users_table.php",
            "line": 18,
        }
    ],
}


# ─── _build_code_map ──────────────────────────────────────────────────────────

class TestBuildCodeMap:
    def test_minimal_payload(self):
        cm = _build_code_map(MINIMAL_PAYLOAD)
        assert isinstance(cm, CodeMap)
        assert cm.framework == Framework.LARAVEL
        assert cm.php_version == "8.1"
        assert cm.total_files == 12
        assert cm.total_lines == 840
        assert cm.config_files == ["config/app.php"]

    def test_full_payload(self):
        cm = _build_code_map(FULL_PAYLOAD)
        assert len(cm.controllers) == 1
        assert len(cm.models) == 1
        assert len(cm.routes) == 3
        assert len(cm.db_schema) == 1

    def test_unknown_framework_falls_back(self):
        payload = {**MINIMAL_PAYLOAD, "framework": "joomla"}
        cm = _build_code_map(payload)
        assert cm.framework == Framework.UNKNOWN

    def test_missing_keys_default_to_empty(self):
        cm = _build_code_map({"php_version": "7.4", "framework": "raw_php"})
        assert cm.classes == []
        assert cm.routes == []
        assert cm.total_files == 0

    @pytest.mark.parametrize("fw_str,expected", [
        ("laravel",     Framework.LARAVEL),
        ("symfony",     Framework.SYMFONY),
        ("codeigniter", Framework.CODEIGNITER),
        ("wordpress",   Framework.WORDPRESS),
        ("raw_php",     Framework.RAW_PHP),
        ("unknown",     Framework.UNKNOWN),
    ])
    def test_all_frameworks_map_correctly(self, fw_str, expected):
        cm = _build_code_map({**MINIMAL_PAYLOAD, "framework": fw_str})
        assert cm.framework == expected


# ─── get_classes_by_type ──────────────────────────────────────────────────────

class TestGetClassesByType:
    def setup_method(self):
        self.cm = _build_code_map(FULL_PAYLOAD)

    def test_returns_controllers(self):
        result = get_classes_by_type(self.cm, "controller")
        assert len(result) == 1
        assert result[0]["name"] == "UserController"

    def test_returns_models(self):
        result = get_classes_by_type(self.cm, "model")
        assert len(result) == 1
        assert result[0]["name"] == "User"

    def test_returns_empty_for_unknown_type(self):
        result = get_classes_by_type(self.cm, "nonexistent")
        assert result == []


# ─── get_routes_by_method ─────────────────────────────────────────────────────

class TestGetRoutesByMethod:
    def setup_method(self):
        self.cm = _build_code_map(FULL_PAYLOAD)

    def test_get_routes(self):
        routes = get_routes_by_method(self.cm, "GET")
        assert len(routes) == 2

    def test_post_routes(self):
        routes = get_routes_by_method(self.cm, "POST")
        assert len(routes) == 1
        assert routes[0]["path"] == "/users"

    def test_case_insensitive(self):
        assert get_routes_by_method(self.cm, "get") == get_routes_by_method(self.cm, "GET")

    def test_nonexistent_method(self):
        assert get_routes_by_method(self.cm, "DELETE") == []


# ─── summarise_code_map ───────────────────────────────────────────────────────

class TestSummariseCodeMap:
    def test_returns_all_keys(self):
        cm = _build_code_map(FULL_PAYLOAD)
        s  = summarise_code_map(cm)
        for key in ["framework", "php_version", "total_files", "total_lines",
                    "classes", "controllers", "models", "services", "routes",
                    "migrations", "config_files"]:
            assert key in s, f"Missing key: {key}"

    def test_counts_correct(self):
        cm = _build_code_map(FULL_PAYLOAD)
        s  = summarise_code_map(cm)
        assert s["controllers"] == 1
        assert s["models"]      == 1
        assert s["routes"]      == 3
        assert s["migrations"]  == 1


# ─── _detect_project_php_version ─────────────────────────────────────────────

class TestDetectProjectPhpVersion:
    def test_reads_php_version_file(self, tmp_path):
        (tmp_path / ".php-version").write_text("8.2.5\n")
        assert _detect_project_php_version(str(tmp_path)) == "8.2"

    def test_reads_composer_json_constraint(self, tmp_path):
        composer = {"require": {"php": ">=8.1.0", "laravel/framework": "^10.0"}}
        (tmp_path / "composer.json").write_text(json.dumps(composer))
        assert _detect_project_php_version(str(tmp_path)) == "8.1"

    def test_php_version_takes_priority_over_composer(self, tmp_path):
        (tmp_path / ".php-version").write_text("8.3")
        composer = {"require": {"php": ">=7.4"}}
        (tmp_path / "composer.json").write_text(json.dumps(composer))
        assert _detect_project_php_version(str(tmp_path)) == "8.3"

    def test_falls_back_to_default(self, tmp_path):
        # No hints, PHP binary also unavailable in mock
        with patch("pipeline.stage1_parse.PHP_BIN", "nonexistent_php_binary_xyz"):
            version = _detect_project_php_version(str(tmp_path))
        assert version == "8.1"

    def test_handles_malformed_composer_json(self, tmp_path):
        (tmp_path / "composer.json").write_text("NOT JSON {{")
        version = _detect_project_php_version(str(tmp_path))
        # Should not crash — falls through to system PHP or default
        assert isinstance(version, str)


# ─── _run_parser (mocked subprocess) ─────────────────────────────────────────

class TestRunParser:
    def _make_proc(self, stdout: str, returncode: int = 0, stderr: str = ""):
        proc = MagicMock()
        proc.stdout     = stdout
        proc.stderr     = stderr
        proc.returncode = returncode
        return proc

    def test_returns_stdout_on_success(self):
        payload = json.dumps(MINIMAL_PAYLOAD)
        with patch("pipeline.stage1_parse.subprocess.run",
                   return_value=self._make_proc(payload)):
            result = _run_parser("/fake/project", "8.1")
        assert result == payload

    def test_raises_on_nonzero_exit(self):
        with patch("pipeline.stage1_parse.subprocess.run",
                   return_value=self._make_proc("", returncode=1, stderr="Fatal error")):
            with pytest.raises(RuntimeError, match="exited with code 1"):
                _run_parser("/fake/project", "8.1")

    def test_raises_on_timeout(self):
        import subprocess as sp
        with patch("pipeline.stage1_parse.subprocess.run",
                   side_effect=sp.TimeoutExpired(cmd="php", timeout=300)):
            with pytest.raises(RuntimeError, match="timed out"):
                _run_parser("/fake/project", "8.1")

    def test_raises_on_missing_binary(self):
        with patch("pipeline.stage1_parse.subprocess.run",
                   side_effect=FileNotFoundError()):
            with pytest.raises(RuntimeError, match="Could not execute"):
                _run_parser("/fake/project", "8.1")

    def test_stderr_warning_does_not_raise(self, capsys):
        payload = json.dumps(MINIMAL_PAYLOAD)
        with patch("pipeline.stage1_parse.subprocess.run",
                   return_value=self._make_proc(payload, stderr="PHP Notice: something")):
            result = _run_parser("/fake/project", "8.1")
        assert result == payload
        captured = capsys.readouterr()
        assert "non-fatal" in captured.err


# ─── Integration Tests (skipped unless --run-integration passed) ──────────────

@pytest.mark.integration
class TestStage1Integration:
    """
    Requires: PHP on PATH, nikic/php-parser installed, a sample PHP project.
    Run with: pytest tests/test_stage1_parse.py -v -m integration --run-integration
    """

    def test_full_parse_laravel_project(self, tmp_path):
        """
        Point at a real Laravel project and assert CodeMap is populated sensibly.
        Set PHPBA_TEST_PROJECT env var to the project path before running.
        """
        import os
        project = os.environ.get("PHPBA_TEST_PROJECT")
        if not project:
            pytest.skip("PHPBA_TEST_PROJECT not set")

        ctx = PipelineContext.create(php_project_path=project, output_base=str(tmp_path))
        from pipeline.stage1_parse import run
        run(ctx)

        assert ctx.code_map is not None
        assert ctx.code_map.framework != Framework.UNKNOWN
        assert ctx.code_map.total_files > 0
        assert ctx.code_map.total_lines > 0
        assert (tmp_path / f"run_{ctx.run_id}" / "code_map.json").exists() or \
               Path(ctx.output_path("code_map.json")).exists()

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""write_analysis confines against the project the command resolved.

The project-root confinement guard in ``write_analysis`` must use the
same root as ``AppState.project()``, so ``--project`` is honoured. Before,
it re-resolved from ``$OASE_PROJECT`` / CWD walk-up, which:

  - skipped the guard when ``--project`` named a valid program but
    CWD/env did not resolve, so ``--out`` outside the project succeeded;
  - refused (exit 9) a write into the ``--project`` program when run
    from inside a different program, because the guard confined against
    that other program.

These tests drive the real CLI (``phenotype analyze``, offline) and do
not mock ``_get_project_root``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

TOOLS_DIR = Path(__file__).resolve().parent.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

# Import click classes from submodules, not ``import click``: some test
# modules replace sys.modules["click"] with a mock and never restore it.
# This only helps when real click was loaded first (the normal collection
# order). Collected after such a module alone, this file fails to import,
# as test_artifact_register and test_verb_aliases already do.
from click.core import Context  # noqa: E402
from click.testing import CliRunner  # noqa: E402

from oase.cli import cli  # noqa: E402
from oase.common import AppState  # noqa: E402
from oase.core import provenance  # noqa: E402
from oase.core.context import init_project  # noqa: E402
from oase.core.errors import ArtifactError, ProjectRootError, Refusal  # noqa: E402

ARTIFACT = {
    "schema": "oase.phenotype.v1",
    "gene": "TP53",
    "phenotypes": [{"phenotype_term": "tumor"}],
}


def _make_program(path: Path) -> Path:
    root = init_project(path)
    raw = root / "raw" / "genomics"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "tp53.phenotype-mgi.artifact.json").write_text(
        json.dumps(ARTIFACT), encoding="utf-8"
    )
    return root


@pytest.fixture()
def fx(tmp_path, monkeypatch):
    """Two programs (good, other), a neutral dir and an outside dir."""
    monkeypatch.delenv("OASE_PROJECT", raising=False)
    monkeypatch.setenv("OASE_NO_DIRTY_WARNING", "1")
    dirs = {
        "good": _make_program(tmp_path / "good"),
        "other": _make_program(tmp_path / "other"),
        "neutral": tmp_path / "neutral",
        "outside": tmp_path / "outside",
    }
    dirs["neutral"].mkdir()
    dirs["outside"].mkdir()
    return dirs


def _analyses(root: Path) -> list[Path]:
    return sorted(root.rglob("*.analysis.json"))


def test_control_walkup_out_outside_is_refused(fx, monkeypatch):
    """Control: no --project, CWD inside the program, --out outside → refused."""
    monkeypatch.chdir(fx["good"])
    result = CliRunner().invoke(
        cli, ["phenotype", "analyze", "TP53", "--out", str(fx["outside"])]
    )
    assert result.exit_code == Refusal.exit_code, result.output
    assert _analyses(fx["outside"]) == []


def test_project_flag_from_neutral_cwd_confines_out(fx, monkeypatch):
    """--project from a CWD with no program must still confine --out."""
    monkeypatch.chdir(fx["neutral"])
    result = CliRunner().invoke(
        cli,
        [
            "--project",
            str(fx["good"]),
            "phenotype",
            "analyze",
            "TP53",
            "--out",
            str(fx["outside"]),
        ],
    )
    assert result.exit_code == Refusal.exit_code, result.output
    assert "outside project root" in result.output
    assert _analyses(fx["outside"]) == []


def test_project_flag_with_unresolvable_env_confines_out(fx, monkeypatch):
    """A broken $OASE_PROJECT overridden by --project must not disable the guard."""
    monkeypatch.setenv("OASE_PROJECT", str(fx["neutral"] / "does-not-exist"))
    monkeypatch.chdir(fx["neutral"])
    result = CliRunner().invoke(
        cli,
        [
            "--project",
            str(fx["good"]),
            "phenotype",
            "analyze",
            "TP53",
            "--out",
            str(fx["outside"]),
        ],
    )
    assert result.exit_code == Refusal.exit_code, result.output
    assert _analyses(fx["outside"]) == []


def test_project_flag_from_inside_other_program_is_allowed(fx, monkeypatch):
    """--project A run from inside program B writes into A, not refused."""
    monkeypatch.chdir(fx["other"])
    result = CliRunner().invoke(
        cli, ["--project", str(fx["good"]), "phenotype", "analyze", "TP53"]
    )
    assert result.exit_code == 0, result.output
    assert len(_analyses(fx["good"])) == 1
    assert _analyses(fx["other"]) == []


def test_project_flag_overrides_env_other_program(fx, monkeypatch):
    """--project A with $OASE_PROJECT=B writes into A, not refused."""
    monkeypatch.setenv("OASE_PROJECT", str(fx["other"]))
    monkeypatch.chdir(fx["neutral"])
    result = CliRunner().invoke(
        cli, ["--project", str(fx["good"]), "phenotype", "analyze", "TP53"]
    )
    assert result.exit_code == 0, result.output
    assert len(_analyses(fx["good"])) == 1
    assert _analyses(fx["other"]) == []


def test_get_project_root_uses_active_app_state(fx, monkeypatch):
    """Inside a click context, the AppState project wins over CWD/env."""
    monkeypatch.setenv("OASE_PROJECT", str(fx["other"]))
    monkeypatch.chdir(fx["other"])
    state = AppState(project_override=str(fx["good"]))
    with Context(cli, obj=state):
        assert provenance._get_project_root() == fx["good"].resolve()


def test_get_project_root_without_cli_context_falls_back(fx, monkeypatch):
    """Library use (no click context) keeps the env/walk-up lookup."""
    monkeypatch.chdir(fx["other"])
    assert provenance._get_project_root() == fx["other"].resolve()
    monkeypatch.chdir(fx["neutral"])
    assert provenance._get_project_root() is None


def test_get_project_root_propagates_resolution_error(fx, monkeypatch):
    """Under the CLI an unresolvable project raises; it is never None."""
    monkeypatch.chdir(fx["other"])
    state = AppState(project_override=str(fx["neutral"] / "does-not-exist"))
    with Context(cli, obj=state):
        with pytest.raises(ProjectRootError):
            provenance._get_project_root()


def test_normalize_source_uses_project_flag_root(fx, monkeypatch):
    """Sources normalise against the --project root, not CWD/env."""
    monkeypatch.chdir(fx["neutral"])
    inside = fx["good"] / "raw" / "genomics" / "tp53.phenotype-mgi.artifact.json"
    outside = fx["outside"] / "x.pdb"
    outside.write_text("x", encoding="utf-8")
    state = AppState(project_override=str(fx["good"]))
    with Context(cli, obj=state):
        assert provenance._normalize_source(str(inside)) == (
            "raw/genomics/tp53.phenotype-mgi.artifact.json"
        )
        with pytest.raises(ArtifactError):
            provenance._normalize_source(str(outside))


def test_get_project_root_ignores_non_appstate_context(fx, monkeypatch):
    """A mocked click (find_object returns a non-AppState) falls back, never a mock root."""
    monkeypatch.setattr(provenance, "click", MagicMock())
    monkeypatch.chdir(fx["neutral"])
    assert provenance._get_project_root() is None

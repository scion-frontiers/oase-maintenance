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

"""Fail-loud detection of pre-rename (DDE) programs and environment variables.

Program roots. Every entry point that accepts or derives a program root
goes through legacy.check_program_root, which refuses any root holding a
`.dde/` directory, with or without `.oase/`. The tests below cover each
entry point from the dev-notes coverage table, for both states:
- legacy: a DDE program as the pre-rename CLI left it
- both markers: that program after `oase init` from main (before this
  change) wrote a fresh `.oase/` beside the `.dde/`
Both are built from real `init_project` and `controlstore.write_record`
output, not hand-made directory trees (see _make_dde_program).

Environment variables. Each REFUSE variable stops the CLI; a WARN
variable warns and continues; a variable set under both names uses the
OASE_* value; an empty OASE_* value does not count as set. `oase doctor`
reports both conditions instead of being refused.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from click.testing import CliRunner

from oase.cli import cli
from oase.commands.doctor import FAIL, INFO, OK, WARN, Report, _check_legacy_env
from oase.core import context, controlstore, legacy
from oase.core.context import init_project, resolve_project
from oase.core.errors import ArtifactError, ProjectRootError, UsageError
from oase.core.legacy import (
    LEGACY_ENV_VARS,
    REFUSE,
    LegacyProgramError,
    check_program_root,
    find_marked_dir,
    legacy_env_messages,
    scan_legacy_env,
)

REFUSE_VARS = sorted(k for k, (_, a) in LEGACY_ENV_VARS.items() if a == REFUSE)


def _clean_environ(**extra: str) -> dict[str, str]:
    """os.environ without any OASE_*/DDE_* variable, plus *extra*."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OASE_", "DDE_"))}
    env["OASE_NO_DIRTY_WARNING"] = "1"
    env.update(extra)
    return env


def _runner_env(**extra: str) -> dict[str, str | None]:
    """CliRunner env that unsets every OASE_*/DDE_* variable, plus *extra*.

    CliRunner overlays its env on os.environ; a None value unsets a key.
    """
    env: dict[str, str | None] = {
        k: None for k in os.environ if k.startswith(("OASE_", "DDE_"))
    }
    env["OASE_NO_DIRTY_WARNING"] = "1"
    env.update(extra)
    return env


_RECORD = {
    "id": "WO-001",
    "revision": 1,
    "state": "scientifically_accepted",
    "decision_question": "Does X hold?",
    "requested_role": "test-agent",
    "stage": "test",
    "cycle": 1,
    "context": {"summary": "test context"},
    "dependencies": [],
    "capabilities": ["test"],
    "deliverables": {"artifact": "test.json"},
    "acceptance_criteria": "tests pass",
    "alert_policy": {"on_failure": "notify"},
    "priority": "normal",
    "resource_class": "standard",
    "report_to": "test-lead",
    "created_at": "2026-01-01T00:00:00Z",
}


def _make_oase_program(root: Path) -> Path:
    """A current program from the real `oase init`, holding one work order."""
    init_project(root)
    controlstore.write_record(root, "work-order", "WO-001-r1", dict(_RECORD))
    return root


def _make_dde_program(root: Path) -> Path:
    """A DDE program as the pre-rename CLI left it, holding one work order.

    The pre-rename `dde init` (3cd9684^) ran the same init_project and
    ensure_control_dirs code as today's, with ".dde" as the marker, and
    the control store layout is unchanged. So the real current init,
    plus a real record write, with the marker renamed back, is exactly
    that tree (verified file by file in the dev notes).
    """
    _make_oase_program(root)
    (root / ".oase").rename(root / ".dde")
    return root


def _add_oase_from_old_init(root: Path, scratch: Path) -> Path:
    """Run `oase init` from main (before this change) on a DDE program.

    That init wrote a fresh `.oase/` (skeleton yaml plus control/ with an
    empty events.ndjson) beside the `.dde/`; every other directory it
    writes already exists. Today's init refuses, so run the real init on
    *scratch* and move its `.oase/` in.
    """
    init_project(scratch)
    (scratch / ".oase").rename(root / ".oase")
    return root


def _tree(root: Path) -> dict[str, bytes]:
    """Every file under *root* with its content, for "nothing changed" checks."""
    return {
        str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }


class _TmpDirCase(unittest.TestCase):
    """Provides self.legacy (.dde/ only) and self.both (.dde/ and .oase/)."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.base = Path(self._tmpdir.name).resolve()
        self.legacy = _make_dde_program(self.base / "legacy-program")
        self.both = _make_dde_program(self.base / "both-program")
        _add_oase_from_old_init(self.both, self.base / "scratch-init")
        self.states = {"legacy": self.legacy, "both": self.both}
        self._cwd = os.getcwd()

    def tearDown(self):
        os.chdir(self._cwd)
        self._tmpdir.cleanup()

    def assert_refused_message(self, text: str, state: str) -> None:
        if state == "legacy":
            self.assertIn("is a legacy DDE program", text)
            self.assertIn("mv ", text)
        else:
            self.assertIn("has both .dde/ and .oase/", text)
            self.assertIn("reconcile them by hand", text)


class TestLegacyTable(unittest.TestCase):
    def test_context_uses_the_single_marker_definition(self):
        self.assertIs(context.PROJECT_MARKER, legacy.PROJECT_MARKER)
        self.assertEqual(context.PROJECT_MARKER, ".oase")

    def test_legacy_error_is_a_project_root_error_with_its_exit_code(self):
        err = legacy.legacy_program_error(Path("/p"), "test")
        self.assertIsInstance(err, LegacyProgramError)
        self.assertIsInstance(err, ProjectRootError)
        self.assertEqual(err.exit_code, ProjectRootError.exit_code)

    def test_every_legacy_var_maps_to_its_oase_name(self):
        for old, (new, action) in LEGACY_ENV_VARS.items():
            self.assertTrue(old.startswith("DDE_"), old)
            self.assertEqual(new, "OASE_" + old[len("DDE_") :])
            self.assertIn(action, (legacy.REFUSE, legacy.WARN))

    def test_refuse_set_is_the_wrong_project_risks(self):
        self.assertEqual(
            REFUSE_VARS, ["DDE_AF3_PROJECT", "DDE_AG_PROJECT", "DDE_PROJECT"]
        )


class TestFixtures(_TmpDirCase):
    """The fixtures are the real states, not approximations."""

    def test_both_markers_fixture_has_the_old_init_skeleton(self):
        oase = self.both / ".oase"
        self.assertTrue((oase / "control" / "events.ndjson").is_file())
        self.assertTrue((oase / "program.yaml").is_file())
        self.assertEqual(list((oase / "control" / "work-orders").iterdir()), [])
        wo = self.both / ".dde" / "control" / "work-orders" / "WO-001-r1.json"
        self.assertTrue(wo.is_file())

    def test_legacy_fixture_has_no_oase_marker(self):
        self.assertFalse((self.legacy / ".oase").exists())
        self.assertTrue((self.legacy / ".dde" / "control" / "events.ndjson").is_file())


class TestCheckProgramRoot(_TmpDirCase):
    """The one legacy check every program-root entry point goes through."""

    def test_no_marker_passes(self):
        check_program_root(self.base, "t")

    def test_oase_only_passes(self):
        check_program_root(_make_oase_program(self.base / "p"), "t")

    def test_legacy_and_both_refuse(self):
        for state, root in self.states.items():
            with self.subTest(state=state):
                with self.assertRaises(LegacyProgramError) as cm:
                    check_program_root(root, "the test")
                err = cm.exception
                self.assertIsInstance(err, ProjectRootError)
                self.assertEqual(err.exit_code, ProjectRootError.exit_code)
                self.assertIn("found via the test", err.detail)
                self.assert_refused_message(f"{err.message}\n{err.remedy}", state)

    def test_remedies_never_delete_or_overwrite(self):
        for state, root in self.states.items():
            with self.subTest(state=state):
                err = legacy.legacy_program_error(root, "t")
                self.assertIsNone(
                    re.search(r"\b(rm|cp|rsync|rmdir)\b|-f\b", err.remedy), err.remedy
                )
                self.assertIn(legacy.UPGRADE_DOCS, err.remedy)
                self.assertIn(".dde/ is the marker the pre-rename DDE CLI", err.detail)

    def test_legacy_remedy_is_a_rename_into_a_free_name(self):
        remedy = legacy.legacy_program_error(self.legacy, "t").remedy
        self.assertIn(f"mv {self.legacy / '.dde'} {self.legacy / '.oase'}", remedy)
        self.assertIn("do NOT run `oase init`", remedy)

    def test_legacy_remedy_is_safe_to_paste_with_awkward_paths(self):
        # The mv line is meant to be pasted into a shell, so each path must
        # come out as exactly one word.
        for name in ("sp ace", "it's", "a;b $HOME"):
            with self.subTest(name=name):
                root = _make_dde_program(self.base / name / "q")
                remedy = legacy.legacy_program_error(root, "t").remedy
                (mv_line,) = [ln.strip() for ln in remedy.splitlines() if "mv " in ln]
                self.assertEqual(
                    shlex.split(mv_line),
                    ["mv", str(root / ".dde"), str(root / ".oase")],
                )

    def test_both_remedy_prints_no_command(self):
        # With .oase/ present, `mv .dde .oase` would nest or fail, so the
        # remedy is prose: reconcile, then remove or rename the old marker.
        remedy = legacy.legacy_program_error(self.both, "t").remedy
        self.assertNotIn("mv ", remedy)
        self.assertIn("Only once .oase/ holds everything", remedy)
        self.assertIn("IDs may", remedy)

    def test_remedy_continuation_lines_are_aligned(self):
        for state, root in self.states.items():
            remedy = legacy.legacy_program_error(root, "t").remedy
            for line in remedy.splitlines()[1:]:
                with self.subTest(state=state, line=line):
                    self.assertTrue(line.startswith(" " * 11), repr(line))

    def test_find_marked_dir_returns_nearest(self):
        deep = self.legacy / "a" / "b"
        self.assertEqual(find_marked_dir(deep), self.legacy)
        (self.legacy / "a" / ".oase").mkdir(parents=True)
        self.assertEqual(find_marked_dir(deep), self.legacy / "a")
        self.assertIsNone(find_marked_dir(self.base))


class TestResolveProject(_TmpDirCase):
    """Entry points 1-3: --project, OASE_PROJECT and walk-up."""

    def test_project_flag(self):
        for state, root in self.states.items():
            with self.subTest(state=state):
                with self.assertRaises(LegacyProgramError) as cm:
                    resolve_project(root)
                self.assertIn("--project", cm.exception.detail)

    def test_oase_project(self):
        for state, root in self.states.items():
            with self.subTest(state=state):
                env = _clean_environ(OASE_PROJECT=str(root))
                with mock.patch.dict(os.environ, env, clear=True):
                    with self.assertRaises(LegacyProgramError) as cm:
                        resolve_project()
                self.assertIn("OASE_PROJECT", cm.exception.detail)

    def test_walk_up(self):
        for state, root in self.states.items():
            with self.subTest(state=state):
                os.chdir(root / "findings" / "regulatory")
                with mock.patch.dict(os.environ, _clean_environ(), clear=True):
                    with self.assertRaises(LegacyProgramError) as cm:
                        resolve_project()
                err = cm.exception
                self.assertIn("walk-up", err.detail)
                self.assert_refused_message(f"{err.message}\n{err.remedy}", state)

    def test_walk_up_refuses_nearer_legacy_program_inside_oase_one(self):
        # A legacy program nested in a current one must not silently
        # resolve to the outer program.
        outer = _make_oase_program(self.base / "outer")
        inner = _make_dde_program(outer / "inner")
        os.chdir(inner)
        with mock.patch.dict(os.environ, _clean_environ(), clear=True):
            with self.assertRaises(LegacyProgramError):
                resolve_project()

    def test_oase_only_program_still_resolves(self):
        program = _make_oase_program(self.base / "current")
        os.chdir(program / "findings")
        with mock.patch.dict(os.environ, _clean_environ(), clear=True):
            self.assertEqual(resolve_project().root, program)

    def test_no_marker_remedy_does_not_send_legacy_users_to_init(self):
        empty = self.base / "empty"
        empty.mkdir()
        os.chdir(empty)
        with mock.patch.dict(os.environ, _clean_environ(), clear=True):
            with self.assertRaises(ProjectRootError) as cm:
                resolve_project()
        self.assertIn("create a new one", cm.exception.remedy)
        self.assertIn("not re-initialised", cm.exception.remedy)


class TestWorkOrderList(_TmpDirCase):
    """Survey E-5: `OASE_PROJECT=<legacy> oase wo list` printed "No work
    orders found." and exited 0. The both-markers state did the same."""

    def _assert_refused(self, result, state):
        self.assertEqual(result.exit_code, ProjectRootError.exit_code, result.output)
        self.assertNotIn("No work orders found", result.stdout)
        self.assert_refused_message(result.stderr, state)

    def test_via_each_resolution_path(self):
        for state, root in self.states.items():
            cases = {
                "--project": (["--project", str(root), "wo", "list"], {}),
                "OASE_PROJECT": (["wo", "list"], {"OASE_PROJECT": str(root)}),
                "walk-up": (["wo", "list"], {}),
            }
            for how, (args, env) in cases.items():
                with self.subTest(state=state, how=how):
                    os.chdir(root if how == "walk-up" else self.base)
                    result = CliRunner().invoke(cli, args, env=_runner_env(**env))
                    self._assert_refused(result, state)

    def test_migrated_program_lists_its_work_orders(self):
        program = _make_dde_program(self.base / "migrated")
        (program / ".dde").rename(program / ".oase")  # the legacy remedy
        result = CliRunner().invoke(
            cli, ["--project", str(program), "wo", "list"], env=_runner_env()
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("WO-001", result.stdout)


class TestInit(_TmpDirCase):
    """Entry points 4-5: the `oase init` target and the program enclosing it."""

    def test_target_refused_and_nothing_changes(self):
        for state, root in self.states.items():
            with self.subTest(state=state):
                before = _tree(root)
                result = CliRunner().invoke(cli, ["init", str(root)], env=_runner_env())
                self.assertEqual(result.exit_code, ProjectRootError.exit_code)
                self.assert_refused_message(result.stderr, state)
                self.assertEqual(_tree(root), before)
                with self.assertRaises(LegacyProgramError):
                    init_project(root)
                self.assertEqual(_tree(root), before)

    def test_subdirectory_of_legacy_or_both_program_refused(self):
        # Otherwise walk-up from below would stop at the new, empty program
        # and never reach the refusal.
        for state, root in self.states.items():
            with self.subTest(state=state):
                sub = root / "findings" / "regulatory"
                with self.assertRaises(LegacyProgramError) as cm:
                    init_project(sub)
                self.assertIn("enclosing", cm.exception.detail)
                self.assertEqual(list(sub.iterdir()), [])

    def test_new_nested_path_inside_legacy_program_refused(self):
        target = self.legacy / "a" / "b"
        result = CliRunner().invoke(cli, ["init", str(target)], env=_runner_env())
        self.assertEqual(result.exit_code, ProjectRootError.exit_code, result.output)
        self.assertIn("legacy DDE program", result.stderr)
        self.assertFalse((self.legacy / "a").exists())

    def test_inside_oase_program_still_allowed(self):
        # Nesting under a current program is unchanged by this check.
        outer = _make_oase_program(self.base / "outer")
        init_project(outer / "sub")
        self.assertTrue((outer / "sub" / ".oase").is_dir())

    def test_inside_oase_program_nested_in_legacy_one_allowed(self):
        # The nearest marker decides, as in walk-up: the OASE program
        # already shadows the legacy one above it.
        # (Built elsewhere and moved in: init now refuses inside .dde/.)
        mid = _make_oase_program(self.base / "mid").rename(self.legacy / "mid")
        init_project(mid / "sub")
        self.assertTrue((mid / "sub" / ".oase").is_dir())


class TestTriage(_TmpDirCase):
    """Entry point 7. `oase triage run` treats "no project" as optional. It
    must not treat a legacy program as "no project" and exit 0 without
    persisting."""

    def setUp(self):
        super().setUp()
        self.concept = self.base / "concept.json"
        self.concept.write_text(json.dumps({"id": "IC-001", "revision": 1}))
        # Spy (not a stub): the refusal must come at project resolution,
        # before any workstream runs.
        from oase.commands import triage as triage_cmd

        patcher = mock.patch.object(
            triage_cmd, "run_triage", wraps=triage_cmd.run_triage
        )
        self.run_triage = patcher.start()
        self.addCleanup(patcher.stop)

    def _invoke(self, args, cwd, **env):
        os.chdir(cwd)
        return CliRunner().invoke(cli, args, env=_runner_env(**env))

    def _assert_refused(self, result, state):
        self.assertEqual(result.exit_code, ProjectRootError.exit_code, result.output)
        self.assert_refused_message(result.stderr, state)
        self.assertNotIn("n_shortlisted", result.stdout)
        self.run_triage.assert_not_called()

    def test_each_resolution_path(self):
        run = ["triage", "run", str(self.concept), "--json"]
        for state, root in self.states.items():
            cases = {
                "walk-up": (run, root, {}),
                "OASE_PROJECT": (run, self.base, {"OASE_PROJECT": str(root)}),
                "--project": (["--project", str(root), *run], self.base, {}),
            }
            for how, (args, cwd, env) in cases.items():
                with self.subTest(state=state, how=how):
                    self._assert_refused(self._invoke(args, cwd, **env), state)
        self.assertFalse((self.legacy / ".oase").exists())

    def test_out_flag(self):
        # With --out, triage never asks for the artifact dir, so the re-raise
        # at project resolution is the only thing that stops it.
        out = self.base / "triage-out"
        args = ["triage", "run", str(self.concept), "--out", str(out), "--json"]
        self._assert_refused(self._invoke(args, self.legacy), "legacy")
        self.assertFalse(out.exists())

    def test_project_flag_with_out(self):
        out = self.base / "triage-out"
        args = ["--project", str(self.both), "triage", "run", str(self.concept)]
        args += ["--out", str(out)]
        self._assert_refused(self._invoke(args, self.base), "both")
        self.assertFalse(out.exists())


class TestProgramResume(_TmpDirCase):
    """Entry point 6. `oase program resume SOURCE` reads `.oase/control/` and
    otherwise falls back to importing markdown work orders, so a legacy
    source used to exit 0 importing nothing."""

    def setUp(self):
        super().setUp()
        self.dest = self.base / "dest"
        init_project(self.dest)

    def _resume(self, source):
        return CliRunner().invoke(
            cli,
            ["--project", str(self.dest), "program", "resume", str(source), "--json"],
            env=_runner_env(),
        )

    def _assert_refused(self, result, state):
        self.assertEqual(result.exit_code, ProjectRootError.exit_code, result.output)
        self.assert_refused_message(result.stderr, state)
        self.assertNotIn("imported_work_orders", result.stdout)

    def test_source_root(self):
        for state, root in self.states.items():
            with self.subTest(state=state):
                self._assert_refused(self._resume(root), state)

    def test_source_root_with_markdown(self):
        for state, root in self.states.items():
            with self.subTest(state=state):
                (root / "README.md").write_text("# notes\n")
                self._assert_refused(self._resume(root), state)

    def test_paths_inside_the_markers(self):
        rels = [
            ".dde",
            ".dde/control",
            ".dde/control/work-orders",
        ]
        for state, root in self.states.items():
            paths = rels + ([".oase", ".oase/control"] if state == "both" else [])
            for rel in paths:
                with self.subTest(state=state, rel=rel):
                    self._assert_refused(self._resume(root / rel), state)

    def test_current_program_imports(self):
        for i, rel in enumerate(("", ".oase", ".oase/control/work-orders")):
            with self.subTest(rel=rel):
                src = _make_oase_program(self.base / f"src{i}")
                self.dest = init_project(self.base / f"dest{i}")
                result = self._resume(src / rel)
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn("WO-001", result.stdout)

    def test_program_stored_under_a_dir_named_like_a_marker(self):
        # Review r4 finding 4: a root holding a marker is used as is, even
        # when an ancestor directory happens to be named .oase or .dde.
        from oase.commands.program import _resolve_source

        for name in (".oase", ".dde"):
            with self.subTest(name=name):
                src = _make_oase_program(self.base / name / "progs" / "src")
                self.assertEqual(_resolve_source(str(src)), src)

    def test_markdown_directory_still_accepted(self):
        md = self.base / "md-only"
        md.mkdir()
        (md / "README.md").write_text("# notes\n")
        result = self._resume(md)
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertNotIn("legacy DDE program", result.stderr)

    def test_directory_without_control_plane_is_still_an_artifact_error(self):
        from oase.commands.program import _resolve_source

        with self.assertRaises(ArtifactError):
            _resolve_source(str(self.base / "dest" / "findings"))


class TestLegacyEnvMessages(unittest.TestCase):
    def test_nothing_set(self):
        self.assertEqual(legacy_env_messages({}), ([], None))

    def test_each_refuse_var_refuses(self):
        for name in REFUSE_VARS:
            with self.subTest(name=name):
                lines, error = legacy_env_messages({name: "my-project"})
                self.assertIsInstance(error, UsageError)
                self.assertIn(name, error.message)
                self.assertIn(LEGACY_ENV_VARS[name][0], error.remedy)
                self.assertEqual(lines, [])

    def test_warn_var_warns_without_refusing(self):
        lines, error = legacy_env_messages({"DDE_RUN_ID": "r1"})
        self.assertIsNone(error)
        self.assertEqual(len(lines), 1)
        self.assertIn("DDE_RUN_ID", lines[0])
        self.assertIn("OASE_RUN_ID", lines[0])

    def test_both_set_same_value_is_silent(self):
        env = {"DDE_AF3_PROJECT": "p", "OASE_AF3_PROJECT": "p"}
        self.assertEqual(legacy_env_messages(env), ([], None))

    def test_both_set_different_values_uses_oase_with_notice(self):
        env = {"DDE_AF3_PROJECT": "old", "OASE_AF3_PROJECT": "new"}
        lines, error = legacy_env_messages(env)
        self.assertIsNone(error)
        self.assertEqual(len(lines), 1)
        self.assertIn("using OASE_AF3_PROJECT", lines[0])

    def test_empty_oase_value_does_not_shadow_refuse_var(self):
        env = {"DDE_PROJECT": "/legacy", "OASE_PROJECT": ""}
        lines, error = legacy_env_messages(env)
        self.assertIsInstance(error, UsageError)
        self.assertEqual(lines, [])

    def test_empty_oase_value_does_not_shadow_warn_var(self):
        lines, error = legacy_env_messages({"DDE_RUN_ID": "r", "OASE_RUN_ID": ""})
        self.assertIsNone(error)
        self.assertEqual(len(lines), 1)
        self.assertIn("Warning", lines[0])

    def test_unlisted_dde_var_is_ignored(self):
        self.assertEqual(scan_legacy_env({"DDE_SOMETHING_ELSE": "1"}), [])


class TestCliStartupEnvCheck(_TmpDirCase):
    def setUp(self):
        super().setUp()
        self.program = self.base / "program"
        init_project(self.program)

    def _invoke(self, args, **extra):
        # --project, not OASE_PROJECT: OASE_PROJECT would shadow DDE_PROJECT.
        args = ["--project", str(self.program), *args]
        return CliRunner().invoke(cli, args, env=_runner_env(**extra))

    def test_each_refuse_var_exits_nonzero(self):
        for name in REFUSE_VARS:
            with self.subTest(name=name):
                result = self._invoke(["wo", "list"], **{name: "my-project"})
                self.assertEqual(result.exit_code, UsageError.exit_code)
                self.assertIn("refusing to run", result.stderr)
                self.assertIn(name, result.stderr)
                self.assertNotIn("No work orders found", result.stdout)

    def test_warn_var_continues(self):
        result = self._invoke(["wo", "list"], DDE_PACE_REQUIRE_SHARED="1")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("DDE_PACE_REQUIRE_SHARED", result.stderr)
        self.assertIn("No work orders found", result.stdout)

    def test_both_set_continues(self):
        result = self._invoke(
            ["wo", "list"], DDE_AF3_PROJECT="old", OASE_AF3_PROJECT="new"
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Notice", result.stderr)

    def test_empty_oase_project_does_not_bypass_refusal(self):
        result = self._invoke(["wo", "list"], DDE_PROJECT="/legacy", OASE_PROJECT="")
        self.assertEqual(result.exit_code, UsageError.exit_code)
        self.assertIn("DDE_PROJECT", result.stderr)
        self.assertNotIn("Notice", result.stderr)

    def test_doctor_is_not_refused(self):
        result = self._invoke(["doctor", "--json"], DDE_AG_PROJECT="x")
        self.assertNotIn("refusing to run", result.stderr)
        self.assertIn('"legacy variable DDE_AG_PROJECT"', result.stdout)
        self.assertEqual(result.exit_code, 1)

    def test_doctor_reports_each_variable_once(self):
        # Doctor has its own report line per variable; the startup check
        # does not also print a stderr warning for it.
        result = self._invoke(["doctor", "--json"], DDE_RUN_ID="r")
        self.assertNotIn("DDE_RUN_ID", result.stderr)
        self.assertIn('"legacy variable DDE_RUN_ID"', result.stdout)


class TestDoctorLegacyEnv(unittest.TestCase):
    def _run(self, **env):
        report = Report()
        with mock.patch.dict(os.environ, _clean_environ(**env), clear=True):
            _check_legacy_env(report)
        return {c.name: c.status for c in report.checks}

    def test_none_set(self):
        self.assertEqual(self._run(), {"legacy DDE_* variables": OK})

    def test_refuse_var_fails(self):
        self.assertEqual(
            self._run(DDE_PROJECT="/p"), {"legacy variable DDE_PROJECT": FAIL}
        )

    def test_warn_var_warns(self):
        self.assertEqual(
            self._run(DDE_TOOLS_HOME="/t"), {"legacy variable DDE_TOOLS_HOME": WARN}
        )

    def test_both_set_is_info(self):
        self.assertEqual(
            self._run(DDE_AG_PROJECT="a", OASE_AG_PROJECT="b"),
            {"legacy variable DDE_AG_PROJECT": INFO},
        )


class TestDoctorLegacyProgram(_TmpDirCase):
    """Entry point 8: doctor reports the refusal as a FAIL line."""

    def test_project_root_failure(self):
        for state, root in self.states.items():
            with self.subTest(state=state):
                result = CliRunner().invoke(
                    cli, ["--project", str(root), "doctor", "--json"], env=_runner_env()
                )
                self.assertEqual(result.exit_code, 1)
                checks = {c["name"]: c for c in json.loads(result.stdout)["checks"]}
                check = checks["project root"]
                self.assertEqual(check["status"], FAIL)
                self.assert_refused_message(
                    f"{check['detail']}\n{check['remedy']}", state
                )
                self.assertNotIn("legacy program marker", checks)


if __name__ == "__main__":
    unittest.main()

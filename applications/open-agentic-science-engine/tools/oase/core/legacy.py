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

"""Detect state left behind by the pre-rename CLI (DDE) and fail loudly.

OASE was previously called DDE. The rename was a hard cut: the ``.dde/``
program marker and every ``DDE_*`` environment variable stopped being
read. Without detection, some of these fail *silently*. For example, a
legacy program under ``OASE_PROJECT`` looks empty, and ``DDE_AF3_PROJECT``
is ignored, so AlphaFold 3 jobs go to the default GCP project.

This module only *detects*. It never honours the old names. Migration
is a separate concern.
"""

from __future__ import annotations

import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .errors import ProjectRootError, UsageError

#: Program marker directory written by the pre-rename CLI.
LEGACY_PROJECT_MARKER = ".dde"

#: Current program marker. Defined here (and re-exported by ``context``)
#: so that ``context`` can import this module without a cycle.
PROJECT_MARKER = ".oase"

REFUSE = "refuse"
WARN = "warn"

#: Every environment variable the pre-rename CLI and its scripts read,
#: mapped to its OASE name and to what happens when only the old name is
#: set. This is the single place to change the refuse/warn policy.
#:
#: REFUSE marks variables whose silent loss sends work to the wrong
#: place: the program root, or the GCP project billed for AlphaFold 3 and
#: AlphaGenome (both fall back to a built-in default project).
LEGACY_ENV_VARS: dict[str, tuple[str, str]] = {
    "DDE_PROJECT": ("OASE_PROJECT", REFUSE),
    "DDE_AF3_PROJECT": ("OASE_AF3_PROJECT", REFUSE),
    "DDE_AG_PROJECT": ("OASE_AG_PROJECT", REFUSE),
    "DDE_TOOLS_HOME": ("OASE_TOOLS_HOME", WARN),
    "DDE_VENV": ("OASE_VENV", WARN),
    "DDE_BIN": ("OASE_BIN", WARN),
    "DDE_PACE_DIR": ("OASE_PACE_DIR", WARN),
    "DDE_PACE_REQUIRE_SHARED": ("OASE_PACE_REQUIRE_SHARED", WARN),
    "DDE_NO_DIRTY_WARNING": ("OASE_NO_DIRTY_WARNING", WARN),
    "DDE_WORK_ORDER_ID": ("OASE_WORK_ORDER_ID", WARN),
    "DDE_RUN_ID": ("OASE_RUN_ID", WARN),
    "DDE_AF3_REGION": ("OASE_AF3_REGION", WARN),
    "DDE_AF3_LOCK": ("OASE_AF3_LOCK", WARN),
    "DDE_AF3_ENDPOINT_NAME": ("OASE_AF3_ENDPOINT_NAME", WARN),
    "DDE_AG_REGION": ("OASE_AG_REGION", WARN),
    "DDE_AG_ENDPOINT_ID": ("OASE_AG_ENDPOINT_ID", WARN),
    "DDE_AG_ENDPOINT_DNS": ("OASE_AG_ENDPOINT_DNS", WARN),
    "DDE_PUBMED_BQ_DATASET": ("OASE_PUBMED_BQ_DATASET", WARN),
    "DDE_HYPEX_SOURCE_DIR": ("OASE_HYPEX_SOURCE_DIR", WARN),
}


# ---------------------------------------------------------------------------
# Legacy program directories
# ---------------------------------------------------------------------------


#: Where the error messages send users for the full upgrade steps.
UPGRADE_DOCS = "the DDE to OASE upgrade notes in the OASE docs"


def find_marked_dir(start: Path) -> Path | None:
    """Return the nearest of *start* and its parents holding either marker.

    The one walk-up used both for project discovery and for the ancestor
    check in ``oase init``, so the two always agree on which program a
    directory belongs to. Directories that do not exist yet are skipped.
    """
    for candidate in (start, *start.parents):
        if (candidate / PROJECT_MARKER).is_dir() or (
            candidate / LEGACY_PROJECT_MARKER
        ).is_dir():
            return candidate
    return None


class LegacyProgramError(ProjectRootError):
    """The program root holds a pre-rename DDE marker (``.dde/``).

    A subclass so that callers which treat ProjectRootError as "no
    project, carry on without one" can let this case through. Running
    on without the project here would silently skip a program's state.
    """


def legacy_program_error(root: Path, source: str) -> LegacyProgramError:
    """The error for a program root that holds ``.dde/``.

    The remedy never prints a command that deletes or overwrites data.
    """
    dde = root / LEGACY_PROJECT_MARKER
    oase = root / PROJECT_MARKER
    markers = (
        f"{PROJECT_MARKER}/ marks an OASE program; {LEGACY_PROJECT_MARKER}/ "
        "is the marker the pre-rename DDE CLI wrote"
    )
    if not oase.is_dir():
        return LegacyProgramError(
            f"{root} is a legacy DDE program ({LEGACY_PROJECT_MARKER}/ found, "
            f"no {PROJECT_MARKER}/)",
            detail=(
                f"found via {source}. {markers}. OASE only reads "
                f"{PROJECT_MARKER}/, so this program's control store, "
                "program.yaml and thresholds would be invisible"
            ),
            remedy=(
                "rename the marker directory (this moves nothing else and "
                "deletes nothing):\n"
                f"           mv {shlex.quote(str(dde))} {shlex.quote(str(oase))}\n"
                "           do NOT run `oase init` on it: that creates a second, "
                f"empty marker beside the old one. See {UPGRADE_DOCS}"
            ),
        )
    return LegacyProgramError(
        f"{root} has both {LEGACY_PROJECT_MARKER}/ and {PROJECT_MARKER}/",
        detail=(
            f"found via {source}. {markers}. Both together usually mean "
            "`oase init` was run on a DDE program. OASE cannot tell which "
            "one holds the program's state, so it will not guess"
        ),
        remedy=(
            f"reconcile them by hand (see {UPGRADE_DOCS}):\n"
            f"           compare {dde}/ with {oase}/ (program.yaml, "
            "thresholds.yaml, and the control/ records, whose IDs may "
            "collide) and move what you need into "
            f"{PROJECT_MARKER}/.\n"
            f"           Only once {PROJECT_MARKER}/ holds everything, remove "
            f"{LEGACY_PROJECT_MARKER}/ or rename it out of the way (for example "
            f"to {LEGACY_PROJECT_MARKER}.migrated)"
        ),
    )


def check_program_root(root: Path, source: str) -> None:
    """The legacy check for every place OASE accepts or derives a program root.

    Raises :class:`LegacyProgramError` (a ProjectRootError, exit code 2)
    when *root* holds a ``.dde/`` directory, whether or not ``.oase/`` is
    also present. A fully migrated program has no ``.dde/``. *source*
    names the entry point for the error message.
    """
    if (root / LEGACY_PROJECT_MARKER).is_dir():
        raise legacy_program_error(root, source)


# ---------------------------------------------------------------------------
# Legacy environment variables
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LegacyEnvFinding:
    """One ``DDE_*`` variable that is set in the environment.

    ``action`` is REFUSE or WARN when the OASE counterpart is unset or
    empty. It is None when the counterpart is set to a non-empty value:
    OASE uses the new name. Empty counts as unset because that is how
    OASE reads these variables (e.g. ``if os.environ.get("OASE_PROJECT")``).
    """

    legacy: str
    current: str
    action: str | None


def scan_legacy_env(environ: Mapping[str, str]) -> list[LegacyEnvFinding]:
    """Return a finding for each variable in LEGACY_ENV_VARS that is set.

    An empty legacy variable is still reported: it is set, and it is
    still a leftover that should be renamed or removed.
    """
    findings = []
    for legacy, (current, action) in LEGACY_ENV_VARS.items():
        if legacy not in environ:
            continue
        shadowed = bool(environ.get(current))
        findings.append(LegacyEnvFinding(legacy, current, None if shadowed else action))
    return findings


def legacy_env_messages(
    environ: Mapping[str, str],
) -> tuple[list[str], UsageError | None]:
    """Render the stderr lines and the refusal (if any) for *environ*.

    Returns ``(warnings, error)``. *warnings* holds one line per WARN
    variable, and one notice per variable set under both names with
    different values. *error* is a UsageError naming every REFUSE
    variable, or None when nothing calls for a refusal.
    """
    lines: list[str] = []
    refused: list[LegacyEnvFinding] = []
    for f in scan_legacy_env(environ):
        if f.action == REFUSE:
            refused.append(f)
        elif f.action == WARN:
            lines.append(
                f"Warning: {f.legacy} is set but OASE no longer reads it; "
                f"rename it to {f.current}."
            )
        elif environ[f.legacy] != environ[f.current]:
            lines.append(
                f"Notice: both {f.legacy} and {f.current} are set; "
                f"using {f.current} and ignoring {f.legacy}."
            )

    if not refused:
        return lines, None
    names = ", ".join(f.legacy for f in refused)
    renames = "\n".join(
        f'           export {f.current}="${f.legacy}"; unset {f.legacy}'
        for f in refused
    )
    error = UsageError(
        f"refusing to run: legacy environment variable(s) set: {names}",
        detail=(
            "OASE was previously called DDE and no longer reads DDE_* "
            "variables. Ignoring these would send work to the wrong program "
            "directory or GCP project"
        ),
        remedy=f"rename them to their OASE_* equivalents:\n{renames}",
    )
    return lines, error

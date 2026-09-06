"""The scaffolder, held to the standard it generates.

The single most important test here is
`test_a_generated_project_passes_its_own_gates`. Everything else checks the
shape of the output; that one checks the claim - a fresh scaffold runs ruff,
mypy --strict and pytest clean with no edits. Without it the templates are
prose that happens to be indented like Python.

The compile check earns its place separately. Every template is a Python source
string embedded in a Python source file, and three separate escaping bugs in
this repo's history came from exactly that shape - a `\\n` that became a real
newline on its way through a here-document and silently split a string literal.
`compile()` catches that class of damage; reading the file does not.
"""

from __future__ import annotations

import compileall
import subprocess
import sys
from pathlib import Path

import pytest

from hybrid_rag.scaffolder import (
    ProjectKind,
    ScaffoldFile,
    build_scaffold,
    normalise_package_name,
    validate_project_name,
    write_scaffold,
)

# The rule `.github/copilot-instructions.md` states. A scaffolder that shipped
# these would be teaching agents the opposite of what it ships beside.
PLACEHOLDERS = ("# TODO", "NotImplementedError", "FIXME", "<placeholder>")

EXPECTED_FILES = {
    "ARCHITECTURE.md",
    "README.md",
    "pyproject.toml",
    "tests/conftest.py",
    "tests/test_service.py",
}


# ------------------------------------------------------------------- naming


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("my-app", "my_app"),
        ("My App", "my_app"),
        ("weather.cli", "weather_cli"),
        ("  spaced  out  ", "spaced_out"),
        ("already_fine", "already_fine"),
        ("v2-parser", "v2_parser"),
    ],
)
def test_package_names_are_normalised(given: str, expected: str) -> None:
    assert normalise_package_name(given) == expected


@pytest.mark.parametrize("given", ["", "   ", "---", "123", "9lives"])
def test_unusable_names_raise_rather_than_producing_a_junk_package(given: str) -> None:
    """Silently creating a package called `_` helps nobody."""
    with pytest.raises(ValueError, match="usable Python package name"):
        normalise_package_name(given)


# -------------------------------------------------------------------- shape


@pytest.mark.parametrize("kind", list(ProjectKind))
def test_every_kind_emits_the_documented_layout(kind: ProjectKind, tmp_path: Path) -> None:
    scaffold = build_scaffold("my-app", kind, tmp_path / "my-app")

    relative = {file.relative_path.as_posix() for file in scaffold.files}

    assert relative >= EXPECTED_FILES
    assert f"src/{scaffold.package}/models.py" in relative
    assert f"src/{scaffold.package}/service.py" in relative
    assert f"src/{scaffold.package}/cli.py" in relative


@pytest.mark.parametrize("kind", list(ProjectKind))
def test_no_generated_source_file_contains_a_placeholder(
    kind: ProjectKind, tmp_path: Path
) -> None:
    """Source only.

    The generated README states the no-placeholder rule, which means it has to
    be able to name `# TODO` and `NotImplementedError` in order to forbid them.
    Documentation that describes a prohibition is not an instance of it, and a
    scan that cannot tell those apart would force the docs to talk around the
    thing they are about.
    """
    scaffold = build_scaffold("my-app", kind, tmp_path / "my-app")

    for file in scaffold.files:
        if file.relative_path.suffix != ".py":
            continue
        for marker in PLACEHOLDERS:
            assert marker not in file.content, f"{file.relative_path} contains {marker!r}"


@pytest.mark.parametrize("kind", list(ProjectKind))
def test_no_generated_module_is_a_bare_pass(kind: ProjectKind, tmp_path: Path) -> None:
    """`pass` as a whole statement line is the placeholder that reads as code."""
    scaffold = build_scaffold("my-app", kind, tmp_path / "my-app")

    for file in scaffold.files:
        if file.relative_path.suffix != ".py":
            continue
        bare = [line for line in file.content.splitlines() if line.strip() == "pass"]
        assert not bare, f"{file.relative_path} has a bare `pass`"


@pytest.mark.parametrize("kind", list(ProjectKind))
def test_the_sentinels_are_all_substituted(kind: ProjectKind, tmp_path: Path) -> None:
    scaffold = build_scaffold("weather-cli", kind, tmp_path / "weather-cli")

    for file in scaffold.files:
        assert "__PKG__" not in file.content, file.relative_path
        assert "__NAME__" not in file.content, file.relative_path


@pytest.mark.parametrize("kind", list(ProjectKind))
def test_the_architecture_contract_states_a_tree_and_a_schema(
    kind: ProjectKind, tmp_path: Path
) -> None:
    """Stage 1 of the Architect pattern has required contents, not just a title."""
    scaffold = build_scaffold("my-app", kind, tmp_path / "my-app")
    architecture = next(
        file for file in scaffold.files if file.relative_path.name == "ARCHITECTURE.md"
    )

    assert "## Directory tree" in architecture.content
    assert "## Core data schemas" in architecture.content
    assert "contracts" in architecture.content
    assert "my_app" in architecture.content, "the tree must name the real package"


# -------------------------------------------------------------------- disk


@pytest.mark.parametrize("kind", list(ProjectKind))
def test_generated_python_compiles(kind: ProjectKind, tmp_path: Path) -> None:
    """Guards the escaping class of bug, which reading the template will not."""
    root = tmp_path / "my-app"
    write_scaffold(build_scaffold("my-app", kind, root))

    assert compileall.compile_dir(str(root), quiet=2, force=True), "generated source is invalid"


def test_writing_refuses_to_overwrite_by_default(tmp_path: Path) -> None:
    root = tmp_path / "my-app"
    scaffold = build_scaffold("my-app", ProjectKind.CLI, root)
    write_scaffold(scaffold)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_scaffold(scaffold)


def test_a_refused_write_changes_nothing(tmp_path: Path) -> None:
    """Every target is checked before any is written.

    A scaffolder that fails halfway leaves a directory that is neither the old
    project nor the new one.
    """
    root = tmp_path / "my-app"
    scaffold = build_scaffold("my-app", ProjectKind.CLI, root)
    (root / "src" / "my_app").mkdir(parents=True)
    marker = root / "src" / "my_app" / "service.py"
    marker.write_text("ORIGINAL", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_scaffold(scaffold)

    assert marker.read_text(encoding="utf-8") == "ORIGINAL"
    assert not (root / "README.md").exists(), "nothing may be written on a refusal"


def test_force_overwrites(tmp_path: Path) -> None:
    root = tmp_path / "my-app"
    scaffold = build_scaffold("my-app", ProjectKind.CLI, root)
    write_scaffold(scaffold)

    written = write_scaffold(scaffold, force=True)

    assert len(written) == len(scaffold.files)


def test_a_generated_project_passes_its_own_gates(tmp_path: Path) -> None:
    """The claim, checked rather than asserted in a docstring.

    One kind rather than all three: this spawns a real interpreter, and the
    other two are covered by the compile check plus a manual run recorded in
    docs/VERIFY.md. If this ever fails, run the same three commands against
    each kind by hand - the templates have diverged.
    """
    root = tmp_path / "gate-check"
    write_scaffold(build_scaffold("gate-check", ProjectKind.CLI, root))

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, f"generated tests failed:\n{result.stdout}\n{result.stderr}"


# ------------------------------------------------------------------ traversal
#
# Found by the local security review of this module's own diff. `build_scaffold`
# normalised the PACKAGE name but the directory was built from the raw one, so
# `../../escaped` normalised to the innocent package `escaped` while the
# directory still resolved two levels up. Measured before the fix: it wrote
# outside the requested parent, and an absolute name wrote to the drive root.
#
# The user typing it is not the whole threat model here. Agents drive this CLI,
# and an agent can take a name from a retrieved passage.


@pytest.mark.parametrize(
    "name",
    ["../escaped", "../../escaped", "sub/dir", r"sub\dir", "/abs/rooted", "..", "."],
)
def test_a_traversing_project_name_is_refused(name: str) -> None:
    with pytest.raises(ValueError, match=r"single|empty"):
        validate_project_name(name)


@pytest.mark.parametrize("name", ["my-app", "My App", "weather_cli", "v2-parser"])
def test_an_ordinary_project_name_is_accepted(name: str) -> None:
    assert validate_project_name(name) == name.strip()


@pytest.mark.parametrize("name", ["../../escaped", "sub/dir"])
def test_build_scaffold_refuses_a_traversing_name(name: str, tmp_path: Path) -> None:
    """The package name alone would have accepted these."""
    assert normalise_package_name(name), "precondition: the package name looks innocent"

    with pytest.raises(ValueError, match="single"):
        build_scaffold(name, ProjectKind.CLI, tmp_path / name)


def test_nothing_is_written_outside_the_root(tmp_path: Path) -> None:
    """Defence in depth at the boundary that actually touches the disk.

    `Scaffold` is a plain dataclass, so a caller can hand-build one that
    `build_scaffold` would never have produced.
    """
    from dataclasses import replace

    root = tmp_path / "app"
    good = build_scaffold("app", ProjectKind.CLI, root)
    escaping = replace(
        good,
        files=(ScaffoldFile(relative_path=Path("../../pwned.txt"), content="x"),),
    )

    with pytest.raises(ValueError, match="escapes the scaffold root"):
        write_scaffold(escaping)

    assert not (tmp_path.parent / "pwned.txt").exists()

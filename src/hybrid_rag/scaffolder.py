"""Project scaffolding for the Architect-Worker workflow.

    uv run hybrid-rag scaffold my-app --kind cli

Emits a small, complete, *running* project: typed source, tests that pass, and
an `ARCHITECTURE.md` that is the contract Stage 1 of the Architect pattern is
supposed to produce before any implementation happens.

Three rules govern what gets generated, and they are the whole point:

1. **No placeholders.** Not `pass`, not `# TODO`, not `raise
   NotImplementedError`. `.github/copilot-instructions.md` forbids agents from
   emitting those, and a scaffolder that emits them teaches the opposite of
   what the instructions say. Every generated file does something, and
   `pytest` passes on a fresh scaffold with no edits.
2. **The service layer is pure.** No display, no socket, no framework import.
   That is what makes the generated tests runnable with nothing installed but
   pytest, and it is the shape the Architect pattern wants: logic that can be
   verified the turn after it is written.
3. **Generated code is typed.** The scaffolds are `mypy --strict` clean on
   their own terms, because the instructions demand that of agents working
   here and a scaffold is the first thing an agent reads for the house style.

This module writes files and, optionally, indexes them. It decides nothing
about routing and spends nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePath

__all__ = [
    "ProjectKind",
    "Scaffold",
    "ScaffoldFile",
    "build_scaffold",
    "normalise_package_name",
    "validate_project_name",
    "write_scaffold",
]

# Templates carry Python source, which is full of braces, so `.format()` and
# f-strings are both the wrong tool. Sentinels survive dict literals and nested
# f-strings without escaping.
_PKG = "__PKG__"
_NAME = "__NAME__"


class ProjectKind(StrEnum):
    """What to generate. Each is a different service layer, same skeleton."""

    FASTAPI = "fastapi"
    PYGAME = "pygame"
    CLI = "cli"

    @property
    def label(self) -> str:
        return {
            ProjectKind.FASTAPI: "FastAPI Backend",
            ProjectKind.PYGAME: "Pygame Arcade",
            ProjectKind.CLI: "CLI Tool",
        }[self]


@dataclass(frozen=True, slots=True)
class ScaffoldFile:
    relative_path: Path
    content: str


@dataclass(frozen=True, slots=True)
class Scaffold:
    name: str
    package: str
    kind: ProjectKind
    root: Path
    files: tuple[ScaffoldFile, ...]

    @property
    def architecture_path(self) -> Path:
        """The file the RAG index actually wants. Stage 1's output."""
        return self.root / "ARCHITECTURE.md"


def validate_project_name(name: str) -> str:
    """Reject anything that is not a single, safe directory name.

    The project name becomes a DIRECTORY name, and the package name it also
    feeds is normalised separately - so validating only the package would let
    `../../escaped` through, because that normalises to the perfectly innocent
    `escaped` while the directory still resolves two levels up. Measured before
    this existed: `scaffold "../../escaped" --path sandbox` wrote outside
    `sandbox`, and `scaffold "/abs/rooted"` wrote to the drive root.

    That matters here beyond the usual argument that the user typed it. Agents
    drive this CLI, and an agent can take a name from a retrieved passage - the
    exact untrusted-input path the rest of this project is built to contain.
    """
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("project name is empty")
    if cleaned in {".", ".."} or any(sep in cleaned for sep in ("/", "\\")):
        raise ValueError(
            f"{name!r} is not a single directory name; path separators and "
            "'..' are refused. Use --path to choose where it goes."
        )
    if len(PurePath(cleaned).parts) != 1 or PurePath(cleaned).is_absolute():
        raise ValueError(f"{name!r} is not a single relative directory name")
    return cleaned


def normalise_package_name(name: str) -> str:
    """Turn a project name into an importable package name.

    `my-app` and `My App` both become `my_app`. Raises rather than guessing
    when nothing importable is left - a package named `_` is not a useful
    thing to have silently created.
    """
    cleaned = "".join(char if char.isalnum() else "_" for char in name.strip().lower())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    if not cleaned or not (cleaned[0].isalpha() or cleaned[0] == "_"):
        raise ValueError(
            f"{name!r} does not yield a usable Python package name; "
            "start it with a letter and include at least one alphanumeric character"
        )
    return cleaned


# --------------------------------------------------------------------------
# Shared templates
# --------------------------------------------------------------------------

_PYPROJECT = '''[build-system]
requires = ["hatchling>=1.25,<2"]
build-backend = "hatchling.build"

[project]
name = "__NAME__"
version = "0.1.0"
description = "Generated by hybrid-rag scaffold."
requires-python = ">=3.12"
dependencies = []

[project.optional-dependencies]
dev = ["pytest>=8.3,<9", "ruff>=0.6,<1", "mypy>=1.11,<2"]

[tool.hatch.build.targets.wheel]
packages = ["src/__PKG__"]

[tool.ruff]
line-length = 100
target-version = "py312"
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "ANN", "RUF"]

[tool.ruff.lint.per-file-ignores]
"tests/**" = ["ANN"]

[tool.mypy]
python_version = "3.12"
strict = true
mypy_path = "src"
packages = ["__PKG__"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
addopts = "-q --strict-markers"
'''

_CONFTEST = '''"""Shared fixtures.

The service layer is pure, so there is nothing to stub and nothing to mock.
That is deliberate: a test that needs a live dependency to assert on business
logic is telling you the logic is in the wrong place.
"""

from __future__ import annotations

import pytest

from __PKG__.service import build_default_service


@pytest.fixture
def service():
    return build_default_service()
'''

_INIT = '''"""__NAME__ - generated by hybrid-rag scaffold."""

from __future__ import annotations

__version__ = "0.1.0"
'''

_README = """# __NAME__

Generated by `hybrid-rag scaffold`. It runs as-is.

```bash
uv sync --extra dev
uv run pytest
uv run python -m __PKG__.cli --help
```

## Layout

| path | responsibility |
|---|---|
| `src/__PKG__/models.py` | data shapes only - no behaviour, no IO |
| `src/__PKG__/service.py` | the logic, pure and synchronously testable |
| `src/__PKG__/cli.py` | argument parsing and printing, nothing else |
| `tests/test_service.py` | asserts the service, not the framework |
| `ARCHITECTURE.md` | the contract. Change it before changing code. |

## Working on this with an agent

`ARCHITECTURE.md` is Stage 1 of the Architect-Worker pattern and it is what the
hybrid-rag index has ingested. An agent should read it, query
`local_rag_search` for the patterns already established here, then implement
**one file per turn** with its test alongside.

The rule that matters: no `pass`, no `# TODO`, no `raise NotImplementedError`
in a file that is claimed as done. A placeholder is an unverified claim wearing
the shape of finished work.
"""

_ARCHITECTURE_HEADER = """# __NAME__ - architecture

Stage 1 of the Architect-Worker pattern: the contract, written before any
implementation. This file is indexed by hybrid-rag, so `local_rag_search`
answers questions about this project rather than returning nothing.

Change this file first when the design changes. A codebase that has drifted
from its contract has no contract.

## Directory tree

```
__NAME__/
  ARCHITECTURE.md      this file - the contract
  README.md            how to run it
  pyproject.toml       package metadata, ruff + mypy + pytest config
  src/__PKG__/
    __init__.py        version only
    models.py          data shapes - frozen dataclasses, no behaviour
    service.py         the logic - pure, no IO, no framework
    cli.py             argument parsing and output, nothing else
  tests/
    conftest.py        fixtures
    test_service.py    asserts the service
```

## Boundaries

`models.py` never imports `service.py`. `service.py` never imports `cli.py`.
The dependency arrow points one way, which is what lets the tests import the
logic without dragging in an argument parser.

`service.py` performs no IO. No file reads, no sockets, no clock, no display.
Anything non-deterministic is passed in as an argument. That is why
`test_service.py` needs no mocks.
"""


# --------------------------------------------------------------------------
# Per-kind templates
# --------------------------------------------------------------------------

_CLI_MODELS = '''"""Data shapes. No behaviour lives here."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class Task:
    """One unit of work. Frozen, so a state change is a new value."""

    identifier: int
    title: str
    done: bool = False

    def completed(self) -> Task:
        return replace(self, done=True)

    def render(self) -> str:
        mark = "x" if self.done else " "
        return f"[{mark}] {self.identifier}. {self.title}"
'''

_CLI_SERVICE = '''"""The logic. Pure: no file IO, no clock, no printing."""

from __future__ import annotations

from dataclasses import dataclass, field

from __PKG__.models import Task


class TaskNotFoundError(LookupError):
    """Raised when an identifier does not name a task in this list."""


@dataclass(slots=True)
class TaskService:
    """An ordered task list. Identifiers are stable and never reused."""

    tasks: list[Task] = field(default_factory=list)
    _next_id: int = 1

    def add(self, title: str) -> Task:
        cleaned = title.strip()
        if not cleaned:
            raise ValueError("a task needs a title")
        task = Task(identifier=self._next_id, title=cleaned)
        self._next_id += 1
        self.tasks.append(task)
        return task

    def complete(self, identifier: int) -> Task:
        for index, task in enumerate(self.tasks):
            if task.identifier == identifier:
                done = task.completed()
                self.tasks[index] = done
                return done
        raise TaskNotFoundError(f"no task with id {identifier}")

    def pending(self) -> list[Task]:
        return [task for task in self.tasks if not task.done]

    def render(self) -> str:
        if not self.tasks:
            return "no tasks"
        return "\\n".join(task.render() for task in self.tasks)


def build_default_service() -> TaskService:
    return TaskService()
'''

_CLI_CLI = '''"""Argument parsing and output. All logic lives in service.py."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from __PKG__.service import TaskNotFoundError, build_default_service


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="__PKG__", description="A small task list.")
    parser.add_argument("titles", nargs="*", help="Tasks to add.")
    parser.add_argument(
        "--complete", type=int, action="append", default=[], help="Identifier to mark done."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    service = build_default_service()
    for title in args.titles:
        service.add(title)
    for identifier in args.complete:
        try:
            service.complete(identifier)
        except TaskNotFoundError as exc:
            print(f"error: {exc}")
            return 1
    print(service.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

_CLI_TESTS = '''from __future__ import annotations

import pytest

from __PKG__.cli import main
from __PKG__.service import TaskNotFoundError, TaskService


def test_added_tasks_get_stable_increasing_identifiers(service: TaskService) -> None:
    first = service.add("write the contract")
    second = service.add("implement one file")

    assert (first.identifier, second.identifier) == (1, 2)


def test_completing_a_task_does_not_mutate_the_original(service: TaskService) -> None:
    original = service.add("verify it")

    done = service.complete(original.identifier)

    assert done.done is True
    assert original.done is False, "Task is frozen; completing returns a new value"


def test_pending_excludes_completed(service: TaskService) -> None:
    service.add("one")
    second = service.add("two")
    service.complete(second.identifier)

    assert [task.title for task in service.pending()] == ["one"]


def test_an_unknown_identifier_raises_rather_than_passing_silently(
    service: TaskService,
) -> None:
    with pytest.raises(TaskNotFoundError):
        service.complete(999)


def test_a_blank_title_is_refused(service: TaskService) -> None:
    with pytest.raises(ValueError, match="needs a title"):
        service.add("   ")


def test_cli_reports_a_missing_identifier_as_a_nonzero_exit() -> None:
    assert main(["--complete", "42"]) == 1


def test_cli_renders_added_tasks() -> None:
    assert main(["alpha", "beta"]) == 0
'''

_PYGAME_MODELS = '''"""Data shapes for the grid. No pygame import - none is needed."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Direction(StrEnum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"

    @property
    def delta(self) -> tuple[int, int]:
        return {
            Direction.UP: (0, -1),
            Direction.DOWN: (0, 1),
            Direction.LEFT: (-1, 0),
            Direction.RIGHT: (1, 0),
        }[self]

    @property
    def opposite(self) -> Direction:
        return {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT,
        }[self]


@dataclass(frozen=True, slots=True)
class Board:
    columns: int
    rows: int

    def contains(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        return 0 <= x < self.columns and 0 <= y < self.rows
'''

_PYGAME_SERVICE = '''"""Game rules, entirely pure.

No pygame import, no display, no clock. The renderer calls `step()`; the tests
call `step()`. That is the only way to assert a collision rule without opening
a window, and it is why this file has tests that run in CI.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from __PKG__.models import Board, Direction


@dataclass(slots=True)
class GameState:
    board: Board
    body: list[tuple[int, int]]
    direction: Direction
    food: tuple[int, int]
    score: int = 0
    over: bool = False

    @property
    def head(self) -> tuple[int, int]:
        return self.body[0]


@dataclass(slots=True)
class GameService:
    """One grid game. `rng` is injected so every test is deterministic."""

    board: Board = field(default_factory=lambda: Board(columns=30, rows=20))
    rng: random.Random = field(default_factory=lambda: random.Random(1337))

    def new_game(self) -> GameState:
        start = (self.board.columns // 2, self.board.rows // 2)
        body = [start, (start[0] - 1, start[1])]
        return GameState(
            board=self.board,
            body=body,
            direction=Direction.RIGHT,
            food=self.place_food(body),
        )

    def place_food(self, occupied: list[tuple[int, int]]) -> tuple[int, int]:
        free = [
            (x, y)
            for x in range(self.board.columns)
            for y in range(self.board.rows)
            if (x, y) not in occupied
        ]
        if not free:
            return occupied[0]
        return free[self.rng.randrange(len(free))]

    def steer(self, state: GameState, direction: Direction) -> None:
        """Reversing into yourself is refused rather than fatal."""
        if len(state.body) > 1 and direction is state.direction.opposite:
            return
        state.direction = direction

    def step(self, state: GameState) -> GameState:
        if state.over:
            return state
        dx, dy = state.direction.delta
        head = (state.head[0] + dx, state.head[1] + dy)

        if not state.board.contains(head) or head in state.body[:-1]:
            state.over = True
            return state

        state.body.insert(0, head)
        if head == state.food:
            state.score += 1
            state.food = self.place_food(state.body)
        else:
            state.body.pop()
        return state


def build_default_service() -> GameService:
    return GameService()
'''

_PYGAME_CLI = '''"""Headless driver. Renders to text so the rules can be watched without pygame.

The real renderer belongs in a separate module that imports `service` and draws
with `pygame.draw` primitives only. Keeping it out of here is what lets the
rules be tested in CI, where there is no display.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from __PKG__.service import GameState, build_default_service


def render(state: GameState) -> str:
    rows: list[str] = []
    for y in range(state.board.rows):
        row = ""
        for x in range(state.board.columns):
            if (x, y) == state.head:
                row += "@"
            elif (x, y) in state.body:
                row += "o"
            elif (x, y) == state.food:
                row += "*"
            else:
                row += "."
        rows.append(row)
    return "\\n".join(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="__PKG__", description="Headless grid game.")
    parser.add_argument("--steps", type=int, default=5, help="Ticks to simulate.")
    args = parser.parse_args(argv)

    service = build_default_service()
    state = service.new_game()
    for _ in range(args.steps):
        service.step(state)
        if state.over:
            break
    print(render(state))
    print(f"score {state.score}  over {state.over}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

_PYGAME_TESTS = '''from __future__ import annotations

from __PKG__.models import Board, Direction
from __PKG__.service import GameService


def test_a_new_game_starts_alive_with_food_off_the_body(service: GameService) -> None:
    state = service.new_game()

    assert state.over is False
    assert state.food not in state.body


def test_walking_into_a_wall_ends_the_game(service: GameService) -> None:
    state = service.new_game()
    for _ in range(state.board.columns + 2):
        service.step(state)

    assert state.over is True


def test_reversing_into_itself_is_refused(service: GameService) -> None:
    state = service.new_game()

    service.steer(state, Direction.LEFT)

    assert state.direction is Direction.RIGHT, "a 180 turn must not be accepted"


def test_eating_food_grows_the_body_and_scores(service: GameService) -> None:
    state = service.new_game()
    dx, dy = state.direction.delta
    state.food = (state.head[0] + dx, state.head[1] + dy)
    length_before = len(state.body)

    service.step(state)

    assert state.score == 1
    assert len(state.body) == length_before + 1


def test_stepping_a_finished_game_changes_nothing(service: GameService) -> None:
    state = service.new_game()
    state.over = True
    before = list(state.body)

    service.step(state)

    assert state.body == before


def test_a_full_board_places_food_without_crashing() -> None:
    tiny = GameService(board=Board(columns=1, rows=1))

    assert tiny.place_food([(0, 0)]) == (0, 0)
'''

_FASTAPI_MODELS = '''"""Data shapes. Frozen dataclasses rather than pydantic on purpose.

The service layer must be importable and testable without installing a web
framework. Pydantic models belong at the HTTP edge, in the router module, where
they do request validation - not in the layer the tests exercise.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class Item:
    identifier: int
    name: str
    quantity: int = 1

    def with_quantity(self, quantity: int) -> Item:
        if quantity < 0:
            raise ValueError("quantity cannot be negative")
        return replace(self, quantity=quantity)
'''

_FASTAPI_SERVICE = '''"""Storage logic. No FastAPI import, no database driver, no IO."""

from __future__ import annotations

from dataclasses import dataclass, field

from __PKG__.models import Item


class ItemNotFoundError(LookupError):
    """Raised when an identifier does not name a stored item."""


class DuplicateItemError(ValueError):
    """Raised when a name is already taken. Names are unique by contract."""


@dataclass(slots=True)
class ItemService:
    items: dict[int, Item] = field(default_factory=dict)
    _next_id: int = 1

    def create(self, name: str, quantity: int = 1) -> Item:
        cleaned = name.strip()
        if not cleaned:
            raise ValueError("an item needs a name")
        if any(item.name == cleaned for item in self.items.values()):
            raise DuplicateItemError(f"{cleaned!r} already exists")
        item = Item(identifier=self._next_id, name=cleaned, quantity=quantity)
        self.items[item.identifier] = item
        self._next_id += 1
        return item

    def get(self, identifier: int) -> Item:
        try:
            return self.items[identifier]
        except KeyError as exc:
            raise ItemNotFoundError(f"no item with id {identifier}") from exc

    def set_quantity(self, identifier: int, quantity: int) -> Item:
        updated = self.get(identifier).with_quantity(quantity)
        self.items[identifier] = updated
        return updated

    def delete(self, identifier: int) -> None:
        if self.items.pop(identifier, None) is None:
            raise ItemNotFoundError(f"no item with id {identifier}")

    def list_items(self) -> list[Item]:
        return sorted(self.items.values(), key=lambda item: item.identifier)


def build_default_service() -> ItemService:
    return ItemService()
'''

_FASTAPI_CLI = '''"""Local driver for the service, with no server running.

The HTTP layer is a separate module (`routes.py`, not yet written) that imports
this service and adds pydantic request models. Keeping the two apart is what
lets `test_service.py` run without installing FastAPI.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from __PKG__.service import DuplicateItemError, build_default_service


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="__PKG__", description="Item store, offline.")
    parser.add_argument("names", nargs="*", help="Item names to create.")
    args = parser.parse_args(argv)

    service = build_default_service()
    for name in args.names:
        try:
            service.create(name)
        except (DuplicateItemError, ValueError) as exc:
            print(f"error: {exc}")
            return 1
    for item in service.list_items():
        print(f"{item.identifier}. {item.name} x{item.quantity}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

_FASTAPI_TESTS = '''from __future__ import annotations

import pytest

from __PKG__.cli import main
from __PKG__.service import DuplicateItemError, ItemNotFoundError, ItemService


def test_created_items_get_increasing_identifiers(service: ItemService) -> None:
    first = service.create("bolt")
    second = service.create("nut")

    assert (first.identifier, second.identifier) == (1, 2)


def test_duplicate_names_are_refused(service: ItemService) -> None:
    service.create("bolt")

    with pytest.raises(DuplicateItemError):
        service.create("bolt")


def test_a_missing_item_raises_rather_than_returning_none(service: ItemService) -> None:
    with pytest.raises(ItemNotFoundError):
        service.get(404)


def test_setting_quantity_replaces_the_stored_value(service: ItemService) -> None:
    item = service.create("bolt")

    updated = service.set_quantity(item.identifier, 12)

    assert updated.quantity == 12
    assert service.get(item.identifier).quantity == 12


def test_a_negative_quantity_is_refused(service: ItemService) -> None:
    item = service.create("bolt")

    with pytest.raises(ValueError, match="negative"):
        service.set_quantity(item.identifier, -1)


def test_deleting_twice_raises_the_second_time(service: ItemService) -> None:
    item = service.create("bolt")
    service.delete(item.identifier)

    with pytest.raises(ItemNotFoundError):
        service.delete(item.identifier)


def test_cli_reports_a_duplicate_as_a_nonzero_exit() -> None:
    assert main(["bolt", "bolt"]) == 1
'''

_ARCHITECTURE_BODY: dict[ProjectKind, str] = {
    ProjectKind.CLI: """
## Core data schemas

```python
@dataclass(frozen=True, slots=True)
class Task:
    identifier: int
    title: str
    done: bool = False
```

Frozen. `Task.completed()` returns a new value rather than mutating, so a
reference held elsewhere cannot observe a change it did not ask for.

## Function contracts

| signature | raises |
|---|---|
| `TaskService.add(title: str) -> Task` | `ValueError` on a blank title |
| `TaskService.complete(identifier: int) -> Task` | `TaskNotFoundError` |
| `TaskService.pending() -> list[Task]` | never |
| `TaskService.render() -> str` | never |

Identifiers are stable and never reused, so a caller holding one can rely on it.

## Next files, in order

1. `storage.py` - persist to JSON. Takes a `Path`; the service stays pure.
2. `filters.py` - due dates and tags. Pure, tested the same turn.
""",
    ProjectKind.PYGAME: """
## Core data schemas

```python
class Direction(StrEnum):        # UP DOWN LEFT RIGHT, with .delta and .opposite

@dataclass(frozen=True, slots=True)
class Board:
    columns: int
    rows: int

@dataclass(slots=True)
class GameState:
    board: Board
    body: list[tuple[int, int]]
    direction: Direction
    food: tuple[int, int]
    score: int = 0
    over: bool = False
```

## Function contracts

| signature | contract |
|---|---|
| `GameService.new_game() -> GameState` | food is never placed on the body |
| `GameService.steer(state, direction) -> None` | a 180 turn is ignored, not fatal |
| `GameService.step(state) -> GameState` | wall or self collision sets `over` |
| `GameService.place_food(occupied) -> tuple[int, int]` | total on a full board |

`GameService.rng` is an injected `random.Random`, which is what makes every
test deterministic without freezing the game to a fixed sequence.

## Rendering is deliberately absent

`service.py` imports no pygame. The renderer is a separate module that reads
`GameState` and draws with `pygame.draw` primitives only - no image or sound
files, so there is no asset to be missing at runtime.

That split is why the rules have tests: a collision assertion needs no display.

## Next files, in order

1. `render.py` - `draw(surface, state)` using `pygame.draw` rects only.
2. `loop.py` - pygame event pump at a fixed tick, calling `service.step`.
""",
    ProjectKind.FASTAPI: """
## Core data schemas

```python
@dataclass(frozen=True, slots=True)
class Item:
    identifier: int
    name: str
    quantity: int = 1
```

Frozen dataclass, not a pydantic model. Pydantic belongs at the HTTP edge doing
request validation; putting it in the service layer means the tests cannot run
until a web framework is installed.

## Function contracts

| signature | raises |
|---|---|
| `ItemService.create(name, quantity=1) -> Item` | `ValueError`, `DuplicateItemError` |
| `ItemService.get(identifier) -> Item` | `ItemNotFoundError` |
| `ItemService.set_quantity(identifier, quantity) -> Item` | `ItemNotFoundError`, `ValueError` |
| `ItemService.delete(identifier) -> None` | `ItemNotFoundError` |
| `ItemService.list_items() -> list[Item]` | never |

Names are unique by contract; identifiers are stable and never reused.

## Route contracts, not yet implemented

`routes.py` maps these one-to-one. Domain errors become status codes at that
boundary and nowhere else.

| method | path | body | success | errors |
|---|---|---|---|---|
| POST | `/items` | `{name, quantity}` | 201 `Item` | 400 blank, 409 duplicate |
| GET | `/items` | - | 200 `list[Item]` | - |
| GET | `/items/{id}` | - | 200 `Item` | 404 |
| PATCH | `/items/{id}` | `{quantity}` | 200 `Item` | 404, 400 negative |
| DELETE | `/items/{id}` | - | 204 | 404 |

## Next files, in order

1. `schemas.py` - pydantic request/response models. Edge only.
2. `routes.py` - `APIRouter` translating domain errors to `HTTPException`.
3. `app.py` - `FastAPI()` and router registration. No logic.
""",
}

_KIND_SOURCES: dict[ProjectKind, tuple[str, str, str, str]] = {
    ProjectKind.CLI: (_CLI_MODELS, _CLI_SERVICE, _CLI_CLI, _CLI_TESTS),
    ProjectKind.PYGAME: (_PYGAME_MODELS, _PYGAME_SERVICE, _PYGAME_CLI, _PYGAME_TESTS),
    ProjectKind.FASTAPI: (_FASTAPI_MODELS, _FASTAPI_SERVICE, _FASTAPI_CLI, _FASTAPI_TESTS),
}


def _fill(template: str, *, package: str, name: str) -> str:
    return template.replace(_PKG, package).replace(_NAME, name)


def build_scaffold(name: str, kind: ProjectKind, root: Path) -> Scaffold:
    """Compute every file without touching the filesystem.

    Separated from `write_scaffold` so the content is assertable in a test
    without a temporary directory, and so a caller can inspect what it is about
    to create before it exists.
    """
    name = validate_project_name(name)
    package = normalise_package_name(name)
    models, service, cli, tests = _KIND_SOURCES[kind]
    architecture = _ARCHITECTURE_HEADER + _ARCHITECTURE_BODY[kind]

    raw: list[tuple[str, str]] = [
        ("ARCHITECTURE.md", architecture),
        ("README.md", _README),
        ("pyproject.toml", _PYPROJECT),
        (f"src/{package}/__init__.py", _INIT),
        (f"src/{package}/models.py", models),
        (f"src/{package}/service.py", service),
        (f"src/{package}/cli.py", cli),
        ("tests/conftest.py", _CONFTEST),
        ("tests/test_service.py", tests),
    ]
    files = tuple(
        ScaffoldFile(
            relative_path=Path(relative),
            content=_fill(template, package=package, name=name),
        )
        for relative, template in raw
    )
    return Scaffold(name=name, package=package, kind=kind, root=root, files=files)


def write_scaffold(scaffold: Scaffold, *, force: bool = False) -> list[Path]:
    """Write the scaffold to disk. Returns the paths written, in order.

    Refuses to overwrite by default, and checks EVERY target before writing
    ANY of them. A scaffolder that fails halfway leaves a directory that is
    neither the old project nor the new one, which is worse than either.
    """
    root = scaffold.root.resolve()
    targets = [(root / file.relative_path, file) for file in scaffold.files]

    # Defence in depth. `build_scaffold` already refuses a traversing name, but
    # a `Scaffold` can be constructed by hand, and this is the boundary that
    # actually touches the disk. Nothing may be written outside the root.
    for path, _ in targets:
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"{path} escapes the scaffold root {root}")

    if not force:
        existing = [path for path, _ in targets if path.exists()]
        if existing:
            listed = ", ".join(str(path.relative_to(root)) for path in existing[:5])
            raise FileExistsError(
                f"refusing to overwrite {len(existing)} existing file(s) under "
                f"{root}: {listed}. Pass force=True to replace them."
            )

    written: list[Path] = []
    for path, file in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        # newline="\n" so a generated file does not acquire CRLF on Windows and
        # then read as modified the moment git touches it.
        path.write_text(file.content, encoding="utf-8", newline="\n")
        written.append(path)
    return written

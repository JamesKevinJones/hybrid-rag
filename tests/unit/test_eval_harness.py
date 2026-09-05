from __future__ import annotations

import orjson
import pytest

from hybrid_rag.config import EvalSettings
from hybrid_rag.eval.eval_harness import (
    EvalCase,
    EvalHarness,
    GroundednessScorer,
    RecordedResponder,
    Response,
    content_words,
    extract_code,
    load_cases,
    overlap_ratio,
    python_syntax_error,
    score_instructions,
    split_sentences,
)

CONTEXT = [
    "The rolling window budget takes a reservation before the cloud call and "
    "settles it with measured spend afterwards.",
    "Stage four outranks the router: a veto is recorded as degraded_local.",
]


class ScriptedResponder:
    """Returns a fixed answer per case. No model, no network."""

    def __init__(self, answers: dict[str, Response], label: str = "scripted") -> None:
        self._answers = answers
        self._label = label

    @property
    def label(self) -> str:
        return self._label

    async def respond(self, case: EvalCase) -> Response:
        return self._answers.get(case.case_id, Response(text="", error="no scripted answer"))


@pytest.fixture
def harness(embedder) -> EvalHarness:
    settings = EvalSettings()
    return EvalHarness(settings=settings, scorer=GroundednessScorer(embedder, settings))


# ------------------------------------------------------------- text helpers


def test_short_boilerplate_sentences_are_not_scored() -> None:
    text = "Sure. Here you go. The rolling window budget reserves capacity before any cloud call."
    assert split_sentences(text, min_chars=25) == [
        "The rolling window budget reserves capacity before any cloud call."
    ]


def test_stopwords_are_excluded_from_overlap() -> None:
    assert content_words("the budget is in the ledger") == {"budget", "ledger"}


def test_verbatim_quotation_scores_full_overlap() -> None:
    assert overlap_ratio("Stage four outranks the router", CONTEXT[1]) == 1.0
    assert overlap_ratio("sourdough bread recipe", CONTEXT[1]) == 0.0


# ------------------------------------------------------------- groundedness


async def test_a_quoted_answer_is_fully_grounded(embedder) -> None:
    scorer = GroundednessScorer(embedder, EvalSettings())
    score, unsupported = await scorer.score(
        "Stage four outranks the router and a veto is recorded as degraded_local.", CONTEXT
    )
    assert score == 1.0
    assert unsupported == []


async def test_an_invented_claim_is_flagged(embedder) -> None:
    scorer = GroundednessScorer(embedder, EvalSettings())
    score, unsupported = await scorer.score(
        "The system uses a distributed Kafka cluster to replicate audit events globally.",
        CONTEXT,
    )
    assert score == 0.0
    assert len(unsupported) == 1


async def test_partial_hallucination_scores_between(embedder) -> None:
    scorer = GroundednessScorer(embedder, EvalSettings())
    answer = (
        "Stage four outranks the router and a veto is recorded as degraded_local. "
        "It also replicates every event to a Kafka cluster in three regions."
    )
    score, unsupported = await scorer.score(answer, CONTEXT)
    assert 0.0 < score < 1.0
    assert len(unsupported) == 1


async def test_an_answer_with_no_context_is_ungrounded(embedder) -> None:
    scorer = GroundednessScorer(embedder, EvalSettings())
    score, unsupported = await scorer.score("The budget window is five hours long.", [])
    assert score == 0.0
    assert unsupported


async def test_an_empty_answer_claims_nothing(embedder) -> None:
    """Nothing asserted means nothing hallucinated. Instruction precision catches this."""
    scorer = GroundednessScorer(embedder, EvalSettings())
    score, unsupported = await scorer.score("", CONTEXT)
    assert score == 1.0
    assert unsupported == []


# -------------------------------------------------------- instruction score


def test_a_case_with_no_constraints_is_vacuously_perfect() -> None:
    score, violations = score_instructions(EvalCase(case_id="c", query="q"), "anything")
    assert score == 1.0
    assert violations == []


def test_negative_constraints_are_enforced() -> None:
    case = EvalCase(case_id="c", query="q", must_not_include=["Kafka"])
    score, violations = score_instructions(case, "We stream it through Kafka.")
    assert score == 0.0
    assert "forbidden" in violations[0]


@pytest.mark.parametrize(
    ("tag", "answer", "ok"),
    [
        ("startswith:ANSWER:", "ANSWER: forty two", True),
        ("startswith:ANSWER:", "Well, forty two", False),
        ("max_words:3", "one two three", True),
        ("max_words:3", "one two three four", False),
        ("min_words:3", "one two", False),
        ("json", '{"a": 1}', True),
        ("json", "not json", False),
        ("no_markdown", "plain prose only", True),
        ("no_markdown", "- a bullet list", False),
        ("cites", "per [1] the window is 5h", True),
        ("cites", "the window is 5h", False),
    ],
)
def test_format_tags(tag: str, answer: str, ok: bool) -> None:
    case = EvalCase(case_id="c", query="q", format_tags=[tag])
    score, _ = score_instructions(case, answer)
    assert (score == 1.0) is ok


def test_partial_credit_across_mixed_constraints() -> None:
    case = EvalCase(
        case_id="c",
        query="q",
        must_include=["reservation"],
        must_not_include=["Kafka"],
        format_tags=["max_words:50"],
    )
    score, violations = score_instructions(case, "A reservation is taken, then Kafka streams it.")
    assert score == pytest.approx(2 / 3, abs=1e-3)
    assert len(violations) == 1


# -------------------------------------------------------------- aggregation


async def test_cost_is_divided_by_passes_not_attempts(harness) -> None:
    """A model twice as fast that fails half the time costs the same per answer."""
    cases = [EvalCase(case_id="a", query="q"), EvalCase(case_id="b", query="q")]
    responder = ScriptedResponder(
        {
            "a": Response(
                text="Stage four outranks the router and a veto is recorded as degraded_local.",
                context_chunks=CONTEXT,
                tokens_in=100,
                tokens_out=50,
                latency_ms=200.0,
            ),
            "b": Response(
                text="It replicates every audit event to a Kafka cluster in three regions.",
                context_chunks=CONTEXT,
                tokens_in=100,
                tokens_out=50,
                latency_ms=200.0,
            ),
        }
    )
    report = await harness.run(responder, cases)

    assert report.n_cases == 2
    assert report.pass_rate == 0.5
    assert report.total_tokens == 300
    assert report.tokens_per_resolution == 300.0, "300 tokens bought exactly one usable answer"
    assert report.ms_per_resolution == 400.0


async def test_a_responder_error_becomes_a_failed_case(harness) -> None:
    report = await harness.run(ScriptedResponder({}), [EvalCase(case_id="a", query="q")])
    assert report.pass_rate == 0.0
    assert report.results[0].error == "no scripted answer"
    assert report.tokens_per_resolution == float("inf")


async def test_an_empty_suite_reports_zeroes(harness) -> None:
    report = await harness.run(ScriptedResponder({}), [])
    assert report.n_cases == 0
    assert report.pass_rate == 0.0


async def test_report_round_trips_to_disk(harness, tmp_path) -> None:
    report = await harness.run(ScriptedResponder({}), [EvalCase(case_id="a", query="q")])
    path = report.save(tmp_path / "reports" / "r.json")
    assert orjson.loads(path.read_bytes())["label"] == "scripted"


# ----------------------------------------------------- the recorded baseline


async def test_the_baseline_is_replayed_and_costs_nothing(tmp_path) -> None:
    """Re-running Opus to check whether we can stop running Opus is self-defeating."""
    path = tmp_path / "baseline.jsonl"
    path.write_bytes(
        orjson.dumps(
            {
                "case_id": "a",
                "text": "Stage four outranks the router.",
                "context_chunks": CONTEXT,
                "tokens_in": 900,
                "tokens_out": 300,
                "latency_ms": 4200.0,
            }
        )
        + b"\n"
    )
    responder = RecordedResponder(path)
    assert responder.case_ids == {"a"}

    response = await responder.respond(EvalCase(case_id="a", query="q"))
    assert response.tokens_in == 900
    assert response.error is None


async def test_a_missing_baseline_case_is_an_error_not_a_crash(tmp_path) -> None:
    responder = RecordedResponder(tmp_path / "absent.jsonl")
    response = await responder.respond(EvalCase(case_id="ghost", query="q"))
    assert response.error is not None and "ghost" in response.error


def test_malformed_baseline_lines_are_skipped(tmp_path) -> None:
    path = tmp_path / "baseline.jsonl"
    path.write_bytes(
        b'{"case_id": "a", "text": "ok"}\n'
        b"{not json\n"
        b'{"missing": "case_id"}\n'
    )
    assert RecordedResponder(path).case_ids == {"a"}


def test_cases_load_from_jsonl_and_tolerate_junk(tmp_path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_bytes(
        orjson.dumps({"case_id": "a", "query": "q", "must_not_include": ["Kafka"]})
        + b"\n{bad\n"
        + orjson.dumps({"no_case_id": True})
        + b"\n"
    )
    cases = load_cases(path)
    assert len(cases) == 1
    assert cases[0].must_not_include == ["Kafka"]


def test_a_malformed_format_tag_fails_the_case_not_the_suite() -> None:
    """Regression: int('abc') used to escape as an uncaught ValueError."""
    case = EvalCase(case_id="c", query="q", format_tags=["max_words:abc"])
    score, violations = score_instructions(case, "some answer")
    assert score == 0.0
    assert "unparseable" in violations[0]


# ------------------------------------------------- AST validation of emitted code


FENCED_OK = "Here you go:\n\n```python\ndef add(a, b):\n    return a + b\n```"
FENCED_TRUNCATED = "Here you go:\n\n```python\ndef add(a, b):\n    return a +\n```"


def test_valid_fenced_python_parses() -> None:
    assert python_syntax_error(FENCED_OK) is None


def test_truncated_code_is_caught() -> None:
    """The failure a small distilled model makes constantly.

    No embedding metric notices this: a truncated function is lexically almost
    identical to a complete one, so groundedness scores it fine.
    """
    problem = python_syntax_error(FENCED_TRUNCATED)
    assert problem is not None
    assert "syntax error" in problem


def test_bare_code_without_a_fence_is_still_checked() -> None:
    assert python_syntax_error("def add(a, b):\n    return a + b") is None
    assert python_syntax_error("def add(a, b):\n    return a +") is not None


@pytest.mark.parametrize(
    "answer",
    [
        "The maximum line length recommended for Python code is 79 characters.",
        "",
        "   ",
        "It depends on whether you are using type hints.",
    ],
)
def test_prose_is_not_a_syntax_error(answer: str) -> None:
    """Prose is not code. Flagging it would fire the tag on every text answer."""
    assert python_syntax_error(answer) is None
    assert extract_code(answer) is None


def test_multiple_blocks_are_all_checked() -> None:
    both_fine = "```python\nx = 1\n```\nand\n```python\ny = 2\n```"
    one_broken = "```python\nx = 1\n```\nand\n```python\ny = (\n```"
    assert python_syntax_error(both_fine) is None
    assert python_syntax_error(one_broken) is not None


def test_the_syntax_tag_scores_through_the_normal_path() -> None:
    case = EvalCase(case_id="c", query="q", format_tags=["python_syntax"])
    assert score_instructions(case, FENCED_OK) == (1.0, [])

    score, violations = score_instructions(case, FENCED_TRUNCATED)
    assert score == 0.0
    assert "syntax error" in violations[0]


def test_syntax_combines_with_other_constraints() -> None:
    case = EvalCase(
        case_id="c",
        query="q",
        must_include=["def add"],
        must_not_include=["eval("],
        format_tags=["python_syntax"],
    )
    score, violations = score_instructions(case, FENCED_OK)
    assert score == 1.0
    assert violations == []

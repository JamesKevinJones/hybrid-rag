"""Module 5b - the evaluation harness. Local metrics, zero cloud spend.

Three scores per case, none of which need a judge model:

  Groundedness         does the answer stay inside the retrieved context, or
                       does it invent? Scored by embedding each answer sentence
                       locally and asking whether any context chunk supports it,
                       with a lexical-overlap fallback for short quotations.

  Instruction precision did it obey positive constraints, NEGATIVE constraints
                       and format tags? Negative constraints are the interesting
                       half - "do not mention X" is where small models fail
                       first, and an LLM judge is unnecessary to check it.

  Cost per resolution  tokens and milliseconds divided by the number of cases
                       that actually PASSED. Averaging cost over all attempts
                       flatters a fast model that is wrong, which is the exact
                       trade this project exists to avoid making by accident.

The Opus baseline is REPLAYED from recorded completions, never re-run. An
evaluation harness that spends frontier tokens every time you check whether you
can stop spending frontier tokens is self-defeating; the orchestrator already
harvests every Level 2 answer it produces, so the baseline is free.
"""

from __future__ import annotations

import ast
import re
import statistics
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import orjson
from pydantic import BaseModel, Field

from hybrid_rag.config import EvalSettings
from hybrid_rag.llm.ollama_client import EmbeddingBackend, LocalInferenceError, TextGenerator
from hybrid_rag.rag.local_rag import LocalRAGEngine

_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n{2,}")
_WORD = re.compile(r"[A-Za-z0-9_]+")
_FENCED = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
# SIM905 wants a literal here rather than `.split()`, so a literal it is.
_STOPWORDS = frozenset(
    [
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "of", "to", "in",
        "on", "for", "with", "and", "or", "not", "it", "its", "this", "that", "as", "at", "by",
        "from", "into", "if", "then", "than", "so", "but", "do", "does", "did", "can", "could",
        "will", "would", "should", "may", "might"
    ]
)


# --------------------------------------------------------------------------
# Cases and results
# --------------------------------------------------------------------------


class EvalCase(BaseModel):
    """One test query plus what a correct answer must and must not contain."""

    case_id: str
    query: str
    must_include: list[str] = Field(default_factory=list)
    # Negative constraints. A model that ignores these is unsafe to promote
    # regardless of how well it scores on groundedness.
    must_not_include: list[str] = Field(default_factory=list)
    # Supported tags: "startswith:<text>", "max_words:<n>", "min_words:<n>",
    # "json", "no_markdown", "cites" (must reference a [n] source marker),
    # "python_syntax" (any emitted code must actually parse).
    format_tags: list[str] = Field(default_factory=list)

    @property
    def constraint_count(self) -> int:
        return len(self.must_include) + len(self.must_not_include) + len(self.format_tags)


class Response(BaseModel):
    """What a responder produced, plus the context it was allowed to use."""

    text: str
    context_chunks: list[str] = Field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    error: str | None = None


class CaseResult(BaseModel):
    case_id: str
    groundedness: float
    instruction_precision: float
    latency_ms: float
    tokens_in: int
    tokens_out: int
    passed: bool
    unsupported: list[str] = Field(default_factory=list)
    violations: list[str] = Field(default_factory=list)
    error: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out


class EvalReport(BaseModel):
    label: str
    n_cases: int
    groundedness_mean: float
    instruction_precision_mean: float
    pass_rate: float
    latency_p50_ms: float
    latency_p95_ms: float
    total_tokens: int
    # The headline number: what one *usable* answer actually costs.
    tokens_per_resolution: float
    ms_per_resolution: float
    results: list[CaseResult] = Field(default_factory=list)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(orjson.dumps(self.model_dump(), option=orjson.OPT_INDENT_2))
        return path

    def summary(self) -> str:
        return (
            f"{self.label}: grounded={self.groundedness_mean:.3f} "
            f"instr={self.instruction_precision_mean:.3f} pass={self.pass_rate:.0%} "
            f"p95={self.latency_p95_ms:.0f}ms tok/res={self.tokens_per_resolution:.0f}"
        )


# --------------------------------------------------------------------------
# Responders
# --------------------------------------------------------------------------


@runtime_checkable
class Responder(Protocol):
    @property
    def label(self) -> str: ...

    async def respond(self, case: EvalCase) -> Response: ...


_EVAL_SYSTEM = (
    "You are a grounded technical assistant. Answer only from the provided "
    "context. If the context does not contain the answer, say so plainly."
)
_EVAL_PROMPT = "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


class LocalResponder:
    """Runs the real local pipeline: retrieve, then generate on the resident model.

    `model` defaults to the resident generator. Pass a candidate adapter's name
    here to evaluate it - that is the only place the promotion gate needs to
    vary, and it is why this parameter exists at all.
    """

    def __init__(
        self,
        *,
        rag: LocalRAGEngine,
        generator: TextGenerator,
        model: str | None = None,
        label: str | None = None,
        top_k: int = 4,
    ) -> None:
        self._rag = rag
        self._generator = generator
        self._model = model
        self._label = label or (model or "local")
        self._top_k = top_k

    @property
    def label(self) -> str:
        return self._label

    async def respond(self, case: EvalCase) -> Response:
        t0 = time.perf_counter()
        try:
            retrieval = await self._rag.query(case.query, top_k=self._top_k)
            chunks = [s.chunk.text for s in retrieval.results]
            context = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(chunks))
            completion = await self._generator.generate(
                _EVAL_PROMPT.format(context=context or "(none)", question=case.query),
                model=self._model,
                system=_EVAL_SYSTEM,
            )
        except LocalInferenceError as exc:
            return Response(
                text="",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                error=f"{type(exc).__name__}: {exc}"[:200],
            )
        return Response(
            text=completion.text,
            context_chunks=chunks,
            tokens_in=completion.tokens_in,
            tokens_out=completion.tokens_out,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )


class RecordedResponder:
    """Replays a baseline from JSONL. Spends nothing.

    Expected shape per line, which is what the orchestrator's harvest writes
    plus the fields the eval needs:

        {"case_id": "...", "text": "...", "context_chunks": [...],
         "tokens_in": 0, "tokens_out": 0, "latency_ms": 0.0}
    """

    def __init__(self, path: Path, *, label: str = "opus-baseline") -> None:
        self._label = label
        self._by_case: dict[str, Response] = {}
        if path.exists():
            for line in path.read_bytes().splitlines():
                if not line.strip():
                    continue
                try:
                    row = orjson.loads(line)
                    self._by_case[str(row["case_id"])] = Response.model_validate(row)
                except (orjson.JSONDecodeError, KeyError, ValueError):
                    continue

    @property
    def label(self) -> str:
        return self._label

    @property
    def case_ids(self) -> set[str]:
        return set(self._by_case)

    async def respond(self, case: EvalCase) -> Response:
        recorded = self._by_case.get(case.case_id)
        if recorded is None:
            return Response(text="", error=f"no recorded baseline for {case.case_id}")
        return recorded


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def split_sentences(text: str, *, min_chars: int) -> list[str]:
    """Sentences worth scoring. Boilerplate is excluded, not counted as ungrounded."""
    return [s.strip() for s in _SENTENCE.split(text) if len(s.strip()) >= min_chars]


def content_words(text: str) -> set[str]:
    return {w for w in (m.group(0).lower() for m in _WORD.finditer(text)) if w not in _STOPWORDS}


def overlap_ratio(sentence: str, chunk: str) -> float:
    """Fraction of the sentence's content words that appear in the chunk.

    The embedding fallback. A short verbatim quotation embeds oddly - too few
    tokens for the vector to be stable - but scores 1.0 here, correctly.
    """
    words = content_words(sentence)
    if not words:
        return 0.0
    return len(words & content_words(chunk)) / len(words)


def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-9, None)
    bn = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-9, None)
    return np.asarray(an @ bn.T)


class GroundednessScorer:
    """Local embeddings plus lexical overlap. No judge model, no cloud call."""

    def __init__(self, embedder: EmbeddingBackend, settings: EvalSettings) -> None:
        self._embedder = embedder
        self._s = settings

    async def score(self, answer: str, context_chunks: Sequence[str]) -> tuple[float, list[str]]:
        sentences = split_sentences(answer, min_chars=self._s.min_scored_sentence_chars)
        if not sentences:
            # Nothing substantive was claimed, so nothing was hallucinated.
            return 1.0, []
        if not context_chunks:
            # Every claim is unsupported by definition - there was no context.
            return 0.0, sentences

        similarity: np.ndarray | None = None
        try:
            embedded = await self._embedder.embed([*sentences, *context_chunks])
            vectors = np.asarray(embedded.vectors, dtype=np.float32)
            similarity = _cosine_matrix(vectors[: len(sentences)], vectors[len(sentences) :])
        except LocalInferenceError:
            # Degrade to lexical overlap alone rather than failing the run.
            similarity = None

        unsupported: list[str] = []
        for i, sentence in enumerate(sentences):
            best_sim = float(similarity[i].max()) if similarity is not None else 0.0
            best_overlap = max(overlap_ratio(sentence, c) for c in context_chunks)
            if (
                best_sim < self._s.support_similarity
                and best_overlap < self._s.support_overlap
            ):
                unsupported.append(sentence[:160])
        supported = len(sentences) - len(unsupported)
        return round(supported / len(sentences), 4), unsupported


def extract_code(answer: str) -> str | None:
    """The Python in an answer, or None if it contains none.

    Prefers fenced blocks. Falls back to treating the whole answer as code only
    when it plainly looks like code - a bare prose sentence is not a syntax
    error, it is simply not code, and reporting it as one would make the
    `python_syntax` tag fire on every text answer.
    """
    blocks = _FENCED.findall(answer)
    if blocks:
        return "\n\n".join(b.strip() for b in blocks)
    stripped = answer.strip()
    if not stripped:
        return None
    looks_like_code = any(
        stripped.startswith(kw)
        for kw in ("def ", "class ", "import ", "from ", "async def ", "@")
    )
    return stripped if looks_like_code else None


def python_syntax_error(answer: str) -> str | None:
    """None if the answer's code parses (or contains none). Otherwise the reason.

    A syntax check is not a correctness check and is not pretending to be one.
    It catches the specific failure a small distilled model makes constantly -
    emitting code that cannot even be parsed - which no amount of embedding
    similarity will notice, because a truncated function is lexically almost
    identical to a complete one.
    """
    code = extract_code(answer)
    if code is None:
        return None
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return f"python syntax error at line {exc.lineno}: {exc.msg}"
    except ValueError as exc:  # null bytes and similar
        return f"python could not be parsed: {exc}"
    return None


def score_instructions(case: EvalCase, answer: str) -> tuple[float, list[str]]:
    """Positive constraints, negative constraints and format tags."""
    if case.constraint_count == 0:
        return 1.0, []

    violations: list[str] = []
    lowered = answer.lower()

    for needle in case.must_include:
        if needle.lower() not in lowered:
            violations.append(f"missing: {needle!r}")
    for needle in case.must_not_include:
        if needle.lower() in lowered:
            violations.append(f"forbidden: {needle!r}")

    for tag in case.format_tags:
        name, _, arg = tag.partition(":")
        if name == "startswith" and not answer.strip().lower().startswith(arg.lower()):
            violations.append(f"format: must start with {arg!r}")
        elif name in {"max_words", "min_words"}:
            # A hand-written case file is a text file; a typo here should fail
            # the case, not crash the whole suite half way through.
            limit = int(arg) if arg.strip().lstrip("-").isdigit() else None
            words = len(answer.split())
            if limit is None:
                violations.append(f"format: unparseable {name} argument {arg!r}")
            elif name == "max_words" and words > limit:
                violations.append(f"format: over {limit} words ({words})")
            elif name == "min_words" and words < limit:
                violations.append(f"format: under {limit} words ({words})")
        elif name == "json":
            try:
                orjson.loads(answer.strip())
            except orjson.JSONDecodeError:
                violations.append("format: not valid JSON")
        elif name == "no_markdown" and re.search(r"(^|\n)\s*[#*\-]\s|```", answer):
            violations.append("format: contains markdown")
        elif name == "cites" and not re.search(r"\[\d+\]", answer):
            violations.append("format: no [n] source citation")
        elif name == "python_syntax":
            problem = python_syntax_error(answer)
            if problem is not None:
                violations.append(f"format: {problem}")

    satisfied = case.constraint_count - len(violations)
    return round(max(0.0, satisfied) / case.constraint_count, 4), violations


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------


class EvalHarness:
    """Runs a responder over a case set and reports the three metrics."""

    def __init__(
        self,
        *,
        settings: EvalSettings,
        scorer: GroundednessScorer,
        pass_groundedness: float = 0.8,
    ) -> None:
        self._s = settings
        self._scorer = scorer
        self._pass_groundedness = pass_groundedness

    async def run(self, responder: Responder, cases: Sequence[EvalCase]) -> EvalReport:
        results: list[CaseResult] = []
        for case in cases:
            response = await responder.respond(case)
            if response.error:
                results.append(
                    CaseResult(
                        case_id=case.case_id,
                        groundedness=0.0,
                        instruction_precision=0.0,
                        latency_ms=response.latency_ms,
                        tokens_in=response.tokens_in,
                        tokens_out=response.tokens_out,
                        passed=False,
                        error=response.error,
                    )
                )
                continue

            grounded, unsupported = await self._scorer.score(
                response.text, response.context_chunks
            )
            precision, violations = score_instructions(case, response.text)
            results.append(
                CaseResult(
                    case_id=case.case_id,
                    groundedness=grounded,
                    instruction_precision=precision,
                    latency_ms=response.latency_ms,
                    tokens_in=response.tokens_in,
                    tokens_out=response.tokens_out,
                    # A pass needs BOTH. An ungrounded answer that follows the
                    # format is still wrong, and a grounded one that ignores a
                    # negative constraint is still unusable.
                    passed=grounded >= self._pass_groundedness and precision >= 1.0,
                    unsupported=unsupported,
                    violations=violations,
                )
            )
        return self._aggregate(responder.label, results)

    @staticmethod
    def _aggregate(label: str, results: Sequence[CaseResult]) -> EvalReport:
        if not results:
            return EvalReport(
                label=label,
                n_cases=0,
                groundedness_mean=0.0,
                instruction_precision_mean=0.0,
                pass_rate=0.0,
                latency_p50_ms=0.0,
                latency_p95_ms=0.0,
                total_tokens=0,
                tokens_per_resolution=0.0,
                ms_per_resolution=0.0,
            )

        latencies = sorted(r.latency_ms for r in results)
        passes = sum(1 for r in results if r.passed)
        total_tokens = sum(r.total_tokens for r in results)
        total_ms = sum(latencies)
        return EvalReport(
            label=label,
            n_cases=len(results),
            groundedness_mean=round(statistics.fmean(r.groundedness for r in results), 4),
            instruction_precision_mean=round(
                statistics.fmean(r.instruction_precision for r in results), 4
            ),
            pass_rate=round(passes / len(results), 4),
            latency_p50_ms=round(_percentile(latencies, 0.50), 2),
            latency_p95_ms=round(_percentile(latencies, 0.95), 2),
            total_tokens=total_tokens,
            # Divided by PASSES, not attempts. A model that answers twice as
            # fast and fails half the time costs the same, and this says so.
            tokens_per_resolution=round(total_tokens / passes, 2) if passes else float("inf"),
            ms_per_resolution=round(total_ms / passes, 2) if passes else float("inf"),
            results=list(results),
        )


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, round(q * (len(sorted_values) - 1))))
    return float(sorted_values[idx])


def load_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    if not path.exists():
        return cases
    for line in path.read_bytes().splitlines():
        if not line.strip():
            continue
        try:
            cases.append(EvalCase.model_validate(orjson.loads(line)))
        except (orjson.JSONDecodeError, ValueError):
            continue
    return cases

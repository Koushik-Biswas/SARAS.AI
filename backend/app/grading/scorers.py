"""Pair scorers: given student answer windows (premises) and key points (hypotheses),
return entailment and contradiction probabilities.

* LexicalScorer  - dependency-free, OCR-noise tolerant fallback (always available).
                   Term overlap weighted by in-question IDF, with the polarity /
                   negation checks of :mod:`app.grading.text` for contradictions.
* SbertScorer    - Sentence-BERT cosine baseline (PRD 12.4; MVP grader).
* NliScorer      - DeBERTa-v3 NLI cross-encoder on PyTorch (R1).
* OnnxNliScorer  - the same kind of NLI cross-encoder on ONNX Runtime (no PyTorch),
                   registered as "nli-onnx" (see :mod:`app.grading.onnx_nli`).
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from . import text as T

log = logging.getLogger(__name__)

DEFAULT_ONNX_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "nli" / "default"

# A clause must mention at least this share of a key point's (weighted) terms before
# an opposite meaning in it counts as a contradiction of that key point.
MIN_CONTRADICTION_ALIGN = 0.45


def lexical_contradiction(hypothesis: str, premise: str, entail_raw: float = 1.0,
                          weights: dict[str, float] | None = None) -> float:
    """Heuristic contradiction signal for the lexical and SBERT scorers: the clause of
    the premise aligned with the key point states one of its terms with the opposite
    meaning (a true negation operator, a POLARITY opposite or a prefix opposite; see
    :func:`app.grading.text.polarity_check`). The NLI models predict contradiction
    directly. ``entail_raw`` is kept for backward compatibility and ignored."""
    del entail_raw
    pol = T.polarity_check(hypothesis, premise, weights)
    if pol.contradicts and pol.align >= MIN_CONTRADICTION_ALIGN:
        return float(min(1.0, 0.3 + 0.7 * pol.align))
    return 0.0


class PairScorer(ABC):
    name = "base"

    @abstractmethod
    def score(self, premises: list[str], hypotheses: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Return (entail, contra) arrays of shape [len(hypotheses), len(premises)]."""

    @abstractmethod
    def similarity(self, a: str, b: str) -> float: ...


class LexicalScorer(PairScorer):
    """Term overlap. ``weights`` (stem -> importance, e.g. in-question IDF) make terms
    shared by several key points count less; without them every term counts 1."""

    name = "lexical"

    def score(self, premises: list[str], hypotheses: list[str],
              weights: dict[str, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        e = np.zeros((len(hypotheses), len(premises)))
        c = np.zeros_like(e)
        for i, h in enumerate(hypotheses):
            slots = T.key_point_slots(h, weights)
            sub_claims = T.kp_units(h)
            units = [T.key_point_slots(u, weights) for u in sub_claims]
            has_subs = len(sub_claims) > 1 or sub_claims[0] != h
            for j, p in enumerate(premises):
                # Recall of the whole key point, or the mean over its sub-claims (asides
                # dropped), or - for an "e.g." list, which is any-of - main claim + example.
                unit_recall = float(np.mean([T.slot_recall(u, p) for u in units])) if has_subs else 0.0
                recall = max(T.slot_recall(slots, p), unit_recall, T.example_recall(h, p, weights))
                raw = 0.85 * recall + 0.15 * T.char_cosine(h, p)
                ent = float(np.clip(1.2 * raw - 0.05, 0, 1))
                pol = T.polarity_check(h, p, units=units)
                if pol.conflicts:
                    ent *= 0.3  # an opposite statement is not evidence for the key point
                if pol.contradicts and pol.align >= MIN_CONTRADICTION_ALIGN:
                    c[i, j] = min(1.0, 0.3 + 0.7 * pol.align)
                e[i, j] = ent
        return e, c

    def similarity(self, a: str, b: str) -> float:
        sa, sb = set(T.content_stems(a)), set(T.content_stems(b))
        if not sa or not sb:
            return 0.0
        matched = sum(1 for x in sb if any(T.fuzzy_equal(x, y) for y in sa))
        recall = matched / len(sb)
        precision = sum(1 for y in sa if any(T.fuzzy_equal(x, y) for x in sb)) / len(sa)
        f1 = 0.0 if recall + precision == 0 else 2 * recall * precision / (recall + precision)
        return float(np.clip(0.6 * f1 + 0.4 * T.char_cosine(a, b), 0, 1))


class SbertScorer(PairScorer):
    name = "sbert"
    _lock = threading.Lock()

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        self._cache: dict[str, np.ndarray] = {}  # embedding cache (PRD 18.2)

    @staticmethod
    def available() -> bool:
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            return False
        return True

    def _embed(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(self.model_name, device="cpu")
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if todo:
            vecs = self._model.encode(todo, normalize_embeddings=True, batch_size=16, show_progress_bar=False)
            for t, v in zip(todo, vecs):
                if len(self._cache) > 5000:
                    self._cache.clear()
                self._cache[t] = v
        return np.stack([self._cache[t] for t in texts])

    def score(self, premises: list[str], hypotheses: list[str]) -> tuple[np.ndarray, np.ndarray]:
        if not premises or not hypotheses:
            return np.zeros((len(hypotheses), len(premises))), np.zeros((len(hypotheses), len(premises)))
        cos = self._embed(hypotheses) @ self._embed(premises).T
        e = np.clip((cos - 0.25) / 0.55, 0, 1)  # map MiniLM cosine onto a 0–1 match score
        c = np.zeros_like(e)
        for i, h in enumerate(hypotheses):
            for j, p in enumerate(premises):
                pol = T.polarity_check(h, p)
                if pol.conflicts:
                    e[i, j] *= 0.3
                if pol.contradicts and pol.align >= MIN_CONTRADICTION_ALIGN:
                    c[i, j] = min(1.0, 0.3 + 0.7 * pol.align)
        return e, c

    def similarity(self, a: str, b: str) -> float:
        if not a.strip() or not b.strip():
            return 0.0
        v = self._embed([a, b])
        return float(np.clip((float(v[0] @ v[1]) - 0.2) / 0.65, 0, 1))


class NliScorer(SbertScorer):
    """Each key point is the hypothesis and each student window the premise;
    entailment and contradiction come straight from the NLI head (PRD 13.2, R1)."""

    name = "nli"

    def __init__(self, nli_model: str, sbert_model: str) -> None:
        super().__init__(sbert_model)
        self.nli_model = nli_model
        self._ce = None
        self._labels: dict[str, int] = {}

    @staticmethod
    def available() -> bool:
        return SbertScorer.available()

    def _load_ce(self) -> None:
        if self._ce is not None:
            return
        with self._lock:
            if self._ce is None:
                from sentence_transformers import CrossEncoder

                ce = CrossEncoder(self.nli_model, device="cpu")
                id2label = getattr(ce.model.config, "id2label", {}) or {}
                labels = {str(v).lower(): int(k) for k, v in id2label.items()}
                if not {"entailment", "contradiction"} <= labels.keys():
                    labels = {"contradiction": 0, "entailment": 1, "neutral": 2}  # cross-encoder/nli-* default
                self._labels = labels
                self._ce = ce

    def score(self, premises: list[str], hypotheses: list[str]) -> tuple[np.ndarray, np.ndarray]:
        e = np.zeros((len(hypotheses), len(premises)))
        c = np.zeros_like(e)
        if not premises or not hypotheses:
            return e, c
        self._load_ce()
        pairs = [(p, h) for h in hypotheses for p in premises]
        logits = np.asarray(self._ce.predict(pairs, batch_size=16, show_progress_bar=False))
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        e[:] = probs[:, self._labels["entailment"]].reshape(len(hypotheses), len(premises))
        c[:] = probs[:, self._labels["contradiction"]].reshape(len(hypotheses), len(premises))
        return e, c


# ------------------------------------------------------------------ registry
def _onnx_scorer_class():  # noqa: ANN202
    """OnnxNliScorer, or None when the module cannot be imported (guarded: optional)."""
    try:
        from .onnx_nli import OnnxNliScorer
    except ImportError as exc:
        log.debug("nli-onnx scorer unavailable: %s", exc)
        return None
    return OnnxNliScorer


def onnx_available(model_dir: Path | str | None = None) -> bool:
    cls = _onnx_scorer_class()
    if cls is None:
        return False
    try:
        return bool(cls.available(Path(model_dir) if model_dir is not None else DEFAULT_ONNX_MODEL_DIR))
    except Exception as exc:  # noqa: BLE001 - a broken install must not break grading
        log.warning("nli-onnx availability check failed: %s", exc)
        return False


_scorers: dict[str, PairScorer] = {}
_scorers_lock = threading.Lock()
_auto_logged: set[str] = set()


def resolve_auto(onnx_model_dir: Path | str | None = None) -> str:
    """The scorer 'auto' picks: NLI on PyTorch (sentence-transformers installed) ->
    NLI on ONNX Runtime (model files + onnxruntime present) -> SBERT -> lexical."""
    if NliScorer.available():
        return "nli"
    if onnx_available(onnx_model_dir):
        return "nli-onnx"
    if SbertScorer.available():
        return "sbert"
    return "lexical"


def get_scorer(kind: str, sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2",
               nli_model: str = "cross-encoder/nli-deberta-v3-xsmall",
               onnx_model_dir: Path | str | None = None) -> PairScorer:
    """Scorer by name: lexical | sbert | nli | nli-onnx | auto. Scorers are cached
    per process (models load lazily on first use)."""
    model_dir = Path(onnx_model_dir) if onnx_model_dir is not None else DEFAULT_ONNX_MODEL_DIR
    with _scorers_lock:
        if kind == "auto":
            kind = resolve_auto(model_dir)
            if kind not in _auto_logged:
                _auto_logged.add(kind)
                log.info("grader=auto selected the %r scorer", kind)
        key = f"nli-onnx:{model_dir}" if kind == "nli-onnx" else kind
        if key not in _scorers:
            if kind == "lexical":
                _scorers[key] = LexicalScorer()
            elif kind == "sbert":
                if not SbertScorer.available():
                    raise RuntimeError("sentence-transformers is not installed")
                _scorers[key] = SbertScorer(sbert_model)
            elif kind == "nli":
                if not NliScorer.available():
                    raise RuntimeError("sentence-transformers is not installed")
                _scorers[key] = NliScorer(nli_model, sbert_model)
            elif kind == "nli-onnx":
                cls = _onnx_scorer_class()
                if cls is None or not cls.available(model_dir):
                    raise RuntimeError(f"nli-onnx is unavailable: install onnxruntime + tokenizers and run "
                                       f"'python -m scripts.download_models' (model dir {model_dir})")
                _scorers[key] = cls(model_dir, max_length=256, batch_size=8, threads=2)
            else:
                raise ValueError(f"Unknown grader {kind!r}")
        return _scorers[key]


def _onnx_installed(model_dir: Path | None = None) -> bool:
    """Cheap capability check (no onnxruntime import): model files and packages present."""
    import importlib.util

    d = model_dir or DEFAULT_ONNX_MODEL_DIR
    files = ("model.onnx", "tokenizer.json", "config.json")
    return all((d / f).is_file() for f in files) and all(
        importlib.util.find_spec(m) is not None for m in ("onnxruntime", "tokenizers"))


def grader_capabilities() -> dict:
    return {"lexical": True, "sbert": SbertScorer.available(), "nli": NliScorer.available(),
            "nli-onnx": _onnx_installed()}

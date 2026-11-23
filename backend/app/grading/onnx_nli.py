"""Local NLI cross-encoder on ONNX Runtime (no PyTorch, no network at runtime).

Meaning-aware entailment / contradiction scoring that fits the 4 GB low-resource
laptop profile: ``onnxruntime`` + the ``tokenizers`` package + an int8 model that
``python -m scripts.download_models`` puts in ``backend/models/nli/default/``.

Default model: cross-encoder/nli-deberta-v3-xsmall (Apache-2.0), dynamic-int8
ONNX export (87 MB). Each key point is the hypothesis and a window of the
student's answer the premise, fed to the model as the pair (premise, hypothesis);
the label order comes from the model's ``config.json`` (id2label), never assumed.

Measured on the 4 GB / 4-core dev laptop (other jobs running, threads=2, batch 8):
model.onnx 87.4 MB (96 MB with tokenizer); load 2-4 s with a transient peak working
set of ~430 MB (ONNX Runtime graph build); ~160-250 MB working set / ~370 MB private
bytes once loaded; 45-85 ms per uncached (premise, hypothesis) pair; cache hits ~0.

Model limitations (also documented for teachers/examiners; see the evaluation
numbers in the grading report):
* Trained on SNLI + MultiNLI (everyday/news English), not exam answers. Domain terms,
  OCR noise and very long premises degrade it; it is a signal for the grader, not a
  judge. Scores are softmax probabilities, not calibrated confidences.
* Unrelated premise/hypothesis pairs often get a HIGH contradiction probability (an
  SNLI artefact: off-topic captions were labelled contradiction). Never take the max
  contradiction over every sentence of an answer; only trust contradiction from a
  window that is about the same thing as the key point (lexical/semantic gate).
* It checks that a window entails the WHOLE hypothesis. Compound key points ("A,
  whereas B"; "A; B"), parenthetical lists and rubric descriptors ("Names a valid
  way to ...", "e.g. ...") are rarely entailed even by correct answers; key points
  should be atomic declarative statements (or be split into sub-claims).
* Partial answers are not separated from missing ones (both get entailment ~0), so
  partial credit needs another signal.
DEV benchmark (42 answers, 148 key points; windows = sentences + adjacent pairs):
matched-vs-missed ROC-AUC 0.84 (0.89 with key points split into sub-claims); at
entailment >= 0.3 precision 0.88 / recall 0.57 for "matched"; max contradiction over
all windows >= 0.5 fires on 52 % of non-contradicted key points (see the grading
report for thresholds).
* The int8 quantisation changes probabilities slightly against the fp32 model.
* The model is *dynamically* quantised: each DynamicQuantizeLinear node picks one
  scale / zero point over the whole activation tensor [batch, tokens, hidden]. In a
  batch, a pair's probabilities therefore depend on its batch-mates and on the padding,
  and with the first-write cache on the order in which pairs were first seen. So pairs
  run one at a time (no padding) by default, which makes every score a function of its
  pair alone. Batching (``batch_size`` > 1) needs ``allow_batch_dependent=True`` and
  gives scores that depend on the batch; use it only when that is acceptable.
* Special-token strings typed in the text ("[SEP]", "[CLS]", ...) would be encoded as
  the real special tokens and move the premise / hypothesis boundary, so they are
  broken up ("[ SEP ]") before tokenisation. Ordinary text is unchanged.
* Input is truncated to ``max_length`` tokens (pair, longest-first), so the grader
  must feed sentence windows, not whole essays.
* It does not do arithmetic or check numbers/units; numeric checks stay in code.
* Questions (interrogatives) are not propositions: NLI scores against a question
  prompt are not meaningful (do not use ``similarity`` for copied-question checks).
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .scorers import PairScorer

log = logging.getLogger(__name__)

REQUIRED_FILES = ("model.onnx", "tokenizer.json", "config.json")
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "nli" / "default"

_deps_ok: bool | None = None
_deps_lock = threading.Lock()


def _deps_importable() -> bool:
    """True when onnxruntime and tokenizers import (checked once per process)."""
    global _deps_ok
    if _deps_ok is None:
        with _deps_lock:
            if _deps_ok is None:
                try:
                    import onnxruntime  # noqa: F401
                    import tokenizers  # noqa: F401
                except Exception:  # ImportError, or a broken native DLL  # noqa: BLE001
                    _deps_ok = False
                else:
                    _deps_ok = True
    return _deps_ok


def read_label_order(config_path: Path) -> dict[str, int]:
    """Map lower-case label name -> logit index from the model's config.json id2label."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    id2label = config.get("id2label") or {}
    labels = {str(v).strip().lower(): int(k) for k, v in id2label.items()}
    if not {"entailment", "contradiction"} <= labels.keys():
        raise ValueError(f"{config_path}: id2label {id2label!r} has no 'entailment'/'contradiction' labels; "
                         "this is not an NLI model the scorer can use")
    return labels


def _norm(text: str) -> str:
    return " ".join(str(text).split())


def _special_pattern(tokenizer) -> re.Pattern | None:  # noqa: ANN001
    """Regex for the tokenizer's special-token strings (longest first), or None."""
    decoder = getattr(tokenizer, "get_added_tokens_decoder", None)
    if decoder is None:
        return None
    specials = {t.content for t in decoder().values() if getattr(t, "special", False) and len(t.content) >= 2}
    if not specials:
        return None
    return re.compile("|".join(re.escape(s) for s in sorted(specials, key=len, reverse=True)))


def _defuse(text: str, pattern: re.Pattern | None) -> str:
    """Break up special-token strings typed in the text ("[SEP]" -> "[ SEP ]") so they are
    encoded as plain words, not as the special tokens that delimit the pair."""
    if pattern is None:
        return text
    return pattern.sub(lambda m: f"{m[0][0]} {m[0][1:-1]} {m[0][-1]}".replace("  ", " "), text)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64)
    z -= z.max(axis=1, keepdims=True)
    np.exp(z, out=z)
    z /= z.sum(axis=1, keepdims=True)
    return z


class OnnxNliScorer(PairScorer):
    """NLI cross-encoder scorer on ONNX Runtime.

    ``score(premises, hypotheses)`` returns (entail, contra), each of shape
    [len(hypotheses), len(premises)], where entail[i, j] = P(entailment | premise j,
    hypothesis i) from the model's softmax (same for contradiction). Empty
    premises/hypotheses score (0, 0) without running the model.

    ``similarity(a, b)`` is the mean of the two entailment directions,
    (P(a => b) + P(b => a)) / 2: about 1 for paraphrases, about 0.5 when one text
    implies the other but not the reverse (e.g. a partial answer against the model
    answer), about 0 for unrelated or contradictory texts. It measures meaning
    equivalence, not topical overlap, so it is not a drop-in replacement for a
    cosine similarity.

    The ONNX session is created lazily on first use (thread-safe) and model runs are
    serialised to keep peak memory bounded. Results are kept in a bounded LRU cache
    keyed by the pair itself, the tuple (premise, hypothesis) after whitespace
    normalisation. A tuple has no separator that two different pairs could share, and the
    key fully determines the model input. Nothing here logs input text.

    Each pair is its own model run (no padding, no batch-mates), so a score depends on the
    pair only, never on what else is scored in the same call or earlier (see the module
    doc). ``batch_size`` > 1 batches pairs only with ``allow_batch_dependent=True``: with
    this dynamically quantised model, batched scores then depend on the batch-mates and the
    padding. Without that flag a larger ``batch_size`` is logged once at load and not used.
    """

    name = "nli-onnx"

    def __init__(self, model_dir: Path, max_length: int = 256, batch_size: int = 1, threads: int = 2,
                 cache_size: int = 4096, allow_batch_dependent: bool = False) -> None:
        self.model_dir = Path(model_dir)
        self.max_length = int(max_length)
        self.batch_size = max(1, int(batch_size))
        self.allow_batch_dependent = bool(allow_batch_dependent)
        self.threads = max(1, int(threads))
        self.cache_size = max(0, int(cache_size))
        self._special_re: re.Pattern | None = None
        self._session = None
        self._tokenizer = None
        self._input_names: tuple[str, ...] = ()
        self._labels: dict[str, int] = {}
        self._pad_id = 0
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._cache: OrderedDict[tuple[str, str], tuple[float, float]] = OrderedDict()
        self._cache_lock = threading.Lock()
        self.stats = {"pairs": 0, "cache_hits": 0, "model_pairs": 0, "batches": 0, "infer_seconds": 0.0}

    # ----------------------------------------------------------------- availability
    @staticmethod
    def available(model_dir: Path | str | None = None) -> bool:
        """True when the model files exist and onnxruntime + tokenizers import."""
        d = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIR
        if not all((d / f).is_file() for f in REQUIRED_FILES):
            return False
        return _deps_importable()

    # ----------------------------------------------------------------- loading
    @property
    def labels(self) -> dict[str, int]:
        """Label name -> logit index (loads config.json on first access)."""
        if not self._labels:
            self._labels = read_label_order(self.model_dir / "config.json")
        return dict(self._labels)

    @property
    def loaded(self) -> bool:
        return self._session is not None

    @property
    def effective_batch_size(self) -> int:
        """Pairs per model run: 1 unless batch-dependent scores were explicitly allowed."""
        return self.batch_size if self.allow_batch_dependent else 1

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        with self._load_lock:
            if self._session is not None:
                return
            missing = [f for f in REQUIRED_FILES if not (self.model_dir / f).is_file()]
            if missing:
                raise RuntimeError(f"NLI model files missing in {self.model_dir}: {missing}. "
                                   "Run: cd backend && .venv/Scripts/python -m scripts.download_models")
            import onnxruntime as ort
            from tokenizers import Tokenizer

            t0 = time.perf_counter()
            labels = read_label_order(self.model_dir / "config.json")
            config = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))
            max_pos = int(config.get("max_position_embeddings") or 512)
            self.max_length = max(8, min(self.max_length, max_pos))

            tok = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
            tok.no_padding()  # padded per batch below (single-pair runs need no padding)
            tok.enable_truncation(self.max_length, strategy="longest_first")
            special_re = _special_pattern(tok)
            pad_token = None
            if config.get("pad_token_id") is not None:
                pad_token = tok.id_to_token(int(config["pad_token_id"]))
            pad_id = tok.token_to_id(pad_token) if pad_token else None
            if pad_id is None:
                for cand in ("[PAD]", "<pad>"):
                    pad_id = tok.token_to_id(cand)
                    if pad_id is not None:
                        break
            self._pad_id = int(pad_id or 0)

            so = ort.SessionOptions()
            so.intra_op_num_threads = self.threads
            so.inter_op_num_threads = 1
            so.enable_cpu_mem_arena = False
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            so.add_session_config_entry("session.intra_op.allow_spinning", "0")  # no busy-wait on a shared CPU
            try:
                session = ort.InferenceSession(str(self.model_dir / "model.onnx"), so,
                                               providers=["CPUExecutionProvider"])
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"cannot load NLI model {self.model_dir / 'model.onnx'}: {exc}") from exc
            names = tuple(i.name for i in session.get_inputs())
            unknown = set(names) - {"input_ids", "attention_mask", "token_type_ids"}
            if unknown or "input_ids" not in names:
                raise RuntimeError(f"unsupported NLI model inputs {names}")
            n_out = session.get_outputs()[0].shape[-1]
            if isinstance(n_out, int) and n_out <= max(labels.values()):
                raise RuntimeError(f"model has {n_out} outputs but id2label needs {max(labels.values()) + 1}")

            self._labels, self._tokenizer, self._input_names = labels, tok, names
            self._special_re = special_re
            self._session = session
            log.info("nli-onnx loaded dir=%s inputs=%s labels=%s max_length=%d threads=%d pairs_per_run=%d "
                     "load_ms=%.0f", self.model_dir.name, ",".join(names), labels, self.max_length, self.threads,
                     self.effective_batch_size, (time.perf_counter() - t0) * 1000)
            if self.batch_size > 1 and not self.allow_batch_dependent:
                log.info("nli-onnx: batch_size=%d not used; each pair runs alone, because batching makes this "
                         "dynamically quantised model's scores depend on the batch-mates "
                         "(allow_batch_dependent=True enables it)", self.batch_size)

    def unload(self) -> None:
        """Drop the ONNX session (frees ~100-150 MB); it reloads lazily on next use."""
        with self._load_lock, self._infer_lock:
            self._session = None
            self._tokenizer = None

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()

    # ----------------------------------------------------------------- inference
    def _run(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Softmax probabilities [len(pairs), n_labels] for (premise, hypothesis) pairs."""
        while True:
            self._ensure_loaded()
            with self._infer_lock:
                session, tokenizer = self._session, self._tokenizer
                if session is None or tokenizer is None:
                    continue  # unload() ran between loading and taking the lock: load again
                return self._run_locked(session, tokenizer, pairs)

    def _run_locked(self, session, tokenizer, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Body of ``_run``; the caller holds ``_infer_lock``. One model run per pair unless
        batch-dependent scores are allowed (see the class doc)."""
        t0 = time.perf_counter()
        sp = self._special_re
        encs = tokenizer.encode_batch([(_defuse(p, sp), _defuse(h, sp)) for p, h in pairs])
        lengths = np.array([len(e.ids) for e in encs])
        size = self.effective_batch_size
        if size == 1:
            runs = [np.array([k]) for k in range(len(encs))]  # exact: no batch-mates, no padding
        else:
            order = np.argsort(lengths, kind="stable")  # similar lengths share a batch: less padding
            runs = [order[start:start + size] for start in range(0, len(order), size)]
        out: np.ndarray | None = None
        for idx in runs:
            width = int(lengths[idx].max())
            ids = np.full((len(idx), width), self._pad_id, dtype=np.int64)
            mask = np.zeros((len(idx), width), dtype=np.int64)
            types = np.zeros((len(idx), width), dtype=np.int64)
            for r, k in enumerate(idx):
                e = encs[k]
                n = len(e.ids)
                ids[r, :n] = e.ids
                mask[r, :n] = 1
                types[r, :n] = e.type_ids
            feed = {"input_ids": ids, "attention_mask": mask, "token_type_ids": types}
            logits = np.asarray(session.run(None, {k: feed[k] for k in self._input_names})[0])
            if logits.ndim != 2 or logits.shape[0] != len(idx) or logits.shape[1] <= max(self._labels.values()):
                raise RuntimeError(f"unexpected NLI logits shape {logits.shape}")
            if out is None:
                out = np.zeros((len(pairs), logits.shape[1]))
            out[idx] = _softmax(logits)
            self.stats["batches"] += 1
        self.stats["model_pairs"] += len(pairs)
        self.stats["infer_seconds"] += time.perf_counter() - t0
        assert out is not None  # pairs is never empty here
        return out

    def score_pairs(self, pairs: list[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray]:
        """(entail, contra) arrays of shape [len(pairs)] for explicit (premise, hypothesis) pairs."""
        keys = [(_norm(p), _norm(h)) for p, h in pairs]
        e = np.zeros(len(keys))
        c = np.zeros(len(keys))
        todo: dict[tuple[str, str], list[int]] = {}
        hits = 0
        with self._cache_lock:
            for i, key in enumerate(keys):
                if not key[0] or not key[1]:
                    continue  # nothing to entail from / nothing to entail: (0, 0)
                cached = self._cache.get(key)
                if cached is not None:
                    self._cache.move_to_end(key)
                    e[i], c[i] = cached
                    hits += 1
                else:
                    todo.setdefault(key, []).append(i)
        if todo:
            uniq = list(todo)
            probs = self._run(uniq)
            ei, ci = self._labels["entailment"], self._labels["contradiction"]
            with self._cache_lock:
                for key, row in zip(uniq, probs):
                    val = (float(row[ei]), float(row[ci]))
                    for i in todo[key]:
                        e[i], c[i] = val
                    if self.cache_size:
                        self._cache[key] = val
                        self._cache.move_to_end(key)
                        while len(self._cache) > self.cache_size:
                            self._cache.popitem(last=False)
        with self._cache_lock:
            self.stats["pairs"] += len(keys)
            self.stats["cache_hits"] += hits
        if todo:
            log.debug("nli-onnx pairs=%d cached=%d computed=%d", len(keys), hits, len(todo))
        return e, c

    # ----------------------------------------------------------------- PairScorer API
    def score(self, premises: list[str], hypotheses: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Return (entail, contra) arrays of shape [len(hypotheses), len(premises)]."""
        shape = (len(hypotheses), len(premises))
        if not premises or not hypotheses:
            return np.zeros(shape), np.zeros(shape)
        e, c = self.score_pairs([(p, h) for h in hypotheses for p in premises])
        return e.reshape(shape), c.reshape(shape)

    def similarity(self, a: str, b: str) -> float:
        """Mean of the two entailment directions, in [0, 1] (see the class docstring)."""
        if not _norm(a) or not _norm(b):
            return 0.0
        e, _ = self.score_pairs([(a, b), (b, a)])
        return float(np.clip(e.mean(), 0.0, 1.0))

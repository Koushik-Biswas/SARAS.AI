"""Local ONNX NLI scorer (app/grading/onnx_nli.py): label order, shapes, orientation,
entailment vs contradiction on clear pairs, the LRU cache, availability checks, and
scores that do not depend on batch-mates or call order.

Most tests need the model files (about 96 MB):
    cd backend && .venv/Scripts/python -m scripts.download_models
Without them those tests are skipped; the fake-session tests still run. One model is
loaded once for the module, configured as scorers.get_scorer builds it (batch_size=8).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from app.grading.onnx_nli import DEFAULT_MODEL_DIR, OnnxNliScorer, read_label_order
from app.grading.scorers import PairScorer

needs_model = pytest.mark.skipif(
    not OnnxNliScorer.available(DEFAULT_MODEL_DIR),
    reason=f"NLI model not downloaded to {DEFAULT_MODEL_DIR} (or onnxruntime/tokenizers missing); "
           "run: cd backend && .venv/Scripts/python -m scripts.download_models",
)

GUITAR = "A man is playing a guitar on stage."
INSTRUMENT = "A person is playing an instrument."
NOBODY = "Nobody is playing music."
OVERFIT = "A model that overfits performs very well on the training data but poorly on new, unseen data."
OVERFIT_PARA = "An overfitted model does badly on data it has not seen before."
OVERFIT_OPP = "An overfitted model performs well on new, unseen data."


@pytest.fixture(scope="module")
def scorer() -> OnnxNliScorer:
    # The arguments scorers.get_scorer("nli-onnx") passes: a batch_size of 8 must not make
    # scores batch-dependent.
    s = OnnxNliScorer(DEFAULT_MODEL_DIR, max_length=256, batch_size=8, threads=2)
    yield s
    s.unload()


def test_is_a_pair_scorer_with_expected_name():
    assert issubclass(OnnxNliScorer, PairScorer)
    assert OnnxNliScorer.name == "nli-onnx"


@needs_model
def test_label_order_comes_from_config(scorer, tmp_path):
    config = json.loads((DEFAULT_MODEL_DIR / "config.json").read_text(encoding="utf-8"))
    expected = {v.lower(): int(k) for k, v in config["id2label"].items()}
    assert scorer.labels == expected
    assert {"entailment", "contradiction"} <= expected.keys()

    # A permuted id2label is honoured, labels are case-insensitive, and a non-NLI head is rejected.
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"id2label": {"0": "ENTAILMENT", "1": "neutral", "2": "Contradiction"}}))
    assert read_label_order(cfg) == {"entailment": 0, "neutral": 1, "contradiction": 2}
    cfg.write_text(json.dumps({"id2label": {"0": "LABEL_0", "1": "LABEL_1"}}))
    with pytest.raises(ValueError):
        read_label_order(cfg)


def test_softmax_uses_the_configured_label_order(tmp_path):
    """Post-processing picks columns by id2label, not by a hard-coded order (fake session)."""

    class Enc:
        ids, type_ids = [1, 5, 2, 6, 2], [0, 0, 0, 1, 1]

    class FakeTokenizer:
        def encode_batch(self, pairs):
            return [Enc() for _ in pairs]

    class FakeSession:
        def run(self, _outputs, feed):
            n = feed["input_ids"].shape[0]
            return [np.tile(np.array([[2.0, 0.0, -1.0]], dtype=np.float32), (n, 1))]

    (tmp_path / "config.json").write_text(
        json.dumps({"id2label": {"0": "entailment", "1": "neutral", "2": "contradiction"}}))
    s = OnnxNliScorer(tmp_path)
    s._labels = read_label_order(tmp_path / "config.json")
    s._input_names = ("input_ids", "attention_mask")
    s._tokenizer, s._session = FakeTokenizer(), FakeSession()
    e, c = s.score(["p1", "p2"], ["h1"])
    probs = np.exp([2.0, 0.0, -1.0]) / np.exp([2.0, 0.0, -1.0]).sum()
    assert e.shape == c.shape == (1, 2)
    np.testing.assert_allclose(e, probs[0], rtol=1e-6)
    np.testing.assert_allclose(c, probs[2], rtol=1e-6)


@needs_model
def test_output_shapes_and_ranges(scorer):
    premises = [GUITAR, "The sky is blue.", OVERFIT]
    hypotheses = [INSTRUMENT, OVERFIT_PARA]
    scorer.clear_cache()
    e, c = scorer.score(premises, hypotheses)
    assert e.shape == c.shape == (len(hypotheses), len(premises))
    assert np.all((e >= 0) & (e <= 1)) and np.all((c >= 0) & (c <= 1))
    assert np.all(e + c <= 1 + 1e-9)  # neutral takes the rest
    # Element [i, j] is the pair (premise j, hypothesis i) -- recomputed on its own (empty
    # cache), it is exactly the value it got among the other five pairs.
    scorer.clear_cache()
    pe, pc = scorer.score_pairs([(premises[2], hypotheses[1])])
    assert e[1, 2] == pe[0] and c[1, 2] == pc[0]
    # Empty inputs keep the shape and never reach the model.
    assert scorer.score([], hypotheses)[0].shape == (2, 0)
    assert scorer.score(premises, [])[1].shape == (0, 3)
    e2, c2 = scorer.score(["  "], [INSTRUMENT])
    assert e2[0, 0] == 0.0 and c2[0, 0] == 0.0


# Pairs of very different lengths: in a padded batch the short ones would share a
# quantisation scale with (and be padded to) the long ones.
LONG = ("Overfitting happens when a model learns the noise and the accidental details of its training "
        "examples instead of the underlying pattern, so it scores very well on the training set but its "
        "accuracy drops sharply on new, unseen data such as a validation set or real use.")
BATCH_MATES = [(LONG, OVERFIT_PARA), (LONG, OVERFIT_OPP), (GUITAR, NOBODY), (LONG, INSTRUMENT),
               ("Yes.", OVERFIT_PARA), (OVERFIT, LONG), (NOBODY, GUITAR)]
TARGETS = [(OVERFIT, OVERFIT_PARA), (OVERFIT, OVERFIT_OPP), (GUITAR, INSTRUMENT)]


def _alone(scorer, pair):
    scorer.clear_cache()
    e, c = scorer.score_pairs([pair])
    return e[0], c[0]


@needs_model
def test_score_does_not_depend_on_batch_mates(scorer):
    """Review finding harness-onnx #3: the dynamically quantised int8 model gave a pair
    different probabilities alone and in a padded batch of 8 (scorers.get_scorer passes
    batch_size=8). Each pair now runs alone, so the scores are exactly equal."""
    assert scorer.batch_size == 8 and scorer.effective_batch_size == 1
    for target in TARGETS:
        alone = _alone(scorer, target)
        for k in (0, 3, len(BATCH_MATES)):  # the target first, in the middle, last among 7 batch-mates
            scorer.clear_cache()
            batch = BATCH_MATES[:k] + [target] + BATCH_MATES[k:]
            e, c = scorer.score_pairs(batch)
            assert (e[k], c[k]) == alone, (target, k)


@needs_model
def test_score_does_not_depend_on_call_order_or_cache_history(scorer):
    pairs = TARGETS + BATCH_MATES
    scorer.clear_cache()
    e_fwd, c_fwd = scorer.score_pairs(pairs)
    scorer.clear_cache()
    e_rev, c_rev = scorer.score_pairs(pairs[::-1])
    assert e_rev[::-1].tolist() == e_fwd.tolist() and c_rev[::-1].tolist() == c_fwd.tolist()
    # Cache filled by different earlier calls (as in a full run vs a filtered run vs a
    # long-lived worker): every pair still gets the same value.
    for history in (BATCH_MATES[:3], BATCH_MATES[3:], TARGETS[::-1]):
        scorer.clear_cache()
        scorer.score_pairs(history)
        e, c = scorer.score_pairs(pairs)
        assert e.tolist() == e_fwd.tolist() and c.tolist() == c_fwd.tolist()
    scorer.clear_cache()


@needs_model
def test_production_scorer_runs_pairs_one_at_a_time():
    from app.grading.scorers import get_scorer

    s = get_scorer("nli-onnx")  # built with batch_size=8; constructing does not load the model
    assert isinstance(s, OnnxNliScorer) and s.effective_batch_size == 1


@needs_model
def test_special_token_text_cannot_move_the_pair_boundary(scorer):
    """'[SEP]' typed in an answer was encoded as the real separator, so ('water boils [SEP] at
    100', 'it is hot') and ('water boils', 'at 100 [SEP] it is hot') reached the model as
    the same input. It is now encoded as plain text: the inputs differ, and the only
    special tokens are the ones the pair template adds."""
    real = scorer._session
    seen: list[list[int]] = []

    class Recording:
        def run(self, outputs, feed):
            seen.extend(feed["input_ids"].tolist())
            return real.run(outputs, feed)

    a = ("Water boils [SEP] at 100 degrees.", "It is hot.")
    b = ("Water boils", "at 100 degrees. [SEP] It is hot.")
    scorer._ensure_loaded()
    scorer.clear_cache()
    scorer._session = Recording()
    try:
        scorer.score_pairs([a])
        scorer.score_pairs([b])
    finally:
        scorer._session = real
        scorer.clear_cache()
    assert len(seen) == 2 and seen[0] != seen[1]
    cls_id, sep_id = scorer._tokenizer.token_to_id("[CLS]"), scorer._tokenizer.token_to_id("[SEP]")
    for ids in seen:
        assert ids[0] == cls_id and ids[-1] == sep_id and ids.count(sep_id) == 2  # [CLS] p [SEP] h [SEP]


@needs_model
def test_pairs_are_fed_premise_first(scorer):
    """Guitar => instrument holds; instrument => guitar does not."""
    fwd, _ = scorer.score([GUITAR], [INSTRUMENT])
    back, _ = scorer.score([INSTRUMENT], [GUITAR])
    assert fwd[0, 0] > 0.8
    assert fwd[0, 0] > back[0, 0] + 0.3


@needs_model
@pytest.mark.parametrize("premise, hypothesis", [(GUITAR, INSTRUMENT), (OVERFIT, OVERFIT_PARA)])
def test_paraphrase_is_entailed(scorer, premise, hypothesis):
    e, c = scorer.score([premise], [hypothesis])
    assert e[0, 0] > c[0, 0]
    assert e[0, 0] > 0.5


@needs_model
@pytest.mark.parametrize("premise, hypothesis", [(GUITAR, NOBODY), (OVERFIT, OVERFIT_OPP)])
def test_opposite_is_contradicted(scorer, premise, hypothesis):
    e, c = scorer.score([premise], [hypothesis])
    assert c[0, 0] > e[0, 0]
    assert c[0, 0] > 0.5


@needs_model
def test_similarity_is_bounded_and_meaning_aware(scorer):
    para = scorer.similarity(OVERFIT, OVERFIT_PARA)
    opp = scorer.similarity(OVERFIT, OVERFIT_OPP)
    assert 0.0 <= opp <= 1.0 and 0.0 <= para <= 1.0
    assert para > opp
    assert scorer.similarity("", OVERFIT) == 0.0


@needs_model
def test_cache_hit_skips_the_model(scorer):
    scorer.clear_cache()
    before = dict(scorer.stats)
    e1, c1 = scorer.score([GUITAR], [INSTRUMENT])
    assert scorer.stats["model_pairs"] == before["model_pairs"] + 1
    # Same pair again (whitespace differences ignored): served from the cache.
    e2, c2 = scorer.score(["  A man is playing a guitar   on stage. "], [INSTRUMENT])
    assert scorer.stats["model_pairs"] == before["model_pairs"] + 1
    assert scorer.stats["cache_hits"] == before["cache_hits"] + 1
    assert e2[0, 0] == e1[0, 0] and c2[0, 0] == c1[0, 0]


@needs_model
def test_cache_is_bounded_lru(scorer):
    old = scorer.cache_size
    scorer.clear_cache()
    scorer.cache_size = 2
    try:
        scorer.score([GUITAR], [INSTRUMENT, NOBODY])
        scorer.score([GUITAR], [INSTRUMENT])  # touch: now most recently used
        scorer.score([OVERFIT], [OVERFIT_PARA])  # evicts the least recently used (GUITAR, NOBODY)
        assert len(scorer._cache) == 2
        assert set(scorer._cache) == {(GUITAR, INSTRUMENT), (OVERFIT, OVERFIT_PARA)}
    finally:
        scorer.cache_size = old
        scorer.clear_cache()


@needs_model
def test_available_is_false_for_missing_or_incomplete_dir(tmp_path):
    assert OnnxNliScorer.available(tmp_path / "does-not-exist") is False
    (tmp_path / "config.json").write_text("{}")
    assert OnnxNliScorer.available(tmp_path) is False  # no model.onnx / tokenizer.json
    assert OnnxNliScorer.available(DEFAULT_MODEL_DIR) is True


def test_missing_model_raises_a_clear_error_on_use(tmp_path):
    s = OnnxNliScorer(tmp_path)  # constructing is free: the model loads lazily
    assert not s.loaded
    with pytest.raises(RuntimeError, match="download_models"):
        s.score(["a premise"], ["a hypothesis"])


# ----------------------------------------------------------------------------- fake session
class _Enc:
    def __init__(self, premise: str, hypothesis: str) -> None:
        a = [10 + len(w) for w in premise.split()]
        b = [10 + len(w) for w in hypothesis.split()]
        self.ids = [1] + a + [2] + b + [2]
        self.type_ids = [0] * (len(a) + 2) + [1] * (len(b) + 1)


class _Tokenizer:
    def encode_batch(self, pairs):  # noqa: ANN001, ANN201
        return [_Enc(p, h) for p, h in pairs]


class _QuantisedSession:
    """Mimics a DynamicQuantizeLinear model: one scale per whole input tensor, so each row's
    logits shift with the largest value and the padded width of the batch it is run in."""

    def __init__(self) -> None:
        self.batch_shapes: list[tuple[int, int]] = []

    def run(self, _outputs, feed):  # noqa: ANN001, ANN201
        ids, mask = feed["input_ids"], feed["attention_mask"]
        self.batch_shapes.append(ids.shape)
        scale = (ids.max() + ids.shape[1]) / 100.0  # depends on batch-mates and padding
        own = (ids * mask).sum(axis=1) / mask.sum(axis=1) / 10.0  # depends on the row only
        logits = np.stack([own, np.zeros_like(own), np.round(own / scale) * scale], axis=1)
        return [logits.astype(np.float32)]


def _fake_scorer(tmp_path, **kwargs) -> OnnxNliScorer:
    (tmp_path / "config.json").write_text(
        json.dumps({"id2label": {"0": "entailment", "1": "neutral", "2": "contradiction"}}))
    s = OnnxNliScorer(tmp_path, **kwargs)
    s._labels = read_label_order(tmp_path / "config.json")
    s._input_names = ("input_ids", "attention_mask")
    s._tokenizer, s._session = _Tokenizer(), _QuantisedSession()
    return s


FAKE_TARGET = ("short premise", "short claim")
FAKE_MATES = [("a much longer premise with many more words in it than the target", "claim"),
              ("tiny", "an extraordinarily long hypothesis sentence containing considerably longer words"),
              ("medium sized premise here", "medium claim")]


def test_fake_quantised_model_scores_do_not_depend_on_batch_or_order(tmp_path):
    """Regression for finding #3 without the real model: with batch_size=8 (as get_scorer
    passes) a pair alone and a pair among longer batch-mates used to get different scores
    from a batch-sensitive session, and the first-write cache made that order-dependent."""
    s = _fake_scorer(tmp_path, batch_size=8)
    e0, c0 = s.score_pairs([FAKE_TARGET])
    for k in range(len(FAKE_MATES) + 1):
        s.clear_cache()
        e, c = s.score_pairs(FAKE_MATES[:k] + [FAKE_TARGET] + FAKE_MATES[k:])
        assert (e[k], c[k]) == (e0[0], c0[0])
    assert all(shape[0] == 1 for shape in s._session.batch_shapes)  # one pair per run, no padding
    # Cache history: the mates first, then everything, gives the same values as a fresh run.
    s.clear_cache()
    s.score_pairs(FAKE_MATES[::-1])
    e, c = s.score_pairs([FAKE_TARGET] + FAKE_MATES)
    s.clear_cache()
    e_fresh, c_fresh = s.score_pairs([FAKE_TARGET] + FAKE_MATES)
    assert e.tolist() == e_fresh.tolist() and c.tolist() == c_fresh.tolist()


def test_batching_is_opt_in_and_batch_dependent(tmp_path):
    """allow_batch_dependent=True restores padded batches -- and with them the dependence on
    batch-mates that the default avoids (this is why it is off by default)."""
    s = _fake_scorer(tmp_path, batch_size=8, allow_batch_dependent=True)
    assert s.effective_batch_size == 8
    _, c_alone = s.score_pairs([FAKE_TARGET])
    s.clear_cache()
    _, c_batch = s.score_pairs([FAKE_TARGET] + FAKE_MATES)
    assert s._session.batch_shapes[-1][0] == 4  # all four pairs in one padded run
    assert c_batch[0] != c_alone[0]
    assert OnnxNliScorer(tmp_path).effective_batch_size == 1  # default batch_size is 1
    assert OnnxNliScorer(tmp_path, batch_size=8).effective_batch_size == 1


def test_cache_key_is_the_pair_tuple_without_separator_collisions(tmp_path):
    """('a b', 'c') and ('a', 'b c') join to the same text; they are different pairs, so they
    get separate cache entries and separate model runs. Whitespace variants share one."""
    s = _fake_scorer(tmp_path)
    s.score_pairs([("a b", "c"), ("a", "b c"), ("a  b ", " c")])
    assert set(s._cache) == {("a b", "c"), ("a", "b c")}
    assert s.stats["model_pairs"] == 2

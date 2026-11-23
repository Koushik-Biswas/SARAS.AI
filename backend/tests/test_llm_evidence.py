"""Evidence quotes (app.grading.llm_grader.match_quote / find_quote): an AI quote counts
only when it is really the student's words. Review findings ai-layer #0 (fuzzy misquotes
that reverse the meaning) and #1 (1-character / in-word substrings)."""

from __future__ import annotations

import pytest

from app.grading.llm_grader import find_quote, match_quote

LR = "Increasing the learning rate does not make training converge faster."


# ------------------------------------------------------------------ finding #0: misquotes are rejected
@pytest.mark.parametrize(("answer", "misquote"), [
    (LR, "Increasing the learning rate does make training converge faster"),  # dropped "not"
    ("At sea level water boils at 90 degrees Celsius.", "water boils at 100 degrees Celsius"),  # changed number
    ("Light travels at 3 x 10^5 km/s in a vacuum.", "Light travels at 3 x 10^8 km/s"),  # changed exponent
    ("Entropy decreases in an isolated system over time.", "Entropy increases in an isolated system over time"),
    ("Mitochondria is not the powerhouse of the cell.", "Mitochondria is the powerhouse of the cell"),
])
def test_misquotes_that_reverse_the_meaning_are_rejected(answer: str, misquote: str) -> None:
    assert match_quote(answer, misquote) is None
    assert find_quote(answer, misquote) is None


@pytest.mark.parametrize(("answer", "misquote"), [
    ("Anaerobic respiration releases less energy.", "Aerobic respiration releases less energy"),  # a-/an- prefix
    ("The model performs poorly on unseen data.", "The model performs well on unseen data"),  # polarity lexicon
    ("The data is unlabelled and comes from sensors.", "The data is labelled and comes from sensors"),
    ("Hypertonic solutions draw water out of cells.", "Hypotonic solutions draw water out of cells"),
    ("Mitosis produces two identical diploid cells.", "Meiosis produces two identical diploid cells"),
    ("The male gamete is called the sperm.", "The female gamete is called the sperm"),
    ("Charcoal adsorbs gases on its surface.", "Charcoal absorbs gases on its surface"),
    ("It never rains in the desert.", "It always rains in the desert"),
    ("Water boils at 100 degrees Celsius.", "Water boils at 100.5 degrees Celsius"),
    ("They cannot fly at night.", "They can fly at night"),
])
def test_opposites_negators_and_numbers_never_align(answer: str, misquote: str) -> None:
    assert match_quote(answer, misquote) is None


def test_a_negation_left_outside_the_quote_is_not_evidence() -> None:
    # a true substring, but the student said the opposite
    assert match_quote(LR, "make training converge faster") is None
    assert match_quote("Mitochondria is not the powerhouse of the cell.", "the powerhouse of the cell") is None
    # quoting the negation itself is fine
    m = match_quote(LR, "does not make training converge faster")
    assert m is not None and not m.fuzzy and m.span == "does not make training converge faster"
    # a negation of another word, or one in an earlier clause, does not count
    assert find_quote("It does not overfit because dropout prevents co-adaptation.",
                      "dropout prevents co-adaptation") == "dropout prevents co-adaptation"
    assert find_quote("No, the model is accurate on test data.", "the model is accurate on test data") == \
        "the model is accurate on test data"
    assert find_quote("It is not only fast but also accurate.", "fast but also accurate") == "fast but also accurate"


def test_merging_a_whole_extra_word_is_not_an_ocr_split() -> None:
    assert match_quote(LR, "does make training converge faster") is None
    assert match_quote("Self-supervised learning uses pretext tasks.", "supervised learning uses pretext tasks") is None


# ------------------------------------------------------------------ finding #1: word boundaries
def test_quotes_must_match_whole_words_and_whole_numbers() -> None:
    assert find_quote("I do not know.", "o") is None  # inside "do" / "not" / "know"
    assert find_quote("The cat sat.", "at sat") is None
    assert find_quote("Temperature was -5 degrees today.", "5 degrees today") is None  # drops the sign
    assert find_quote("Temperature was -5 degrees today.", "-5 degrees today") == "-5 degrees today"
    assert find_quote("The value 3.5 metres was measured.", "5 metres was measured") is None
    assert find_quote("Light travels at 3 x 10^8 m/s.", "Light travels at 3 x 10") is None
    assert find_quote("The data is un-labelled data.", "labelled data") is None  # inside a compound
    assert find_quote("Self-supervised learning uses pretext tasks.", "supervised learning") is None
    assert find_quote("They can't fly at night.", "They can") is None  # cuts "can't"
    assert find_quote("Newton's laws explain motion.", "Newton's laws explain motion") == \
        "Newton's laws explain motion"


# ------------------------------------------------------------------ OCR noise is still tolerated
@pytest.mark.parametrize(("answer", "quote", "span"), [
    # 'm' read as 'rn'
    ("Plants need the rnitochondria to make energy.", "Plants need the mitochondria to make energy",
     "Plants need the rnitochondria to make energy"),
    # 'l' read as '1'
    ("The ce11 wall protects the plant cell.", "The cell wall protects the plant cell",
     "The ce11 wall protects the plant cell"),
    # one dropped letter
    ("Photosynthsis converts light energy into chemical energy.",
     "Photosynthesis converts light energy into chemical energy",
     "Photosynthsis converts light energy into chemical energy"),
    # the quote has the noise, the answer is clean
    ("Water boils at 100 degrees Celsius.", "Water boi1s at 100 degrees Celsius", "Water boils at 100 degrees Celsius"),
    # an OCR-split word
    ("In photo synthesis plants make food.", "In photosynthesis plants make food", "In photo synthesis plants make food"),
    ("Overfiting: the model lerns the noize in the trainig data, so it fails.",
     "the model learns the noise in the training data", "the model lerns the noize in the trainig data"),
])
def test_realistic_ocr_noise_is_accepted_as_fuzzy_evidence(answer: str, quote: str, span: str) -> None:
    m = match_quote(answer, quote)
    assert m is not None and m.fuzzy
    assert m.span == span  # always the student's own text, never the AI's corrected wording
    assert answer[m.start: m.end] == span


def test_exact_matches_are_not_fuzzy() -> None:
    answer = "Overfiting: the model lerns the noize in the trainig data, so it fails on new data."
    m = match_quote(answer, "  THE MODEL lerns\nthe noize  ")
    assert m is not None and not m.fuzzy and m.span == "the model lerns the noize" and m.words == 5
    assert match_quote(answer, "“so it fails on new data.”").span == "so it fails on new data"
    # only punctuation differs: the same words, so not "fuzzy"
    m = match_quote("Cells, which divide by mitosis, grow.", "Cells which divide by mitosis")
    assert m is not None and not m.fuzzy and m.span == "Cells, which divide by mitosis"
    assert match_quote(answer, "") is None and match_quote(answer, "...") is None and match_quote("", "x y") is None

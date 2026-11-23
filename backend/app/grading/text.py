"""Small, dependency-free text utilities shared by the graders.

Besides tokenising, stemming and sentence splitting, this module holds the
lexical *meaning* checks every scorer can rely on:

* :data:`NEGATIONS` - true negation operators only ("not", "never", "without", n't ...).
  Evaluative words ("poorly", "unseen") are not negations.
* :data:`POLARITY` - an easy-to-extend table of opposite word groups (well vs poorly,
  increase vs decrease, arteries' "away" vs veins' "back" ...), plus a generic prefix
  rule (un-/in-/non-/dis- forms and over-/under- swaps of a key-point word).
* :func:`polarity_check` - compares a key point with the clause of a premise that
  is aligned with it, slot by slot, and reports opposite meanings.
* :func:`role_pairs` / :func:`find_role_swap` - detect a clause that attributes key
  point *i*'s facts to the entity of a parallel key point *j* (arteries vs veins).
* :func:`keyword_present`, :func:`number_values` - rubric keyword and number checks.

Everything here is deterministic, has no model dependencies and never logs text.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache

STOPWORDS = frozenset("""
a an the and or but if then so of to in on at by for with from as is are was were be been being am
it its this that these those there here which who whom whose what when where why how into onto over under
about above below than too very can could should would may might will shall do does did done has have had
having we you they he she i me my our your their his her them us also such just only some any each every
both either neither more most other own same s t
""".split())

# True negation operators (A). "unable", "fails", "lacks", "poorly", "unseen" are
# words with a meaning, not operators: they are handled by POLARITY / the prefix rule.
NEGATIONS = frozenset({"not", "no", "never", "none", "nothing", "nor", "neither", "without", "cannot",
                       "nobody", "nowhere"})
# n't contractions; OCR and fast typists often drop the apostrophe.
_NT_FORMS = frozenset({"dont", "doesnt", "didnt", "isnt", "arent", "wasnt", "werent", "cant", "couldnt", "wont",
                       "wouldnt", "shouldnt", "hasnt", "havent", "hadnt", "mustnt", "neednt", "aint"})
# "not only X but also Y" and hedges do not negate X.
_NOT_NEGATED = frozenset({"only", "just", "merely", "necessarily", "always"})
# Skipped when looking for the word a negation applies to.
_SCOPE_SKIP = frozenset({"really", "actually", "quite", "even", "yet", "still", "any", "a", "an", "the", "be",
                         "been", "being", "is", "are", "was", "were", "do", "does", "did", "have", "has", "had",
                         "very", "too", "so", "at", "all"})
# "without it", "not this": the negation applies to a pronoun, not to the next content word.
_NEG_PRONOUNS = frozenset({"it", "they", "them", "this", "that", "these", "those", "him", "her", "one"})

# Opposite groups judged in the clause aligned with a key point (A). Each row is
# (name, anchored, side A, side B). Words on the same side are interchangeable here
# (badly ~ poorly) and never contradict each other. An *anchored* group only
# conflicts when the opposite words modify the same thing (a neighbouring content
# word matches): "more complex" does not contradict "too little data". Un-anchored
# rows name whole technical concepts. To extend, add a row.
POLARITY: tuple[tuple[str, bool, str, str], ...] = (
    ("quality", True, "well good better best accurate accurately correct correctly effective effectively",
     "poorly poor badly bad worse worst inaccurate inaccurately incorrect incorrectly ineffective wrong wrongly"),
    ("level", True, "high higher highest greater", "low lower lowest"),
    ("size", True, "large larger largest big bigger biggest huge", "small smaller smallest tiny"),
    ("amount", True, "more much many", "less fewer few little"),
    ("direction", True, "increase increases increased increasing rise rises rising rose grow grows growing grew "
                        "raise raises raised",
     "decrease decreases decreased decreasing fall falls falling fell drop drops dropped dropping decline "
     "declines declined reduce reduces reduced reducing shrink shrinks"),
    ("speed", True, "fast faster fastest quick quickly rapid rapidly", "slow slower slowest slowly"),
    ("flow", True, "away", "back towards toward"),
    ("thickness", True, "thick thicker thickest", "thin thinner thinnest"),
    ("labels", False, "labelled labeled", "unlabelled unlabeled"),
    ("supervision", False, "supervised", "unsupervised"),
    ("fitting", False, "overfitting overfit overfits overfitted", "underfitting underfit underfits underfitted"),
    ("complexity", True, "complex complicated", "simple simpler simplest simplistic"),
    ("order", True, "before prior earlier", "after later"),
    ("sign", True, "positive", "negative"),
    ("truth", True, "true", "false"),
    ("redox", True, "oxidation oxidised oxidized oxidise oxidize oxidises oxidizes oxidising oxidizing",
     "reduction reduced reduce reduces reducing"),
    ("division", False, "mitosis mitotic", "meiosis meiotic"),
    ("extremum", True, "maximum maximise maximize maximises maximizes maximising maximizing maximal",
     "minimum minimise minimize minimises minimizes minimising minimizing minimal"),
    ("sameness", True, "same identical similar", "opposite different reverse"),
    ("position", True, "inside internal inner interior", "outside external outer exterior"),
    ("transfer", True, "absorb absorbs absorbed absorbing absorption", "release releases released releasing "
                                                                       "emit emits emitted emission"),
    ("gain", True, "gain gains gained gaining", "loss lose loses lost losing"),
    ("temperature", True, "hot hotter warm warmer", "cold colder cool cooler"),
    ("volume", True, "expand expands expanded expansion", "contract contracts contracted contraction"),
    ("acceptance", True, "accept accepts accepted", "reject rejects rejected"),
    ("heat", False, "exothermic", "endothermic"),
    ("relation", True, "direct directly", "inverse inversely"),
    ("slope", True, "upward upwards ascending", "downward downwards descending"),
)


@dataclass(frozen=True)
class _Pole:
    group: int
    side: int  # +1 side A, -1 side B
    anchored: bool


def _build_poles() -> dict[str, list[_Pole]]:
    poles: dict[str, list[_Pole]] = {}
    for g, (_, anchored, a, b) in enumerate(POLARITY):
        for side, wordlist in ((1, a), (-1, b)):
            for w in wordlist.split():
                poles.setdefault(w, []).append(_Pole(g, side, anchored))
    return poles


_POLES = _build_poles()

FUNCTION_WORDS = frozenset({"is", "are", "was", "were", "be", "it", "the", "a", "an", "this", "that", "when", "because",
                            "so", "which", "can", "will", "has", "have", "means", "by", "on", "to", "of", "in", "we"})
# Common verbs of exam answers: "Mitosis produces two identical diploid cells" is a
# statement even without a function word; a keyword list has no verb.
COMMON_VERBS = frozenset("""
produce produces produced give gives gave carry carries carried form forms formed use uses used make makes made
cause causes caused occur occurs occurred contain contains follow follows increase increases decrease decreases
take takes become becomes lead leads result results depend depends create creates convert converts move moves
flow flows prevent prevents allow allows help helps require requires refer refers involve involves divide divides
store stores absorb absorbs release releases do does did should must learn learns predict predicts find finds
work works reduce reduces get gets keep keeps trap traps emit emits pass passes rise rises fall falls show shows
states says say happens happen gain gains lose loses push pushes pop pops
""".split())

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_SENT = re.compile(r"(?<=[\.\!\?;])\s+|\n+")
_ABBREV_END = re.compile(r"(?:\b(?:i\.e|e\.g|cf|vs|viz|fig|approx|eq|no|resp)\.)$", re.IGNORECASE)
_CLAUSE = re.compile(r"[,;:!?()\[\]]|(?<!\.[a-z])\.(?![a-z0-9])|\b(?:and|but|so|because|while|whilst|whereas|although|though|therefore|"
                     r"hence|thus|which|however)\b", re.IGNORECASE)
_HYPHEN_PREFIX = re.compile(r"\b(over|under|non|un|in|im|il|ir|dis)-(?=[a-z])")
_NUM = re.compile(r"(?<![a-z])-?\d+(?:\.\d+)?(?:\s*(?:%|kg|g|m|cm|mm|km|s|ms|hz|khz|mhz|v|a|w|j|n|k|°c|c))?", re.I)
_NUMVAL = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")


def _norm(text: str) -> str:
    text = (text or "").lower().replace("’", "'").replace("‘", "'").replace("−", "-")
    return _HYPHEN_PREFIX.sub(r"\1", text)  # "over-fitting" -> "overfitting", "un-labelled" -> "unlabelled"


_OCR_DIGITS = str.maketrans({"1": "l", "0": "o"})


def _ocr_fix(w: str) -> str:
    """OCR confusions inside a word: "b1ood" -> "blood", "wa11s" -> "walls", "0hm" -> "ohm".
    Numbers ("10", "10n") and chemistry ("co2") are left alone."""
    if w.isalpha() or w.isdigit() or sum(ch.isalpha() for ch in w) < 2:
        return w
    fixed = w.translate(_OCR_DIGITS)
    return fixed if fixed.replace("'", "").isalpha() else w


def words(text: str) -> list[str]:
    return [_ocr_fix(w) for w in _WORD.findall(_norm(text))]


@lru_cache(maxsize=20000)
def stem(w: str) -> str:
    for suf in ("ational", "ization", "isation", "ations", "ation", "ness", "ments", "ment", "ingly", "ing",
                "edly", "ies", "ied", "ly", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            base = w[: -len(suf)]
            if suf in ("ies", "ied"):
                return base + "y"
            # mapping -> map, fitted -> fit, stopping -> stop (but labelled keeps its l's)
            if suf in ("ing", "ingly", "ed", "edly") and len(base) >= 4 and base[-1] == base[-2] \
                    and base[-1] in "bdgmnprt":
                base = base[:-1]
            return base
    return w


def is_content(w: str) -> bool:
    return w not in STOPWORDS and len(w) > 1 and not is_negator(w)


def content_stems(text: str) -> list[str]:
    return [stem(w) for w in words(text) if is_content(w)]


# ------------------------------------------------------------------ sentences and clauses
def _strip_span(text: str, s: int, e: int) -> tuple[int, int]:
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) offsets of the sentences of ``text``; ``text[s:e]`` is the sentence,
    so evidence quoted from these spans always occurs literally in the answer."""
    text = text or ""
    raw: list[tuple[int, int]] = []
    pos = 0
    for m in _SENT.finditer(text):
        if _ABBREV_END.search(text[max(0, m.start() - 8):m.start()]) or \
                (text[max(0, m.start() - 4):m.start()].lower() == "etc." and text[m.end():m.end() + 1].islower()):
            continue  # "i.e. it fails", "e.g. dropout": not a sentence end
        raw.append((pos, m.start()))
        pos = m.end()
    raw.append((pos, len(text)))
    out: list[tuple[int, int]] = []
    for s, e in raw:
        s, e = _strip_span(text, s, e)
        if s >= e:
            continue
        if len(text[s:e].split()) > 40:
            # Very long run-on OCR lines: split further on commas so key points can align.
            cs = s
            for i in range(s, e + 1):
                if i == e or text[i] == ",":
                    a, b = _strip_span(text, cs, i)
                    if a < b:
                        out.append((a, b))
                    cs = i + 1
        else:
            out.append((s, e))
    return out


def sentences(text: str) -> list[str]:
    return [text[s:e] for s, e in sentence_spans(text)]


def clause_spans(text: str) -> list[tuple[int, int]]:
    """Clauses of a premise: split on punctuation and clause conjunctions."""
    out: list[tuple[int, int]] = []
    pos = 0
    for m in _CLAUSE.finditer(text):
        s, e = _strip_span(text, pos, m.start())
        if s < e and _WORD.search(text[s:e].lower()):
            out.append((s, e))
        pos = m.end()
    s, e = _strip_span(text, pos, len(text))
    if s < e and _WORD.search(text[s:e].lower()):
        out.append((s, e))
    return out or ([_strip_span(text, 0, len(text))] if text.strip() else [])


# ------------------------------------------------------------------ fuzzy matching
@lru_cache(maxsize=50000)
def trigrams(w: str) -> frozenset[str]:
    w = f"  {w} "
    return frozenset(w[i:i + 3] for i in range(len(w) - 2))


def fuzzy_equal(a: str, b: str, threshold: float = 0.55) -> bool:
    """Token match robust to OCR noise ("regulanzation" ≈ "regularization")."""
    if a == b:
        return True
    if min(len(a), len(b)) < 4:
        return False
    ta, tb = trigrams(a), trigrams(b)
    return len(ta & tb) / len(ta | tb) >= threshold


@lru_cache(maxsize=100000)
def stems_match(a: str, b: str) -> bool:
    """Same term: equal stems, a short inflection ("map"/"mapp", "label"/"labell") or OCR noise."""
    if a == b:
        return True
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) >= 3 and long_.startswith(short) and \
            (len(long_) == len(short) + 1 or (len(short) >= 4 and len(short) / len(long_) >= 0.6)):
        return True
    if fuzzy_equal(a, b):
        return True
    if "rn" in a or "rn" in b:  # OCR: "dernand" = "demand"
        a2, b2 = a.replace("rn", "m"), b.replace("rn", "m")
        return a2 == b2 or fuzzy_equal(a2, b2)
    return False


def char_cosine(a: str, b: str, n: int = 3) -> float:
    def grams(s: str) -> Counter:
        s = " ".join(words(s))
        return Counter(s[i:i + n] for i in range(max(0, len(s) - n + 1)))

    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 0.0
    dot = sum(ga[k] * gb[k] for k in ga.keys() & gb.keys())
    return dot / (math.sqrt(sum(v * v for v in ga.values())) * math.sqrt(sum(v * v for v in gb.values())))


# ------------------------------------------------------------------ numbers
def numbers(text: str) -> set[str]:
    return {re.sub(r"\s+", "", m.group(0).lower()) for m in _NUM.finditer(text or "")}


def number_values(text: str) -> set[float]:
    """Numeric values in ``text`` (units ignored): "10 N" and "10N" are both 10."""
    out: set[float] = set()
    for m in _NUMVAL.finditer(_norm(text)):
        try:
            out.add(round(float(m.group(0)), 9))
        except ValueError:
            continue
    return out


def missing_numbers(required: set[float], present: set[float]) -> set[float]:
    return {r for r in required if not any(math.isclose(r, p, rel_tol=1e-6, abs_tol=1e-9) for p in present)}


def same_number(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)


# ------------------------------------------------------------------ equations
_UNIT_NAMES = {"newton": "n", "newtons": "n", "volt": "v", "volts": "v", "amp": "a", "amps": "a", "ampere": "a",
               "amperes": "a", "watt": "w", "watts": "w", "joule": "j", "joules": "j", "ohm": "\u03c9",
               "ohms": "\u03c9", "\u03c9": "\u03c9", "kg": "kg", "kilogram": "kg", "kilograms": "kg"}
_UNIT_TOKENS = frozenset({"kg", "kw", "kv", "mv", "ma", "mw", "kj", "kn", "hz", "khz", "mhz", "ms", "cm", "mm", "km",
                          "m/s", "m/s\u00b2", "m/s2", "\u00b0c"}) | frozenset(_UNIT_NAMES)
_MATHY = re.compile(r"^[0-9A-Za-z\u00b2\u00b3/\u00d7*\u00b7+\-^().,\u03a9\u03c9\u00b0']+$")
_NUM_UNIT = re.compile(r"^\(?(-?\d+(?:\.\d+)?)\s*([a-z\u00b0\u03c9/\u00b2]*)")


@dataclass(frozen=True)
class Equation:
    """``lhs = parts[0] = parts[1] ...`` as written in a text (compact, lower-case)."""
    lhs: str
    parts: tuple[str, ...]
    result: float | None  # the number the chain ends with
    unit: str  # unit after the result ("" if none)
    in_parens: bool  # written inside (...) - an aside of the key point

    @property
    def forms(self) -> set[str]:
        """The symbolic right-hand sides, alternatives "(or I\u00b2R or V\u00b2/R)" included."""
        out: set[str] = set()
        for p in self.parts:
            inner = re.findall(r"\(([^()]*)\)", p)
            base = re.sub(r"\([^()]*\)", "", p)
            for f in [base] + [x for i in inner for x in re.split(r"\bor\b", i)]:
                f = f.strip()[2:] if f.strip().startswith("or") else f
                f = re.sub(r"[\s\u00d7*\u00b7]", "", f).strip(".,")
                f = re.sub(r"(?<=[a-z0-9\u00b2\u00b3)])x(?=[a-z0-9(])", "", f)  # "V x I" = "VI"
                while f.endswith(")") and f.count(")") > f.count("("):
                    f = f[:-1]
                if f and not re.fullmatch(r"[-\d./]+[a-z\u00b0\u03c9/\u00b2]*", f):
                    out.add(f)
        return out


def _mathy(tok: str) -> bool:
    t = tok.strip(".,;:")
    if not t or not _MATHY.match(t):
        return False
    letters = sum(ch.isalpha() for ch in t)
    return letters <= 2 or t.lower().strip("()") in _UNIT_TOKENS or t.lower().startswith("(or")


def equations(text: str) -> list[Equation]:
    """Equations such as "F = ma", "I = V/R = 24/8 = 3 A" or "P = VI = 72 W" in a text."""
    spaced = re.sub(r"\s*=\s*", " = ", (text or "").replace("\u2212", "-"))
    toks = spaced.split()
    out: list[Equation] = []
    k = 0
    depth_before: list[int] = []
    depth = 0
    for t in toks:
        depth_before.append(depth)
        depth += t.count("(") - t.count(")")
    while k < len(toks):
        if toks[k] != "=" or k == 0:
            k += 1
            continue
        lhs = toks[k - 1].strip("(),.;:").lower()
        if not lhs or not _mathy(toks[k - 1]):
            k += 1
            continue
        parts: list[list[str]] = [[]]
        m = k + 1
        while m < len(toks) and (toks[m] == "=" or _mathy(toks[m])):
            if toks[m] == "=":
                parts.append([])
            else:
                parts[-1].append(toks[m])
                if re.search(r"[.;:]$", toks[m]) and not re.search(r"\d\.$", toks[m][-2:]) and \
                        toks[m].count("(") <= toks[m].count(")"):
                    m += 1
                    break
            m += 1
        raw = [" ".join(p).strip(" .,;:") for p in parts if p]
        if raw:
            last = raw[-1].lower()
            num = _NUM_UNIT.match(last)
            result, unit = None, ""
            if num and re.fullmatch(r"\(?-?\d+(?:\.\d+)?\s*[a-z\u00b0\u03c9/\u00b2]*\)?", last):
                result = round(float(num.group(1)), 9)
                unit = _UNIT_NAMES.get(num.group(2), num.group(2))
            compact = tuple(p.lower() for p in raw)
            out.append(Equation(lhs, compact, result, unit, depth_before[k] > 0))
        k = max(m, k + 1)
    return out


def formula_match(kp_eq: Equation, ans_eqs: list[Equation]) -> bool:
    """The answer writes the key point's formula: same left side and one of its forms."""
    forms = kp_eq.forms
    return bool(forms) and any(a.lhs == kp_eq.lhs and (forms & a.forms) for a in ans_eqs)


# ------------------------------------------------------------------ propositions
def is_proposition(sentence: str) -> bool:
    """A key point needs an entailing sentence, not a bare keyword list (PRD 12.11)."""
    ws = words(re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", "", sentence))
    if len(ws) < 4:
        return False
    if sum(1 for w in ws if w in FUNCTION_WORDS or w in COMMON_VERBS) == 0:
        return False
    commas = sentence.count(",")
    return not (commas >= 3 and commas >= len(ws) / 3)


# ------------------------------------------------------------------ negation and polarity
def is_negator(w: str) -> bool:
    return w in NEGATIONS or w in _NT_FORMS or w.endswith("n't")


def poles(w: str) -> list[_Pole]:
    return _POLES.get(w, [])


def _is_word_pole(ws: list[str], i: int) -> bool:
    """A polarity word at i ("as well as" is not the evaluative "well")."""
    if not _POLES.get(ws[i]):
        return False
    if ws[i] == "well" and i > 0 and ws[i - 1] == "as" and i + 1 < len(ws) and ws[i + 1] == "as":
        return False
    return True


def negation_targets(ws: list[str]) -> set[int]:
    """Indices of the words that a negation operator applies to.

    A negation applies to the first content word after it ("not caused", "without
    any labels", "never seen"), or to a polarity word right after that ("does not
    perform well" negates "well", so it agrees with "performs poorly")."""
    out: set[int] = set()
    n = len(ws)
    for i, w in enumerate(ws):
        if not is_negator(w):
            continue
        if i + 1 < n and ws[i + 1] in _NEG_PRONOUNS:
            continue
        j = i + 1
        while j < n and j <= i + 4 and (ws[j] in _SCOPE_SKIP or (ws[j] in STOPWORDS and not _is_word_pole(ws, j))):
            if ws[j] in _NOT_NEGATED:
                break
            j += 1
        if j >= n or j > i + 4 or ws[j] in _NOT_NEGATED or is_negator(ws[j]):
            continue
        if not _is_word_pole(ws, j):
            for k in (j + 1, j + 2):
                if k < n and _is_word_pole(ws, k):
                    j = k
                    break
        out.add(j)
    return out


_NEG_PREFIXES = ("non", "un", "dis", "in", "im", "il", "ir")
_ADJ_ENDINGS = ("ent", "ant", "ate", "ble", "ive", "al", "ect", "ete", "ous", "ic", "id", "ar", "ite", "ful", "ed")


def _strip_neg_prefix(w: str) -> str | None:
    """The base of a negated-prefix word: unseen -> seen, invalid -> valid, nonlinear -> linear."""
    for p in _NEG_PREFIXES:
        if w.startswith(p) and len(w) - len(p) >= 4:
            base = w[len(p):]
            if p in ("un", "non"):
                return base
            # in-/im-/il-/ir-/dis- are only negations on adjective-like bases: inaccurate,
            # independent, irrelevant - not information, increase, import, discover.
            if len(base) >= 5 and base.endswith(_ADJ_ENDINGS):
                return base
    return None


@lru_cache(maxsize=50000)
def prefix_opposite(a: str, b: str) -> bool:
    """Generic prefix rule: a and b are opposite forms of the same word
    (labelled/unlabelled, valid/invalid, overfitting/underfitting)."""
    if a == b:
        return False
    for x, y in ((a, b), (b, a)):
        base = _strip_neg_prefix(x)
        if base is not None and (base == y or stem(base) == stem(y)):
            return True
    for p, q in (("over", "under"), ("under", "over")):
        if a.startswith(p) and b.startswith(q) and len(a) - len(p) >= 3 and \
                (a[len(p):] == b[len(q):] or stem(a[len(p):]) == stem(b[len(q):])):
            return True
    return False


def relation(kp_word: str, word: str) -> int:
    """+1 same term (or same-side synonym), -1 opposite, 0 unrelated."""
    if kp_word == word:
        return 1
    pk, pw = _POLES.get(kp_word), _POLES.get(word)
    if pk and pw:
        for a in pk:
            for b in pw:
                if a.group == b.group:
                    return 1 if a.side == b.side else -1
    if prefix_opposite(kp_word, word):
        return -1
    if stems_match(stem(kp_word), stem(word)):
        return 1
    return 0


@dataclass
class _Tokens:
    ws: list[str]
    stems: list[str]
    content: list[bool]
    negated: set[int]


def _tokens(text: str) -> _Tokens:
    ws = words(text)
    return _Tokens(ws, [stem(w) for w in ws], [is_content(w) or _is_word_pole(ws, i) for i, w in enumerate(ws)],
                   negation_targets(ws))


def _anchors(tok: _Tokens, i: int) -> list[str]:
    """The content word a polarity word modifies: the next word when it follows
    directly ("less elastic", "too little data", "directly proportional"), else the
    nearest content word before it ("performs poorly", "price rises")."""
    def plain(k: int) -> bool:
        return tok.content[k] and not _POLES.get(tok.ws[k])

    if i + 1 < len(tok.ws) and plain(i + 1):
        return [tok.stems[i + 1]]
    for k in range(i - 1, -1, -1):
        if plain(k):
            return [tok.stems[k]]
    for k in range(i + 1, len(tok.ws)):
        if plain(k):
            return [tok.stems[k]]
    return []


@dataclass
class Slot:
    """One key-point term whose meaning a premise clause can agree or disagree with."""
    word: str
    stem: str
    sign: int  # -1 when the key point itself negates it
    pole: _Pole | None
    anchors: list[str]
    weight: float = 1.0
    in_recall: bool = True


def key_point_slots(kp_text: str, weights: dict[str, float] | None = None) -> list[Slot]:
    tok = _tokens(kp_text)
    slots: list[Slot] = []
    seen: set[tuple] = set()
    for i, w in enumerate(tok.ws):
        if not tok.content[i]:
            continue
        pls = [p for p in _POLES.get(w, []) if _is_word_pole(tok.ws, i)]
        pole = pls[0] if pls else None
        anchors = _anchors(tok, i) if pole is not None and pole.anchored else []
        key = ("p", pole.group, pole.side, tuple(anchors)) if pole is not None else ("t", tok.stems[i])
        if key in seen:
            continue
        seen.add(key)
        s = tok.stems[i]
        weight = (weights or {}).get(s, 1.0)
        slots.append(Slot(w, s, -1 if i in tok.negated else 1, pole, anchors, weight,
                          in_recall=w not in STOPWORDS))
    return slots


def _anchor_ok(slot: Slot, tok: _Tokens, k: int) -> bool:
    if slot.pole is None or not slot.pole.anchored or not slot.anchors:
        return True
    return any(stems_match(a, b) for a in slot.anchors for b in _anchors(tok, k))


def _slot_detail(slot: Slot, tok: _Tokens) -> tuple[int, str, set[int]]:
    """(+1 agree / -1 conflict / 0 not mentioned, conflict kind, related token indices).

    Conflict kinds: "opposite" (a POLARITY or prefix opposite), "clause_neg" (the
    clause negates a term the key point asserts) and "kp_neg" (the clause asserts a
    term the key point negates)."""
    agree = False
    kinds: list[str] = []
    related: set[int] = set()
    for k, w in enumerate(tok.ws):
        if not tok.content[k]:
            continue
        if slot.pole is not None:
            rel = 0
            for p in _POLES.get(w, []):
                if p.group == slot.pole.group and _is_word_pole(tok.ws, k):
                    rel = 1 if p.side == slot.pole.side else -1
                    break
            else:
                # not a word of this polarity row: the same term ("label" ~ "labelled")?
                rel = relation(slot.word, w)
        else:
            rel = relation(slot.word, w)
        if rel == 0:
            continue
        related.add(k)
        negated = k in tok.negated
        sign = rel * (-1 if negated else 1)
        if sign == slot.sign:
            agree = True
        elif rel == 1:
            kinds.append("clause_neg" if negated else "kp_neg")
        elif _anchor_ok(slot, tok, k):
            kinds.append("opposite")
    if agree:
        return 1, "", related
    if kinds:
        return -1, ("opposite" if "opposite" in kinds else kinds[0]), related
    return 0, "", related


def _slot_state(slot: Slot, tok: _Tokens) -> int:
    """+1 agree, -1 conflict, 0 not mentioned."""
    return _slot_detail(slot, tok)[0]


def related_share(slots: list[Slot], text: str) -> float:
    """Weighted share of the key point's terms that ``text`` mentions with either
    meaning: how much the text is *about* the key point, right or wrong."""
    tok = _tokens(text)
    total = sum(s.weight for s in slots if s.in_recall)
    if total <= 0:
        return 0.0
    return sum(s.weight for s in slots if s.in_recall and _slot_state(s, tok) != 0) / total


def ocr_normalise(text: str) -> str:
    """The text with in-word OCR digit confusions fixed ("b1ood" -> "blood"), for
    model input only (evidence is always quoted from the original answer)."""
    return re.sub(r"[A-Za-z0-9]+", lambda m: m.group(0) if m.group(0).isalpha() or m.group(0).isdigit()
                  else (_ocr_fix(m.group(0).lower()) if _ocr_fix(m.group(0).lower()) != m.group(0).lower()
                        else m.group(0)), text or "")


def slot_recall(slots: list[Slot], text: str) -> float:
    """Weighted share of the key point's terms that ``text`` states with the same meaning."""
    tok = _tokens(text)
    total = sum(s.weight for s in slots if s.in_recall)
    if total <= 0:
        return 0.0
    return sum(s.weight for s in slots if s.in_recall and _slot_state(s, tok) == 1) / total


_UNIT_SPLIT = re.compile(r";|\b(?:whereas|while|whilst|but|however|unlike)\b|\bon the other hand\b|"
                         r"\bin contrast\b|\bi\.\s?e\.|\s[-\u2013\u2014]\s|,\s*so\b", re.IGNORECASE)
_PARENS = re.compile(r"\([^()]*\)|\[[^\[\]]*\]")
_EXAMPLES = re.compile(r"(?:,\s*)?\b(?:e\.\s?g\.|for example|for instance|such as)[^;]*", re.IGNORECASE)
_EXAMPLE_HEAD = re.compile(r"\b(?:e\.\s?g\.|for example|for instance|such as)", re.IGNORECASE)
# Rubric meta-language ("Names a valid way to ...") that students never write.
_META = frozenset({"name", "names", "gives", "give", "mention", "mentions", "valid", "suitable", "way", "ways",
                   "example", "examples", "one", "statement", "states", "state"})


def kp_units(kp_text: str) -> list[str]:
    """A key point split into its sub-claims: parenthetical asides and "e.g." lists
    dropped, then split on ";", contrast words (whereas, while, but ...), "i.e.", a
    dash and ", so". "Arteries carry blood away from the heart, whereas veins carry
    blood back" is two statements, each checked against its own clause of the answer."""
    text = _PARENS.sub(" ", kp_text or "")
    text = _EXAMPLES.sub(" ", text)
    parts = [p.strip(" ,.;:-") for p in _UNIT_SPLIT.split(text) if p]
    merged: list[str] = []
    for part in parts:
        n = len(content_stems(part))
        if merged and n < 3:
            merged[-1] = f"{merged[-1]} {part}"  # "an inverse relationship" belongs to the law before it
        elif n >= 1:
            merged.append(part)
    parts = [p for p in merged if len(content_stems(p)) >= 2]
    if parts:
        return parts
    cleaned = " ".join(text.split())
    return [cleaned] if len(content_stems(cleaned)) >= 2 else [kp_text]


def example_items(kp_text: str) -> tuple[str, list[str]]:
    """(main claim, example items) of a key point with an "e.g." list: "Names a way
    to reduce overfitting, e.g. dropout, early stopping or more data" ->
    ("Names a way to reduce overfitting", ["dropout", "early stopping", "more data"]).
    Any one example is enough for the list part of the key point."""
    m = _EXAMPLE_HEAD.search(kp_text or "")
    if not m:
        return kp_text, []
    main = kp_text[:m.start()].strip(" ,;:")
    rest = _PARENS.sub(" ", kp_text[m.end():])
    items = [i.strip(" .") for i in re.split(r",|;|/|\bor\b|\band\b", rest) if i and i.strip(" .")]
    return main, [i for i in items if content_stems(i)]


_ITEM_SKIP = frozenset({"a", "an", "the", "of", "to", "in", "on", "for", "with", "by", "or", "and", "its", "their"})


def example_recall(kp_text: str, text: str, weights: dict[str, float] | None = None) -> float:
    """Recall of a key point with an example list: half for the main claim (rubric
    meta-words such as "names a valid way" ignored), half for the best example."""
    main, items = example_items(kp_text)
    if not items:
        return 0.0
    ans = [stem(w) for w in words(text)]
    best = 0.0
    for it in items:
        # every word of the example counts, "more" included: "more data" is not "data"
        st = [stem(w) for w in words(it) if w not in _ITEM_SKIP and len(w) > 1]
        if st:
            best = max(best, sum(1 for x in st if any(stems_match(x, a) for a in ans)) / len(st))
    slots = [sl for sl in key_point_slots(main, weights) if sl.word not in _META]
    if len(slots) <= 1:  # the claim is only "give examples": two examples make it complete
        scores = sorted((_item_recall(it, ans) for it in items), reverse=True)
        return sum(scores[:2]) / 2 if len(scores) > 1 else best
    return 0.5 * slot_recall(slots, text) + 0.5 * best


def _item_recall(item: str, ans: list[str]) -> float:
    st = [stem(w) for w in words(item) if w not in _ITEM_SKIP and len(w) > 1]
    return sum(1 for x in st if any(stems_match(x, a) for a in ans)) / len(st) if st else 0.0


@dataclass
class PolarityResult:
    conflicts: int = 0  # key-point terms the aligned clause states with the opposite meaning
    align: float = 0.0  # weighted share of key-point terms the clause mentions (either polarity)
    clause: tuple[int, int] | None = None  # span of the aligned clause inside the premise
    conflict_terms: list[str] = field(default_factory=list)

    @property
    def contradicts(self) -> bool:
        """An odd number of flips states the opposite fact. Two flips ("performs well on
        seen data" vs "performs poorly on unseen data") is a different, parallel
        statement: not a contradiction, but not support either."""
        return self.conflicts % 2 == 1


_PRONOUNS = frozenset({"it", "they", "this", "these", "he", "she", "its", "their", "them", "which"})


def aligned_clause(slots: list[Slot], premise: str) -> tuple[tuple[int, int] | None, list[int]]:
    """The clause of ``premise`` that best matches the key point (direct mentions count
    double opposite ones), with per-slot states for it."""
    best: tuple[float, tuple[int, int] | None, list[int]] = (-1.0, None, [])
    spans = clause_spans(premise)
    cands = list(spans)
    for k in range(1, len(spans)):
        first = words(premise[spans[k][0]:spans[k][1]])[:1]
        if first and first[0] in _PRONOUNS:  # "..., but they are not genetically different"
            cands.append((spans[k - 1][0], spans[k][1]))
    for s, e in cands:
        tok = _tokens(premise[s:e])
        states = [_slot_state(sl, tok) for sl in slots]
        score = sum(sl.weight * (1.0 if st == 1 else 0.5 if st == -1 else 0.0) for sl, st in zip(slots, states))
        if score > best[0]:
            best = (score, (s, e), states)
    return best[1], best[2]


def _clause_conflicts(slots: list[Slot], clause: str) -> tuple[list[str], float]:
    """Conflicting key-point terms in an aligned clause, and the clause's alignment.

    A negation that moved to a neighbouring word is not a flip: "data not used for
    training" and "data it was not trained on" carry one negation each, on different
    words. So a key-point negation whose word the clause does not mention cancels one
    clause negation of another key-point term (and vice versa)."""
    tok = _tokens(clause)
    details = [_slot_detail(sl, tok) for sl in slots]
    total = sum(s.weight for s in slots) or 1.0
    align = sum(s.weight for s, d in zip(slots, details) if d[0] != 0) / total
    related = set().union(*(d[2] for d in details)) if details else set()
    content = [k for k in range(len(tok.ws)) if tok.content[k]]
    if len(content) >= 3:
        # A clause made (almost) only of the key point's terms is about it, even when
        # the key point itself is long: count its precision too.
        align = max(align, 0.9 * sum(1 for k in content if k in related) / len(content))
    kp_orphans = sum(1 for s, d in zip(slots, details) if s.sign == -1 and d[0] == 0)
    clause_orphans = sum(1 for k in tok.negated if k not in related)
    terms: list[str] = []
    subject = content[0] if content else -1
    for s, (st, kind, rel_idx) in zip(slots, details):
        if st != -1:
            continue
        if kind == "opposite" and s.pole is not None and not s.pole.anchored and subject in rel_idx:
            # "In mitosis the cells ..." is about the other concept, not a contradiction of
            # "meiosis has two divisions": the clause must also state this key point's own
            # (distinctive) content - "Unsupervised learning uses labelled data" does.
            agreeing = [o.weight for o, d in zip(slots, details) if o is not s and d[0] == 1]
            if sum(1 for w in agreeing if w >= 1.5) < 2 and len(agreeing) < 3:
                continue
        if kind == "clause_neg" and kp_orphans > 0:
            kp_orphans -= 1
            continue
        if kind == "kp_neg" and clause_orphans > 0:
            clause_orphans -= 1
            continue
        terms.append(s.word)
    return terms, align


def polarity_check(kp_text: str, premise: str, weights: dict[str, float] | None = None,
                   units: list[list[Slot]] | None = None) -> PolarityResult:
    """Compare a key point with the clause of ``premise`` aligned to it (A). A key point
    with contrasted statements is checked statement by statement; the result is the
    statement with an odd number of flips (a contradiction) if any, else the best aligned."""
    if units is None:
        units = [key_point_slots(u, weights) for u in kp_units(kp_text)]
    if not premise.strip():
        return PolarityResult()
    results: list[PolarityResult] = []
    for slots in units:
        if not slots:
            continue
        span, _states = aligned_clause(slots, premise)
        if span is None:
            continue
        terms, align = _clause_conflicts(slots, premise[span[0]:span[1]])
        results.append(PolarityResult(len(terms), align, span, terms))
    if not results:
        return PolarityResult()
    flipped = [r for r in results if r.contradicts]
    if flipped:
        return max(flipped, key=lambda r: r.align)
    return max(results, key=lambda r: (r.conflicts > 0, r.align))


# ------------------------------------------------------------------ role swaps (B)
@dataclass(frozen=True)
class RolePair:
    """In two parallel statements u and v, ``mine`` (of u) and ``theirs`` (of v) fill the
    same role; ``others`` are u's remaining terms that v does not use."""
    unit: int
    mine: str
    theirs: str
    other_unit: int
    others: frozenset[str]


def distinct_terms(a: list[str], b: list[str]) -> set[str]:
    """Stems of ``a`` that ``b`` does not use (inflections and OCR variants included)."""
    return {x for x in a if not any(stems_match(x, y) for y in b)}


def role_pairs(unit_stems: list[list[str]]) -> list[list[RolePair]]:
    """For each statement u (a key point or one contrasted half of one), the role pairs
    against every *parallel* statement v of the same question. Distinctive terms are
    judged pairwise (terms of u that v does not use), so "arteries" stays distinctive
    against "veins" even when other key points mention both. Pairs come from aligning
    the content words of u and v: one-for-one and two-for-two substitutions, plus the
    first content words (the subject position). No subject words are hard-coded."""
    out: list[list[RolePair]] = [[] for _ in unit_stems]
    for u, a in enumerate(unit_stems):
        for v, b in enumerate(unit_stems):
            if u == v or len(a) < 2 or len(b) < 2:
                continue
            ops = SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
            shared = sum(i2 - i1 for tag, i1, i2, _, _ in ops if tag == "equal")
            if shared < 2 and shared / max(len(a), len(b)) < 0.34:
                continue  # not parallel statements
            da, db = distinct_terms(a, b), distinct_terms(b, a)
            cand: list[tuple[str, str]] = []
            for tag, i1, i2, j1, j2 in ops:
                if tag == "replace" and (i2 - i1) == (j2 - j1) <= 2:
                    cand.extend(zip(a[i1:i2], b[j1:j2]))
            cand.append((a[0], b[0]))
            seen: set[tuple[str, str]] = set()
            for x, y in cand:
                if x in da and y in db and (x, y) not in seen:
                    seen.add((x, y))
                    out[u].append(RolePair(u, x, y, v, frozenset(da - {x})))
    return out


def find_role_swap(clause: str, pairs: list[RolePair]) -> RolePair | None:
    """A clause aligned with statement u that keeps u's own terms but puts the parallel
    statement's term in u's role ("Veins carry blood away from the heart")."""
    if not pairs:
        return None
    cs = content_stems(clause)
    ws = [w for w in words(clause) if _POLES.get(w)]

    def has(t: str) -> bool:
        return any(stems_match(t, c) for c in cs) or any(relation(t, w) == 1 for w in ws)

    for p in pairs:
        if has(p.theirs) and not has(p.mine) and any(has(d) for d in p.others):
            return p
    return None


# ------------------------------------------------------------------ concept inversion
_CONTRAST = frozenset({"unlike", "than", "whereas", "versus", "vs", "opposite", "contrast", "compared", "not",
                       "never", "instead", "rather", "differs", "different"})


def _over_under(a: str, b: str) -> bool:
    """over-/under- forms of the same word (overfitting / underfitting)."""
    for p, q in (("over", "under"), ("under", "over")):
        if a.startswith(p) and b.startswith(q) and len(a) - len(p) >= 3 and \
                (a[len(p):] == b[len(q):] or stem(a[len(p):]) == stem(b[len(q):])):
            return True
    return False


def opposite_concept(prompt: str, answer: str) -> tuple[str, str] | None:
    """The answer talks about the opposite of the concept the question asks about
    ("Underfitting is ..." for "Explain overfitting"): an opposite of a question term
    (an over-/under- swap or an un-anchored, i.e. whole-concept, POLARITY row such as
    mitosis/meiosis) is used outside a contrast, and the question term itself never
    appears. The generic un-/in-/dis- prefix rule is deliberately not used here: an
    answer may say "unnatural" about a question on the natural greenhouse effect."""
    if not prompt or not answer:
        return None
    pw = [w for w in dict.fromkeys(words(prompt)) if is_content(w)]
    aw = words(answer)
    neg = negation_targets(aw)
    for p in pw:
        hits = []
        same = False
        for k, w in enumerate(aw):
            if not is_content(w):
                continue
            pole_rel = 0
            for a in _POLES.get(p, []):
                for b in _POLES.get(w, []):
                    if a.group == b.group and not a.anchored:
                        pole_rel = 1 if a.side == b.side else -1
            if pole_rel == 1 or w == p or (pole_rel == 0 and not prefix_opposite(p, w)
                                          and stems_match(stem(p), stem(w))):
                same = True
                break
            if pole_rel == -1 or _over_under(p, w):
                window = aw[max(0, k - 3):k]
                if k not in neg and not (_CONTRAST & set(window)):
                    hits.append(w)
        if hits and not same:
            return p, hits[0]
    return None


# ------------------------------------------------------------------ keywords (C)
def keyword_present(keyword: str, text: str) -> bool:
    """A teacher keyword is present: same stem, inflection or OCR variant; a phrase
    ("last in first out", "V = IR") by its words in order or all its content words;
    and "unseen" is also said as "never seen" / "not seen"."""
    kw = words(keyword)
    if not kw:
        return False
    ws = words(text)
    if not ws:
        return False
    sts = [stem(w) for w in ws]
    if len(kw) > 1:
        n = len(kw)
        for s in range(len(ws) - n + 1):
            if all(ws[s + k] == kw[k] or stems_match(stem(kw[k]), sts[s + k]) for k in range(n)):
                return True
        kc = [stem(w) for w in kw if is_content(w)]
        return bool(kc) and all(any(stems_match(k, s) for s in sts) for k in kc)
    k = kw[0]
    ks = stem(k)
    neg = negation_targets(ws)
    if any(w == k and i not in neg for i, w in enumerate(ws)):
        return True
    if len(k) > 2 and any(stems_match(ks, s) and i not in neg and not prefix_opposite(k, ws[i])
                          for i, s in enumerate(sts) if len(s) > 2):
        return True
    # a polarity keyword: a same-side synonym ("badly" for "poorly") or a negated
    # opposite ("not ... well") says the same thing
    for p in _POLES.get(k, []):
        for i, w in enumerate(ws):
            for q in _POLES.get(w, []):
                same_meaning = (q.side == p.side) == (i not in neg)
                if q.group == p.group and _is_word_pole(ws, i) and same_meaning:
                    return True
    # overfitted ~ fits: strip a non-negating prefix from the answer word.
    for w in ws:
        for p in ("over", "under", "re", "pre"):
            if w.startswith(p) and len(w) - len(p) >= 3 and stems_match(ks, stem(w[len(p):])):
                return True
    base = _strip_neg_prefix(k)
    if base is not None:
        bs = stem(base)
        if any(i in neg and stems_match(bs, sts[i]) for i in range(len(ws))):
            return True
    return False

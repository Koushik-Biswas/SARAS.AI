"""Anonymise text before it leaves the machine (PRD 13.5, spec P-5).

Removes e-mail addresses, URLs, phone numbers, student IDs and names that the text
itself marks as personal, plus any caller-supplied terms (student_ref, names). The
student_ref and names are never put into prompts in the first place; this is the
second line of defence for what a student wrote inside the answer itself, and for the
header of a continuation page ("Name: ... Roll No: ...") that OCR put into an answer.

What counts as personal data
* names after a label: "Name: Priya Sharma", "NAME PRIYA SHARMA" at the start of a
  line or after a field separator (| ; , ( ), "Student name - ...", "Candidate's name
  is ...", "Name of the student: ..."; and after "my name is", "I am" / "I'm" and
  "Myself" (Indian English). A name is 1-4 capitalised words (initials such as "K."
  included) on the same line, stopping at the next label ("Roll", "USN", ...);
* IDs after a label: Roll (No|Number|#), Reg(istration|d) No, USN, PRN, SRN, UID,
  Enrol(l)ment No, Admission No, Student / Candidate / Exam / Seat / Index / Hall
  ticket / Admit card / Application ID|No, and a bare "ID". The value may be split by
  OCR ("Roll No: 21 BCE 1234"): up to 4 chunks of digits, upper-case letters or mixed
  letters and digits. After a label with "No"/":"/"#", 2+ digits go; after a bare
  label ("Roll 2021001234"), 4+ digits ("roll 30 m" and "the id 7 row" stay);
* mixed letter-digit tokens shaped like IDs with 5+ digits anywhere: 21BCE1234,
  1RV20CS045, 2021UCS0045, 2021CS10234, CS2021-00123, CS2021001, ab-123456, 2021/CS/123;
* phone numbers: a +country code, an (area) code, a trunk 0 with digit groups ("020
  7946 0958"), a dashed 3-3-4 number, or a 10-digit mobile in any usual grouping right
  after a phone cue word ("mobile", "call me at", "ph:");
* extra_terms (student_ref, names), matched case-insensitively and tolerantly: an ID
  with digits also when spaces, hyphens, dots, slashes or underscores were put between
  its characters ("21 BCE 1234", "STU 017"); a name also with such separators between
  its words ("Asha-Rao", "ASHA  RAO").

Numbers in answers are content: "300000000 m/s", "1000000 J", "6.022 x 10^23",
"7800000000 people", "70000-80000 km/h", "1,250,000 rows", years, dates and lists of
values must reach the grader unchanged. So a digit run is NEVER removed for its length
alone. Ordinary codes and formulas stay: H2O, CO2, SHA256, ISO9001, RTX3090, DDR4-3200,
C6H12O6, COVID-19, i7-12700H, 6.022e23, ID3; so does prose such as "the name of the
element is Sodium", "Roll the ball", "the IUPAC name: ethanol".

Known limits
* An unlabelled name ("Priya wrote that ...", a signature line with no label) cannot
  be told apart from other capitalised words without a named-entity model, which this
  offline, dependency-free step does not have. It is NOT removed. The app stores no
  student name, so callers can only pass student_ref; teachers are told not to rely on
  redaction for anything but accidental personal data in answers.
* A line that starts with "Name:" is always treated as a person's name, so an answer
  written as "Name: Methane" loses that word for the AI (the offline checker still
  reads the original text).
* A phone number written without a +code, area code, trunk 0 or cue word ("98765
  43210") is kept, because it cannot be told apart from two numbers; a few standard
  names with 5+ digits after letters (ISO27001) are removed.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

EMAIL = "[EMAIL]"
URL = "[URL]"
PHONE = "[PHONE]"
ID = "[ID]"
NAME = "[NAME]"

_URL = re.compile(r"(?<![\w@])(?:https?://|www\.)[^\s<>\"'()\[\]]+", re.I)
_EMAIL = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")

_END = r"(?![\w.,]?\d)(?!\w)"  # not followed by more digits (or a decimal / thousands part)
_PHONES = [
    # +country code, then digit groups: +91 98765 43210, +4915112345678, +1 (202) 555-0143
    re.compile(r"(?<![\w+])\+\d{1,3}(?:[\s.-]?\(?\d{1,5}\)?){1,5}" + _END),
    # (area code) then groups of 3+ digits: (555) 123-4567, (080) 2345 6789
    re.compile(r"(?<![\w(])\(\d{2,5}\)[\s.-]?\d{3,5}(?:[\s.-]\d{3,5}){0,2}" + _END),
    # trunk 0 then groups: 020 7946 0958, 011-2345-6789, 080 23456789
    re.compile(r"(?<![\w.,])0\d{2,4}[\s-]\d{3,4}(?:[\s-]?\d{3,4})" + _END),
    # dashed / dotted 3-3-4 with the same separator: 555-123-4567, 555.123.4567
    re.compile(r"(?<![\w.,-])\d{3}([-.])\d{3}\1\d{4}" + _END),
]
_PHONE_DIGITS = (8, 15)
# A mobile number without a prefix counts as a phone only right after a phone cue word.
_MOBILE = r"(?:\+?91[\s-]?|0[\s-]?)?(?:\d{10}|\d{5}[\s-]\d{5}|\d{3}[\s-]\d{3}[\s-]\d{4}|\d{4}[\s-]\d{3}[\s-]\d{3})"
_PHONE_CUE = re.compile(
    r"(?i)\b(?:phone|ph|mobile|mob|cell|tel|telephone|call|contact|whatsapp)\b\.?"
    r"(?:\s*(?:me|us|at|on|to|is|no|number|num)\b\.?)*\s*[:#=-]?\s*"
    rf"(?P<num>{_MOBILE}){_END}")

# Mixed letter-digit tokens (letters, digits and inner - or /) that look like student IDs.
_TOKEN = re.compile(r"(?<![\w/-])[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*(?![\w/-])")
_ID_SHAPES = (
    re.compile(r"[A-Za-z]{2,}[-/]?\d{5,}"),  # CS2021001, ab-123456, ROLL/2023045X, 2021CS10234
    re.compile(r"\d+[-/]?[A-Za-z]{2,}[-/]?\d+"),  # 21BCE1234, 1MS20CS001, U19CS001, 2021/CS/123
    re.compile(r"[A-Za-z]{2,}\d{2,}[-/]\d{3,}"),  # CS2021-00123
)
_ID_TOKEN_MIN_DIGITS = 5

# ---------------------------------------------------------------- labelled IDs
_NUMWORD = r"(?:no|num|number|nr)\b\.?"
_ID_LABEL_WORDS = (
    rf"roll(?:\s*(?:{_NUMWORD}|\#))?"
    rf"|reg(?:istration|ister|d)?\.?(?:\s*(?:{_NUMWORD}|\#))?"
    rf"|enrol{{1,2}}ment(?:\s*(?:{_NUMWORD}|\#))?"
    rf"|admission(?:\s*(?:{_NUMWORD}|\#))?"
    rf"|adm\.?\s*(?:{_NUMWORD}|\#)"
    r"|usn|prn|srn|uid"
    r"|(?:student|candidate|exam(?:ination)?|seat|index|hall\s*ticket|ht|admit\s*card|application)"
    rf"\s*(?:id\b|{_NUMWORD}|\#)"
    r"|id"
)
# A chunk of an ID value: digits / mixed letters+digits (with inner - or /), or a short
# upper-case letter group ("21 BCE 1234").
_ID_CHUNK = r"(?:(?=[A-Za-z0-9/-]*\d)[A-Za-z0-9]+(?:[/-][A-Za-z0-9]+)*|[A-Z]{1,6}(?![A-Za-z0-9]))"
_ID_LABEL = re.compile(
    rf"(?<![\w-])(?P<label>(?i:{_ID_LABEL_WORDS}))(?![\w])"
    r"(?P<sep>[ \t]*[:#=\-–]?[ \t]*)"
    rf"(?P<val>{_ID_CHUNK}(?:[ \t]+{_ID_CHUNK}){{0,3}})" + _END)
_ID_MIN_DIGITS_PLAIN = 4  # "roll 30 m" and "the id 7 row" stay; "Roll 2021001234" goes
_ID_MIN_DIGITS_CONNECTED = 2  # "Roll No. 23" / "ID: 23" go
_CONNECTED_LABEL = re.compile(r"(?i)(?:\b(?:no|num|number|nr)\.?|#|\S\s*\bid)$")

# ---------------------------------------------------------------- names
# Words that start the next header field, never part of a name.
_FIELD_WORDS = (r"roll|reg|regd|registration|register|usn|prn|srn|uid|id|enrol{1,2}ment|admission|adm|student|"
                r"candidate|seat|hall|index|class|section|sec|div|division|date|subject|course|exam|examination|"
                r"paper|q|question|ans|answer|sign|signature|no|number|branch|dept|department|semester|sem|year|"
                r"batch|marks|page|std|grade|name|mobile|phone|email|e-mail|time|centre|center|college|school")
_UPPER = "A-ZÀ-ÖØ-Þ"
# A capitalised word ("Priya", "D'Souza", "PRIYA"), or a short one with a dot: an
# initial or title ("K.", "K.R.", "Dr."). A long word never takes the sentence's period.
_NAME_WORD = (rf"(?!(?i:{_FIELD_WORDS})(?![\w'’-]))"
              rf"(?:[{_UPPER}](?:[^\W\d_]|['’-]){{3,}}|[{_UPPER}][^\W\d_]{{0,2}}\.?(?:[{_UPPER}]\.)*)"
              r"(?![\w'’-])")
_NAME_VALUE = rf"(?P<val>{_NAME_WORD}(?:[ \t]+{_NAME_WORD}){{0,3}})"
_NAME_SEP = r"(?P<sep>[ \t]*[:=\-–][ \t]*|[ \t]+(?i:is)[ \t]+|[ \t]+)"
_NAME_LABELS = [
    # "Student name", "Candidate's name", "Name of the student" - anywhere.
    re.compile(r"(?<![\w-])(?P<label>(?i:(?:student|candidate|pupil)(?:'s|’s)?[ \t]+name"
               r"|name[ \t]+of[ \t]+(?:the[ \t]+)?(?:student|candidate)))(?![\w'’])" + _NAME_SEP + _NAME_VALUE),
    # A bare "Name" label only at the start of a line or after a field separator, so
    # "the IUPAC name: ethanol" and "its name is Sodium" stay.
    re.compile(r"(?:^|(?<=[\n|;,(\[]))[ \t]*(?P<label>(?i:name))(?![\w'’])" + _NAME_SEP + _NAME_VALUE, re.M),
    # "my name is X", "My full name is X".
    re.compile(r"(?<![\w-])(?P<label>(?i:my[ \t]+(?:full[ \t]+)?name[ \t]+is))(?P<sep>[ \t]+)" + _NAME_VALUE),
]
# "I am X", "I'm X", "Myself X": only a capitalised word that is not a common sentence word.
_SELF_STOP = ("Not|The|A|An|Sure|Going|Here|There|Writing|Answering|Explaining|Trying|Unable|Able|In|At|On|From|"
              "Of|With|Also|So|Very|Now|Glad|Happy|Sorry|Done|Fine|Ok|OK|Okay|It|This|That|Doing|Using|Taking|"
              "Confident|Certain|Aware|Thankful|Grateful|Ready|Afraid|Really|Just|Still|Only|Always|Never|Telling|"
              "Giving|Attempting|Attaching|Submitting|Studying|Learning")
_SELF = re.compile(r"(?<![\w-])(?P<label>(?:I[ \t]+(?i:am)|I['’](?i:m)|(?i:myself)))(?P<sep>[ \t]+)"
                   rf"(?!(?:{_SELF_STOP})(?![\w'’-]))" + _NAME_VALUE)

_TERM_SEP = r"[\s.\-_/]"


def _url(m: re.Match[str]) -> str:
    s = m.group(0)
    tail = len(s) - len(s.rstrip(".,;:!?"))  # sentence punctuation after a URL is kept
    return URL + (s[-tail:] if tail else "")


def _digits(s: str) -> int:
    return sum(ch.isdigit() for ch in s)


def _phone(m: re.Match[str]) -> str:
    s = m.group(0)
    if set(ch for ch in s if ch.isdigit()) <= {"0", "1"}:
        return s  # binary groups ("0110 1010 1111") are answer content, not a phone number
    return PHONE if _PHONE_DIGITS[0] <= _digits(s) <= _PHONE_DIGITS[1] else s


def _replace_group(m: re.Match[str], group: str, placeholder: str) -> str:
    start, end = m.span(group)
    whole = m.group(0)
    return whole[: start - m.start()] + placeholder + whole[end - m.start():]


def is_id_token(token: str) -> bool:
    """A mixed letter-digit token shaped like a student / registration number."""
    if _digits(token) < _ID_TOKEN_MIN_DIGITS or not any(c.isalpha() for c in token):
        return False
    return any(p.search(token) for p in _ID_SHAPES)


def _id_token(m: re.Match[str]) -> str:
    return ID if is_id_token(m.group(0)) else m.group(0)


def _id_label(m: re.Match[str]) -> str:
    label, sep, val = m.group("label"), m.group("sep"), m.group("val")
    # "Roll No. 23", "ID: 23", "Student ID 23": a number word, a symbol or a compound
    # "... ID" label. A bare "roll" / "id" needs a longer number ("roll 30 m" stays).
    connected = bool(sep.strip(" \t")) or bool(_CONNECTED_LABEL.search(label.strip()))
    needed = _ID_MIN_DIGITS_CONNECTED if connected else _ID_MIN_DIGITS_PLAIN
    if _digits(val) >= needed or any(is_id_token(c) for c in val.split()):
        return _replace_group(m, "val", ID)
    return m.group(0)


def _term_pattern(term: str) -> re.Pattern[str] | None:
    """A tolerant, case-insensitive pattern for one extra term: an ID (it has digits and
    5+ letters/digits) with optional separators between any characters; anything else
    with optional separators where the term has one or switches letters <-> digits."""
    core = [c for c in term if c.isalnum()]
    if len(core) < 2:
        return None
    if any(c.isdigit() for c in core) and len(core) >= 5:
        body = f"{_TERM_SEP}{{0,2}}".join(re.escape(c) for c in core)
    else:
        segments = re.findall(r"[^\W\d_]+|\d+", term)
        body = f"{_TERM_SEP}{{0,3}}".join(re.escape(s) for s in segments)
    return re.compile(rf"(?<![^\W_]){body}(?![^\W_])", re.I)


def redact(text: str | None, extra_terms: Iterable[str | None] = ()) -> str:
    """Return `text` with personal data replaced by placeholders such as [EMAIL]."""
    if not text:
        return ""
    out = _URL.sub(_url, text)
    out = _EMAIL.sub(EMAIL, out)
    for pattern in (*_NAME_LABELS, _SELF):
        out = pattern.sub(lambda m: _replace_group(m, "val", NAME), out)
    out = _PHONE_CUE.sub(lambda m: _replace_group(m, "num", PHONE), out)
    for pattern in _PHONES:
        out = pattern.sub(_phone, out)
    out = _ID_LABEL.sub(_id_label, out)
    out = _TOKEN.sub(_id_token, out)
    terms = {t.strip() for t in extra_terms if t and len(t.strip()) >= 2}
    for term in sorted(terms, key=len, reverse=True):
        pattern = _term_pattern(term)
        if pattern is not None:
            out = pattern.sub(NAME, out)
    return out

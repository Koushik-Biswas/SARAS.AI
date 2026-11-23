"""Redaction before anything leaves the machine (PRD 13.5, spec P-5): e-mails,
phones, URLs, ID numbers and caller-supplied names go; numeric answer content stays
exactly as written (a digit run is never removed for its length alone)."""

from __future__ import annotations

import pytest

from app.llm.redact import EMAIL, ID, NAME, PHONE, URL, is_id_token, redact


@pytest.mark.parametrize(("text", "gone", "placeholder"), [
    ("Contact asha.rao+exam@college.edu.in now", "asha.rao+exam@college.edu.in", EMAIL),
    ("mail: A_B-9@x.io.", "A_B-9@x.io", EMAIL),
    ("see https://example.com/notes?id=42 for more", "example.com", URL),
    ("visit www.my-site.org/page", "my-site.org", URL),
    # phones: +country code, (area) code, trunk 0, dashed 3-3-4, or a mobile after a cue word
    ("call +91 98765 43210.", "98765 43210", PHONE),
    ("intl +4915112345678", "4915112345678", PHONE),
    ("US (555) 123-4567 office", "123-4567", PHONE),
    ("UK 020 7946 0958", "7946 0958", PHONE),
    ("dial 555-123-4567 now", "555-123-4567", PHONE),
    ("call 98765-43210 today", "98765-43210", PHONE),
    ("mobile 9876543210", "9876543210", PHONE),
    ("Call me at 98765 43210 after six", "98765 43210", PHONE),
    ("my phone number is +91-9876543210", "9876543210", PHONE),
    ("Ph. No.: 0 9876543210", "9876543210", PHONE),
    # mixed letter-digit ID tokens
    ("Roll CS2021001 present", "CS2021001", ID),
    ("roll no: ab-123456", "ab-123456", ID),
    ("id ROLL/2023045X", "ROLL/2023045X", ID),
    ("I am 21BCE1234 from B section", "21BCE1234", ID),
    ("student 2021CS10234 wrote this", "2021CS10234", ID),
    ("ref CS2021-00123.", "CS2021-00123", ID),
    ("USN 1MS20CS001", "1MS20CS001", ID),
    ("file 2021/CS/123 attached", "2021/CS/123", ID),
    # digit runs after an ID cue word
    ("reg 20231234567", "20231234567", ID),
    ("Roll No. 23, section B", "23", ID),
    ("roll no: 12345", "12345", ID),
    ("Registration number: 2021001234", "2021001234", ID),
    ("enrolment no. 123456", "123456", ID),
    ("Enrollment 98765432", "98765432", ID),
    ("Admission No 4567", "4567", ID),
    ("ID: 98765", "98765", ID),
    ("student no 778899", "778899", ID),
    ("PRN 2020123456", "2020123456", ID),
])
def test_personal_data_is_removed(text: str, gone: str, placeholder: str) -> None:
    out = redact(text)
    assert gone not in out and placeholder in out, out


def test_cue_words_are_kept_and_only_the_value_is_replaced() -> None:
    assert redact("Roll No. 23, section B") == f"Roll No. {ID}, section B"
    assert redact("mobile 9876543210.") == f"mobile {PHONE}."
    assert redact("reg 20231234567 and done") == f"reg {ID} and done"


def test_names_and_student_ref_are_removed_case_insensitively() -> None:
    out = redact("My name is ASHA RAO (asha rao), ref STU-001; Rao's answer.", ["Asha Rao", "STU-001", "", None])
    assert "asha" not in out.lower() and "STU-001" not in out
    assert out.count(NAME) == 3
    assert redact("Rao", ["R"]) == "Rao"  # single letters are not treated as names
    assert redact("Ramesh and Ram", ["Ram"]) == f"Ramesh and {NAME}"  # word boundaries
    assert redact("by ML001_full_marks.", ["ML001_full_marks"]) == f"by {NAME}."


def test_sentence_punctuation_after_url_is_kept() -> None:
    assert redact("Read www.x.com/a, then stop.") == f"Read {URL}, then stop."


@pytest.mark.parametrize("text", [
    # numeric content that the old "8+ digit run" rule destroyed
    "The speed of light is 300000000 m/s.",
    "The work done is 1000000 J and the power 25000000 W.",
    "c = 299792458 m/s exactly; Avogadro's number is 602214076000000000000000.",
    "The world population is about 7800000000 people.",
    "A 12345678 byte file; checksum 9876543210 matched.",
    "The answer is 9876543210.",
    "Speeds of 70000-80000 km/h and 98765 43210 are just two numbers.",
    "Values 100 200 300 400 and 1 000 000 J.",
    "Binary 0110 1010 1111 0000 and nibbles 0100-1100-1010.",
    # ordinary numbers and codes
    "The speed of sound is 343 m/s and light is 3 x 10^8 m/s.",
    "Learning rate 0.01, batch 32, 100 epochs; accuracy rose from 72.5% to 91.25%.",
    "In 2024 the dataset had 1,250,000 rows.",
    "Dates like 2024-11-23 and times like 10:30 stay.",
    "Numbers 10 20 30 40 and H2O, CO2, IPv6, SHA256, Python3 stay.",
    "Pi is 3.14159 and e is 2.71828; 6.022e23 and 1.6e-19 C.",
    "ISO9001, RTX3090, DDR4-3200, C6H12O6, COVID-19, i7-12700H, x86-64 and IEEE-754 stay.",
    "The ball will roll 30 m and the id 7 row was deleted.",
    "Options (10) 20 30 40 50 and 0.5 0.25 0.125.",
    "The pH 7 solution; call 5 times; a 192.168.100.1 address.",
    "",
])
def test_ordinary_answer_text_is_untouched(text: str) -> None:
    assert redact(text) == text


@pytest.mark.parametrize(("token", "is_id"), [
    ("21BCE1234", True), ("2021CS10234", True), ("CS2021-00123", True), ("CS2021001", True),
    ("ab-123456", True), ("1MS20CS001", True), ("U19CS001", True),
    ("SHA256", False), ("ISO9001", False), ("DDR4-3200", False), ("C6H12O6", False), ("i7-12700H", False),
    ("COVID-19", False), ("300000000", False), ("2021CS", False),
])
def test_id_token_shapes(token: str, is_id: bool) -> None:
    assert is_id_token(token) is is_id


def test_redact_handles_none() -> None:
    assert redact(None) == ""
    assert ID and EMAIL and PHONE  # placeholders are non-empty


# ------------------------------------------------------------------ review finding ai-layer #2
@pytest.mark.parametrize(("text", "gone", "kept"), [
    ("I am Priya Sharma (USN 1RV20CS045). Photosynthesis makes food.", ["Priya", "Sharma", "1RV20CS045"],
     ["USN", "Photosynthesis makes food."]),
    ("Reg no 2021UCS0045 - answer: plants make food.", ["2021UCS0045"], ["Reg no", "plants make food."]),
    # a roll number as OCR split it
    ("Roll 21 BCE 1234 Photosynthesis needs light.", ["21", "BCE", "1234"], ["Roll", "Photosynthesis needs light."]),
    ("Roll No: 21 BCE 1234\nPhotosynthesis needs light.", ["21", "BCE", "1234"], ["Roll No:", "Photosynthesis"]),
    # a continuation-page header OCR'd into an answer
    ("Name: Priya Sharma Roll No: 21BCE1234", ["Priya", "Sharma", "21BCE1234"], ["Name:", "Roll No:"]),
    ("NAME PRIYA SHARMA\nQ2 continued: the heart pumps blood.", ["PRIYA", "SHARMA"], ["the heart pumps blood."]),
    ("Candidate Name - RAHUL K. Seat No 4521", ["RAHUL", "4521"], ["Seat No"]),
    ("Student's name is Priya. She wrote this.", ["Priya"], ["She wrote this."]),
    ("Name of the student: Anil Kumar", ["Anil", "Kumar"], ["Name of the student:"]),
    ("My name is Asha Rao and I study ML.", ["Asha", "Rao"], ["and I study ML."]),
    ("Myself Priya Sharma from CSE.", ["Priya", "Sharma"], ["from CSE."]),
    ("Hall Ticket No: 2134 5678 is printed.", ["2134", "5678"], ["is printed."]),
    ("Student ID 4521 attended.", ["4521"], ["attended."]),
])
def test_labelled_names_and_ids_are_removed(text: str, gone: list[str], kept: list[str]) -> None:
    out = redact(text)
    for word in gone:
        assert word not in out, out
    for word in kept:
        assert word in out, out


def test_extra_terms_match_case_insensitively_and_across_ocr_splits() -> None:
    out = redact("Ref 21 BCE 1234, also 21-bce-1234 and 21BCE1234.", ["21BCE1234"])
    for fragment in ("21", "BCE", "bce", "1234"):
        assert fragment not in out, out
    assert redact("STU 017 and S.T.U.017 and stu-017.", ["STU-017"]) == f"{NAME} and {NAME} and {NAME}."
    assert redact("Asha-Rao and ASHA  RAO and AshaRao wrote.", ["Asha Rao"]) == \
        f"{NAME} and {NAME} and {NAME} wrote."
    # a name is matched word by word, never letter by letter across ordinary words
    assert redact("put the tin a little closer", ["Tina"]) == "put the tin a little closer"


@pytest.mark.parametrize("text", [
    "The name of the element is Sodium.",
    "Roll the ball down the slope and it will roll 30 m.",
    "The ID3 algorithm splits on information gain.",
    "The IUPAC name: Ethanol is C2H5OH.",
    "Its name is Bernoulli's principle.",
    "Name the parts of a cell: nucleus, membrane.",
    "Admission to the course needs 60 marks.",
    "Student ID cards are issued in 2024.",
    "I am Not sure, but I am confident about Newton's laws.",
    "Exam no. 3 was hard.",
    "The register keeps a record of every payroll entry.",
])
def test_ordinary_prose_with_label_words_is_untouched(text: str) -> None:
    assert redact(text) == text


def test_unlabelled_names_are_a_documented_limitation() -> None:
    # Without a label (or a named-entity model) a name cannot be told from other words.
    text = "Priya wrote that plants make food."
    assert redact(text) == text
    assert redact(text, ["Priya"]) == f"{NAME} wrote that plants make food."

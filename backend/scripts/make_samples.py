"""Generate SARAS AI sample data: a "Machine Learning Unit Test" rubric plus typed
student answer sheets (PDF text layer) of varied quality, one image-only sheet
(no text layer: exercises the manual-transcription path) and the invalid files
the upload validator must reject.

    cd backend
    .venv/Scripts/python.exe scripts/make_samples.py          (Windows)
    .venv/Scripts/python.exe -m scripts.make_samples          (same, as a module)

Writes backend/samples/ (rubric.json, sheets/*.pdf, invalid/*). Everything is
generated with PyMuPDF; no ML models, no network. The tests import the builders
below directly, so they never depend on files on disk.
"""

from __future__ import annotations

import json
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

BACKEND = Path(__file__).resolve().parent.parent
SAMPLES_DIR = BACKEND / "samples"

# ------------------------------------------------------------------ exam + rubric
EXAM = {"title": "Machine Learning Unit Test", "subject": "Machine Learning", "date": "2026-11-20",
        "total_marks": 15}

Q1_PROMPT = "Define supervised learning."
Q2_PROMPT = "Explain overfitting in machine learning."
Q3_PROMPT = "Describe how gradient descent is used to train a model."
Q4_PROMPT = "What is the purpose of a validation set?"

RUBRIC = {"questions": [
    {
        "question_no": 1, "type": "short", "max_marks": 3, "prompt": Q1_PROMPT,
        "model_answer": "Supervised learning is a type of machine learning in which a model is trained on "
                        "labelled data. It learns a mapping from inputs to known outputs so that it can predict "
                        "the outputs for new unseen inputs.",
        "key_points": [
            {"text": "the model is trained on labelled data", "weight": 1, "keywords": ["labelled", "trained"]},
            {"text": "it learns a mapping from inputs to known outputs", "weight": 1,
             "keywords": ["mapping", "inputs", "outputs"]},
            {"text": "it predicts outputs for new unseen inputs", "weight": 1, "keywords": ["predict", "unseen"]},
        ],
    },
    {
        # PRD worked example (overfitting, five 1-mark key points).
        "question_no": 2, "type": "long", "max_marks": 5, "prompt": Q2_PROMPT,
        "model_answer": "Overfitting is when a model fits the training data too closely. It learns the noise "
                        "or random patterns in the training data, so it performs poorly on new unseen data. "
                        "It is caused by an overly complex model or too little data. A remedy is "
                        "regularisation, dropout, early stopping or collecting more data.",
        "key_points": [
            {"text": "the model fits the training data too closely", "weight": 1,
             "keywords": ["fits", "training", "closely"]},
            {"text": "it learns noise or random patterns", "weight": 1, "keywords": ["noise", "random"]},
            {"text": "it performs poorly on new unseen data", "weight": 1, "keywords": ["poorly", "unseen"]},
            {"text": "it is caused by an overly complex model or too little data", "weight": 1,
             "keywords": ["complex", "little"]},
            {"text": "a remedy such as regularisation, dropout, early stopping or more data", "weight": 1,
             "keywords": ["regularisation", "dropout", "stopping"]},
        ],
    },
    {
        "question_no": 3, "type": "paragraph", "max_marks": 5, "prompt": Q3_PROMPT,
        "model_answer": "Gradient descent is an iterative optimisation algorithm that minimises a loss "
                        "function. It computes the gradient of the loss with respect to the model parameters "
                        "and updates the parameters in the opposite direction of the gradient. The learning "
                        "rate controls the size of each step; a learning rate that is too large overshoots the "
                        "minimum while one that is too small converges slowly.",
        "key_points": [
            {"text": "gradient descent is an iterative optimisation algorithm that minimises a loss function",
             "weight": 1, "keywords": ["iterative", "minimises", "loss"]},
            {"text": "it computes the gradient of the loss with respect to the parameters", "weight": 1,
             "keywords": ["gradient", "parameters"]},
            {"text": "the parameters are updated in the opposite direction of the gradient", "weight": 1,
             "keywords": ["opposite", "direction"]},
            {"text": "the learning rate controls the size of each step", "weight": 1,
             "keywords": ["learning", "rate", "step"]},
            {"text": "a learning rate that is too large overshoots the minimum and one that is too small "
                     "converges slowly", "weight": 1, "keywords": ["overshoots", "slowly"]},
        ],
    },
    {
        "question_no": 4, "type": "short", "max_marks": 2, "prompt": Q4_PROMPT,
        "model_answer": "A validation set is used to tune hyperparameters and to evaluate the model on data "
                        "that was not used for training.",
        "key_points": [
            {"text": "the validation set is used to tune hyperparameters", "weight": 1,
             "keywords": ["tune", "hyperparameters"]},
            {"text": "it evaluates the model on data not used for training", "weight": 1,
             "keywords": ["evaluate", "training"]},
        ],
    },
]}

# ------------------------------------------------------------------ answer texts
# Reused by the grading tests, so each text is named after what it demonstrates.
ANSWERS: dict[str, str] = {
    "q1_full": "Supervised learning is machine learning where a model is trained on labelled data. It learns a "
               "mapping from inputs to known outputs and can then predict outputs for new unseen inputs.",
    "q1_partial": "Supervised learning means the model is trained on labelled data.",
    "q1_paraphrased": "In supervised learning every training example comes with the correct answer (a label), "
                      "the algorithm learns how inputs map to those outputs, and afterwards it predicts the "
                      "outputs for new inputs it has never seen.",
    "q1_noisy": "Supervized lerning is when a model is trainned on labeled dota. It lerns a maping from inputs to "
                "known outputs and can predict outputs for new unseen inputs.",
    # PRD worked example: KP1, KP2, KP3 and KP5 present, KP4 (cause) missing -> 4/5.
    "q2_worked_example": "Overfitting happens when the model fits the training data too closely. It learns the "
                         "noise and random patterns in that data instead of the general trend. As a result it "
                         "performs poorly on new unseen data. We can reduce it with regularisation, dropout or "
                         "early stopping.",
    "q2_full": "Overfitting is when a model fits the training data too closely. It learns noise or random "
               "patterns, so it performs poorly on new unseen data. It is caused by an overly complex model or "
               "too little data. A remedy is regularisation, dropout, early stopping or more data.",
    "q2_partial": "Overfitting is when the model fits the training data too closely and learns the noise.",
    "q2_paraphrased": "An overfitted model has memorised its training examples far too closely, including the "
                      "random noise and patterns in them. Because of this it performs poorly on new data it has "
                      "not seen. Overfitting is usually caused by a model that is too complex or by too little "
                      "training data. To prevent it we can use regularisation, dropout or early stopping, or we "
                      "can gather more data.",
    "q2_noisy": "Overfiting is when the modle fits the trainng dota too closely. It lerns noise or random "
                "paterns, so it performs poorly on new unseen dota. It is caused by an overly complex model or "
                "too little dota. A remedy is regulanzation, dropout or early stoping.",
    "q2_flipped": "Overfitting is when a model fits the training data too closely. It learns noise or random "
                  "patterns in the data. An overfitted model performs well on new unseen data.",
    "q2_keyword_list": "overfitting, noise, training data, unseen data, complex model, regularisation, dropout",
    "q2_copied_question": "Explain overfitting in machine learning.",
    "q2_off_topic": "The French Revolution began in 1789. The people of Paris stormed the Bastille and the "
                    "monarchy was later abolished.",
    "q2_duplicate_first": "Overfitting is when the model learns",
    "q2_continued_p1": "Overfitting is when a model fits the training data too closely. It learns noise or "
                       "random patterns, so it performs poorly on new unseen data.",
    "q2_continued_p2": "It is caused by an overly complex model or too little data. A remedy is regularisation, "
                       "dropout, early stopping or more data.",
    "q3_full": "Gradient descent is an iterative optimisation algorithm that minimises a loss function. It "
               "computes the gradient of the loss with respect to the parameters. The parameters are updated in "
               "the opposite direction of the gradient. The learning rate controls the size of each step. A "
               "learning rate that is too large overshoots the minimum and one that is too small converges "
               "slowly.",
    "q3_partial": "Gradient descent is an iterative optimisation algorithm that minimises a loss function. The "
                  "learning rate controls the size of each step.",
    "q3_paraphrased": "Gradient descent repeatedly adjusts a model to minimise its loss function. At each "
                      "iteration it computes the gradient of the loss with respect to every parameter and then "
                      "moves the parameters in the opposite direction of that gradient. How big each step is "
                      "depends on the learning rate; a learning rate that is too large overshoots the minimum, "
                      "and one that is too small converges very slowly.",
    "q3_noisy": "Gradiant descent is an iterative optimisation algoritm that minimises a loss functon. It "
                "computes the gradient of the loss with respect to the paramters. The paramters are updated in "
                "the opposite direction of the gradient. The lerning rate controls the size of each step.",
    "q3_off_topic": "My favourite football team won the league last season and the final match was very "
                    "exciting to watch.",
    "q4_full": "A validation set is used to tune hyperparameters. It evaluates the model on data that was not "
               "used for training.",
    "q4_partial": "It is used to tune the hyperparameters.",
    "q4_paraphrased": "We keep a validation set aside to tune hyperparameters and to evaluate the model on data "
                      "it was not trained on.",
    "q1_copied_question": "Define supervised learning.",
}

ADVERSARIAL_Q2 = {
    "empty": "",
    "copied_question": ANSWERS["q2_copied_question"],
    "keyword_list": ANSWERS["q2_keyword_list"],
    "off_topic": ANSWERS["q2_off_topic"],
}


# ------------------------------------------------------------------ sheets
@dataclass
class SheetSpec:
    student_ref: str
    description: str
    pages: list[list[str]]  # per page: paragraphs; a paragraph starting with a marker begins an answer
    expect: dict = field(default_factory=dict)  # hints for tests (e.g. {"blank": [3]})

    @property
    def filename(self) -> str:
        return f"{self.student_ref}.pdf"


def _a(marker: str, key: str) -> str:
    return f"{marker} {ANSWERS[key]}"


TYPED_SHEETS: list[SheetSpec] = [
    SheetSpec("ML001_full_marks", "every key point covered", [[
        _a("Q1.", "q1_full"), _a("Q2.", "q2_full"), _a("Q3.", "q3_full"), _a("Q4.", "q4_full")]],
        {"min_total": 13}),
    SheetSpec("ML002_partial", "partly correct answers", [[
        _a("Q1.", "q1_partial"), _a("Q2.", "q2_partial"), _a("Q3.", "q3_partial"), _a("Q4.", "q4_partial")]]),
    SheetSpec("ML003_paraphrased", "correct answers in the student's own words, mixed marker styles", [[
        _a("1.", "q1_paraphrased"), _a("2(a)", "q2_paraphrased"), _a("Question 3:", "q3_paraphrased"),
        _a("Q4", "q4_paraphrased")]]),
    SheetSpec("ML004_ocr_noisy", "misspellings such as 'regulanzation' and 'dota'", [[
        _a("Q1.", "q1_noisy"), _a("Q2.", "q2_noisy"), _a("Q3.", "q3_noisy"), _a("Q4.", "q4_full")]]),
    SheetSpec("ML005_skipped_question", "Q3 left empty under its marker", [[
        _a("Q1.", "q1_full"), _a("Q2.", "q2_partial"), "Q3.", _a("Q4.", "q4_full")]],
        {"blank": [3]}),
    SheetSpec("ML006_duplicate_marker", "Q2 started, abandoned and restarted under a second Q2 marker", [[
        _a("Q1.", "q1_full"), _a("Q2.", "q2_duplicate_first"), _a("Q2.", "q2_full"), _a("Q3.", "q3_partial"),
        _a("Q4.", "q4_full")]], {"flags": ["duplicate_marker"]}),
    SheetSpec("ML007_out_of_order", "answers written in the order Q1, Q3, Q2, Q4", [[
        _a("Q1.", "q1_full"), _a("Q3.", "q3_full"), _a("Q2.", "q2_full"), _a("Q4.", "q4_full")]],
        {"flags": ["out_of_order"]}),
    SheetSpec("ML008_continuation", "Q2 continues onto page 2 without a new marker", [
        [_a("Q1.", "q1_full"), _a("Q2.", "q2_continued_p1")],
        [ANSWERS["q2_continued_p2"], _a("Q3.", "q3_full"), _a("Q4.", "q4_full")],
    ], {"pages": 2}),
    SheetSpec("ML009_flipped_fact", "claims an overfitted model performs well on new data", [[
        _a("Q1.", "q1_full"), _a("Q2.", "q2_flipped"), _a("Q3.", "q3_partial"), _a("Q4.", "q4_partial")]]),
    # Q4 has no marker after the last one found (Q3): it may be skipped or its marker
    # misread, so it scores 0 but goes to review instead of being silently blank.
    SheetSpec("ML010_adversarial", "copied question, keyword list, off-topic text, Q4 missing", [[
        _a("Q1.", "q1_copied_question"), _a("Q2.", "q2_keyword_list"), _a("Q3.", "q3_off_topic")]],
        {"missing": [4]}),
    SheetSpec("ML011_empty_sheet", "a blank page", [[]], {"blank": [1, 2, 3, 4]}),
]

IMAGE_ONLY_SHEET = SheetSpec("ML012_scanned_image_only", "rasterised page without a text layer", [[
    _a("Q1.", "q1_full"), _a("Q2.", "q2_full"), _a("Q3.", "q3_full"), _a("Q4.", "q4_full")]])

# ------------------------------------------------------------------ PDF builders
_FONT_SIZE = 11
_LINE_STEP = 16  # points
_LEFT, _TOP, _BOTTOM = 60, 72, 800
_WRAP = 88  # characters per line at 11 pt Helvetica on A4


def _layout(paragraphs: list[str]) -> list[tuple[float, str]]:
    y = _TOP
    out = []
    for para in paragraphs:
        for line in textwrap.wrap(para, _WRAP) or [para]:
            if y > _BOTTOM:
                raise ValueError("sample page overflow; split the paragraphs over more pages")
            out.append((y, line))
            y += _LINE_STEP
        y += _LINE_STEP // 2  # gap between answers
    return out


def build_text_pdf(pages: list[list[str]]) -> bytes:
    """A typed sheet with a real text layer (one PDF page per entry)."""
    doc = pymupdf.open()
    try:
        for paragraphs in pages:
            page = doc.new_page(width=595, height=842)  # A4 in points
            for y, line in _layout(paragraphs):
                page.insert_text((_LEFT, y), line, fontsize=_FONT_SIZE)
        return doc.tobytes(garbage=3, deflate=True)
    finally:
        doc.close()


def build_image_only_pdf(pages: list[list[str]], dpi: int = 110) -> bytes:
    """Render a typed sheet to pixels and embed only the image: what a scanner
    produces. There is no text layer, so OCR (or the teacher) must read it."""
    src = pymupdf.open("pdf", build_text_pdf(pages))
    out = pymupdf.open()
    try:
        for page in src:
            pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False)
            new = out.new_page(width=page.rect.width, height=page.rect.height)
            new.insert_image(new.rect, stream=pix.tobytes("png"))
        return out.tobytes(garbage=3, deflate=True)
    finally:
        src.close()
        out.close()


def corrupted_pdf_bytes() -> bytes:
    return b"%PDF-1.4 garbage"


def not_a_pdf_bytes() -> bytes:
    """A real PNG saved with a .pdf extension."""
    pix = pymupdf.Pixmap(pymupdf.csGRAY, pymupdf.IRect(0, 0, 16, 16), False)
    pix.clear_with(200)
    return pix.tobytes("png")


def password_protected_pdf_bytes(user_pw: str = "student", owner_pw: str = "teacher") -> bytes:
    doc = pymupdf.open("pdf", build_text_pdf([[_a("Q1.", "q1_full")]]))
    try:
        return doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw=user_pw, owner_pw=owner_pw)
    finally:
        doc.close()


def many_pages_pdf_bytes(pages: int = 51) -> bytes:
    doc = pymupdf.open()
    try:
        for i in range(pages):
            doc.new_page(width=595, height=842).insert_text((_LEFT, _TOP), f"Page {i + 1}", fontsize=_FONT_SIZE)
        return doc.tobytes(garbage=3, deflate=True)
    finally:
        doc.close()


def oversized_pdf_bytes(megabytes: float) -> bytes:
    """Starts like a PDF so only the size check can reject it."""
    return b"%PDF-1.4\n" + b"0" * int(megabytes * 1024 * 1024)


def sheet_bytes(spec: SheetSpec) -> bytes:
    return build_text_pdf(spec.pages)


def valid_sample_files(include_image_only: bool = True) -> list[tuple[str, bytes]]:
    files = [(s.filename, sheet_bytes(s)) for s in TYPED_SHEETS]
    if include_image_only:
        files.append((IMAGE_ONLY_SHEET.filename, build_image_only_pdf(IMAGE_ONLY_SHEET.pages)))
    return files


def invalid_sample_files() -> list[tuple[str, bytes, int, str]]:
    """(filename, bytes, expected HTTP status, expected error code)."""
    return [
        ("corrupted.pdf", corrupted_pdf_bytes(), 422, "unreadable_pdf"),
        ("not_a_pdf.pdf", not_a_pdf_bytes(), 415, "unsupported_media_type"),
        ("password_protected.pdf", password_protected_pdf_bytes(), 422, "unreadable_pdf"),
        ("too_many_pages_51.pdf", many_pages_pdf_bytes(51), 413, "file_too_large"),
    ]


def write_samples(out_dir: Path = SAMPLES_DIR) -> dict[str, list[Path]]:
    sheets_dir, invalid_dir = out_dir / "sheets", out_dir / "invalid"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    invalid_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rubric.json").write_text(json.dumps(RUBRIC, indent=2) + "\n", encoding="utf-8")
    (out_dir / "exam.json").write_text(json.dumps(EXAM, indent=2) + "\n", encoding="utf-8")
    written: dict[str, list[Path]] = {"sheets": [], "invalid": []}
    for name, data in valid_sample_files():
        path = sheets_dir / name
        path.write_bytes(data)
        written["sheets"].append(path)
    for name, data, _status, _code in invalid_sample_files():
        path = invalid_dir / name
        path.write_bytes(data)
        written["invalid"].append(path)
    manifest = {
        "exam": EXAM,
        "sheets": [{"file": f"sheets/{s.filename}", "student_ref": s.student_ref, "description": s.description}
                   for s in [*TYPED_SHEETS, IMAGE_ONLY_SHEET]],
        "invalid": [{"file": f"invalid/{n}", "expected_status": st, "expected_code": c}
                    for n, _d, st, c in invalid_sample_files()],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return written


def main() -> int:
    written = write_samples()
    print(f"Wrote {SAMPLES_DIR / 'rubric.json'}")
    for kind, paths in written.items():
        for p in paths:
            print(f"Wrote {p.relative_to(BACKEND).as_posix()} ({p.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

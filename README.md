<div align="center">

<img src="docs/images/saras-banner.svg" alt="SARAS AI: Smart Assessment and Recognition Automation System" width="100%">

<br>

**AI-powered exam answer-sheet evaluation and marks assessment, with the teacher always in control.**

[![CI](https://github.com/Koushik-Biswas/SARAS.AI/actions/workflows/ci.yml/badge.svg)](https://github.com/Koushik-Biswas/SARAS.AI/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.165-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-strict-3178C6?logo=typescript&logoColor=white)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-CPU-005CED?logo=onnx&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Overview](#-project-description--overview) ·
[Features](#-key-features) ·
[AI/ML concepts](#-aimldl-core-engineering-concepts) ·
[Screenshots](#-screenshots--demo) ·
[Architecture](#-system-architecture) ·
[Setup](#-installation--setup) ·
[API](#-api-documentation) ·
[Deploy](#-deployment-instructions)

</div>

---

## 📑 Table of contents

1. [Project description / overview](#-project-description--overview)
2. [Problem statement](#-problem-statement)
3. [Key features](#-key-features)
4. [AI/ML/DL core engineering concepts](#-aimldl-core-engineering-concepts)
5. [Screenshots & demo](#-screenshots--demo)
6. [Technology stack](#-technology-stack)
7. [Full engineering mechanism](#-full-engineering-mechanism)
8. [System architecture](#-system-architecture)
9. [Project directory structure](#-project-directory-structure)
10. [Prerequisites](#-prerequisites)
11. [Installation & setup](#-installation--setup)
12. [Environment variables & configuration](#-environment-variables--configuration)
13. [Database schema & design](#-database-schema--design)
14. [API documentation](#-api-documentation)
15. [Authentication & security](#-authentication--security)
16. [Testing & code quality](#-testing--code-quality)
17. [Deployment instructions](#-deployment-instructions)
18. [Known issues & limitations](#-known-issues--limitations)
19. [Contributing guidelines](#-contributing-guidelines)
20. [License](#-license)
21. [Contact & support](#-contact--support)

---

## 🧭 Project description / overview

**SARAS AI** (*Smart Assessment and Recognition Automation System*) is a full-stack web
application that evaluates scanned exam answer sheets. It started as the final-year project
*AI-Powered Exam Answer Sheet Evaluation and Marks Assessment System*.

The workflow:

1. A teacher creates an exam and its **rubric**. Each question has a number, a type, maximum
   marks, and optionally a model answer and weighted **key points**. The rubric can also be
   drafted automatically from an uploaded question paper.
2. The teacher uploads the students' **answer-sheet PDFs** (up to 50 per request).
3. A background pipeline **renders, cleans and reads** every page (OCR). It **splits the
   text into one answer per question** using the question numbers written on the sheet.
4. Each answer is **graded against the rubric** by an ensemble of local NLP graders and,
   optionally, an LLM marker. The result is explainable marks: every key point is reported
   as matched, partial, missed or contradicted, with the student's own sentence as evidence.
5. A **confidence score** decides the route. Confident answers are auto-accepted, and
   uncertain or flagged ones go to a **review queue**, riskiest first.
6. The teacher reviews, corrects, overrides and **approves** the marks, then **exports a CSV**
   of the approved results.

> **The AI suggests and the teacher decides.** Every change to marks, text or answer regions
> is written to an append-only audit log, and only approved results can be exported.

The whole system runs as **one Python process on a 4 GB laptop** (SQLite, in-process job
worker, CPU-only ONNX model). The same code scales up to Docker + PostgreSQL + HTTPS on a
server.

---

## ❓ Problem statement

Evaluating descriptive (written) answers is one of the most time-consuming parts of teaching:

- **It is slow.** A teacher marking a stack of multi-page scripts spends evenings on
  repetitive reading, and results reach students late.
- **It is inconsistent.** The same answer can get different marks from different examiners,
  or from the same examiner when tired, and partial credit is applied unevenly.
- **It gives students little feedback.** A bare number does not say which points were
  covered, which were missed and which statements were wrong.
- **Existing automation does not fit.** OMR and MCQ tools cannot read free-text answers.
  Black-box "AI graders" give a mark without evidence and leave no room for the teacher's
  judgement, which is unacceptable for real exams.
- **Real-world constraints are tight.** Scripts are scanned or photographed (skewed, noisy,
  ruled paper, handwriting), students number their answers inconsistently, and many
  institutions only have modest hardware and must protect student data.

**SARAS AI's goal:** automate the reading and first-pass marking of answer sheets with
*explainable*, rubric-based grading, measure how confident each mark is, and route every
uncertain case to the teacher. The result should be faster, more consistent evaluation
that stays auditable and under human control.

---

## ✨ Key features

| Area | What it does |
|---|---|
| 📄 **End-to-end pipeline** | PDF → page images → deskew / denoise / contrast → OCR → segmentation by question marker → key-point grading → confidence routing → totals |
| 🧾 **Rubric from a question paper** | Upload the question paper (PDF or photo). An AI reader (or a rule-based parser for typed papers) drafts the questions, types and marks for the teacher to confirm. |
| 👁️ **Multi-engine OCR** | PDF text layer → vision-LLM reader → TrOCR handwriting model → teacher transcription, picked automatically per page |
| 🧠 **Explainable grading** | Each key point is *matched / partial / missed / contradicted*, with the evidence sentence quoted from the student's answer |
| 🤝 **Grader ensemble** | A local NLI cross-encoder (ONNX) plus a lexical scorer, and an optional LLM marker. Disagreements are flagged, never silently resolved. |
| 🎯 **Confidence-based routing** | OCR × grading × segmentation confidence, temperature-scaled. Low-confidence or flagged answers go to `needs_review`. |
| 👩‍🏫 **Teacher in control** | Edit OCR text (re-graded instantly), override marks (a reason is required), comment, flag, approve / lock / reopen a sheet |
| ✂️ **Region editor** | Crop, reassign, split, merge, blank or draw answer regions on the page image, with undo |
| 📦 **Robust batch uploads** | Up to 50 PDFs per request, streamed to disk, with signature / size / page / password checks and an `Idempotency-Key`. One bad page never fails the batch. |
| 📊 **Results & analytics** | Results grid, mark distribution, per-question difficulty, auto-accept and override rates, CSV export of approved sheets |
| 🔐 **Secure by default** | Argon2 hashing, short-lived JWTs with rotating refresh tokens, lock-out, rate limiting, CSP / security headers, per-teacher data isolation |
| 🕵️ **Privacy-first AI** | Nothing leaves the machine without explicit consent flags. Answers are anonymised before any LLM call, and AI replies are cached and logged. |
| 💻 **Runs on a 4 GB laptop** | SQLite, one process, an in-process worker and a CPU-only int8 ONNX model. It degrades gracefully when optional engines are missing. |
| 🎨 **Polished UI** | React + TypeScript SPA with light / dark / system themes, a self-hosted Inter font (works offline), accessible components and live processing status |

---

## 🧠 AI/ML/DL core engineering concepts

SARAS AI combines **classical computer vision**, **deep-learning OCR**, **Natural Language
Inference**, **sentence embeddings**, **LLM structured output** and **confidence
calibration**. Every concept below is implemented in this repository. The figures were
generated from the project's own code or reflect its exact configuration values.

### 1. Computer-vision preprocessing (OpenCV)

Image pages (scans and photos) are cleaned before OCR. The review screen always shows the
original deskewed page, and only the OCR stage sees the cleaned copy.

<p align="center"><img src="docs/images/preprocessing-pipeline.png" alt="Preprocessing pipeline: input scan, deskew, clean, text-line detection" width="100%"></p>

<sub>*Produced by the real `deskew()`, `clean()` and `detect_lines()` functions in
[`backend/app/pipeline/preprocess.py`](backend/app/pipeline/preprocess.py), applied to a
sample sheet with a simulated phone-scan tilt, uneven light and ruled paper.*</sub>

| Step | Technique | Why |
|---|---|---|
| Quality check | Otsu ink ratio, **variance of the Laplacian** (blur), contrast (σ) | Flags `blank_page` / `poor_quality` pages |
| Deskew | Otsu threshold → `minAreaRect` over ink pixels → affine rotation (≤ 10°) | Straight lines for segmentation and OCR |
| Denoise | Median blur (3×3); a stronger retry uses non-local means | Removes scanner speckle |
| Contrast | **CLAHE** (contrast-limited adaptive histogram equalisation, 8×8 tiles) | Evens out shadows and uneven lighting |
| Binarise | Adaptive Gaussian threshold (block 31) | Robust to local lighting changes |
| Ruled lines | Morphological opening with a long horizontal kernel, then subtraction | Removes notebook lines that confuse OCR |
| Line detection | Horizontal **projection profile** → text-line bounding boxes | Line crops for the TrOCR model |

### 2. OCR: a cascade of recognisers (deep learning + vision LLMs)

<p align="center"><img src="docs/images/concept-ocr-cascade.svg" alt="OCR cascade: PDF text layer, vision LLM reader, TrOCR, teacher transcription" width="100%"></p>

- **TrOCR** ([`pipeline/ocr.py`](backend/app/pipeline/ocr.py)) is a *Vision Transformer
  encoder + Transformer text decoder* (`microsoft/trocr-small-handwritten`). It runs on line
  crops with PyTorch dynamic int8 quantisation, batch 4, loaded once per process. If the mean
  confidence is below 0.5, it retries once with stronger cleaning.
- **Vision-LLM reading** ([`pipeline/ai_ocr.py`](backend/app/pipeline/ai_ocr.py)) sends a
  downscaled page JPEG to a multimodal model and receives **schema-validated JSON lines with
  bounding boxes**. Pages are read concurrently. Lines the model marks `unclear` or
  `[illegible]` get a lower confidence, which raises the `low_ocr_confidence` review flag.
- When no engine can read a page, the line boxes are kept with empty text and the teacher
  types the answer. It is graded immediately.

### 3. Answer segmentation (rule-based NLP + AI fallback)

<p align="center"><img src="docs/images/segmentation-regions.png" alt="Answer regions detected from question markers on a sample sheet" width="70%"></p>

<sub>*Real output of `segment()` in [`backend/app/pipeline/segment.py`](backend/app/pipeline/segment.py)
on a sample sheet that mixes marker styles.*</sub>

Regular expressions find question markers at the start of OCR lines (`Q1`, `Q.2`,
`Question 3:`, `Ans 4`, `5.`, `6)`, `2(a)`). Text runs until the next marker, across pages,
and sub-parts merge into the parent question. Anomalies lower the segmentation confidence
and raise flags: `duplicate_marker`, `out_of_order`, `gap_in_sequence`, `no_markers_found`,
`continues_across_pages` and `unassigned_text`. When a sheet has **no markers at all**, an
LLM is shown the anonymised numbered lines and returns which line ranges answer which
question ([`pipeline/ai_segment.py`](backend/app/pipeline/ai_segment.py)).

### 4. Natural Language Inference: a transformer cross-encoder on ONNX Runtime

<p align="center"><img src="docs/images/concept-nli-cross-encoder.svg" alt="NLI cross-encoder: premise and hypothesis to entailment, neutral and contradiction" width="100%"></p>

The main local grader treats grading as **textual entailment**. The student's answer window
is the *premise* and the rubric key point is the *hypothesis*. A **DeBERTa-v3-xsmall
cross-encoder** (trained on SNLI + MultiNLI) reads both together and outputs P(entailment),
P(neutral) and P(contradiction). It runs as a **dynamic-int8 ONNX export (87 MB)** on
ONNX Runtime + the `tokenizers` package, with no PyTorch. Measured on the 4 GB laptop:
45–85 ms per uncached pair ([`grading/onnx_nli.py`](backend/app/grading/onnx_nli.py)).
The label order is read from the model's `config.json`, never assumed.

### 5. Sentence embeddings and the lexical scorer

<p align="center"><img src="docs/images/concept-similarity-scorers.svg" alt="Sentence-BERT bi-encoder and lexical scorer" width="100%"></p>

- **Sentence-BERT** (`all-MiniLM-L6-v2`, a *bi-encoder*) embeds the sentence and the key
  point separately into 384-dimensional vectors and compares them with **cosine
  similarity**. It is available with the optional ML extras.
- The **lexical scorer** needs no model and is always available. It uses stemming,
  stop-word removal, **fuzzy matching for OCR noise**, **in-question IDF weighting**, and a
  **polarity / negation / antonym** check that detects contradictions. It is the guaranteed
  fallback and the default cross-checker.

All scorers implement the same `PairScorer` interface
([`grading/scorers.py`](backend/app/grading/scorers.py)). Each returns an entailment matrix
*E* and a contradiction matrix *C* over (sentence × key point).

### 6. Key-point rubric grading (explainable partial credit)

<p align="center"><img src="docs/images/concept-keypoint-grading.svg" alt="Key-point rubric grading with entailment matrix and partial credit" width="100%"></p>

For each key point, the best-supporting sentence decides its status:
**matched** (E ≥ τ, full weight), **partial** (E ≥ τ_low, half weight), **contradicted**
(C ≥ c, 0 marks + `contradiction` flag) or **missed**. Marks are the sum of the awarded
weights, capped at the maximum. Thresholds are per scorer: lexical / SBERT use
τ = 0.6, τ_low = 0.4, c = 0.5, and NLI uses τ = 0.5, τ_low = 0.15, c = 0.95. The engine also
detects `numeric_mismatch`, `keyword_list` (keyword stuffing), `copied_question`, `off_topic`
and `role_swap` ([`grading/engine.py`](backend/app/grading/engine.py)).

### 7. The LLM marker: an LLM as a judge, with code deciding the marks

<p align="center"><img src="docs/images/concept-llm-marker.svg" alt="LLM marker pipeline: anonymise, prompt, router, strict JSON, quote verification, marks in code" width="100%"></p>

When enabled, one LLM request per answer sheet judges every key point
([`grading/llm_grader.py`](backend/app/grading/llm_grader.py)). The model **never gives
marks**. It returns a verdict and a **verbatim quote** per criterion through a **strict JSON
schema** generated from a Pydantic model ([`llm/schema.py`](backend/app/llm/schema.py)).
Every quote is then **located in the student's own text** using exact or fuzzy word
alignment. Numbers and negations must match, and opposite words never align. Unsupported
verdicts become `missed`. The **LLM router** ([`llm/router.py`](backend/app/llm/router.py))
provides:

- a SHA-256 response cache, so regrading an unchanged sheet makes no new request
- provider fallback (Groq → Gemini → OpenRouter → local Ollama)
- token-bucket rate limits and rolling daily budgets ([`llm/limiter.py`](backend/app/llm/limiter.py))
- retries with `Retry-After` and an overload cooldown
- a wall-clock deadline

Answers are anonymised first ([`llm/redact.py`](backend/app/llm/redact.py)). Question-only
rubrics (no model answer or key points) are marked directly by the AI, with marks clamped and
rounded in code ([`grading/direct_ai.py`](backend/app/grading/direct_ai.py)).

### 8. Ensemble learning, calibration and selective prediction

<p align="center"><img src="docs/images/concept-ensemble-routing.svg" alt="Ensemble confidence and routing" width="100%"></p>

- **Ensemble** ([`grading/ensemble.py`](backend/app/grading/ensemble.py)): a *decider*
  (the AI marker when it judged the question, otherwise the local primary grader) and a
  *cross-checker*. Any criterion where they disagree about awarding marks is **disputed**
  and flagged `graders_disagree`.
- **Grading confidence** = prior(decider) × agreement factor (1.0 / 0.85 / 0.45).
- **Combined confidence** = calibrate(OCR × grading × segmentation factor, *T*). This uses
  **temperature scaling**, σ(logit(p) / T), from [`pipeline/routing.py`](backend/app/pipeline/routing.py).
- **Selective prediction:** an answer is auto-accepted only when the confidence is ≥ 0.80
  and it has no review flag. Everything else is deferred to the teacher, lowest confidence
  first.

### 9. Evaluation methodology

A dedicated harness ([`app/evaluation/`](backend/app/evaluation)) runs the **production
code paths** over a labelled benchmark. It reports **MAE, RMSE, within-±0.5 / ±1,
quadratic weighted kappa (QWK), Pearson, Cohen's κ** on key-point statuses,
**precision / recall / F1**, contradiction metrics, **ECE, Brier score, risk-coverage / AURC**,
and **CER / WER**, all with **bootstrap 95% confidence intervals**.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/eval-heldout-dark.svg">
  <img src="docs/images/eval-heldout-light.svg" alt="Held-out test results: MAE and QWK for baseline formula, lexical scorer and NLI cross-encoder" width="100%">
</picture>

| Grader (held-out test, N = 111, AI off) | MAE (marks) | Within ±0.5 | QWK (normalised) | Key-point κ |
|---|---|---|---|---|
| Baseline formula | 1.088 [0.899, 1.275] | 0.468 | 0.565 [0.435, 0.675] | n/a |
| Lexical scorer | 0.775 [0.574, 1.023] | 0.631 | 0.714 [0.592, 0.816] | 0.543 |
| **NLI cross-encoder (ONNX)** | **0.669 [0.493, 0.874]** | **0.667** | **0.770 [0.657, 0.863]** | **0.606** |

> ⚠️ **Read before quoting these numbers.** The benchmark
> ([`backend/evaluation/data/`](backend/evaluation/data/README.md)) is **synthetic**: an AI
> agent wrote and labelled the answers, and the labels are **not teacher-validated**. Use it
> for regression testing and for comparing engines, not as a claim of real-world accuracy.
> Full report: [`backend/evaluation/reports/2026-11-23-test-local/report.md`](backend/evaluation/reports/2026-11-23-test-local/report.md).

---

## 📸 Screenshots & demo

<sub>Captured from a local run seeded with the bundled synthetic demo data
(`python -m scripts.seed_demo`). No real student data is shown. These captures ran with
`SARAS_REVIEW_REQUIRED=false`, so confident AI marks went straight to the results.</sub>

| Sign in | Dashboard |
|---|---|
| ![Login screen](docs/images/screenshots/01-login.png) | ![Dashboard](docs/images/screenshots/02-dashboard.png) |
| **Step 1: Rubric** (typed, or drafted from a question paper) | **Step 2: Upload** answer-sheet PDFs |
| ![Rubric](docs/images/screenshots/03-rubric.png) | ![Upload](docs/images/screenshots/04-upload.png) |
| **Step 3: AI status**, a live per-stage processing monitor | **Step 5: Review workspace** with page image, regions, marks, confidence and evidence |
| ![Processing status](docs/images/screenshots/05-status.png) | ![Review workspace](docs/images/screenshots/06-review.png) |
| **Region editor:** reassign, resize, split, merge, draw | **Step 4: Results** per student |
| ![Region editor](docs/images/screenshots/07-region-editor.png) | ![Results](docs/images/screenshots/08-results.png) |
| **Step 6: Final approval** and CSV export | **Analytics** |
| ![Final approval](docs/images/screenshots/09-approval.png) | ![Analytics](docs/images/screenshots/10-analytics.png) |
| **Dark theme:** dashboard | **Dark theme:** review |
| ![Dashboard dark](docs/images/screenshots/11-dashboard-dark.png) | ![Review dark](docs/images/screenshots/12-review-dark.png) |

### Try the demo yourself

```powershell
cd backend
.venv\Scripts\python -m scripts.seed_demo       # demo teacher + "Machine Learning Unit Test" + 12 sample sheets
```

Sign in with **`demo@saras.local` / `demo-password-123`** (development only; the script
refuses to run with `SARAS_ENV=prod`). The 12 sample sheets cover full marks, partial
answers, paraphrases, OCR noise, a skipped question, a duplicate marker, out-of-order
answers, a continuation page, a flipped fact (contradiction), an adversarial answer, an
empty sheet and an image-only scan. A step-by-step viva demo script is in
[docs/DEMO.md](docs/DEMO.md).

---

## 🧰 Technology stack

| Layer | Technologies |
|---|---|
| **Frontend** | React 18, TypeScript (strict), Vite 5, React Router 6, lucide-react icons, self-hosted Inter font, hand-written CSS design tokens (light / dark / system themes) |
| **Backend API** | Python 3.12, FastAPI, Starlette, Uvicorn, Pydantic v2 + pydantic-settings, python-multipart |
| **Persistence** | SQLAlchemy 2 ORM. SQLite (WAL) on the laptop, PostgreSQL 16 (psycopg 3) in Docker. Append-only audit log. |
| **Auth & security** | argon2-cffi (Argon2 password hashing), PyJWT (access + rotating refresh tokens), custom rate-limit, upload-guard and security-header middleware |
| **Document & vision** | PyMuPDF (PDF rendering + text layer), OpenCV (quality, deskew, CLAHE, thresholding, morphology, line detection), NumPy |
| **OCR** | PDF text layer · vision LLMs (Gemini Flash-Lite, a Groq vision model) · TrOCR-small (`transformers`, optional) |
| **NLP / grading** | ONNX Runtime + `tokenizers` (DeBERTa-v3-xsmall NLI, int8), Sentence-Transformers (`all-MiniLM-L6-v2`, optional), PyTorch CPU (optional), custom lexical scorer |
| **LLM integration** | Plain `httpx` adapters (no vendor SDKs) for Groq, Google Gemini, OpenRouter and Ollama; strict JSON-schema output |
| **Background jobs** | Database-backed job table + in-process worker thread (replaces Celery + Redis) |
| **Evaluation** | NumPy-only metrics (QWK, κ, ECE, AURC, CER/WER, bootstrap CIs) |
| **DevOps** | Docker (multi-stage, non-root), Docker Compose (PostgreSQL + scale profile), Caddy (automatic HTTPS), GitHub Actions CI, Vercel (optional frontend), Cloudflare Tunnel (demo link) |
| **Testing** | pytest, httpx / Starlette TestClient, an in-process end-to-end smoke script, `tsc --noEmit` type-checking |

---

## ⚙️ Full engineering mechanism

This section follows one answer sheet from upload to export.
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/GRADING_DESIGN.md](docs/GRADING_DESIGN.md)
contain the full detail.

### Step 1: Upload (`POST /api/v1/exams/{id}/submissions`)

1. **Upload guard middleware** refuses unauthenticated requests before reading the body, and
   cuts off any request larger than the budget (50 files × 20 MB) with **413** while it
   streams.
2. The exam must belong to the caller and have at least one question. A repeated
   `Idempotency-Key` returns the original submissions and creates nothing new.
3. Each file streams to `storage_dir/submissions/<uuid>/original.pdf` in 1 MB chunks.
   `validate_pdf` checks the `%PDF-` signature (415), that the file opens (422), password
   protection, zero pages, and the page limit (50 → 413).
4. Rejected files are listed in `rejected[]` with their own error code, and the others carry
   on. A `Submission` row (`uploaded`), a `Job` row (`queued`) and an audit entry are
   written, and the response is **202 Accepted**.

### Step 2: The background job ([`worker.py`](backend/app/worker.py))

- The worker **claims** the oldest queued job, or a `running` job with a stale heartbeat
  (crash recovery). It uses an optimistic `UPDATE … WHERE status = <seen> AND attempts = <seen>`,
  so two workers never take the same job.
- The heartbeat is refreshed after every page. Failures are retried with exponential
  back-off up to 3 attempts, after which the sheet becomes `failed` and keeps a readable
  error message.
- Processing is **idempotent**: previous OCR lines, regions and answers are deleted before
  rebuilding.

### Step 3: Pipeline ([`pipeline/processor.py`](backend/app/pipeline/processor.py))

```text
original.pdf
  │ for each page (isolated: a failing page is flagged "page_failed", the sheet continues)
  ├─► render with PyMuPDF (150 DPI, grayscale) + text-layer lines with pixel boxes
  ├─► quality check → blank_page / poor_quality
  ├─► image pages only: deskew → median denoise → CLAHE → adaptive threshold → ruled-line removal
  ├─► OCR cascade: text layer │ vision LLM │ TrOCR │ empty boxes for manual transcription
  ▼
all OCR lines (table ocr_lines; regions can be re-cut later without OCR)
  ├─► segmentation by question marker → AnswerRegion rows (+ AI split when no markers)
  ├─► per question: text = lines inside its regions
  │     • no letters or digits      → "blank", 0 marks, never graded
  │     • unreadable lines only     → needs_transcription (teacher types the answer)
  │     • otherwise                 → grade (ensemble) and route
  └─► totals, per-stage timings, live progress events, status "processed"
```

### Step 4: Grading and routing

1. The **local primary grader** (NLI-ONNX when its model is installed, otherwise lexical)
   and a **cross-checker** grade every answer. If the AI marker is enabled and consented,
   **one LLM request per sheet** judges every key point.
2. `ensemble.combine()` applies the per-criterion rules: verified quotes, disputes and
   contradictions. It then sets the marks, verdict (`correct` / `partially_correct` /
   `incorrect` / `ungraded`), flags, feedback and a privacy-safe **grading trace**.
3. `route_graded()` computes the combined confidence and the route (`auto_accepted` or
   `needs_review`). Maths questions are always `maths_manual_review`.
4. If any configured grader fails, it is retried 3 times. The pipeline then falls back to
   the lexical scorer (`graded_with_fallback`). If the AI is unavailable, the offline result
   stands (`ai_unavailable_fallback`), so **a sheet never fails because of the AI**.

### Step 5: Teacher review, approval and export

- `PATCH /answers/{id}/review` re-grades corrected text immediately and stores a
  `Correction` row for future model improvement. Overrides must lie in `[0, max]` and need
  a reason.
- Region edits (crop / reassign / split / merge / blank) mark regions `manual`, recompute
  the affected answers and totals, and snapshot the old regions for **undo**.
- **Approval** is blocked (409, listing the pending questions) while answers still need
  review. It freezes `final_marks` and locks the sheet, and every later write returns 409
  until the sheet is **reopened with a reason**.
- The **CSV export** contains approved sheets only, with exactly the frozen marks.

### Submission lifecycle

```mermaid
stateDiagram-v2
    [*] --> uploaded: POST /submissions (202)
    uploaded --> processing: worker claims job
    processing --> processed: pipeline + grading done
    processing --> failed: 3 attempts failed
    failed --> uploaded: POST /retry
    processed --> in_review: first teacher review
    processed --> approved: approve
    in_review --> approved: approve (no pending answers)
    approved --> in_review: POST /reopen (reason required)
```

---

## 🏗️ System architecture

```mermaid
flowchart LR
    subgraph Browser
        UI["React 18 SPA<br/>TypeScript · Vite"]
    end

    subgraph Server["FastAPI process (uvicorn)"]
        MW["Middleware<br/>security headers · rate limit<br/>upload guard · CORS"]
        API["REST API /api/v1<br/>auth · exams · submissions<br/>review · regions · results"]
        WK["Job worker thread<br/>claim · heartbeat · retry"]
        PL["Pipeline<br/>PyMuPDF · OpenCV · OCR<br/>segmentation"]
        GR["Grading ensemble<br/>NLI-ONNX · lexical · SBERT<br/>LLM marker"]
        RT["LLM router<br/>cache · budgets · fallback"]
    end

    DB[("SQLite / PostgreSQL<br/>users · exams · answers<br/>jobs · audit_log · ai_runs")]
    FS[("File storage<br/>PDFs · page PNGs")]
    LLM["Groq · Gemini · OpenRouter · Ollama<br/>(optional, consent-gated)"]

    UI -- "JSON + Bearer JWT" --> MW --> API
    API --> DB
    API --> FS
    WK -- "poll jobs" --> DB
    WK --> PL --> GR
    PL --> FS
    GR --> RT --> LLM
    GR --> DB
```

**Request flow, from upload to export:**

```mermaid
sequenceDiagram
    autonumber
    actor T as Teacher
    participant UI as React SPA
    participant API as FastAPI
    participant DB as Database
    participant W as Worker
    T->>UI: Upload answer-sheet PDFs
    UI->>API: POST /exams/{id}/submissions (Idempotency-Key)
    API->>API: Upload guard + PDF validation
    API->>DB: Submission (uploaded) + Job (queued) + audit
    API-->>UI: 202 Accepted
    loop every 2-5 s
        UI->>API: GET /submissions/{id}/status
    end
    W->>DB: Claim job (optimistic UPDATE)
    W->>W: Render, clean, OCR, segment, grade, route
    W->>DB: OCR lines, regions, answers, totals
    T->>UI: Review flagged answers
    UI->>API: PATCH /answers/{id}/review
    API->>DB: Re-grade + Review + Correction + audit
    T->>UI: Approve and export
    UI->>API: POST /submissions/{id}/approve
    UI->>API: GET /exams/{id}/export?format=csv
```

| Design decision | Reason |
|---|---|
| **One process, two roles** (API + worker thread) | Fits a 4 GB laptop. The DB job table lets you add worker processes with no broker (`docker compose --profile scale up --scale worker=2`). |
| **Database job queue instead of Celery + Redis** | Fewer moving parts, crash recovery through heartbeats, optimistic claiming |
| **FastAPI serves the built SPA** | Production is a single origin and a single process. Unknown non-`/api` paths fall back to `index.html`. |
| **Models load lazily, once per process** | No ML dependency is required. Each engine is used only if it is installed. |
| **Pluggable OCR and scorers** | Swap or upgrade an engine without touching the pipeline |
| **AI is never on the critical path** | Every AI failure falls back to the offline graders |

---

## 🗂️ Project directory structure

```text
SARAS.AI/
├── backend/
│   ├── app/
│   │   ├── main.py               FastAPI app factory: routers, middleware, /health, /ready, SPA serving
│   │   ├── config.py             Settings (SARAS_* env vars, backend/.env)
│   │   ├── db.py · models.py     SQLAlchemy engine, ORM schema, audit-log triggers
│   │   ├── schemas.py            Pydantic request/response contracts
│   │   ├── errors.py             ApiError + uniform error format
│   │   ├── security.py           Argon2, JWT, current_user, login lock-out
│   │   ├── ratelimit.py          Sliding-window rate limiter middleware
│   │   ├── uploadguard.py        Streaming upload-size guard middleware
│   │   ├── security_headers.py   CSP (hash-based), HSTS, X-Frame-Options, …
│   │   ├── audit.py · worker.py  Audit writer · DB job queue + worker thread
│   │   ├── api/                  auth.py · exams.py · submissions.py · deps.py (ownership checks)
│   │   ├── pipeline/             pdf · preprocess · ocr · ai_ocr · segment · ai_segment ·
│   │   │                         question_paper · routing · progress · processor
│   │   ├── grading/              text · scorers · onnx_nli · engine · ensemble · llm_grader · direct_ai
│   │   ├── llm/                  router · providers · limiter · redact · schema · fakes (tests)
│   │   └── evaluation/           metrics.py (QWK, κ, ECE, AURC, CER…) · grading.py (benchmark runner)
│   ├── evaluation/               data/ (dev + held-out benchmark) · reports/ · ai_benchmark.py · pipeline_benchmark.py
│   ├── scripts/                  make_samples · seed_demo · smoke_api · download_models · backup
│   ├── samples/                  rubric.json · exam.json · manifest.json (PDFs are generated)
│   ├── tests/                    pytest suite (27 files)
│   ├── requirements.txt          runtime (CPU only, no PyTorch)
│   ├── requirements-dev.txt      + pytest, httpx
│   ├── requirements-ml.txt       optional TrOCR / SBERT / PyTorch NLI extras
│   └── .env.example              every setting with its default and an explanation
├── frontend/
│   ├── src/
│   │   ├── api/                  typed API client (token refresh, retries) + endpoint helpers
│   │   ├── auth/ · theme/        AuthContext · ThemeProvider (light / dark / system)
│   │   ├── components/           Button, Badge, DataTable, Modal, Tabs, Toast, Tooltip, Logo, …
│   │   ├── layout/               AppShell (sidebar, breadcrumbs)
│   │   ├── pages/                Login, Dashboard, Rubric, QuestionPaperImport, Upload, Status,
│   │   │                         PipelineMonitor, Review, Results, Approval, Analytics, RegionEditor
│   │   ├── review/               ReviewWorkspace, ScanViewer, RegionCanvas, RegionEditor, KeyPointList, …
│   │   ├── hooks/                useAsync, usePolling, useLocalStorage, …
│   │   └── styles/               design tokens + component / page CSS
│   ├── vite.config.ts            dev server :5173, /api proxy → :8000
│   └── vercel.json               optional Vercel deployment of the SPA
├── deploy/                       Caddyfile + docker-compose.https.yml (automatic HTTPS)
├── docs/                         ARCHITECTURE · GRADING_DESIGN · DEPLOYMENT · DEMO · PRODUCT_SPEC · images/
├── scripts/                      run_dev.ps1 · run_prod.ps1 · saras.mjs (cross-platform launcher)
├── .github/workflows/ci.yml      backend tests, frontend build, Docker build + container smoke test
├── Dockerfile                    multi-stage image (SPA + API + worker), non-root
├── docker-compose.yml            PostgreSQL 16 + app (+ optional worker replicas)
└── package.json                  root npm scripts: setup · dev · start · test · seed
```

Created at run time (git-ignored): `backend/.venv/`, `backend/data/` (database, uploads,
page images), `backend/models/` (ONNX model), `backend/samples/sheets/`,
`frontend/node_modules/` and `frontend/dist/`.

---

## 📋 Prerequisites

| Requirement | Version / notes |
|---|---|
| **Python** | 3.12 |
| **Node.js** | 18.18 or newer (CI uses Node 22) |
| **OS** | Windows 10/11, Linux or macOS |
| **RAM** | 4 GB is enough for the default profile. Install the optional ML extras only with 8 GB or more. |
| **Disk** | ~1 GB for dependencies + ~100 MB for the optional ONNX NLI model |
| **Docker** *(optional)* | Docker Engine / Desktop with Compose v2, for the extended profile (8 GB+ RAM) |
| **AI API keys** *(optional)* | Groq and/or Google AI Studio (Gemini), only for AI reading and marking |

---

## 🚀 Installation & setup

### Option A: one-command launcher (any OS)

From the repository root:

```bash
git clone https://github.com/Koushik-Biswas/SARAS.AI.git
cd SARAS.AI
npm run setup        # creates backend/.venv, installs Python + web dependencies (add -- --dev for test tools)
npm run seed         # optional: demo teacher, exam and 12 sample sheets
npm start            # builds the SPA if needed and serves everything on http://127.0.0.1:8000
```

Other commands: `npm run dev` (API with auto-reload on :8000 + Vite on :5173), `npm run dev:api`,
`npm run dev:web`, `npm run build`, `npm test`, and `node scripts/saras.mjs help`.

### Option B: manual setup (Windows, 4 GB low-resource profile)

```powershell
# 1. Backend: virtual environment and dependencies (CPU only, no PyTorch)
cd backend
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt      # or requirements-dev.txt for pytest
copy .env.example .env                                        # optional: every value has a default

# 2. Optional: the local NLI grader model (~90 MB, int8 ONNX)
.venv\Scripts\python -m scripts.download_models

# 3. Optional: demo data
.venv\Scripts\python -m scripts.seed_demo

# 4. Frontend: build the SPA once
cd ..\frontend
npm install
npm run build

# 5. Run everything in one process on http://127.0.0.1:8000
cd ..
powershell -ExecutionPolicy Bypass -File scripts\run_prod.ps1 -SkipBuild
```

- **Development mode** (hot reload): `powershell -ExecutionPolicy Bypass -File scripts\run_dev.ps1`,
  or two terminals: `cd backend; .venv\Scripts\python -m uvicorn app.main:app --reload --port 8000`
  and `cd frontend; npm run dev` (http://localhost:5173 proxies `/api` to :8000).
- **Linux / macOS:** use `.venv/bin/python` in place of `.venv\Scripts\python`.
- **Sample PDFs only:** `python -m scripts.make_samples` writes the rubric, 12 typed sheets of
  varied quality, an image-only sheet and invalid files to `backend/samples/`.

### Optional ML extras (8 GB RAM or more)

```powershell
cd backend
.venv\Scripts\python -m pip install -r requirements-ml.txt    # CPU PyTorch, transformers, sentence-transformers
```

This enables TrOCR handwriting OCR and the SBERT / PyTorch-NLI graders. Models download
from Hugging Face on first use. **Do not install these on a 4 GB machine.** Use Kaggle,
Colab or a bigger machine for heavy work.

### Enabling AI reading and marking (optional)

Add to `backend/.env` (never commit it):

```ini
SARAS_AI_GRADING=ensemble          # AI marker + offline cross-check
SARAS_AI_CONSENT=true              # nothing is sent to any provider without this
SARAS_GROQ_API_KEY=<your-groq-key>
SARAS_GEMINI_API_KEY=<your-gemini-key>
SARAS_GEMINI_CONSENT=true          # Gemini's free tier may use data for training: separate consent
SARAS_AI_GRADING_STYLE=moderate    # lenient | moderate | strict
```

`GET /api/v1/capabilities` reports which OCR engines, graders and AI providers the current
install can use.

---

## 🔧 Environment variables & configuration

All settings use the `SARAS_` prefix and are read from the environment or `backend/.env`.
Real environment variables override `.env`. [`backend/.env.example`](backend/.env.example)
documents every value. The most important ones:

<details open>
<summary><b>Runtime, storage and auth</b></summary>

| Variable | Default | Description |
|---|---|---|
| `SARAS_ENV` | `dev` | `dev`, `test` or `prod`. In `prod` the app **refuses to start** with the development JWT secret. |
| `SARAS_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `SARAS_DATABASE_URL` | SQLite `backend/data/saras.db` | or `postgresql+psycopg://user:pass@host:5432/saras` |
| `SARAS_STORAGE_DIR` | `backend/data/storage` | Uploaded PDFs and rendered page images |
| `SARAS_JWT_SECRET` | dev-only value | **Required outside dev.** Generate one with `python -c "import secrets; print(secrets.token_urlsafe(48))"` |
| `SARAS_ACCESS_TOKEN_MINUTES` | `15` | Access-token lifetime |
| `SARAS_REFRESH_TOKEN_DAYS` | `7` | Refresh-token lifetime (rotated on every use) |
| `SARAS_LOGIN_MAX_FAILURES` / `SARAS_LOGIN_LOCK_MINUTES` | `5` / `5` | Login lock-out policy |
| `SARAS_MAX_UPLOAD_MB` / `SARAS_MAX_PAGES` | `20` / `50` | Per-file upload limits |

</details>

<details>
<summary><b>Pipeline, grading and routing</b></summary>

| Variable | Default | Description |
|---|---|---|
| `SARAS_RENDER_DPI` | `150` | Page render resolution (kept low for 4 GB RAM) |
| `SARAS_OCR_ENGINE` | `auto` | `auto` · `textlayer` · `trocr` |
| `SARAS_TROCR_MODEL` | `microsoft/trocr-small-handwritten` | TrOCR checkpoint |
| `SARAS_GRADER` | `auto` | Local semantic grader: `auto` · `lexical` · `sbert` · `nli` · `nli-onnx` |
| `SARAS_LOCAL_PRIMARY` | `auto` | Local grader that decides when the AI does not: `auto` · `lexical` · `nli-onnx` |
| `SARAS_SBERT_MODEL` / `SARAS_NLI_MODEL` | MiniLM-L6-v2 / nli-deberta-v3-xsmall | Optional PyTorch models |
| `SARAS_TAU` / `SARAS_TAU_LOW` | `0.6` / `0.4` | Entailment for *matched* / *partial*. A changed value overrides every scorer's own default. |
| `SARAS_CONTRADICTION_THRESHOLD` | `0.5` | Contradiction probability that marks a key point contradicted |
| `SARAS_AUTO_ACCEPT_THRESHOLD` | `0.80` | Combined confidence needed to auto-accept |
| `SARAS_LOW_OCR_CONFIDENCE` | `0.5` | Below this, the answer gets `low_ocr_confidence` |
| `SARAS_CALIBRATION_TEMPERATURE` | `1.0` | Temperature scaling (1.0 = identity) |
| `SARAS_REVIEW_REQUIRED` | `true` | `true`: uncertain answers block approval until checked. `false`: AI marks go straight to results, and review stays optional. |

</details>

<details>
<summary><b>Worker and API</b></summary>

| Variable | Default | Description |
|---|---|---|
| `SARAS_WORKER_ENABLED` | `true` | Run the job worker thread inside the API process |
| `SARAS_WORKER_POLL_SECONDS` | `1.0` | Queue polling interval |
| `SARAS_JOB_MAX_ATTEMPTS` | `3` | Retries before a sheet is `failed` |
| `SARAS_JOB_STALE_SECONDS` | `300` | A running job with no heartbeat for this long is re-queued |
| `SARAS_CORS_ORIGINS` | `["http://localhost:5173","http://127.0.0.1:5173"]` | JSON list of allowed browser origins |
| `SARAS_RATE_LIMIT_ENABLED` | `true` | Sliding-window rate limiting |
| `SARAS_RATE_LIMIT_PER_MINUTE` / `SARAS_UPLOAD_RATE_PER_MINUTE` | `100` / `10` | Per caller |
| `SARAS_SECURITY_HEADERS` | `true` | CSP, X-Frame-Options, nosniff, Referrer-Policy, HSTS (HTTPS in prod) |
| `SARAS_FRONTEND_DIST` | `frontend/dist` | Built SPA served at `/` when present |

</details>

<details>
<summary><b>AI (LLM) reading and marking</b></summary>

| Variable | Default | Description |
|---|---|---|
| `SARAS_AI_GRADING` | `off` | `off` · `ensemble` (AI marker + offline cross-check) |
| `SARAS_AI_CONSENT` | `false` | Master switch: **nothing is sent to any provider unless `true`** |
| `SARAS_GEMINI_CONSENT` / `SARAS_OPENROUTER_CONSENT` | `false` | Extra consent for providers whose free tiers may log or train on data |
| `SARAS_GROQ_API_KEY` · `SARAS_GEMINI_API_KEY` · `SARAS_OPENROUTER_API_KEY` | — | Keys (only in `backend/.env`; never logged or returned) |
| `SARAS_OLLAMA_BASE_URL` | — | Local Ollama (used only when the URL is local or on a private network) |
| `SARAS_AI_GRADING_PROVIDERS` | Groq gpt-oss-120b → gpt-oss-20b → Gemini Flash-Lite… | Ordered `provider:model` list |
| `SARAS_AI_OCR` / `SARAS_AI_OCR_PROVIDERS` | `true` / Gemini → Groq vision… | Vision-model reading of pages without a text layer |
| `SARAS_AI_OCR_CONCURRENCY` | `3` | Pages read in parallel |
| `SARAS_AI_GRADING_STYLE` | `moderate` | `lenient` · `moderate` · `strict` |
| `SARAS_AI_REASONING_EFFORT` | `low` | For reasoning models: `low` · `medium` · `high` |
| `SARAS_AI_TIMEOUT_SECONDS` / `SARAS_AI_MAX_RETRIES` / `SARAS_AI_CALL_DEADLINE_SECONDS` | `30` / `2` / `60` | Request limits |
| `SARAS_GROQ_TOKENS_PER_MINUTE` / `_PER_DAY` | `6000` / `170000` | Client-side budgets below the free tier |
| `SARAS_GEMINI_REQUESTS_PER_MINUTE` / `_PER_DAY` | `10` / `400` | Client-side budgets |
| `SARAS_AI_MAX_INPUT_TOKENS_PER_CALL` | `3500` | A sheet is split into several requests only above this |
| `SARAS_AI_VERIFY` / `SARAS_AI_DIRECT_SCHEME` | `false` | **Experimental.** Measured *less* accurate on the dev benchmark, so off by default. |

</details>

Docker Compose additionally reads `SARAS_DB_PASSWORD`, `SARAS_DOMAIN` (HTTPS overlay) and
`SARAS_APP_CPUS`. The SPA on Vercel reads `VITE_API_ORIGIN`.

---

## 🗄️ Database schema & design

SQLAlchemy 2 models live in [`backend/app/models.py`](backend/app/models.py). UUID primary keys
are used throughout, timestamps are UTC, and `CHECK` constraints enforce every status and
range.

```mermaid
erDiagram
    USERS ||--o{ EXAMS : creates
    USERS ||--o{ REFRESH_TOKENS : holds
    USERS ||--o{ REVIEWS : writes
    USERS |o--o{ AUDIT_LOG : "acts in"
    EXAMS ||--o{ QUESTIONS : has
    EXAMS ||--o{ SUBMISSIONS : receives
    EXAMS |o--o{ AI_RUNS : "logged for"
    SUBMISSIONS ||--o{ OCR_LINES : "read into"
    SUBMISSIONS ||--o{ ANSWER_REGIONS : "segmented into"
    SUBMISSIONS ||--o{ ANSWERS : contains
    SUBMISSIONS ||--o{ JOBS : "processed by"
    SUBMISSIONS |o--o{ AI_RUNS : "logged for"
    QUESTIONS ||--o{ ANSWERS : "answered by"
    QUESTIONS |o--o{ ANSWER_REGIONS : "assigned to"
    ANSWERS ||--o{ REVIEWS : "reviewed in"
    ANSWERS ||--o{ CORRECTIONS : "corrected in"

    USERS {
        uuid id PK
        string email UK
        string password_hash "Argon2"
        string role "teacher | admin"
    }
    EXAMS {
        uuid id PK
        uuid teacher_id FK
        string title
        date exam_date
        float total_marks "> 0"
    }
    QUESTIONS {
        uuid id PK
        uuid exam_id FK
        int question_no "unique per exam"
        string type "short | long | paragraph | maths"
        float max_marks
        text model_answer
        json key_points "text, weight, keywords"
    }
    SUBMISSIONS {
        uuid id PK
        uuid exam_id FK
        string student_ref "unique per exam"
        string status "uploaded to approved | failed"
        float total_ai_marks
        float total_final_marks
        json timings
        json progress
    }
    OCR_LINES {
        uuid id PK
        uuid submission_id FK
        int page_no
        json bbox
        text text
        float confidence
    }
    ANSWER_REGIONS {
        uuid id PK
        uuid question_id FK "nullable = unassigned"
        int page_no
        json bbox
        string source "auto | manual"
        float segmentation_confidence
    }
    ANSWERS {
        uuid id PK
        uuid submission_id FK
        uuid question_id FK
        text ocr_text
        float ai_marks
        float final_marks
        float confidence
        json evidence "per key point"
        json flags
        string status "auto_accepted | needs_review | reviewed | blank | failed"
        string verdict
        json grading_trace
    }
    REVIEWS {
        uuid id PK
        uuid answer_id FK
        uuid teacher_id FK
        float final_marks
        text reason
    }
    CORRECTIONS {
        uuid id PK
        uuid answer_id FK
        text original_text
        text corrected_text
    }
    AUDIT_LOG {
        int id PK
        string entity
        string action
        json old_value
        json new_value
        text reason
    }
    JOBS {
        uuid id PK
        uuid submission_id FK
        string status "queued | running | done | failed"
        int attempts
        datetime heartbeat_at
    }
    AI_RUNS {
        uuid id PK
        string provider
        string model
        string cache_key "sha256"
        string status
        int input_tokens
        int output_tokens
    }
    REFRESH_TOKENS {
        uuid id PK "JWT jti"
        uuid user_id FK
        datetime expires_at
        bool revoked
    }
```

**Design notes**

- **Unique constraints:** `(exam_id, question_no)`, `(exam_id, student_ref)` and
  `(submission_id, question_id)` prevent duplicate questions, sheets and answers.
- **Append-only audit log:** on SQLite, database **triggers reject `UPDATE` and `DELETE`** on
  `audit_log`. On PostgreSQL the same guarantee must come from role privileges.
- **OCR lines are kept** so regions can be re-cropped or reassigned without running OCR again.
- **`ai_runs`** is both the LLM **response cache** (by SHA-256 `cache_key`) and the source of
  the rolling daily budgets. It stores only the anonymised prompt, never keys or names.
- **Additive migrations:** `db.ensure_additive_columns()` adds newer nullable columns
  (`verdict`, `feedback`, `grading_trace`) to existing databases on start-up.
- **Cascades:** deleting an exam or sheet removes its answers, regions, jobs and AI runs.

---

## 📡 API documentation

Interactive **OpenAPI / Swagger** docs are at **`/api/docs`**, and the schema is at
`/api/openapi.json`. All routes are under **`/api/v1`** and need
`Authorization: Bearer <access token>`, except register, login, refresh and health.

**Conventions**
- Errors always have the shape `{"error": {"code", "message", "details"}}`. The codes are
  `validation_error`, `unauthorized`, `forbidden`, `not_found`, `conflict`, `file_too_large`,
  `unsupported_media_type`, `unreadable_pdf`, `rate_limited`, `service_unavailable` and
  `internal_error`.
- List endpoints accept `page` and `page_size` (default 20, max 100).
- Timestamps are UTC ISO 8601, and marks have at most 2 decimal places.

| Area | Method & path | Purpose |
|---|---|---|
| **Auth** | `POST /auth/register` | Create a teacher account → tokens |
| | `POST /auth/login` | Email + password → access + refresh token |
| | `POST /auth/refresh` | Rotate the refresh token |
| | `POST /auth/logout` | Revoke the refresh token |
| | `GET /auth/me` | Current user |
| **Exams** | `GET` / `POST /exams` | List / create exams |
| | `GET` / `PATCH` / `DELETE /exams/{id}` | Read / update / delete an exam |
| **Rubric** | `GET` / `PUT` / `POST /exams/{id}/questions` | Read / replace / add questions and key points |
| | `POST /exams/{id}/questions/extract` | Question paper (PDF / image) → **draft** rubric |
| | `POST /exams/{id}/regrade` | Re-grade every sheet after a rubric change |
| **Uploads** | `POST /exams/{id}/submissions` | Multipart PDFs (`files`, optional `student_refs`, `Idempotency-Key`) → **202** |
| | `GET /exams/{id}/submissions` | List sheets |
| | `GET /submissions/{id}` · `/status` | Sheet detail · live processing status |
| | `POST /submissions/{id}/retry` · `DELETE /submissions/{id}` | Retry a failed sheet · delete a sheet |
| **Review** | `GET /submissions/{id}/answers?status=&sort=priority` | Answers, riskiest first |
| | `GET /answers/{id}` · `PATCH /answers/{id}/review` | Answer detail · correct text / override marks / comment / approve |
| | `POST /answers/{id}/flag` | Flag an answer for attention |
| **Approval** | `POST /submissions/{id}/approve` · `/reopen` | Lock marks · reopen with a reason |
| | `POST /exams/{id}/approve-all` | Approve every ready sheet |
| **Pages & regions** | `GET /submissions/{id}/pages` · `/pages/{n}/image?thumb=1` | Page list · page PNG |
| | `GET` / `POST /submissions/{id}/regions` · `POST …/regions/undo` | List / draw regions · undo the last edit |
| | `PATCH /regions/{id}` | Crop · reassign · split · merge · blank · unblank |
| **Results** | `GET /exams/{id}/results` · `/analytics` | Results grid · class analytics |
| | `GET /exams/{id}/export?format=csv` | CSV of **approved** sheets only |
| **System** | `GET /audit-log?entity=&id=` · `/capabilities` | Audit trail · installed engines |
| | `GET /health` · `/ready` | Liveness · readiness (DB + worker); also at the root |

<details>
<summary><b>Example: the full flow with <code>curl</code></b></summary>

```bash
BASE=http://127.0.0.1:8000/api/v1

# Sign in
TOKEN=$(curl -s -X POST $BASE/auth/login -H "content-type: application/json" \
  -d '{"email":"demo@saras.local","password":"demo-password-123"}' | python -c "import sys,json;print(json.load(sys.stdin)['access_token'])")
AUTH="Authorization: Bearer $TOKEN"

# Create an exam
EXAM=$(curl -s -X POST $BASE/exams -H "$AUTH" -H "content-type: application/json" \
  -d '{"title":"ML Unit Test","subject":"Machine Learning","date":"2026-11-20","total_marks":5}' \
  | python -c "import sys,json;print(json.load(sys.stdin)['id'])")

# Set the rubric (key-point weights must add up to max_marks)
curl -s -X PUT $BASE/exams/$EXAM/questions -H "$AUTH" -H "content-type: application/json" -d '{
  "questions": [{
    "question_no": 1, "type": "long", "max_marks": 5,
    "prompt": "Explain overfitting in machine learning.",
    "model_answer": "Overfitting is when a model fits the training data too closely ...",
    "key_points": [
      {"text": "the model fits the training data too closely", "weight": 2, "keywords": ["training", "closely"]},
      {"text": "it performs poorly on new unseen data",        "weight": 2, "keywords": ["unseen"]},
      {"text": "a remedy is regularisation or more data",      "weight": 1, "keywords": ["regularisation"]}
    ]}]}'

# Upload answer sheets (202 Accepted), then poll the status
curl -s -X POST $BASE/exams/$EXAM/submissions -H "$AUTH" -H "Idempotency-Key: batch-001" \
  -F "files=@sheet1.pdf" -F "files=@sheet2.pdf"
curl -s $BASE/submissions/<submission_id>/status -H "$AUTH"

# Review: override a mark (a reason is required when it differs from the AI mark)
curl -s -X PATCH $BASE/answers/<answer_id>/review -H "$AUTH" -H "content-type: application/json" \
  -d '{"final_marks": 4, "reason": "Remedy explained in the diagram", "approve": true}'

# Approve the sheet and export the class CSV
curl -s -X POST $BASE/submissions/<submission_id>/approve -H "$AUTH"
curl -s "$BASE/exams/$EXAM/export?format=csv" -H "$AUTH" -o results.csv
```

</details>

---

## 🔐 Authentication & security

| Control | Implementation |
|---|---|
| **Password hashing** | **Argon2** (`argon2-cffi`). Passwords must be at least 8 characters. |
| **Tokens** | Short-lived **JWT access token** (15 min, kept in memory by the SPA) + **rotating refresh token** (7 days, `jti` stored in `refresh_tokens`, revoked on logout and rotated on every use) |
| **Brute-force protection** | Login **lock-out** after 5 failures for 5 minutes, plus a sliding-window **rate limiter** (100 requests/min and 10 uploads/min per caller). Callers are identified by verified token or IP, so forged headers never earn a fresh allowance. |
| **Authorization** | **Per-teacher data isolation**: every exam, sheet and answer goes through an ownership check ([`api/deps.py`](backend/app/api/deps.py)). An `admin` role can read everything but cannot change marks. |
| **Upload hardening** | Token checked before the body is read, a streaming size budget (413), `%PDF-` signature, page / password / size checks, random UUID storage names, page-render pixel caps against "PDF bombs" |
| **HTTP security headers** | Hash-based **CSP** (no `unsafe-inline` scripts), `X-Frame-Options: DENY`, `nosniff`, `Referrer-Policy`, `Permissions-Policy`, COOP, and **HSTS** over HTTPS in prod |
| **Production guard** | The app refuses to start in `prod` with the development JWT secret. Docker runs as a non-root user. |
| **Integrity** | **Append-only audit log** (who, when, old value, new value, reason). Approval **locks** marks, and reopening requires a reason. |
| **AI privacy** | Explicit consent flags per provider. **PII redaction** (e-mails, URLs, phone numbers, roll / registration numbers, labelled names) before any LLM call. Answer text is treated as data, never as instructions. API keys live only in `.env` and are never logged, stored or returned. Logs contain no answer text or student identifiers. |
| **Secrets** | `.env`, keys, databases, uploads, backups and model weights are all git-ignored |

---

## 🧪 Testing & code quality

```powershell
cd backend
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest -q             # single process on purpose (no xdist)
.venv\Scripts\python scripts\smoke_api.py     # in-process end-to-end API smoke test

cd ..\frontend
npm run typecheck                             # tsc --noEmit (strict mode)
npm run build                                 # type-check + production build
```

- **365 test functions in 27 files** cover auth, the API contract, uploads, segmentation,
  the rubric, grading engines, the ONNX NLI grader, the ensemble, the LLM router, redaction
  and evidence matching, the question-paper reader, regions, the review flow, the worker,
  production hardening, evaluation metrics and the demo seed.
- The tests use a temporary SQLite database, the lexical grader, PDF text-layer OCR and
  **fake LLM providers** ([`llm/fakes.py`](backend/app/llm/fakes.py)). They need **no ML
  models and no network**.
- **Continuous integration** ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs on
  every push and pull request:
  1. backend tests + smoke test (Python 3.12)
  2. frontend type-check + build (Node 22)
  3. `docker compose config` for both compose files, `docker build`, and a **container smoke
     test** that checks `/health`, `/ready`, the SPA and the CSP / X-Frame-Options headers

**Grading benchmark** (measures the production code paths and never re-implements grading):

```powershell
cd backend
.venv\Scripts\python -m app.evaluation.grading --data evaluation/data/grading_benchmark_dev.json `
    --graders lexical,nli-onnx,baseline-formula --out evaluation/runs/dev --run-label "dev run"
```

The dev split (42 answers) is for tuning. The held-out test split (111 answers) is for the
final evaluation only. `evaluation/ai_benchmark.py` compares versions of the AI marker on
identical data.

---

## ☁️ Deployment instructions

Full guide: **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**. SARAS AI needs a **long-running server
with persistent disk** (background worker, uploads, database). Serverless-only hosting cannot
run the backend.

| Option | Cost | Best for |
|---|---|---|
| **A. Linux VM + Docker + Caddy (HTTPS)** | VM cost | Real use (recommended) |
| **B. Laptop + Cloudflare quick tunnel** | Free | Demos, low RAM |
| **C. Frontend on Vercel + backend from A or B** | Free (Hobby) | A Vercel URL |

**A. Docker with automatic HTTPS**

```bash
git clone https://github.com/Koushik-Biswas/SARAS.AI.git && cd SARAS.AI
cat > .env <<'EOF'
SARAS_JWT_SECRET=<long random string>
SARAS_DB_PASSWORD=<database password>
SARAS_DOMAIN=saras.example.edu
SARAS_APP_CPUS=2
EOF
docker compose -f docker-compose.yml -f deploy/docker-compose.https.yml up -d --build
docker compose exec app python -m scripts.download_models     # optional offline NLI grader
```

- **Plain HTTP for testing:** `docker compose up -d --build` → http://localhost:8000
  (PostgreSQL 16 + app, with resource limits).
- **More processing capacity:** `docker compose --profile scale up -d --scale worker=2`.
- **ML extras:** build with `--build-arg INSTALL_ML=1`.
- **Single container with SQLite:**
  `docker run -d -p 8000:8000 -v saras-data:/data -e SARAS_JWT_SECRET=<secret> saras-ai`.

**B. Laptop + Cloudflare tunnel**

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_prod.ps1      # http://127.0.0.1:8000
cloudflared tunnel --url http://127.0.0.1:8000                    # prints a public https URL
```

**C. Vercel frontend:** import the repo with **Root Directory = `frontend`**, set
`VITE_API_ORIGIN=https://<your-backend>`, and add the Vercel origin to `SARAS_CORS_ORIGINS`.

**After deploying:** `/health` returns `{"status":"ok"}`, `/ready` reports the database and
worker, and the response headers include `Content-Security-Policy`. Never run
`scripts.seed_demo` in production. Schedule `python -m scripts.backup --keep 14` (database +
uploads; backups contain student work, so keep them private).

---

## ⚠️ Known issues & limitations

- **The evaluation numbers are synthetic.** The benchmark was written and labelled by an AI
  agent, not by teachers. Thresholds, priors, agreement factors and the calibration
  temperature (1.0) are **provisional** and have not been fitted on teacher-marked data.
- **With the default settings and no AI marker, most answers go to review.** Lexical-only
  confidence (0.6 × 0.85 ≈ 0.51) and NLI + lexical agreement (≈ 0.79) both sit below the 0.80
  auto-accept threshold. This is deliberate until calibration is measured.
- **Deskew only corrects one tilt direction with OpenCV 5.x.** `cv2.minAreaRect` can return
  angles near −90°, which [`deskew()`](backend/app/pipeline/preprocess.py) does not
  normalise, so such pages are left uncorrected. A fix is pending.
- **Handwriting needs an engine.** On the 4 GB profile without AI keys or ML extras, scanned
  pages without a text layer must be transcribed by the teacher. TrOCR is used off the shelf
  (not fine-tuned).
- **Maths answers** (FR-11) are routed to manual review (`maths_manual_review`), and there is
  no automatic step marking. **PDF report cards** are not built (CSV export only).
- **Segmentation** depends on question numbers at the start of a line. Sub-parts merge into
  the parent question. Sheets without markers rely on the AI splitter or the region editor.
- **The NLI model** was trained on SNLI / MultiNLI (everyday English, not exam answers).
  Domain terms, OCR noise and long premises degrade it, and unrelated pairs can get high
  contradiction probabilities, which is why it is always cross-checked.
- **Lexical grading is a heuristic.** It can miss paraphrases, and it detects contradictions
  only through negation and antonyms.
- **Redaction** cannot remove a student name written without a label, or phone numbers split
  into bare digit groups.
- **Free-tier AI quotas** (e.g. ~200K tokens/day on Groq) can run out on busy days. The app
  then falls back to the next model and finally to the offline graders.
- **Experimental AI features** (`SARAS_AI_VERIFY`, `SARAS_AI_DIRECT_SCHEME`) were measured
  *less* accurate on the dev benchmark and are off by default.
- **Single-process state:** the rate limiter and login lock-out are kept in memory per process.
- **PostgreSQL** is supported, but the test suite runs on SQLite, and the append-only audit
  guarantee on PostgreSQL needs role privileges that the Compose file does not configure.
- **No admin API:** registration always creates a teacher. To make an admin, set
  `users.role = 'admin'` in the database.
- **Teacher corrections** are stored (FR-19), but there is no retraining pipeline yet.

---

## 🤝 Contributing guidelines

Contributions, bug reports and ideas are welcome.

1. **Fork** the repository and create a branch from `main`:
   `git checkout -b feature/short-description` (or `fix/...`, `docs/...`).
2. **Set up** with `npm run setup -- --dev` (or the manual steps above).
3. **Keep the conventions:**
   - **Backend:** Python 3.12 type hints, Pydantic schemas for every request and response,
     the uniform `ApiError` format, and an audit entry for every change to marks, text or
     regions.
   - **Grading:** marks are always computed in code. AI output is only ever a judgement
     with verified evidence.
   - **Privacy:** never log answer text, student references or keys, and never send data to
     an AI provider without the consent flags.
   - **Frontend:** strict TypeScript and the existing components and CSS tokens (no new UI
     framework).
   - **Configuration:** every new setting gets a `SARAS_*` variable with a safe default,
     documented in `backend/.env.example`.
4. **Test:** add or update tests in `backend/tests/`. `pytest -q`, `scripts/smoke_api.py` and
   `npm run build` must pass.
5. **Never commit** `.env` files, API keys, databases, uploaded sheets or model weights.
6. **Open a pull request** that describes the change, why it is needed, and how you tested
   it (screenshots for UI changes). CI must be green.

To report a bug, open an issue with steps to reproduce, the expected and actual behaviour,
and logs (with any student data removed).

---

## 📄 License

Released under the **MIT License**. See [LICENSE](LICENSE).

Copyright © 2026 **Koushik Biswas**

---

## 📬 Contact & support

**Koushik Biswas**

- 📧 Email: [koushikbiswas2423@gmail.com](mailto:koushikbiswas2423@gmail.com)
- 🐙 GitHub: [@Koushik-Biswas](https://github.com/Koushik-Biswas)
- 🐞 Issues: [github.com/Koushik-Biswas/SARAS.AI/issues](https://github.com/Koushik-Biswas/SARAS.AI/issues)

For questions, collaboration or support, email or open an issue. If SARAS AI is useful to
you, please consider giving the repository a ⭐.

<div align="center">
<sub>Built for teachers 👩‍🏫, with the AI suggesting and the teacher deciding.</sub>
</div>

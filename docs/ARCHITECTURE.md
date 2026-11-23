# SARAS AI — Architecture

This document explains how an uploaded answer sheet becomes reviewed, approved marks. The
code references point into `backend/app/`.

## 1. Components

```
 Browser (React SPA)                 FastAPI process (uvicorn, one per machine on the laptop)
 ───────────────────                 ─────────────────────────────────────────────────────────
 pages/*  ── fetch /api/v1 ──►  RateLimit → CORS → routers (api/auth, api/exams, api/submissions)
   polls status every 2-5 s             │                     │
                                        ▼                     ▼
                                  SQLAlchemy session     storage_dir/submissions/<uuid>/
                                  (SQLite or Postgres)     original.pdf, page_001.png ...
                                        ▲
                                        │  jobs table (queued → running → done/failed)
                                  Worker thread (worker.py) ── process_submission() ──► pipeline/, grading/
```

- **One process, two roles.** The HTTP API and a background worker thread run in the same
  uvicorn process (low-resource profile, PRD 18.6). The worker claims jobs from the
  `jobs` table, so you can add more processes (see `docker-compose.yml`, profile `scale`)
  without adding a broker. This codebase replaces the PRD's Celery + Redis with that
  table.
- **The SPA is served by FastAPI** from `frontend/dist`. Any unknown non-`/api` path returns
  `index.html`.
- **Models are loaded once per process**, lazily (`get_ocr_engine`, `get_scorer`), and only
  if the optional ML extras are installed.

## 2. Upload (`POST /api/v1/exams/{id}/submissions`)

1. The exam must belong to the caller (an admin can read but not upload) and have at least
   one question (409 otherwise).
2. If an `Idempotency-Key` header was already used for this exam, the request returns the
   submissions created the first time and creates nothing new.
3. Each file is streamed to `storage_dir/submissions/<uuid>/original.pdf` in 1 MB chunks.
   The size limit (20 MB) is checked while copying, so an oversized file is never fully
   written.
4. `pipeline/pdf.validate_pdf` checks the `%PDF-` signature (415), whether the file opens
   (422 `unreadable_pdf`), password protection (422), zero pages (422) and the page limit
   (50, 413).
5. A duplicate `student_ref` in the same exam is rejected with `conflict`. Rejected files
   are listed in `rejected[]` with their own code, and the other files carry on. The
   request fails as a whole only when **every** file is rejected.
6. A `Submission` row (status `uploaded`), a `Job` row and an audit entry are written,
   and the response is **202 Accepted**. The client then polls
   `GET /submissions/{id}/status`.

## 3. Jobs and failure handling (`worker.py`)

- `_claim` picks the oldest queued job, or a `running` job whose heartbeat is older than
  `job_stale_seconds` (crash recovery). It takes the job with an optimistic `UPDATE ...
  WHERE status = <seen> AND attempts = <seen>`, so two workers can never claim the same
  job.
- While a job runs, its heartbeat is refreshed after every page.
- On an exception the job is re-queued with exponential back-off (2, 4 s) up to
  `job_max_attempts` (3). After that the submission becomes `failed` and the error message
  is kept for the status screen. `POST /submissions/{id}/retry` queues it again.
- `process_submission` is **idempotent**. It deletes the submission's previous OCR lines,
  regions and answers before rebuilding them, so re-running a job is always safe.

## 4. The processing pipeline (`pipeline/processor.py`)

```
 original.pdf
   │  for each page (isolated: an exception marks the page "page_failed", the sheet continues)
   ├─► render (PyMuPDF, 150 DPI, grayscale) + PDF text layer lines with pixel boxes   pipeline/pdf.py
   ├─► quality check: ink ratio, Laplacian blur, contrast → blank_page / poor_quality pipeline/preprocess.py
   ├─► if no text layer: deskew (≤ 10°) and clean: median denoise → CLAHE contrast →
   │     adaptive threshold → ruled-line removal (OCR sees the cleaned copy;
   │     the review screen shows the uncleaned, deskewed page PNG)
   ├─► OCR (pipeline/ocr.py): text layer | TrOCR | "no recogniser" (empty line boxes)
   │     TrOCR mean confidence < 0.5 → one retry with stronger cleaning, keep the better
   ▼
 all OCR lines of the sheet (stored in ocr_lines)
   ├─► segmentation by question marker                                          pipeline/segment.py
   │     "Q1", "Q.2", "Question 3:", "Ans 4", "5.", "6)", "2(a)" at the start of a line
   │     text runs until the next marker, across pages; sub-parts merge into the parent
   │     → AnswerRegion rows (one per question per page, bbox in page pixels)
   ├─► per question: compute_answer()
   │     text = OCR lines whose centre lies inside the question's regions (marker stripped)
   │     ocr_confidence = mean line confidence weighted by text length (unread lines count fully)
   │     segmentation_confidence = lowest confidence of the question's regions
   │     no letter or digit → "blank": 0 marks, never sent to the grader
   │     lines found but no text → "needs_transcription" (teacher types the answer)
   │     otherwise grade (section 5) and route (section 6); some unread lines add
   │     "needs_transcription", fewer than 3 characters add "short_answer" (both review)
   └─► totals (refresh_totals), timings per stage, status "processed"
```

### Segmentation flags (FR-32)

| Flag | Raised when | Effect |
|---|---|---|
| `duplicate_marker` | the same question number appears twice | segment confidence ≤ 0.55, review |
| `out_of_order` | a lower number follows a higher one | segment confidence ≤ 0.8, informational |
| `gap_in_sequence` / `neighbour_missing` | a number inside the found range is missing (likely misread) | neighbours' confidence ≤ 0.7, review |
| `no_markers_found` | text was read but no marker matched | answers without regions go to review |
| `missing_marker` | a question has no region (skipped) | `blank` with 0 marks; review (still 0 marks, never graded) if the sheet had no markers, failed pages or unassigned text, or the number is missing after the last marker found |
| `continues_across_pages` | an answer spans pages without a new marker | informational |
| `unassigned` / `unassigned_text` | text before the first marker | region with `question_id = null`, fixed in the region editor; questions without a region go to review |
| `page_failed` | the page raised an exception | full-page unassigned region, review |
| `next_page_failed` | the answer was still running when the next page failed | its text may be cut short: review |

## 5. Grading (`grading/engine.py`, `grading/scorers.py`)

> **Current design:** grading is now an ensemble of the local graders and the optional AI marker, with per-criterion disputes, verified evidence and the partial-credit rule. See [GRADING_DESIGN.md](GRADING_DESIGN.md). The engine notes below describe the local graders only; where they disagree with GRADING_DESIGN.md, GRADING_DESIGN.md is authoritative.

The rubric for a question is a list of key points `{text, weight, keywords?}`. If no key
points are given, the model answer is used as a single key point worth the full marks.

1. **Split** the answer into sentences. Sentences that are not propositions (keyword lists,
   fragments) count only 0.3 as evidence. An answer made only of keywords gets the
   `keyword_list` flag.
2. **Score** every (sentence, key point) pair with the active scorer and get an entailment
   matrix *E* and a contradiction matrix *C*.
   - `lexical`: stemmed and fuzzy content-word overlap that tolerates OCR noise.
     Contradiction comes from a negation-parity or antonym flip in the clause that best
     matches the key point.
   - `sbert`: cosine similarity with the same contradiction heuristic.
   - `nli`: a DeBERTa-v3 cross-encoder that predicts entailment and contradiction
     directly.
3. **Per key point**, take the best-supporting sentence:
   - contradiction ≥ `contradiction_threshold` (0.5) → **contradicted**, 0 marks, and
     the `contradiction` flag. The sentence is kept as `evidence_sentence` (FR-33).
   - entailment ≥ `tau` (0.6) → **matched**, full weight.
   - entailment ≥ `tau_low` (0.4) → **partial**, half weight, and the `partial_credit` flag.
   - otherwise → **missed**.
   - If the key point contains numbers the answer lacks, entailment is capped below `tau`
     and the answer gets the `numeric_mismatch` flag.
4. **Marks**
   - `long` / `paragraph`: the sum of the awarded key-point weights.
   - `short` (3- and 5-mark answers, PRD 12.4): `max × clip(0.6·similarity + 0.4·keyword
     coverage)`, rounded to the nearest 0.5. Contradicted weight is subtracted, and keyword
     lists are scaled by 0.3.
   - `maths`: not graded automatically. 0 suggested marks and the
     `maths_manual_review` flag (FR-11 is manual-only).
   - An answer that only restates the question gets the `copied_question` flag and 0 marks.
   - The result is clipped to `[0, max_marks]` and rounded to 2 decimals.
5. **Grading confidence** is the weighted mean, over key points, of how far each score is
   from the nearest decision threshold: `0.5 + 0.5 · min(1, distance / 0.2)`. A score
   sitting on `tau` gives 0.5, and a clear match or miss gives 1.0. For `short` questions
   it is averaged with how far the raw mark is from a 0.5-mark rounding boundary.
6. **Fallback**: if the configured scorer raises an exception, grading is retried 3 times
   with back-off. After that the lexical scorer is used and the answer gets the
   `graded_with_fallback` flag.

## 6. Confidence routing (`pipeline/routing.py`, FR-13)

```
segmentation factor = min(1, 0.4 + 0.6 · segmentation_confidence / 0.95)
confidence          = calibrate(ocr_confidence × grading_confidence × segmentation factor, T)
calibrate(p, T)     = sigmoid(logit(p) / T)          # temperature scaling; T = 1.0 → identity
```

Routing adds these flags:

- `low_ocr_confidence` when `ocr_confidence < 0.5` (`SARAS_LOW_OCR_CONFIDENCE`).
- `low_confidence` when `confidence < 0.80` (`SARAS_AUTO_ACCEPT_THRESHOLD`).

An answer is **`auto_accepted`** only if it carries **no review flag**. The informational
flags `continues_across_pages`, `out_of_order` and `partial_text` do not count. Any other
flag sends it to **`needs_review`**: `low_confidence`, `low_ocr_confidence`,
`partial_credit`, `contradiction`, `numeric_mismatch`, `keyword_list`, `copied_question`,
`segmentation_warning`, `needs_transcription`, `maths_manual_review`, `teacher_flag`, and so
on. Blank answers get the status `blank`.

Worked examples:

| Case | OCR | Grading | Segmentation | Confidence | Route |
|---|---|---|---|---|---|
| Typed sheet, clear match | 0.99 (text layer) | 0.90 | 0.95 → factor 1.0 | 0.891 | auto_accepted (if no other flag) |
| Handwriting via TrOCR | 0.70 | 0.90 | 0.95 → 1.0 | 0.630 | needs_review (`low_confidence`) |
| Duplicate marker | 0.99 | 0.90 | 0.55 → 0.747 | 0.666 | needs_review (`low_confidence`, `segmentation_warning`) |
| Borderline key point | 0.99 | 0.55 | 0.95 → 1.0 | 0.545 | needs_review (`low_confidence`, `partial_credit`) |

The review queue (`GET /submissions/{id}/answers?sort=priority`) lists `needs_review` and
`failed` first, then answers by ascending confidence, then by question number. The
teacher's time goes to the riskiest answers first, and auto-accepted answers stay
available for spot checks.

`T` (`SARAS_CALIBRATION_TEMPERATURE`) and all thresholds are meant to be fitted on a
labelled validation set and frozen before the test set is used (PRD 12.11). This repository
ships untuned defaults.

## 7. Teacher review, approval and audit

- **Review** (`PATCH /answers/{id}/review`):
  - Corrected text is re-graded immediately (`compute_answer(override_text=...)`), stored
    as a `Correction` row (FR-19) and marks the answer `text_corrected`.
  - A final mark that differs from the AI mark needs a `reason` (400 otherwise) and must
    lie in `[0, max_marks]`.
  - `approve=true` sets the answer to `reviewed`, and the first review moves the sheet from
    `processed` to `in_review`.
- **Regions** (`PATCH /regions/{id}`):
  - The actions are crop, reassign, split, merge, blank and unblank. Each one marks the
    region `source = "manual"` and recomputes the affected questions' answers and the
    totals.
  - The previous region set is snapshotted in the audit log, which is what
    `POST /submissions/{id}/regions/undo` restores.
- **Approval** (`POST /submissions/{id}/approve`, `/exams/{id}/approve-all`):
  - Blocked (409, `details.pending` = question numbers) while any answer is `needs_review`
    or `failed`.
  - Approval freezes `final_marks` on every answer and locks the sheet. Every write then
    returns 409 until `POST /submissions/{id}/reopen` is called with a reason.
- **Export**: `GET /exams/{id}/export?format=csv` contains approved sheets only, and its
  marks are exactly the frozen final marks.
- **Audit**:
  - Every change to marks, text, regions or status is written to `audit_log` (who, when,
    old value, new value, reason).
  - On SQLite, triggers reject `UPDATE` and `DELETE` on that table. On PostgreSQL the same
    guarantee has to come from role privileges.
  - The log is read with `GET /audit-log?entity=&id=`.

## 8. Data model (`models.py`)

`User` 1─* `Exam` 1─* `Question`
`Exam` 1─* `Submission` 1─* `OcrLine`, `AnswerRegion`, `Answer`, `Job`
`Answer` 1─* `Review`, `Correction`
`AuditLog` is append-only. `RefreshToken` records each refresh token's id (JWT `jti`), so
tokens can be rotated and revoked.

Submission status: `uploaded → processing → processed → in_review → approved` (or
`failed`). Answer status: `auto_accepted | needs_review | reviewed | blank | failed`.

## 9. Resource profiles

| | Low-resource (4 GB laptop) | Extended (8 GB+) |
|---|---|---|
| Run | `scripts/run_prod.ps1`: one uvicorn process | `docker compose up` |
| Database | SQLite (WAL mode) | PostgreSQL 16 |
| Jobs | in-process worker thread, one sheet and one page at a time | same, plus optional `worker` replicas |
| OCR / grader | text layer or manual transcription / lexical | TrOCR-small / SBERT or NLI (`INSTALL_ML=1`) |
| Limits | 150 DPI, models load lazily and only if installed | app 4 CPU / 6 GB, PostgreSQL 1 CPU / 1 GB |

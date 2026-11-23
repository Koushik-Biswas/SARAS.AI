# SARAS AI — Smart Assessment and Recognition Automation System

SARAS AI is the final-year project *AI-Powered Exam Answer Sheet Evaluation and Marks
Assessment System* (PRD v2.5). A teacher creates an exam and its rubric (questions, maximum
marks, model answer, weighted key points), then uploads scanned student answer-sheet PDFs.
The pipeline renders each page, cleans it, reads the text, splits it into one answer per
question using the question numbers written on the sheet, and grades every answer against
the key points. Answers it is unsure about go to a review queue. The teacher reviews,
corrects, overrides and approves the marks, then exports a CSV of the approved results.

The AI suggests marks and the teacher decides. Every change to marks, text or answer
regions is written to an append-only audit log, and only approved results can be exported.

## Highlights

- **End-to-end grading pipeline**: PDF → page images → deskew / denoise / contrast →
  OCR → segmentation by question marker → key-point grading → confidence routing → totals.
- **Explainable marks**: each key point is reported as matched, partial, missed or
  contradicted, with the student sentence used as evidence.
- **Confidence-based routing**: OCR, grading and segmentation confidence are combined
  into one score. Low-confidence or flagged answers go to `needs_review` and are shown first.
- **Teacher always in control**: edit the OCR text (the answer is re-graded), override
  marks (a reason is required), comment, flag, approve or lock a sheet, reopen it with a
  reason, and fix answer regions (crop, reassign, split, merge, mark blank, undo).
- **Robust batch uploads**: up to 50 PDFs per request, streamed to disk under random UUID
  names, with signature, size, page-count and password checks. `Idempotency-Key` prevents
  duplicate uploads, and each page is processed in isolation so one bad page or file never
  fails the batch.
- **Runs on a 4 GB laptop**: SQLite, one process, an in-process job worker and no ML
  models required. It degrades gracefully, and installing the optional ML extras turns on
  TrOCR, SBERT and NLI.
- **Polished web UI**: React + TypeScript single-page app with dark, light and system
  themes, a self-hosted Inter font (works offline), accessible components, live
  processing status, a results grid and class analytics.
- **Secure by default**: Argon2 password hashing, short-lived JWT access tokens with
  rotating refresh tokens, login lock-out, rate limiting, per-teacher data isolation and a
  read-only admin role.

## Architecture

| Layer | Technology | Where |
|---|---|---|
| Web UI | React 18, Vite, TypeScript, react-router, lucide-react | `frontend/src/` |
| REST API | FastAPI (all routes under `/api/v1`), Pydantic v2 contracts, uniform error format | `backend/app/main.py`, `backend/app/api/`, `backend/app/schemas.py`, `backend/app/errors.py` |
| Auth and security | Argon2, PyJWT (access + rotating refresh), login lock-out, rate limiting | `backend/app/security.py`, `backend/app/ratelimit.py`, `backend/app/api/auth.py` |
| Background jobs | Database job table + one worker thread per process (retries, back-off, stale-job recovery) | `backend/app/worker.py` |
| Processing pipeline | PyMuPDF (render), OpenCV (quality, deskew, clean, line detection), pluggable OCR, regex segmentation | `backend/app/pipeline/` |
| Grading | Key-point rubric engine and pluggable pair scorers (lexical / SBERT / NLI) | `backend/app/grading/` |
| Persistence | SQLAlchemy 2: SQLite (laptop) or PostgreSQL (Compose); append-only audit log | `backend/app/db.py`, `backend/app/models.py`, `backend/app/audit.py` |
| Files | Uploaded PDFs and rendered page PNGs under `storage_dir/submissions/<uuid>/` | `backend/data/storage/` (created at run time) |

When `frontend/dist/` exists, FastAPI serves the built SPA at `/` (any non-`/api` path falls
back to `index.html`), so production is a single Python process. In development the Vite
dev server on port 5173 proxies `/api` to FastAPI on port 8000.

The processing pipeline and the confidence routing are explained in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quick start (Windows, 4 GB laptop: low-resource profile)

Requirements: Python 3.12 and Node.js 18.18 or newer (CI uses Node 22). Close other
programs and browser tabs first, because the laptop has very little free RAM.

```powershell
# 1. Backend: virtual environment and dependencies (no ML models, CPU only)
cd backend
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt       # use requirements-dev.txt to also get pytest
copy .env.example .env                                         # optional: every value has a default

# 2. Demo data: the demo teacher, a "Machine Learning Unit Test" exam and all sample sheets
#    (generates backend\samples\ if needed, uploads and processes the sheets; dev only)
.venv\Scripts\python -m scripts.seed_demo
#    Log in as demo@saras.local / demo-password-123

# 3. Frontend: build the SPA once (output: frontend\dist)
cd ..\frontend
npm install
npm run build

# 4. Run everything in one process on http://127.0.0.1:8000 (Ctrl+C to stop)
cd ..
powershell -ExecutionPolicy Bypass -File scripts\run_prod.ps1 -SkipBuild
```

- **Development mode** (API auto-reload and the Vite dev server with hot reload on
  http://localhost:5173): `powershell -ExecutionPolicy Bypass -File scripts\run_dev.ps1`.
  If you prefer two terminals, run
  `cd backend; .venv\Scripts\python -m uvicorn app.main:app --reload --port 8000` in one and
  `cd frontend; npm run dev` in the other.
- **Cross-platform launcher**: the repository root has a dependency-free Node launcher
  (`scripts/saras.mjs`). It provides `npm run setup`, `npm run dev`, `npm start`,
  `npm test` and `npm run seed`. Run `node scripts/saras.mjs help` to see the options.
- **Sample PDFs only**: `cd backend; .venv\Scripts\python -m scripts.make_samples` writes
  the rubric, typed sheets of varied quality, an image-only sheet and invalid files to
  `backend/samples/`. `scripts.seed_demo` runs this automatically when the samples are
  missing. The generated PDFs are git-ignored; the small JSON files (rubric, exam,
  manifest) can be committed.
- **Linux/macOS**: use `.venv/bin/python` in place of `.venv\Scripts\python`.

### Optional ML extras (8 GB RAM or more)

```powershell
cd backend
.venv\Scripts\python -m pip install -r requirements-ml.txt     # CPU PyTorch, transformers, sentence-transformers
```

This turns on TrOCR handwriting OCR and the SBERT / NLI graders. The models download from
Hugging Face the first time they are used. Do **not** install the extras on the 4 GB
laptop. Run heavy work on Kaggle or Colab instead (PRD 18.6).

## Running tests

```powershell
cd backend
.venv\Scripts\python -m pytest -q          # single process on purpose (no xdist)
.venv\Scripts\python scripts\smoke_api.py  # end-to-end API smoke test, in-process
cd ..\frontend
npm run build                              # TypeScript type-check + production build
```

The tests and the smoke script use a temporary SQLite database, the lexical grader and
PDF text-layer OCR, so they need no ML models and no network.
[.github/workflows/ci.yml](.github/workflows/ci.yml) runs these steps, plus
`docker compose config` and `docker build`, on every push and pull request.

## Docker (extended profile: 8 GB RAM or more)

```bash
export SARAS_JWT_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
export SARAS_DB_PASSWORD="choose-a-password"
docker compose up -d --build              # PostgreSQL 16 + app on http://localhost:8000
```

The [Dockerfile](Dockerfile) is multi-stage. Node 22 builds the SPA, and the Python 3.12
slim image runs as a non-root user and has a `/health` healthcheck.
[docker-compose.yml](docker-compose.yml) adds PostgreSQL with a healthcheck and persistent
volumes, and sets the PRD 18.5 resource limits (app 4 CPUs / 6 GB, PostgreSQL 1 CPU / 1 GB).
To include the ML extras, build with `SARAS_INSTALL_ML=1`. The PRD's Celery + Redis worker
is replaced by the database job table. To add processing capacity, run
`docker compose --profile scale up -d --scale worker=2`. Do not use Docker on the 4 GB
laptop: Docker Desktop alone needs 1–2 GB.

## Configuration

Settings are read from `SARAS_*` environment variables or `backend/.env`.
[backend/.env.example](backend/.env.example) lists every setting with its default and an
explanation. The most important ones:

| Setting | Default | Meaning |
|---|---|---|
| `SARAS_ENV` | `dev` | `dev`, `test` or `prod`. In `prod` the app refuses to start with the development JWT secret. |
| `SARAS_JWT_SECRET` | dev-only value | **Must** be set outside development. |
| `SARAS_DATABASE_URL` | SQLite file `backend/data/saras.db` | Or `postgresql+psycopg://user:pass@host:5432/saras` |
| `SARAS_STORAGE_DIR` | `backend/data/storage` | Uploaded PDFs and rendered page images |
| `SARAS_OCR_ENGINE` | `auto` | `auto`, `textlayer` or `trocr` |
| `SARAS_GRADER` | `auto` | `auto`, `lexical`, `sbert` or `nli` |
| `SARAS_TAU` / `SARAS_TAU_LOW` | `0.6` / `0.4` | Entailment needed for a full (matched) / half (partial) key point |
| `SARAS_CONTRADICTION_THRESHOLD` | `0.5` | Contradiction probability that marks a key point as contradicted |
| `SARAS_AUTO_ACCEPT_THRESHOLD` | `0.80` | Combined confidence needed to auto-accept an answer |
| `SARAS_MAX_UPLOAD_MB` / `SARAS_MAX_PAGES` | `20` / `50` | Per-file upload limits |
| `SARAS_RENDER_DPI` | `150` | Page render resolution (kept low for 4 GB RAM) |
| `SARAS_WORKER_ENABLED` | `true` | Run the job worker thread inside the API process |

## API overview

Interactive OpenAPI docs are at **`/api/docs`**, and the schema is at `/api/openapi.json`.
Every route is under `/api/v1` and needs `Authorization: Bearer <access token>`, except
register, login, refresh and health. Errors always have the shape
`{"error": {"code", "message", "details"}}`, using the codes `validation_error`,
`unauthorized`, `forbidden`, `not_found`, `conflict`, `file_too_large`,
`unsupported_media_type`, `unreadable_pdf`, `rate_limited`, `service_unavailable` and `internal_error`. List
endpoints take `page` and `page_size` (default 20, max 100). Timestamps are UTC ISO 8601
and marks have at most 2 decimal places.

| Area | Endpoints |
|---|---|
| Auth | `POST /auth/register`, `/auth/login`, `/auth/refresh`, `/auth/logout`; `GET /auth/me` |
| Exams and rubric | `GET/POST /exams`; `GET/PATCH/DELETE /exams/{id}`; `GET/PUT/POST /exams/{id}/questions`; `POST /exams/{id}/regrade` |
| Uploads and processing | `POST /exams/{id}/submissions` (multipart, `Idempotency-Key`, returns 202); `GET /exams/{id}/submissions`; `GET /submissions/{id}`, `/submissions/{id}/status`; `POST /submissions/{id}/retry`; `DELETE /submissions/{id}` |
| Review | `GET /submissions/{id}/answers?status=&sort=priority`; `GET /answers/{id}`; `PATCH /answers/{id}/review`; `POST /answers/{id}/flag` |
| Approval | `POST /submissions/{id}/approve`, `/submissions/{id}/reopen`; `POST /exams/{id}/approve-all` |
| Pages and regions | `GET /submissions/{id}/pages`, `/submissions/{id}/pages/{n}/image?thumb=1`, `/submissions/{id}/regions`; `POST /submissions/{id}/regions`, `/submissions/{id}/regions/undo`; `PATCH /regions/{id}` (crop, reassign, split, merge, blank, unblank) |
| Results | `GET /exams/{id}/results`, `/exams/{id}/export?format=csv` (approved sheets only), `/exams/{id}/analytics` |
| System | `GET /audit-log?entity=&id=`, `/capabilities`, `/health`, `/ready` (also at the root) |

## OCR and grading modes (graceful degradation)

| Mode | Needs | Behaviour |
|---|---|---|
| OCR `textlayer` | nothing | Reads the PDF's own text layer (typed or digitally written sheets, demo data). Confidence 0.99. |
| OCR `trocr` | ML extras | `microsoft/trocr-small-handwritten` on detected text lines, with PyTorch dynamic int8 quantisation, batch 4. Loaded once per process. If confidence is very low, it retries once with stronger cleaning. |
| OCR `auto` (default) | — | Uses the text layer when the page has one, otherwise TrOCR if installed. Otherwise it keeps the detected line boxes with **empty text** and flags the answer `needs_transcription`. The teacher types the answer on the review screen, and the typed text is graded straight away. |
| Grader `lexical` | nothing | Dependency-free scorer that tolerates OCR noise (stemming, fuzzy matching). It detects contradictions with a negation/antonym heuristic. |
| Grader `sbert` | ML extras | Sentence-BERT (`all-MiniLM-L6-v2`) cosine similarity. |
| Grader `nli` | ML extras | DeBERTa-v3-xsmall NLI cross-encoder, which predicts entailment and contradiction directly. |
| Grader `auto` (default) | — | `sbert` if sentence-transformers is installed, otherwise `lexical`. |

If the configured grader fails, the answer is retried 3 times with back-off and then graded
with the lexical scorer and flagged `graded_with_fallback`. Blank or skipped answers (fewer
than 3 characters) get 0 marks and are never sent to the grader. `GET /api/v1/capabilities`
reports which engines this install can run.

## PRD traceability

Status is based on the code in this repository. **Done** means the requirement works end
to end (API and UI). **Partial** means part of it is missing, as explained in the row.

| FR | Requirement (short) | Implemented in | Status |
|---|---|---|---|
| FR-1 | Register, log in, log out securely | `api/auth.py`, `security.py`; `pages/LoginPage.tsx`, `auth/AuthContext.tsx` | Done |
| FR-2 | Create exam with subject, date, total marks | `POST /exams` (`api/exams.py`); `pages/ExamFormModal.tsx` | Done |
| FR-3 | Upload question paper (PDF or typed), enter question no, type, max marks | `PUT/POST /exams/{id}/questions`; `pages/RubricPage.tsx` | Partial: typed entry only. There is no question-paper PDF upload. |
| FR-4 | Model answer and weighted key points per question | `QuestionIn.key_points` (`schemas.py`); `pages/RubricPage.tsx` | Done |
| FR-5 | Upload one or many answer-sheet PDFs | `POST /exams/{id}/submissions` (`api/submissions.py`); `pages/UploadPage.tsx` | Done |
| FR-6 | PDF to page images, deskew, denoise, contrast | `pipeline/pdf.py`, `pipeline/preprocess.py` | Done (cleaning runs on image pages; text-layer pages need none) |
| FR-7 | Split sheet into per-question regions by question number | `pipeline/segment.py`, `pipeline/processor.py` | Done (sub-parts such as 2(a) are merged into the parent question) |
| FR-8 | Recognise handwritten text per answer region | `pipeline/ocr.py` (`TrOCREngine`) | Partial: off-the-shelf TrOCR-small, only with the ML extras. It has not been fine-tuned or evaluated here. On the 4 GB profile, handwriting is transcribed by the teacher. |
| FR-9 | Grade 3- and 5-mark answers against model answer and key points | `grading/engine.py` (`short`: 0.6·similarity + 0.4·keyword coverage), `grading/scorers.py` | Done (lexical by default; SBERT / NLI with the extras) |
| FR-10 | Paragraph answers by key-point rubric, matched and missed points | `grading/engine.py`; `AnswerOut.key_points`, `matched_points`, `missed_points` | Done |
| FR-11 | Maths answers (final answer and step marks) | `grade_answer` routes `maths` questions to the teacher (`maths_manual_review` flag) | Not started: maths is manual review only |
| FR-12 | Sum question marks into the overall total | `refresh_totals` (`pipeline/processor.py`); `GET /exams/{id}/results`; `pages/ResultsPage.tsx` | Done |
| FR-13 | Confidence score and low-confidence flags | `pipeline/routing.py`, `compute_answer` | Done (calibration temperature defaults to 1.0 and has not been fitted on validation data) |
| FR-14 | Answer image, OCR text and suggested marks side by side | API: `/submissions/{id}/pages/{n}/image`, `/regions`, `/answers` | Partial: API done. The review workspace UI (`pages/ReviewPage.tsx`) currently lists answers, status, confidence and marks without the page image. |
| FR-15 | Edit OCR text, override marks, add comment | `PATCH /answers/{id}/review` (re-grades corrected text, requires a reason for overrides, audited) | Partial: API done. The edit form is part of the unfinished review workspace UI. |
| FR-16 | Approve results and lock them | `POST /submissions/{id}/approve`, `/reopen`, `/exams/{id}/approve-all`; `pages/ResultsPage.tsx` (approve all) | Done (approval is blocked while flagged answers are unreviewed; approved sheets are read-only) |
| FR-17 | Export results as CSV | `GET /exams/{id}/export?format=csv` (approved only); `pages/ResultsPage.tsx` | Done (PDF report cards are Phase 2 and not built) |
| FR-18 | Processing status and clear failure messages | `GET /submissions/{id}/status`, `/retry`, uniform errors (`errors.py`), job retries (`worker.py`); `pages/StatusPage.tsx` | Done |
| FR-19 | Store teacher corrections for later model improvement | `Correction` model (`models.py`), written by `PATCH /answers/{id}/review` | Done (stored only; no retraining pipeline) |
| FR-31 | Adjust crop, reassign, split, merge regions, mark blank | `PATCH /regions/{id}`, `POST /submissions/{id}/regions`, `/regions/undo` | Partial: API done. The region editor UI (`pages/RegionEditorPage.tsx`) is still a placeholder. |
| FR-32 | Flag segmentation problems (skipped, duplicate, out of order, continues across pages) | `pipeline/segment.py` (`duplicate_marker`, `out_of_order`, `gap_in_sequence`, `no_markers_found`, `continues_across_pages`, `unassigned`), `pipeline/processor.py` (`missing_marker`, `segmentation_warning`) | Done |
| FR-33 | Flag contradictions with evidence sentence | `grading/engine.py` (`contradicted` key points, `contradiction` flag, `evidence_sentence`) | Done (heuristic with the lexical grader; model-based with `nli`) |

FR-14, FR-15 and FR-31 depend on the review workspace and region editor screens. Those were
still being built when this table was written. Update the three rows when
`frontend/src/pages/ReviewPage.tsx` and `RegionEditorPage.tsx` are finished.

## Known limitations

- **No handwriting recognition on the 4 GB profile.** Without the ML extras, scanned pages
  that have no text layer produce empty lines, and the teacher types each answer (it is
  then graded automatically). TrOCR is used off the shelf. It has not been fine-tuned,
  and it is quantised with PyTorch dynamic int8, not ONNX Runtime as PRD 18.6 suggests.
  `onnxruntime` is installed but not used yet.
- **Thresholds are untuned defaults.** `tau`, `tau_low`, the contradiction threshold, the
  auto-accept threshold and the calibration temperature (1.0) have not been fitted on a
  labelled validation set.
- **The research track (R1–R6) is not in this repository.** That covers the NLI
  week-9 gate evaluation, the OCR noise model, deep-ensemble selective prediction, active
  learning, per-teacher adaptation, the legibility study and the evaluation
  notebooks/datasets (QWK, MAE, CER).
- **Maths grading (FR-11)** is manual review only. **Question-paper PDF upload (FR-3)**
  and **PDF report cards** are not implemented.
- **Segmentation** depends on question numbers written at the start of a line.
  Sub-parts are merged into the parent question, and an answer without a marker needs the
  region editor.
- **Lexical grading is a heuristic.** It can miss paraphrases, and it detects
  contradictions only through negation and antonyms. Use `sbert` or `nli` where RAM allows.
- **Deviations from the PRD stack.** React instead of Streamlit, and a database job table
  instead of Celery + Redis. There is no MLflow, Prometheus or Grafana: monitoring is
  Python logging plus per-stage timings stored on each submission.
- **Single-process state.** The rate limiter and login lock-out are held in memory per
  process.
- **PostgreSQL** is supported (driver, Compose file), but the test suite runs on SQLite. The
  append-only audit log is enforced by triggers on SQLite. On PostgreSQL it needs database
  role privileges, which the Compose file does not configure.
- **No admin API.** Registration always creates a teacher. To make an admin (who can read
  everything but cannot change marks), set `users.role = 'admin'` in the database.

## Project structure

```
SARAS.AI/
├── backend/
│   ├── app/
│   │   ├── main.py              FastAPI app: routers, middleware, /health, /ready, SPA serving
│   │   ├── config.py            Settings (SARAS_* env vars, backend/.env)
│   │   ├── db.py, models.py     SQLAlchemy engine, schema, audit-log triggers
│   │   ├── schemas.py           Pydantic request/response contracts
│   │   ├── errors.py            ApiError + uniform error handlers
│   │   ├── security.py          Argon2, JWT, current_user, login lock-out
│   │   ├── ratelimit.py, audit.py
│   │   ├── worker.py            Database job queue + worker thread
│   │   ├── api/                 auth.py, exams.py, submissions.py, deps.py (ownership checks)
│   │   ├── pipeline/            pdf, preprocess, ocr, segment, routing, processor
│   │   └── grading/             text utilities, scorers (lexical/SBERT/NLI), rubric engine
│   ├── scripts/                 make_samples.py, smoke_api.py, seed_demo.py
│   ├── tests/                   pytest suite (pytest.ini: single process)
│   ├── requirements.txt         runtime (CPU, no ML models)
│   ├── requirements-dev.txt     + pytest, httpx
│   ├── requirements-ml.txt      optional TrOCR / SBERT / NLI extras
│   └── .env.example             every setting with its default
├── frontend/
│   ├── src/                     api/, auth/, components/, hooks/, layout/, pages/, styles/, theme/
│   ├── vite.config.ts           dev server :5173, /api proxy to :8000
│   └── package.json             npm run dev | build | typecheck
├── scripts/
│   ├── run_dev.ps1              API (--reload) + Vite dev server
│   ├── run_prod.ps1             build SPA, then one uvicorn process (4 GB profile)
│   └── saras.mjs                cross-platform launcher behind the root npm scripts
├── docs/
│   ├── ARCHITECTURE.md          pipeline and confidence routing
│   └── DEMO.md                  step-by-step viva demo script
├── .github/workflows/ci.yml     backend tests, frontend build, docker checks
├── Dockerfile                   multi-stage image (SPA + API + worker)
├── docker-compose.yml           extended profile: PostgreSQL 16 + app
└── package.json                 root npm scripts (setup, dev, start, test, seed)
```

Created at run time (git-ignored): `backend/.venv/`, `backend/data/` (SQLite database,
uploads, page images), `backend/samples/` (from `scripts.make_samples`),
`frontend/node_modules/` and `frontend/dist/` (from `npm run build`).

## Documentation

- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md): **production deployment** (VM + Docker + HTTPS,
  laptop + Cloudflare tunnel, Vercel frontend), required settings, backups, post-deploy checklist.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): processing pipeline, confidence
  routing, data model, jobs and failure handling.
- [docs/DEMO.md](docs/DEMO.md): viva demo script.
- [frontend/README.md](frontend/README.md): frontend development notes.

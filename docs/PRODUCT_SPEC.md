# SARAS AI — Feature Specification and Implementation Roadmap

*Smart Assessment and Recognition Automation System*
Version 3.0 (draft for team and mentor review) · 23 Nov 2026 · Extends PRD v2.5

This document turns PRD v2.5 and the requirements added on 23 Nov 2026 into one
specification. Those additions are AI-assisted "Autopilot" checking, Ask AI,
institution isolation, step-wise marking, the evaluation framework and the release
checklist. It also sets the order of work.

Status labels used below:

- **Done** — implemented and covered by automated tests in the repository today.
- **Partial** — usable, but missing parts listed in the same row.
- **Planned (Stage N)** — not built yet. Nothing marked Planned may be shown to users as working.

---

## 1. Decision record

| Topic | Decision | Source |
|---|---|---|
| Frontend | React + Vite + TypeScript, prebuilt and served by FastAPI. Premium design system with Light, Dark and System themes. The existing UI and UX are kept; new features use the same components. | Team decision; it replaces the PRD's Streamlit because a static bundle adds no RAM on the 4 GB laptop |
| Job queue | Database-backed job table with retries, a heartbeat and idempotent jobs, plus a standalone worker process for the extended profile. Redis and Celery are not required. | PRD 11.2 low-resource profile |
| Grading engine | The offline checker (lexical, Sentence-BERT or NLI) always runs. AI (LLM) grading is layered on top, and the offline checker is the fallback whenever no AI is available. | PRD 13.5: "APIs are never in the critical path" |
| AI providers | Groq (`openai/gpt-oss-120b`) is the main AI marker; Groq's terms say it does not train on customer data. Google Gemini is used for handwriting reading, question-paper import and as a backup marker, and only with a separate consent, because the free tier may use data for training and human review. Optional providers: Cloudflare Workers AI, OpenRouter free models and a local Ollama. | Verified provider research, 23 Nov 2026 |
| Marks arithmetic | The AI only judges each criterion (met / partly met / missed / contradicted) and quotes the student's words as evidence. **SARAS computes the marks in code** from the teacher's criterion weights. A quote that does not appear in the answer is rejected. | Hallucination guard |
| Approval | **One teacher click**: "Approve all ready sheets". Fully automatic approval is an option and is off by default. The teacher has final authority. | User decision 23 Nov 2026; PRD non-goal "fully replacing the teacher" |
| Ask AI | Teachers only. Answers are grounded in stored marks, criteria and evidence. Ask AI never changes marks by itself. | User decision 23 Nov 2026 |
| Diagram grading | Included as **Beta**: AI vision only, and always routed to teacher review. | User decision 23 Nov 2026 (exception to PRD non-goal) |
| Languages | English answers only. | PRD scope freeze; user decision 23 Nov 2026 |
| Accuracy claims | No accuracy figure appears anywhere (UI, README, report) unless it comes from the evaluation framework run on real teacher-marked data. | Development rule 12 |

---

## 2. The Autopilot workflow (what a teacher does)

```
1 Exam ──► 1 question paper (+ answer key / marking scheme / teaching material)
        ──► AI drafts questions, model answers and step-wise criteria ──► teacher edits and saves
        ──► many answer sheets (PDF, photos or a ZIP)
        ──► automatic: read handwriting → split by question → AI + offline checker grade every criterion
        ──► confident answers accepted; only doubtful ones queued for review
        ──► teacher reviews the queue, can Ask AI "why?", corrects text or criterion marks
        ──► "Approve all ready sheets" (one click) ──► CSV export + student-wise PDF reports
```

| Step | Teacher effort | What SARAS does automatically |
|---|---|---|
| Create exam | Title, subject, date, total marks | — |
| Upload question paper and key | Upload 1–3 PDFs or photos | Extracts questions and marks. Drafts the model answer and step-wise criteria with weights that sum to each question's marks. Uses an uploaded marking scheme when one is given. |
| Check the rubric | Edit, then click Save | Validates that weights sum to max marks; shows a running total |
| Choose grading style | Strict / Moderate / Lenient (default Moderate) | Adjusts the match thresholds and the instructions given to the AI. "Try on 3 sheets first" previews marks before the full batch. |
| Upload answer sheets | Drag and drop PDFs, images or a ZIP | Validates files, builds one PDF per student, queues jobs, tracks progress |
| Processing | None | Reads text (PDF text layer, then AI vision with consent, then local TrOCR if installed, then manual typing as the last resort). Segments answers, grades step by step, cross-checks, scores confidence and routes each answer. |
| Review | Only flagged answers | Shows evidence, the AI explanation, the offline checker's view, and why the answer was flagged |
| Approve and export | One click, then download | Locks marks, writes the audit log, exports CSV and generates PDF reports |

---

## 3. Feature specification

### 3.1 Accounts, institutions and users

| ID | Requirement | Acceptance criteria | Status |
|---|---|---|---|
| U-1 | Register, log in, log out, refresh tokens (FR-1) | Wrong credentials are always rejected. 401 without a valid token. Argon2 password hashes. Lockout after 5 failures. | **Done** |
| U-2 | Institutions: each user belongs to one institution; all data is isolated per institution | A user can never read or change another institution's exams, sheets, files or audit entries (403/404). Covered by isolation tests on every route. | Planned (Stage 1) |
| U-3 | Roles: `institution_admin`, `teacher`; platform `superadmin` only through the CLI | An admin manages teachers and can read every exam in the institution but cannot edit marks. A teacher sees only their own exams. | Partial: teacher/admin roles exist, without institutions |
| U-4 | User management UI: invite or create teachers, deactivate, reset password, change role | Every action is audited. A deactivated user cannot log in, and their sessions are revoked. | Planned (Stage 1) |
| U-5 | First-run setup: the first registered account creates the institution and becomes its admin; later users join with an invite link or code | No orphan users | Planned (Stage 1) |

### 3.2 Exams, question papers and rubrics

| ID | Requirement | Acceptance criteria | Status |
|---|---|---|---|
| E-1 | Create, edit and delete an exam: subject, date, total marks (FR-2) | Field-level validation errors. Deleting removes files, text and marks (deletion check). | **Done** |
| E-2 | Typed rubric entry: questions, type, max marks, model answer, weighted criteria (FR-3, FR-4) | Weights must sum to max marks; Save stays disabled until they do | **Done** |
| E-3 | Upload a question paper, answer key or marking scheme, and teaching material (PDF or images) | Stored with the exam; text extracted from the text layer or by AI vision (with consent) | Planned (Stage 2) |
| E-4 | AI draft: questions, max marks, model answers, **step-wise criteria** with weights and keywords | The draft is shown as a suggestion. Weights are normalised in code to the exact max marks. Nothing is saved until the teacher clicks Save. Works without AI: the teacher types the rubric (E-2). | Planned (Stage 2) |
| E-5 | Grading style Strict / Moderate / Lenient per exam (PRD FR-24) | Changes the thresholds τ and τ_low and the AI instructions. "Preview on N sheets" shows marks before the full batch. | Planned (Stage 2) |
| E-6 | Question types: short, long, paragraph, maths, **diagram (Beta)** | Maths: final-answer check with SymPy plus step marks, with unparseable answers routed to review (FR-11). Diagram: AI vision criteria only, always reviewed. | Partial: maths is always manual today; diagram is planned (Stage 4) |
| E-7 | Optional questions ("attempt any N of M") | The best N attempted answers count (PRD 12.10) | Planned (Stage 4) |

### 3.3 Answer-sheet intake and OCR

| ID | Requirement | Acceptance criteria | Status |
|---|---|---|---|
| S-1 | Upload one or many PDFs; Idempotency-Key; 20 MB / 50 page limits; signature, corruption and password checks (FR-5) | A bad file is rejected within 2 s with a clear message. With 1 corrupted file in a batch of 10, the other 9 finish. | **Done** |
| S-2 | Upload photos (JPG, PNG, WEBP) and ZIP archives; group images per student by folder or file name | One PDF is built per student, in page order. ZIP-bomb, path-traversal and type checks. | Planned (Stage 3) |
| S-3 | Preprocessing: deskew, denoise, contrast, ruled-line removal; blank and poor-page flags (FR-6) | Original images are kept for review | **Done** |
| S-4 | OCR chain: PDF text layer → AI vision (Gemini, with consent; the header strip is cropped first) → local TrOCR (if installed) → manual transcription (FR-8) | Per-line text and confidence are stored. AI vision lines carry approximate boxes. On any provider failure the chain moves to the next step without crashing. | Partial: text layer, TrOCR and manual are done; AI vision is planned (Stage 3) |
| S-5 | Segmentation by question markers, with flags for skipped, duplicate, out-of-order and cross-page answers (FR-7, FR-32) | At least 90% correct on the segmentation test set (measured by the evaluation framework) | **Done** (the accuracy figure needs real sheets) |
| S-6 | Region editor: crop, reassign, split, merge, mark blank, undo (FR-31) | Every change is audited and answers are regraded | **Done** (final verification in progress) |

### 3.4 AI evaluation and marks

| ID | Requirement | Acceptance criteria | Status |
|---|---|---|---|
| G-1 | Offline checker: key-point matching with partial credit, contradiction, number checks and gaming guards (FR-9, FR-10, FR-33) | The adversarial set (empty, copied question, keyword list, off-topic) scores at most 20% of max | **Done** |
| G-2 | AI marker: one request per sheet. Per criterion: verdict, evidence quote, short reason. Per answer: Correct / Partially correct / Incorrect and feedback. | Strict JSON schema output. Quotes are verified against the answer text. Marks are computed in code and capped at max. Temperature 0, cached by content hash, prompts stored. | Planned (Stage 4) |
| G-3 | Cross-check and routing: the AI and the offline checker both grade | Auto-accept only when they agree (difference ≤ 0.5 marks and no criterion conflict) AND the calibrated confidence ≥ threshold AND nothing is flagged. Otherwise the answer goes to review with the reason "AI and checker disagree". | Partial: confidence routing done; the cross-check is planned (Stage 4) |
| G-4 | Fallbacks: provider order, retries with backoff on 429/503, 30 s timeout, per-provider rate limiter and daily budget, then the offline checker | A pipeline with no network finishes every sheet. The answer shows "graded with fallback". | Partial: the offline fallback exists; providers are planned (Stage 1/4) |
| G-5 | Confidence and flags (FR-13); calibration via temperature scaling | ECE ≤ 0.10 on the test set (measured, never assumed) | Partial: combination and routing done; calibration fitting is planned (Stage 7) |
| G-6 | Totals (FR-12): sum of question marks, capped per question; best-N for optional questions | AI marks and teacher final marks are shown separately | **Done** (best-N planned) |
| G-7 | Copy detection (FR-22): near-identical answers across students listed as hints | Never penalises automatically | Planned (Stage 6) |

### 3.5 Teacher review, Ask AI and approval

| ID | Requirement | Acceptance criteria | Status |
|---|---|---|---|
| R-1 | Review screen: scan, OCR text, criteria with evidence, marks, confidence and flags side by side (FR-14) | Flagged answers come first. Keyboard shortcuts. Unsaved-change warning. | **Done** (final verification in progress) |
| R-2 | Edit OCR text (regrades), override marks with a reason, comment (FR-15) | Every change is audited, and corrections are stored for learning (FR-19, FR-20) | **Done** |
| R-3 | **Per-criterion editing**: the teacher changes one criterion's mark and the total is recalculated | Reason required; audited | Planned (Stage 5) |
| R-4 | AI explanation panel: the AI verdict, feedback and the offline checker's verdict, with the disagreement highlighted | Shown only when AI grading ran | Planned (Stage 5) |
| R-5 | **Ask AI** (teachers): questions at three scopes, about an answer ("Why 3.5/4?"), a sheet, or the exam/class ("Which question was hardest?", "Which students need help on Q2?") | Answers are grounded in stored data only and cite criteria and evidence. If unsure, Ask AI says so. It can *suggest* a change, which the teacher must apply explicitly (with a reason, audited). Rate-limited, and threads are stored per exam. Without AI configured, it shows an honest "AI is not set up" state with a link to Settings. | Planned (Stage 5) |
| R-6 | Re-check requests (FR-28 subset): the teacher logs a marks complaint, Ask AI drafts a response, and the teacher re-evaluates | Re-opens the sheet with a reason; the history is kept | Planned (Stage 5) |
| R-7 | Approval: per sheet, plus **Approve all ready sheets**; blocked while flagged answers remain; reopen needs a reason (FR-16) | One click for all confident sheets. The optional full auto-approve is off by default. | Partial: per-sheet and approve-all done; the "Ready" state and auto-approve option are planned (Stage 5) |

### 3.6 Reports, exports and analytics

| ID | Requirement | Acceptance criteria | Status |
|---|---|---|---|
| X-1 | CSV export of approved marks (FR-17) | Matches the approved marks exactly (automated test) | **Done** |
| X-2 | Student-wise PDF report: question-wise and criterion-wise marks, evidence, feedback, teacher comments (FR-29) | Only approved sheets; a ZIP of all reports | Planned (Stage 6) |
| X-3 | Class report PDF: distribution, per-question averages, hardest questions, weak criteria | Generated from approved data | Planned (Stage 6) |
| X-4 | Analytics dashboard (FR-25): distribution, averages, hardest questions, most-missed criteria, auto-accept share, override rate, AI-vs-teacher agreement | Empty state until data exists | Partial: core done; AI agreement and criteria view planned (Stage 6) |

### 3.7 Evaluation framework (no unmeasured claims)

| ID | Requirement | Acceptance criteria | Status |
|---|---|---|---|
| Q-1 | Dataset format: sheets, transcriptions, teacher marks (two teachers optional), split by writer into train/val/test, with the test set locked | Documented. A validator rejects writer leakage across splits. | Planned (Stage 7) |
| Q-2 | One command produces: OCR CER/WER; segmentation accuracy; MAE, RMSE, % within ±1, QWK/Cohen's κ and Pearson vs teacher marks; ECE, Brier, reliability bins; deferral rate and risk–coverage; with bootstrap 95% CIs; for each grader (offline, AI, combined) | Writes Markdown and JSON reports. Thresholds are tuned on validation data only. | Planned (Stage 7) |
| Q-3 | "Model quality" page shows the latest real report, or an honest "no evaluation data yet" state | Never shows invented numbers | Planned (Stage 7) |

### 3.8 Security, privacy and reliability

| ID | Requirement | Status |
|---|---|---|
| P-1 | JWT access and refresh tokens, Argon2, lockout, rate limits (100 req/min, 10 uploads/min) | **Done** |
| P-2 | Role and institution authorization on every route, with isolation tests | Partial: ownership checks done; institution isolation planned (Stage 1) |
| P-3 | File validation, UUID storage names, streamed uploads, path-traversal protection on page images | **Done** |
| P-4 | Append-only audit log for every mark, text, region, approval, AI and settings change | **Done** (AI events planned, Stage 4) |
| P-5 | AI privacy: one-time consent screen; separate Gemini consent; names, roll numbers, emails and phone numbers redacted; header strip cropped; keys only in `backend/.env` | Planned (Stage 1) |
| P-6 | Security headers and CSP (the theme script moves to a file), HTTPS via Caddy outside localhost, production settings checks | Planned (Stage 8) |
| P-7 | Optional encryption at rest for stored sheets; retention and purge job with a removal report; deletion on request | Partial: deletion API done; encryption and retention planned (Stage 8) |
| P-8 | Backups and restore (SQLite online backup or pg_dump plus storage), with a restore drill command | Planned (Stage 8) |
| P-9 | Logs never contain answer text or student names; 30-day rotation | Planned (Stage 8) |

### 3.9 Operations and deployment

| ID | Requirement | Status |
|---|---|---|
| O-1 | One-command local run (`npm run dev`, `npm start`), Windows-friendly | **Done** |
| O-2 | Alembic migrations (SQLite and PostgreSQL), auto-migrate on start | Planned (Stage 1) |
| O-3 | Docker images plus Compose (app, worker, PostgreSQL, Caddy); CI with lint, types, tests on SQLite and PostgreSQL, image build and an API schema snapshot | Partial: Dockerfile, Compose and CI written but never run; the rest is planned (Stage 8) |
| O-4 | Deployment guides: laptop + Cloudflare Quick Tunnel / ngrok (live demo), GitHub Codespaces (cloud backup), Azure for Students VM (always-on). No Colab/Kaggle hosting: Colab's FAQ forbids it on free runtimes. | Planned (Stage 8) |
| O-5 | Docs: API reference, ER diagram, feature checklist, known limitations, user manual with screenshots of both themes | Partial (Stage 8) |

---

## 4. AI design (applies to Stages 1–5)

- **Provider adapter** (`backend/app/llm/`): one interface, `generate_json(task, prompt, schema, images=None)`. Adapters:
  - Gemini native REST (`generateContent`, `x-goog-api-key`, `responseMimeType` + `responseJsonSchema`)
  - OpenAI-compatible (Groq, OpenRouter, Cloudflare, Ollama, xAI Grok)

  Only `httpx` is used, with no vendor SDKs. Model IDs come from settings, not code. "Test connection" lists the models actually available to the key.
- **Defaults**, verified live on 23 Nov 2026:

  | Task | Provider order |
  |---|---|
  | Grading | Groq `openai/gpt-oss-120b` (`reasoning_effort` low, strict JSON schema) → Gemini `gemini-3.5-flash-lite` → `gemini-3.1-flash-lite` → offline checker |
  | Vision (handwriting, question paper, diagrams) | Gemini `gemini-3.5-flash-lite` → `gemini-3.1-flash-lite` → manual |
  | Rubric drafting | Gemini `gemini-3.8-flash` (low daily quota) → Groq |

- **Free-tier limits**: a per-provider token bucket below the published or observed limits (Groq about 8K tokens/min and 200K/day; Gemini Flash-Lite about 10 requests/min and 400/day). One request per sheet. A daily budget per institution. Clear "quota reached — using offline checker" status.
- **Reproducibility**: temperature 0 where supported (newer Gemini models ignore it), so every response is cached by hash(provider, model, prompt version, input). The prompt version is stored with each result.
- **Grounding rules**: the AI sees the question, the rubric and the anonymised answer text (or a cropped image) only. It must quote the evidence. SARAS checks the quotes and computes the marks. Ask AI receives only stored facts for the scope asked about.
- **Tests never call real providers**: a fake provider replays recorded JSON. Live checks run only from the "Test connection" button or an explicit command.

---

## 5. Data model additions (summary; full ER diagram in docs/DATABASE.md, Stage 8)

| Table | Purpose |
|---|---|
| `institutions`, `invites` | Tenancy and onboarding; `users.institution_id`, `users.is_active` |
| `exam_materials` | Question paper, answer key, marking scheme, teaching material (file plus extracted text) |
| `ai_settings` (per institution) | Consent flags, provider order, daily budgets, auto-approve option |
| `ai_runs` | One row per AI call: task, provider, model, prompt version, tokens, latency, status, cache key, request/response JSON |
| `answer_ai` / columns on `answers` | AI verdict, criterion verdicts, feedback, offline-vs-AI agreement |
| `criterion_overrides` | Per-criterion teacher marks with reason |
| `ask_threads`, `ask_messages` | Ask AI conversations with scope (exam, sheet, answer) and grounding references |
| `recheck_requests` | Marks complaints and their resolution |
| `similarity_flags` | Copy-detection hints (PRD A.5) |
| `eval_datasets`, `eval_runs` | Evaluation framework inputs and reports |

---

## 6. Implementation roadmap

Each stage is built by a multi-agent workflow: build → automated tests → independent review → fix → verification. Two stages run per workflow, with a check-in after each pair. A stage is complete only when its acceptance criteria pass in automated tests and in a live run.

| Stage | Scope (maps to the brief's Phases 1–8) | Exit gate |
|---|---|---|
| 0 | Audit, competitor research, PRD baseline build | **In progress**: core build, 131 tests passing, final review/fix/end-to-end running |
| 1 | Alembic migrations; institutions and user management (U-2…U-5); AI settings and consent; provider adapter with fake-provider tests; standalone worker | Isolation tests pass; migrations upgrade a fresh DB and the existing dev DB; "Test connection" works with the real keys |
| 2 | Exam materials upload; AI drafting of questions, model answers and step-wise criteria; grading style plus preview | Uploading a question paper produces a valid editable rubric; works with AI off |
| 3 | Photo and ZIP intake; AI vision OCR with header crop and consent; OCR chain fallbacks | An image-only handwritten sheet is transcribed and segmented without manual typing when AI is on |
| 4 | AI marker (batched per sheet), cross-check routing, diagram Beta, maths with SymPy, best-N, AI events in the audit log | The 10-sheet sample batch is graded end to end; the adversarial set still scores ≤ 20%; no network still finishes |
| 5 | Review upgrades (per-criterion edits, AI panel), Ask AI, re-check requests, "Ready" state and Approve all ready | A teacher can ask "why 3.5/4?" and get a grounded answer; one click approves all ready sheets |
| 6 | Student and class PDF reports, ZIP download, copy-detection hints, analytics extensions | Reports match the approved marks exactly (automated test) |
| 7 | Evaluation framework and Model-quality page | One command regenerates every metric table from the dataset folder |
| 8 | Security headers/CSP, HTTPS (Caddy), encryption at rest, retention/purge, backups and restore drill, log hygiene, CI on SQLite + PostgreSQL, Docker, deployment guides, docs (API, ER diagram, checklist, limitations, user manual), screenshot QA of both themes | All 13 final acceptance steps pass on a live server with the real keys |

## 7. Final acceptance (all must pass before the project is called complete)

1. A teacher can register and log in.
2. They can create an examination.
3. They can upload a question paper.
4. They can configure model answers and marking rubrics (AI-drafted or typed).
5. They can upload handwritten answer sheets.
6. Handwritten answers are extracted with OCR (AI vision or local; manual only as the last resort).
7. Question-wise AI evaluation is generated, with step-wise criteria.
8. Marks and evaluation explanations can be viewed.
9. OCR and AI-generated results can be corrected.
10. Low-confidence answers can be reviewed.
11. Final marks can be approved (one click for ready sheets).
12. Student performance can be viewed (analytics and reports).
13. Approved results can be exported (CSV and PDF reports).

Each step is checked by an automated end-to-end test (API level) and by a recorded live run. The results go into `docs/FEATURE_CHECKLIST.md` with honest statuses.

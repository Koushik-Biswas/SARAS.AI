# SARAS AI — Viva demo script

The demo takes about 12 minutes and runs on the 4 GB laptop in the low-resource profile:
one Python process, SQLite and no ML models. Every step below uses only what is in this
repository.

## Before the viva (the evening before)

1. Close every other program and browser tab. The laptop has about 300 MB of free RAM.
2. Prepare the data:
   ```powershell
   cd backend
   .venv\Scripts\python -m scripts.make_samples     # writes backend\samples\ (rubric, 12 sheets, 4 invalid files)
   .venv\Scripts\python -m scripts.seed_demo        # demo teacher + exam + all 12 sample sheets, processed
   cd ..
   ```
3. Copy two sample sheets under **new** names. Student references come from file names,
   so these copies will be accepted as new students during the live upload:
   ```powershell
   copy backend\samples\sheets\ML009_flipped_fact.pdf $env:USERPROFILE\Desktop\LIVE01_flipped_fact.pdf
   copy backend\samples\sheets\ML012_scanned_image_only.pdf $env:USERPROFILE\Desktop\LIVE02_scanned.pdf
   ```
4. Build and start the app, then leave it running:
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\run_prod.ps1
   ```
   Open http://127.0.0.1:8000, log in once, and check that the dashboard loads.
5. Safety net (PRD 18.5, demo-day safety): record a screen video of this whole script, and
   keep the seeded database (`backend\data\`) as a backup copy.

## The demo

| # | Show | Say (key point) | FR |
|---|---|---|---|
| 1 | Log in as **demo@saras.local / demo-password-123**. Switch the theme between light and dark (top bar). | Argon2-hashed passwords, short-lived JWT plus rotating refresh token; five wrong passwords lock the account for five minutes. | FR-1 |
| 2 | Dashboard → the *Machine Learning Unit Test* exam (subject, date, total marks). Briefly open **New exam** to show the form. | Each teacher sees only their own exams; an admin can read but never change marks. | FR-2 |
| 3 | **Rubric** tab: four questions with type, max marks, model answer and weighted key points. | Grading is explainable: marks come from key points, not a black box. Maths questions are always routed to the teacher. | FR-3, FR-4 |
| 4 | **Upload** tab: drop `LIVE01_flipped_fact.pdf`, `LIVE02_scanned.pdf` and the four files in `backend\samples\invalid\`. Then drop `backend\samples\sheets\ML001_full_marks.pdf` again. | Valid files are accepted (202). Each invalid file is rejected with its own reason: not a PDF (415), corrupted or password-protected (422), more than 50 pages (413). A sheet already uploaded for the same student is a conflict. One bad file never blocks the batch, and the Idempotency-Key stops double submissions. | FR-5, FR-18 |
| 5 | **Status** tab: watch the two new sheets go *uploaded → processing → processed* (the page polls every few seconds). | A database job queue with retries and crash recovery; each page is processed in isolation. On this laptop: one sheet and one page at a time at 150 DPI. | FR-6, FR-18 |
| 6 | **Review** tab → the queue: sheets ordered by how many answers need review. Open **ML006_duplicate_marker** or **ML007_out_of_order**. | Segmentation by question number: duplicate, out-of-order, skipped and continued answers are flagged, never silently guessed. | FR-7, FR-32 |
| 7 | Open **LIVE01** (flipped fact). Show Q2's key point marked *contradicted*, with the student's sentence as evidence. | The grader checks meaning, not just keywords. Saying an overfitted model "performs well on new data" contradicts the rubric and earns nothing for that point. | FR-9, FR-10, FR-33 |
| 8 | Same sheet: point out the confidence of each answer and the needs-review / auto-accepted split. | Confidence = OCR × grading × segmentation, calibrated. Below 0.80, or with any review flag, the answer goes to the teacher first (docs/ARCHITECTURE.md §6). | FR-13 |
| 9 | Open **LIVE02** (scanned image, no text layer): the answer is flagged *needs transcription*. Type the answer text and save, and it is graded immediately. | Graceful degradation: without the handwriting model (not installed on a 4 GB laptop) the teacher transcribes and the AI still grades. With the ML extras, TrOCR reads it. | FR-8, FR-15 |
| 10 | Override one mark: change it, leave the reason empty → error; add a reason → saved. Add a comment. | A reason is required whenever a mark changes. Every change goes to the append-only audit log (who, when, old, new, why). | FR-15, FR-19 |
| 11 | Edit answer regions on a sheet (reassign or split a region, then undo). | Teacher can fix segmentation; affected answers are re-graded at once. | FR-31 |
| 12 | **Results** tab: per-question marks grid, AI vs final totals. Click **Approve all**: sheets with unreviewed flagged answers are listed as blocked. | Approval is blocked until every flagged answer has been reviewed. Approved sheets are locked, and reopening one needs a reason. | FR-12, FR-16 |
| 13 | **Export CSV** and open it. | Only approved sheets are exported, and the marks match the approved marks exactly. | FR-17 |
| 14 | **Analytics** tab: distribution, per-question averages, hardest question, most-missed key points, override rate. | Feedback for the teacher and evidence of how much the AI was corrected. | — |
| 15 | Open http://127.0.0.1:8000/api/docs. | Every feature is also a documented REST API under /api/v1 with one error format. | — |

If a UI screen is not available (for example, the region editor in step 11 while it is
still under construction), show the same action in `/api/docs`. Authorise with the access
token from `POST /api/v1/auth/login`, then call `PATCH /api/v1/regions/{id}` and
`POST /api/v1/submissions/{id}/regions/undo`.

## Likely questions

- **"Is it accurate on real handwriting?"** The default install does not read handwriting:
  the teacher transcribes and the system grades. TrOCR-small runs only with the optional
  ML extras on 8 GB or more. Fine-tuning it and measuring CER, MAE and QWK is research
  work (R1–R6) that is not in this repository. See *Known limitations* in the README.
- **"What stops the AI giving wrong marks?"** Several things:
  - Confidence routing sends doubtful answers to review.
  - Approval is blocked while flagged answers are unreviewed.
  - The teacher can override any mark with a reason.
  - Every change is audited.
  - Only approved marks are exported.
- **"Why not Celery or Streamlit as in the PRD?"** On a 4 GB laptop, a database-backed job
  table inside the API process avoids Redis and Celery. The React app is served by the same
  process, so production is a single process.
- **"How does it scale?"** Use `docker compose up` (PostgreSQL 16, PRD 18.5 resource
  limits). Running `--profile scale` adds worker replicas that claim jobs from the same
  table.

## After the demo

Press Ctrl+C in the server window and check that nothing is still listening on port 8000:
```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
```

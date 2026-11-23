# SARAS AI — Grading design (ensemble, AI marker, routing)

Status: implemented and tested. All priors and thresholds below are **provisional**: they have not been
calibrated against real teacher marks (spec G-5). No accuracy figure in this document is a claim about real
exams. The DEV benchmark is 42 synthetic answers and is not teacher-validated.

## 1. Pipeline

```
OCR text per answer (pipeline/processor.py)
  ├─ no letter or digit            → status "blank", 0 marks, never graded
  ├─ unreadable lines only         → needs_review (needs_transcription), never graded
  └─ gradable text
       ├─ local primary grader     (ensemble.grade_local_primary, via processor._grade_with_fallback)
       ├─ local cross-check grader (ensemble.grade_local_secondary)
       ├─ AI marker, ONE request per sheet (ensemble.ai_grade_items → llm_grader.grade_sheet_llm),
       │  only when SARAS_AI_GRADING=ensemble and SARAS_AI_CONSENT=true
       ├─ ensemble.combine()       → per-criterion statuses, marks, verdict, flags, confidence, feedback, trace
       └─ ensemble.route_graded()  → confidence × OCR × segmentation (routing.combine), routing.route
```

The pipeline and the evaluation harness call the same helpers in `app/grading/ensemble.py`, so they cannot
drift apart. Only the engine's public API is used (grade_answer, GradeResult/KeyPointResult,
default_thresholds, get_scorer, routing.combine/route/is_review_flag).

## 2. Graders and roles

| Grader | What it is | Prior (provisional) |
|---|---|---|
| `llm` (A) | AI marker: verdict + quote per criterion; marks computed in code | 0.9 |
| `nli`, `nli-onnx` (N) | NLI cross-encoder via engine.grade_answer (unions lexical evidence) | 0.8 |
| `sbert` | Sentence-BERT similarity | 0.7 |
| `lexical` (L) | IDF-weighted term overlap with polarity checks | 0.6 |

* Primary P = A when it judged the question; else the local primary; else L.
* `settings.local_primary` (`SARAS_LOCAL_PRIMARY`): `auto` (default) = the semantic grader chosen by
  `SARAS_GRADER` (`auto`: torch NLI → nli-onnx → SBERT → lexical), `lexical`, or `nli-onnx`. The default was
  chosen from the engine agent's DEV numbers: nli-onnx MAE 0.238 / QWK 0.957 / KP κ 0.771 vs lexical
  0.381 / 0.918 / 0.685 (N = 42, synthetic).
* Secondary S = the local primary when A is primary; the other local grader when P is local. Other
  results (L when P = A and S = N) are extras: shown, and their contradictions count.
* Scorers are cached per process by `get_scorer`; the ONNX model loads once.
* Key-point thresholds come from `engine.default_thresholds(scorer)` (lexical 0.6/0.4/0.5, NLI
  0.5/0.15/0.95). A Settings value an operator changes (SARAS_TAU, SARAS_TAU_LOW,
  SARAS_CONTRADICTION_THRESHOLD) overrides that field for every scorer (`ensemble.thresholds_for`).
  Before this change the pipeline passed 0.6/0.4/0.5 to every scorer, which was wrong for NLI.

## 3. Ensemble rules (`ensemble.combine`, pure and deterministic)

Per criterion (key point):
* Final status = the decider's status (P; S where the AI did not judge that criterion).
* Every non-missed status needs a quote that is verified against the answer text (llm_grader.find_quote).
  The decider's quote, or the cross-checker's when the decider has none and both award marks. Otherwise the
  criterion becomes `missed` with reason `no_verified_evidence` and the answer gets that flag.
* Dispute: decider and cross-checker disagree on whether the criterion earns marks (award vs nothing, or
  award vs contradiction). The decider's status is kept, the criterion is marked disputed, and the answer
  gets `graders_disagree`. matched vs partial is a minor disagreement, recorded in the trace only.
* A contradiction found by any grader with a verified quote flags `contradiction`. If the decider did not
  contradict, that criterion is also disputed.
* Criteria the AI did not judge, and questions it left out of its reply, fall back to the offline graders.
  They are never given a made-up 0. The answer gets `ai_incomplete` (review).

Per answer:
* Marks: matched = full weight, partial = half, missed = 0, contradicted = 0. Summed, capped at max marks,
  rounded to 2 dp. There is no other marks formula.
* Verdict: `correct` at max, `incorrect` at 0, `partially_correct` otherwise, `ungraded` for maths or a
  question with no rubric (always reviewed).
* Spec G-3: if P's and S's marks differ by more than 0.5, the answer gets `graders_disagree`.
* Grading confidence = prior(decider) × agreement factor. The factor is 1.0 when every criterion was
  cross-checked and agrees, 0.85 when something was not cross-checked, and 0.45 when the graders disagree.
  With mixed deciders the lower prior counts.
* Feedback: the AI's feedback is used only when the AI decided every criterion and nothing changed its
  verdicts. That rules out a withdrawn award, a disputed criterion, a marks gap, and the AI-side flags
  `ai_unverified_evidence` / `ai_incomplete`. Otherwise feedback is a deterministic summary
  ("Covered: …; Partly: …; Missing: …; Contradicts: …"), so it never contradicts the marks.

## 4. Flags and routing

`route_graded` computes confidence = routing.combine(OCR, grading confidence, segmentation, T) and adds
`short_answer` (< 3 characters) and `needs_transcription` (partly unread). Then routing.route runs.
`graders_disagree`, `contradiction`, `no_verified_evidence`, `ai_uncertain`, `ai_unverified_evidence`,
`ai_incomplete` and `ai_contradiction` are never auto-accepted. routing.REVIEW_FLAGS lists them, and
route_graded also enforces it. `ai_unavailable_fallback` (AI enabled but failed) and
`low_reliability_grader` are informational. Engine flags (possible_paraphrase, off_topic, role_swap,
copied_question, keyword_list, numeric_mismatch, …) keep their routing.

**Consequence of the provisional priors (known, deliberate for now).** Clean typed answers read with OCR
confidence 0.99 and segmentation 0.95. Results by grader set:
* lexical only: 0.6 × 0.85 = 0.51. Never auto-accepted at the default 0.80.
* nli-onnx + lexical agreeing: 0.8 × 1.0 = 0.8, then × 0.99 = 0.792. Also below 0.80, so without AI every
  uploaded answer is reviewed.
* AI + offline checker agreeing: 0.9 × 0.99 = 0.891. Can be auto-accepted.

Only measured calibration should change this. Tests that exercise approval flows lower the threshold
explicitly (`conftest.accept_lexical`). `tests/test_pipeline_ai.py` asserts that lexical-only answers go
to review at the default.

## 5. Persistence and API (additive)

* `answers.verdict`, `answers.feedback` and `answers.grading_trace` (JSON) are nullable columns.
  `db.ensure_additive_columns()` adds them on start-up to databases created earlier. It is idempotent and
  works on SQLite and PostgreSQL.
* `AnswerOut` gains `verdict`, `feedback` and `grading_trace`. `KeyPointOut` gains `reason`, `source`,
  `disputed` and `statuses_by_grader`. Existing fields are unchanged.
* `Answer.grader` = the graders used, primary first (e.g. `llm+nli-onnx+lexical`). `Submission.grader` =
  the sorted set of grader names.

## 6. Logging and privacy

* One INFO line per sheet: sheet id, graders, AI provider/model or the fallback reason, counts per answer
  and criterion status, disputes, and duration in ms. One DEBUG line per answer: ids, statuses, flags,
  confidence, routing reason.
* Logs never contain answer text, student_ref, names or keys. The trace holds codes and scores only;
  the AI's free-text reasons stay in the per-criterion evidence.
* Sent to the AI: question prompt, model answer, criteria and weights, and the answer text after
  `llm/redact.py`. Redaction removes e-mails, URLs, phone numbers, ID-shaped tokens, digits after ID cue
  words, and the student_ref. Numeric answer content is never removed for its length alone.
  Student names are not stored, so they cannot be redacted by name.
* Consent: nothing is sent unless SARAS_AI_CONSENT=true. Gemini also needs SARAS_GEMINI_CONSENT=true:
  its free tier may use data for training and human review. Keys live only in `backend/.env`.

## 7. Caching and failure behaviour

* AI replies are cached in `ai_runs` by content hash. Regrading an unchanged sheet makes no new request.
* The worker commits before the AI request, so the router can write `ai_runs` in its own sessions.
  Request handlers (text correction, region edit, exam regrade) write through the request's own
  connection: no SQLite lock wait, and the rows commit together with the change and its audit entry.
* Any AI failure leaves the offline result in place and never fails a sheet. This covers no consent or
  keys, a 5xx, a timeout, invalid output, and an unexpected exception. The `ai_unavailable_fallback`
  flag and trace `ai.reason` record the failure.

## 8. How to enable AI marking

```
SARAS_AI_GRADING=ensemble
SARAS_AI_CONSENT=true
SARAS_GROQ_API_KEY=...            # in backend/.env only
SARAS_GEMINI_CONSENT=true         # optional; Gemini free tier may train on data
SARAS_AI_GRADING_STYLE=moderate   # lenient | moderate | strict
```

## 9. Limitations

* The priors, factors and the 0.5 marks-gap rule are hand-set. They need calibration on teacher-marked data.
* NLI-ONNX takes 2.5–4.4 s per answer on the 4 GB laptop. AI marking adds one request per sheet.
* The DEV benchmark is small and synthetic. The held-out test set must be run once, with the settings frozen.
* Redaction cannot remove names, or phone numbers written as two bare digit groups.

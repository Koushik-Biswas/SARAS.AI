# SARAS AI grading benchmark

This folder holds the labelled answer sets used to measure the answer-sheet grader.

| File | Answers | Use |
|---|---|---|
| `grading_benchmark_dev.json` | 42 | **Development split.** Use it freely while building or tuning the grader. |
| `grading_benchmark_test.json` | 111 | **Held-out test split.** Only the final evaluation step may open it. Do not read it, print it, tune thresholds on it or copy examples from it during development. |

Both files contain the same 10 questions. Answer ids (`d###` and `t###`) are unique, and no answer text appears twice, in either file or across both.

## Provenance (read this before quoting any number)

This is a **synthetic benchmark authored by an AI agent acting as a teacher**. The agent labelled each answer once and then did one self-review pass. **The labels are NOT teacher-validated.** Use the benchmark for regression testing and for comparing grader versions against each other. It does **not** show how accurate the grader is on real student scripts. The answers imitate the English of Indian college students, but no real student wrote them.

The author did not look at the grading engine (`backend/app/grading`, `backend/app/pipeline`) while writing or labelling. Each label records what a careful teacher would award, not what the software is expected to output.

## Format

```json
{
  "version": 1,
  "provenance": "free text",
  "questions": [
    {
      "id": "q03_phy_newton_second_law",
      "domain": "physics",
      "type": "short | long | paragraph",
      "max_marks": 3,
      "prompt": "question text shown to the student",
      "model_answer": "teacher's reference answer",
      "key_points": [
        {"text": "what must be said", "weight": 1.5, "keywords": ["10"]}
      ]
    }
  ],
  "answers": [
    {
      "id": "t099",
      "question_id": "q03_phy_newton_second_law",
      "category": "numeric_mismatch",
      "text": "student answer (plain text, possibly OCR output)",
      "expected_kp": ["matched", "matched", "contradicted"],
      "expected_marks": 1.5,
      "expected_flag": true,
      "ambiguous": false,
      "rationale": "one sentence explaining the label"
    }
  ]
}
```

- The key-point `weight`s of a question sum exactly to its `max_marks`.
- `keywords` (optional, may be empty) lists terms a teacher genuinely expects for that key point. A key point has keywords only where no natural paraphrase avoids them, as with `momentum`, `electron`, `valve`, `diploid`/`haploid`, `labelled`/`unlabelled`, `LIFO`/`FIFO`, `proportional`, `temperature` and the numeric result `10`. Treat the list as **any-of**, and accept inflections and obvious variants (`trained` for `training`, `valves` for `valve`). Every non-OCR answer with a key point labelled `matched` contains at least one of that key point's keywords. Containing keywords never earns a mark by itself (see `keyword_stuffing`).
- `expected_kp` has one status per key point, in key-point order.

### Questions

| id | domain | type | marks | key points (weights) | notes |
|---|---|---|---|---|---|
| q01_ml_overfitting | machine learning | short | 2 | 0.5 / 1 / 0.5 | |
| q02_bio_arteries_veins | biology | short | 3 | 1 / 1 / 1 | role-symmetric (arteries vs veins) |
| q03_phy_newton_second_law | physics | short | 3 | 1 / 0.5 / 1.5 | numeric + unit (10 N) |
| q04_chem_redox | chemistry | short | 2 | 1 / 1 | role-symmetric (oxidation vs reduction) |
| q05_cs_stack_queue | computer science | long | 5 | 1.5 / 1.5 / 1 / 1 | role-symmetric (LIFO vs FIFO) |
| q06_eco_demand_supply | economics | long | 4 | 1.5 / 1.5 / 1 | role-symmetric (demand vs supply) |
| q07_bio_mitosis_meiosis | biology | long | 5 | 1 / 1.5 / 1.5 / 1 | role-symmetric, numeric (2 vs 4 cells) |
| q08_phy_ohms_law | physics | long | 4 | 1.5 / 0.5 / 1 / 1 | numeric + unit (3 A, 72 W) |
| q09_ml_supervised_unsupervised | machine learning | paragraph | 6 | 1.5 / 1.5 / 1 / 1 / 1 | role-symmetric (labelled vs unlabelled) |
| q10_env_greenhouse_effect | environmental science | paragraph | 8 | 1.5 / 2 / 1 / 1.5 / 2 | numeric (about 15 °C vs -18 °C) |

## Labelling rules

**Mark rule (PRD).** Each key point is awarded according to its status:

| status | award | meaning |
|---|---|---|
| `matched` | full weight | the idea is stated correctly; wording does not matter |
| `partial` | half weight | the idea is stated incompletely, vaguely, for only one side of a two-sided point, or with a unit error |
| `missed` | 0 | the idea is absent |
| `contradicted` | 0 | the answer states the opposite or a conflicting fact, such as a swapped role, a negation or a wrong number |

`expected_marks` = sum of awards, capped at `max_marks`. The validator re-derives this for every answer.

**Flag rule.** `expected_flag` is `true` when a teacher should look at the script. That is the case when any key point is `contradicted` (including role swaps and wrong numbers), the answer is off-topic, a copied question or a wrong concept, OCR noise makes the meaning unclear, or the label is genuinely ambiguous. Every `ambiguous: true` answer is also flagged. Clearly-zero answers (blank, "don't know", keyword lists, plain incorrect answers) are not flagged: the teacher would give them 0 without needing a second look.

**Extra wrong statements.** An extra claim that is false but contradicts no key point is **not penalised** (for example "the aorta is a vein" in an otherwise correct arteries/veins answer). It is penalised only when it conflicts with a key point.

**Numbers and units.** A wrong value for a numeric key point is labelled `contradicted`. The right value with a missing or wrong unit (`10`, `10 J`) is labelled `partial`. A formula with no value (`P = VI`) is also `partial`. A wrong unit prefix (`72 kW`) is labelled `partial` and marked ambiguous.

**Question-specific conventions**

- q02: a difference stated for only one vessel ("Arteries have thick walls", "Veins have valves" with neither purpose nor contrast) is `partial`. Pure/impure (oxygenated/deoxygenated) blood does not answer the direction of flow (labelled `missed`, ambiguous).
- q03: giving the law only as "F = ma" / "force = mass × acceleration" earns `partial` for the statement key point and `matched` for the F = ma key point.
- q04: "losing"/"gaining" without electrons is `partial`.
- q05: a bare "Stack is LIFO" with no explanation is `partial` (5-mark question). "Added and removed at the top" with no LIFO principle is `partial` for key point 1. One application only is `partial`.
- q07: key point 1 needs both division count and cell count. Key points 2 and 3 each need both chromosome number (diploid/haploid) and genetic identity/difference. Key point 4 needs both processes. Half of any of these is `partial`.
- q08: "V = IR" alone, without the proportionality statement, is `partial`.
- q09: problem types for only one learning type is `partial`. An algorithm without an application, or the reverse, is `partial`.
- q10: key point 2 needs absorption/re-radiation. Gases that merely "trap heat" are `partial`. Key point 3 is `matched` by a qualitative "keeps Earth warm enough for life / otherwise frozen" and is `partial` if warmth is not linked to life. Stated wrong temperatures, or a claim that Earth would be hotter without it, are `contradicted`. Key point 5 needs at least two distinct consequences; one consequence is `partial`. "Causes global warming" is the enhancement itself, not a consequence.

## Categories

| category | definition | typical labels |
|---|---|---|
| `correct_verbatim` | correct and complete, close to the model-answer wording | all matched |
| `correct_paraphrase` | correct and complete, in different words (synonyms, restructured sentences, analogies) | all matched |
| `partial` | on-topic and correct as far as it goes, but key points are vague or half-done | mix of matched / partial / missed |
| `incorrect` | on-topic attempt that is simply wrong or misconceived, without directly negating a key point | mostly missed, flag false |
| `contradiction_flipped_fact` | mostly correct, but one fact is reversed (direction, size, relationship) | one or more contradicted, flag true |
| `contradiction_negation` | mostly correct, but explicitly negates a key point ("veins do not have valves") | contradicted, flag true |
| `role_swap` | swaps the two roles of a role-symmetric question (arteries/veins, LIFO/FIFO, ...) | swapped points contradicted, flag true |
| `wrong_concept` | explains a different, real concept (first law instead of second, osmosis for mitosis, underfitting for overfitting) | missed (contradicted where it opposes), flag true |
| `missing_key_concepts` | correct but whole key points are omitted | matched + missed |
| `keyword_stuffing` | lists relevant terms with no propositions connecting them | all missed, 0 marks |
| `copied_question` | restates or copies the question without answering | all missed, flag true |
| `off_topic` | about something else entirely (another topic, a note to the examiner) | all missed, flag true |
| `empty_or_trivial` | blank, "NA", "don't know", or only names the topic | all missed, flag false |
| `ocr_noise` | an otherwise correct answer with realistic OCR errors (rn↔m, l↔1, O↔0, dropped characters) | graded on recoverable meaning; flagged and ambiguous only when meaning is lost |
| `numeric_mismatch` | wrong numeric value (contradicted) or wrong/missing unit (partial) | |
| `extra_incorrect_statement` | correct answer plus an extra false claim that contradicts no key point | all matched, not penalised |

## Validation checks

The authoring validator (kept outside the repo) checked the following:

- both JSON files load and have `version` 1;
- questions are identical in both files;
- key-point weights sum to `max_marks`, with 2 to 6 key points per question;
- each answer has one status per key point and uses only valid statuses;
- `expected_marks` equals the sum of awards and never exceeds `max_marks`;
- the flag rules above hold (any `contradicted` means flagged, `role_swap`/`off_topic`/`copied_question`/`wrong_concept` are flagged, `ambiguous` means flagged);
- every category is present in both files and has at least 5 answers in the test file;
- answer ids are unique and there are no duplicate texts after normalising case, whitespace and punctuation;
- the keyword invariant above holds.

## Adding real teacher-marked data

Real data is far more valuable than this synthetic set. To add it:

1. Use the **same JSON format** in a new file, for example `grading_benchmark_real_<course>_<date>.json`. Set `provenance` to a description of the source: course, exam, number of teachers, and whether OCR text came from the production pipeline. Never put student names or roll numbers in the file. Use opaque ids.
2. Keep the teacher's own `key_points` and weights. Have the teacher mark each key point with the four statuses and record the total in `expected_marks`. If the teacher's real total differs from the sum of awards (holistic marking), record the teacher's total and explain the difference in `rationale`.
3. `category` is optional for real data. Use one of the categories above when it clearly applies, otherwise `"real_unclassified"` (extend the validator to allow it).
4. **Optional second teacher.** Add `"second_teacher_kp": [...]` and `"second_teacher_marks": n` to an answer to measure inter-rater agreement. Where the teachers disagree, set `ambiguous: true` and `expected_flag: true`. Treat the agreement between the two teachers as the realistic ceiling for the grader.
5. Decide the split **before** anyone looks at grader output on the data: put about 25% in a dev file and the rest in a held-out test file, then re-run the validation checks above.

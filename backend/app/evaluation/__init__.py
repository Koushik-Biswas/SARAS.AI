"""Evaluation framework (PRD 3.7, Q-2): metrics and the grading benchmark runner.

* ``app.evaluation.metrics`` -- numpy-only metric functions (marks agreement, key-point
  agreement, contradictions, deferral, calibration, bootstrap CIs, OCR CER/WER,
  segmentation accuracy).
* ``app.evaluation.grading`` -- ``python -m app.evaluation.grading --data ... --graders ...``
  runs graders over a benchmark file and writes results.jsonl, report.json, report.md.
"""

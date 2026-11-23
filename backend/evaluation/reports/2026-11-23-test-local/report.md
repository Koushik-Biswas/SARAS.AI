# SARAS AI grading evaluation

| Item | Value |
|---|---|
| Run label | held-out test, local graders (AI off), after all fixes |
| Timestamp (from caller) | 2026-11-23T21:30+05:30 |
| Benchmark | `evaluation/data/grading_benchmark_test.json` (version 1, sha256 5346b536b256) |
| Provenance | Synthetic benchmark authored by an AI agent acting as an independent teacher, labelled once with a self-review pass. NOT teacher-validated. Use for regression testing and relative comparison of grading engines, not as a claim of real-world accuracy. HELD-OUT TEST split: only the final evaluation step may open this file; do not use it to develop or tune the grader. |
| Answers evaluated | 111 of 111 (10 questions; filters: none) |
| Thresholds | mode engine (lexical: tau=0.6, tau_low=0.4, contradiction=0.5; nli-onnx: tau=0.5, tau_low=0.15, contradiction=0.95); auto-accept 0.8; calibration T 1.0 |
| Bootstrap | 1000 resamples, seed 42, 95% percentile CI, unit: answer (key points resampled with their answer) |

Values are shown as `estimate [95% CI low, high]`; n/a = undefined or not produced by the grader.

- **ci_support**: ci95_defined of ci95_resamples bootstrap resamples gave a defined value (e.g. a precision is undefined in a resample with no predicted positive); the CI uses only those (shown as 'd=NN%' when below 100%) and is n/a when fewer than 50% are defined
- **correct**: |predicted - expected| <= 0.5 marks
- **qwk_normalised**: headline QWK: marks / max marks binned to 0.1 steps (ratings 0-10, round half up), pooled across questions; a constant fraction of the maximum scores 0. It still credits differences between questions, and on questions worth more than 5 marks two adjacent half marks can share a bin
- **qwk_per_question_mean**: unweighted mean of QWK within each question on its own 0.5-mark scale, over questions with >= 5 answers whose expected marks are not all equal (gold-only rule, the same questions for every grader); n = number of such questions
- **qwk_absolute_pooled**: QWK on absolute marks binned to 0.5-mark steps, pooled across questions: scale-confounded (predicting max/2 for every answer scores above 0); 'qwk' is the same value under its old name. Quarter-mark labels round half up on the 0.5 grid
- **pearson**: pearson_normalised is on marks / max marks; 'pearson' is on absolute pooled marks (scale-confounded)
- **kp_kappa**: Cohen's unweighted kappa, expected vs predicted key-point status
- **awarded**: key point awarded = matched or partial
- **false_contradiction_rate**: share of key points not expected 'contradicted' that were predicted 'contradicted'
- **flag**: routed to needs_review by pipeline.routing (perfect OCR/segmentation assumed)
- **error_rate_unflagged**: share of incorrect answers among answers not flagged for review
- **calibration**: ECE (10 equal-width bins), Brier and AURC of the routed confidence vs correctness; risk-coverage thresholds sit at distinct confidences (tied answers are accepted together, so the order of answers does not matter)
- **partial_credit_rule**: matched = full weight, partial = half, missed = 0, contradicted = 0
- **errors**: answers whose grader raised are excluded from metrics and counted in n_errors

## Graders

| Grader | Status | Scored | Errors | Reported as | Grading time (s) |
|---|---|---|---|---|---|
| baseline-formula | available | 111 | 0 | baseline-formula/lexical ×111 | 1.06 |
| lexical | available | 111 | 0 | lexical ×111 | 11.15 |
| nli-onnx | available | 111 | 0 | nli-onnx ×111 | 232.97 |

## All answers (N = 111)

### Marks

| Grader | Scored | Errors | MAE | RMSE | Within ±0.5 | Within ±1 | QWK (normalised) | Pearson (normalised) |
|---|---|---|---|---|---|---|---|---|
| baseline-formula | 111 | 0 | 1.088 [0.899, 1.275] | 1.440 [1.215, 1.657] | 0.468 [0.369, 0.568] | 0.676 [0.586, 0.757] | 0.565 [0.435, 0.675] | 0.629 [0.500, 0.745] |
| lexical | 111 | 0 | 0.775 [0.574, 1.023] | 1.430 [1.031, 1.826] | 0.631 [0.532, 0.721] | 0.802 [0.721, 0.865] | 0.714 [0.592, 0.816] | 0.753 [0.631, 0.852] |
| nli-onnx | 111 | 0 | 0.669 [0.493, 0.874] | 1.232 [0.922, 1.546] | 0.667 [0.577, 0.748] | 0.793 [0.712, 0.856] | 0.770 [0.657, 0.863] | 0.793 [0.682, 0.880] |

Other agreement scales (absolute pooled values are scale-confounded; see method):

| Grader | QWK per question, mean (questions) | qwk_absolute_pooled (scale-confounded) | Pearson, absolute pooled (scale-confounded) |
|---|---|---|---|
| baseline-formula | 0.550 [0.360, 0.633] (10) | 0.692 [0.584, 0.774] | 0.774 [0.663, 0.853] |
| lexical | 0.721 [0.568, 0.819] (10) | 0.758 [0.616, 0.873] | 0.792 [0.669, 0.892] |
| nli-onnx | 0.762 [0.619, 0.856] (10) | 0.817 [0.710, 0.901] | 0.846 [0.748, 0.919] |

### Key points

| Grader | KPs (answers) | Not comparable | Cohen κ | Accuracy | Awarded P | Awarded R | Awarded F1 | Contradiction P | Contradiction R | False-contradiction rate |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline-formula | 0 (0) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| lexical | 408 (111) | 0 | 0.543 [0.457, 0.623] | 0.689 [0.624, 0.751] | 0.879 [0.810, 0.931] | 0.801 [0.728, 0.866] | 0.838 [0.782, 0.884] | 0.758 [0.545, 0.952] | 0.610 [0.458, 0.741] | 0.022 [0.003, 0.048] |
| nli-onnx | 408 (111) | 0 | 0.606 [0.522, 0.680] | 0.735 [0.680, 0.789] | 0.872 [0.803, 0.925] | 0.814 [0.740, 0.876] | 0.842 [0.783, 0.884] | 0.683 [0.500, 0.853] | 0.683 [0.540, 0.810] | 0.035 [0.014, 0.063] |

### Review routing and calibration

| Grader | Flag rate | Flag P vs expected | Flag R vs expected | Error rate, non-flagged | ECE | Brier | AURC |
|---|---|---|---|---|---|---|---|
| baseline-formula | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| lexical | 1.000 [1.000, 1.000] | 0.387 [0.297, 0.477] | 1.000 [1.000, 1.000] | n/a (n=0) | 0.192 [0.133, 0.277] | 0.238 [0.221, 0.256] | 0.226 [0.136, 0.324] |
| nli-onnx | 0.946 [0.901, 0.982] | 0.410 [0.315, 0.505] | 1.000 [1.000, 1.000] | 0.167 [0.000, 0.500] d=99% (n=6) | 0.102 [0.061, 0.190] | 0.210 [0.184, 0.237] | 0.208 [0.131, 0.306] |

KP confusion, **lexical** (rows expected, columns predicted):

| expected \ predicted | matched | partial | missed | contradicted |
|---|---|---|---|---|
| matched | 116 | 51 | 24 | 7 |
| partial | 2 | 12 | 14 | 0 |
| missed | 5 | 7 | 128 | 1 |
| contradicted | 7 | 6 | 3 | 25 |

KP confusion, **nli-onnx** (rows expected, columns predicted):

| expected \ predicted | matched | partial | missed | contradicted |
|---|---|---|---|---|
| matched | 135 | 34 | 17 | 12 |
| partial | 2 | 13 | 13 | 0 |
| missed | 5 | 11 | 124 | 1 |
| contradicted | 6 | 5 | 2 | 28 |

## Excluding ambiguous answers (N = 108)

### Marks

| Grader | Scored | Errors | MAE | RMSE | Within ±0.5 | Within ±1 | QWK (normalised) | Pearson (normalised) |
|---|---|---|---|---|---|---|---|---|
| baseline-formula | 108 | 0 | 1.074 [0.898, 1.245] | 1.433 [1.206, 1.634] | 0.481 [0.389, 0.583] | 0.685 [0.602, 0.769] | 0.592 [0.466, 0.701] | 0.653 [0.522, 0.769] |
| lexical | 108 | 0 | 0.748 [0.546, 0.989] | 1.400 [1.014, 1.819] | 0.639 [0.546, 0.722] | 0.806 [0.731, 0.871] | 0.731 [0.603, 0.839] | 0.764 [0.638, 0.865] |
| nli-onnx | 108 | 0 | 0.639 [0.472, 0.845] | 1.191 [0.886, 1.525] | 0.676 [0.593, 0.759] | 0.796 [0.713, 0.870] | 0.788 [0.675, 0.877] | 0.806 [0.696, 0.894] |

Other agreement scales (absolute pooled values are scale-confounded; see method):

| Grader | QWK per question, mean (questions) | qwk_absolute_pooled (scale-confounded) | Pearson, absolute pooled (scale-confounded) |
|---|---|---|---|
| baseline-formula | 0.572 [0.380, 0.663] (10) | 0.700 [0.591, 0.776] | 0.779 [0.665, 0.858] |
| lexical | 0.738 [0.596, 0.844] (10) | 0.772 [0.626, 0.880] | 0.802 [0.680, 0.899] |
| nli-onnx | 0.779 [0.641, 0.872] (10) | 0.833 [0.720, 0.912] | 0.857 [0.764, 0.927] |

### Key points

| Grader | KPs (answers) | Not comparable | Cohen κ | Accuracy | Awarded P | Awarded R | Awarded F1 | Contradiction P | Contradiction R | False-contradiction rate |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline-formula | 0 (0) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| lexical | 397 (108) | 0 | 0.554 [0.467, 0.637] | 0.698 [0.628, 0.758] | 0.881 [0.811, 0.940] | 0.812 [0.743, 0.877] | 0.845 [0.793, 0.891] | 0.750 [0.512, 0.958] | 0.615 [0.450, 0.771] | 0.022 [0.003, 0.049] |
| nli-onnx | 397 (108) | 0 | 0.619 [0.536, 0.693] | 0.746 [0.684, 0.801] | 0.874 [0.805, 0.931] | 0.826 [0.760, 0.886] | 0.849 [0.799, 0.894] | 0.675 [0.450, 0.857] | 0.692 [0.543, 0.829] | 0.036 [0.014, 0.066] |

### Review routing and calibration

| Grader | Flag rate | Flag P vs expected | Flag R vs expected | Error rate, non-flagged | ECE | Brier | AURC |
|---|---|---|---|---|---|---|---|
| baseline-formula | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| lexical | 1.000 [1.000, 1.000] | 0.370 [0.278, 0.463] | 1.000 [1.000, 1.000] | n/a (n=0) | 0.195 [0.133, 0.278] | 0.241 [0.222, 0.259] | 0.226 [0.135, 0.337] |
| nli-onnx | 0.944 [0.898, 0.981] | 0.392 [0.300, 0.485] | 1.000 [1.000, 1.000] | 0.167 [0.000, 0.500] d=99% (n=6) | 0.102 [0.066, 0.199] | 0.210 [0.182, 0.237] | 0.208 [0.123, 0.313] |

KP confusion, **lexical** (rows expected, columns predicted):

| expected \ predicted | matched | partial | missed | contradicted |
|---|---|---|---|---|
| matched | 115 | 49 | 22 | 7 |
| partial | 2 | 11 | 12 | 0 |
| missed | 5 | 7 | 127 | 1 |
| contradicted | 6 | 6 | 3 | 24 |

KP confusion, **nli-onnx** (rows expected, columns predicted):

| expected \ predicted | matched | partial | missed | contradicted |
|---|---|---|---|---|
| matched | 134 | 32 | 15 | 12 |
| partial | 2 | 12 | 11 | 0 |
| missed | 5 | 11 | 123 | 1 |
| contradicted | 5 | 5 | 2 | 27 |

## By category

| Category | N | Grader | Scored | MAE | Within ±0.5 | QWK (normalised) | KP κ | Flag rate | Error rate, non-flagged |
|---|---|---|---|---|---|---|---|---|---|
| correct_verbatim | 6 | baseline-formula | 6 | 0.417 [0.167, 0.667] | 0.833 [0.500, 1.000] | 0.000 [0.000, 0.000] | n/a | n/a | n/a |
| correct_verbatim | 6 | lexical | 6 | 0.167 [0.000, 0.500] | 0.833 [0.500, 1.000] | 0.000 [0.000, 0.000] d=66% | 0.000 [0.000, 0.000] d=66% | 1.000 [1.000, 1.000] | n/a |
| correct_verbatim | 6 | nli-onnx | 6 | 0.083 [0.000, 0.250] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] d=66% | 0.000 [0.000, 0.000] d=66% | 0.500 [0.167, 0.833] | 0.000 [0.000, 0.000] d=98% |
| correct_paraphrase | 10 | baseline-formula | 10 | 1.900 [1.250, 2.550] | 0.100 [0.000, 0.300] | 0.000 [0.000, 0.000] | n/a | n/a | n/a |
| correct_paraphrase | 10 | lexical | 10 | 1.900 [0.975, 3.175] | 0.100 [0.000, 0.300] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | n/a |
| correct_paraphrase | 10 | nli-onnx | 10 | 0.800 [0.175, 1.576] | 0.700 [0.400, 0.902] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | n/a |
| partial | 12 | baseline-formula | 12 | 0.812 [0.500, 1.146] | 0.500 [0.250, 0.750] | 0.050 [-0.233, 0.301] | n/a | n/a | n/a |
| partial | 12 | lexical | 12 | 1.146 [0.729, 1.625] | 0.417 [0.167, 0.667] | 0.121 [-0.112, 0.500] | 0.270 [0.104, 0.445] | 1.000 [1.000, 1.000] | n/a |
| partial | 12 | nli-onnx | 12 | 0.917 [0.542, 1.354] | 0.500 [0.250, 0.752] | 0.179 [-0.125, 0.627] | 0.321 [0.052, 0.544] | 1.000 [1.000, 1.000] | n/a |
| incorrect | 6 | baseline-formula | 6 | 0.833 [0.417, 1.333] | 0.500 [0.167, 0.833] | 0.000 [0.000, 0.000] | n/a | n/a | n/a |
| incorrect | 6 | lexical | 6 | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | n/a | 0.000 [0.000, 0.000] d=66% | 1.000 [1.000, 1.000] | n/a |
| incorrect | 6 | nli-onnx | 6 | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | n/a | 0.000 [0.000, 0.000] d=66% | 1.000 [1.000, 1.000] | n/a |
| contradiction_flipped_fact | 7 | baseline-formula | 7 | 0.929 [0.571, 1.357] | 0.429 [0.143, 0.857] | -0.615 [-0.791, 0.000] | n/a | n/a | n/a |
| contradiction_flipped_fact | 7 | lexical | 7 | 0.929 [0.286, 1.857] | 0.571 [0.143, 1.000] | 0.381 [-0.207, 0.838] | 0.440 [0.301, 0.622] | 1.000 [1.000, 1.000] | n/a |
| contradiction_flipped_fact | 7 | nli-onnx | 7 | 0.929 [0.357, 1.571] | 0.429 [0.000, 0.857] | 0.457 [0.000, 0.792] | 0.430 [0.295, 0.586] | 1.000 [1.000, 1.000] | n/a |
| contradiction_negation | 7 | baseline-formula | 7 | 0.571 [0.214, 1.143] | 0.857 [0.571, 1.000] | -0.105 [-0.705, 0.701] | n/a | n/a | n/a |
| contradiction_negation | 7 | lexical | 7 | 0.571 [0.250, 0.857] | 0.429 [0.000, 0.857] | 0.475 [-0.444, 0.610] | 0.352 [0.205, 0.559] | 1.000 [1.000, 1.000] | n/a |
| contradiction_negation | 7 | nli-onnx | 7 | 0.786 [0.464, 1.071] | 0.286 [0.000, 0.571] | 0.425 [-0.223, 0.573] | 0.486 [0.314, 0.655] | 1.000 [1.000, 1.000] | n/a |
| role_swap | 7 | baseline-formula | 7 | 2.286 [1.500, 3.000] | 0.143 [0.000, 0.429] | 0.022 [-0.036, 0.108] | n/a | n/a | n/a |
| role_swap | 7 | lexical | 7 | 0.679 [0.000, 1.321] | 0.571 [0.143, 1.000] | 0.414 [-0.000, 1.000] | 0.282 [0.016, 0.688] | 1.000 [1.000, 1.000] | n/a |
| role_swap | 7 | nli-onnx | 7 | 0.643 [0.000, 1.286] | 0.571 [0.143, 1.000] | 0.519 [0.000, 1.000] | 0.234 [0.024, 0.623] | 1.000 [1.000, 1.000] | n/a |
| wrong_concept | 6 | baseline-formula | 6 | 1.167 [0.833, 1.583] | 0.167 [0.000, 0.500] | 0.000 [0.000, 0.000] | n/a | n/a | n/a |
| wrong_concept | 6 | lexical | 6 | 0.083 [0.000, 0.250] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] d=66% | 0.732 [0.672, 0.734] d=66% | 1.000 [1.000, 1.000] | n/a |
| wrong_concept | 6 | nli-onnx | 6 | 0.167 [0.000, 0.333] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] d=91% | 0.569 [0.000, 0.734] d=91% | 1.000 [1.000, 1.000] | n/a |
| missing_key_concepts | 8 | baseline-formula | 8 | 0.500 [0.250, 0.750] | 0.750 [0.375, 1.000] | -0.042 [-0.706, 0.600] | n/a | n/a | n/a |
| missing_key_concepts | 8 | lexical | 8 | 0.500 [0.125, 1.125] | 0.875 [0.625, 1.000] | 0.526 [-0.111, 0.868] | 0.494 [0.199, 0.774] | 1.000 [1.000, 1.000] | n/a |
| missing_key_concepts | 8 | nli-onnx | 8 | 0.531 [0.062, 1.125] | 0.750 [0.375, 1.000] | 0.611 [0.150, 0.967] | 0.535 [0.289, 0.783] | 0.750 [0.500, 1.000] | 0.000 [0.000, 0.000] d=90% |
| keyword_stuffing | 6 | baseline-formula | 6 | 1.583 [0.667, 2.750] | 0.500 [0.167, 0.833] | 0.000 [0.000, 0.000] | n/a | n/a | n/a |
| keyword_stuffing | 6 | lexical | 6 | 1.375 [0.333, 2.625] | 0.500 [0.167, 0.833] | 0.000 [0.000, 0.000] d=98% | 0.000 [0.000, 0.000] d=98% | 1.000 [1.000, 1.000] | n/a |
| keyword_stuffing | 6 | nli-onnx | 6 | 1.375 [0.333, 2.625] | 0.500 [0.167, 0.833] | 0.000 [0.000, 0.000] d=98% | 0.000 [0.000, 0.000] d=98% | 1.000 [1.000, 1.000] | n/a |
| copied_question | 5 | baseline-formula | 5 | 1.200 [0.600, 1.900] | 0.400 [0.000, 0.800] | 0.000 [0.000, 0.000] | n/a | n/a | n/a |
| copied_question | 5 | lexical | 5 | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | n/a | n/a | 1.000 [1.000, 1.000] | n/a |
| copied_question | 5 | nli-onnx | 5 | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | n/a | n/a | 1.000 [1.000, 1.000] | n/a |
| off_topic | 5 | baseline-formula | 5 | 0.300 [0.000, 0.600] | 0.800 [0.400, 1.000] | 0.000 [0.000, 0.000] d=92% | n/a | n/a | n/a |
| off_topic | 5 | lexical | 5 | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | n/a | n/a | 1.000 [1.000, 1.000] | n/a |
| off_topic | 5 | nli-onnx | 5 | 0.100 [0.000, 0.300] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] d=67% | 0.000 [0.000, 0.000] d=67% | 1.000 [1.000, 1.000] | n/a |
| empty_or_trivial | 5 | baseline-formula | 5 | 0.100 [0.000, 0.300] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] d=65% | n/a | n/a | n/a |
| empty_or_trivial | 5 | lexical | 5 | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | n/a | n/a | 1.000 [1.000, 1.000] | n/a |
| empty_or_trivial | 5 | nli-onnx | 5 | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | n/a | n/a | 1.000 [1.000, 1.000] | n/a |
| ocr_noise | 8 | baseline-formula | 8 | 1.844 [1.219, 2.532] | 0.125 [0.000, 0.375] | 0.061 [0.000, 0.189] | n/a | n/a | n/a |
| ocr_noise | 8 | lexical | 8 | 1.781 [0.624, 3.281] | 0.375 [0.125, 0.750] | 0.083 [0.000, 0.233] | 0.033 [-0.009, 0.109] | 1.000 [1.000, 1.000] | n/a |
| ocr_noise | 8 | nli-onnx | 8 | 1.625 [0.405, 3.157] | 0.500 [0.125, 0.875] | 0.090 [0.000, 0.260] | 0.056 [0.000, 0.179] | 0.875 [0.625, 1.000] | 1.000 [1.000, 1.000] d=68% |
| numeric_mismatch | 6 | baseline-formula | 6 | 0.708 [0.333, 1.083] | 0.500 [0.167, 0.833] | -0.105 [-0.444, 0.273] d=99% | n/a | n/a | n/a |
| numeric_mismatch | 6 | lexical | 6 | 0.708 [0.125, 1.500] | 0.667 [0.333, 1.000] | 0.277 [-0.500, 0.883] | 0.345 [0.107, 0.724] | 1.000 [1.000, 1.000] | n/a |
| numeric_mismatch | 6 | nli-onnx | 6 | 0.667 [0.208, 1.208] | 0.500 [0.167, 0.833] | 0.333 [-0.218, 0.783] | 0.341 [0.063, 0.716] | 1.000 [1.000, 1.000] | n/a |
| extra_incorrect_statement | 7 | baseline-formula | 7 | 1.500 [1.000, 2.000] | 0.286 [0.000, 0.571] | 0.000 [0.000, 0.000] | n/a | n/a | n/a |
| extra_incorrect_statement | 7 | lexical | 7 | 0.821 [0.179, 1.714] | 0.571 [0.143, 0.857] | 0.000 [0.000, 0.000] d=99% | 0.000 [0.000, 0.000] d=99% | 1.000 [1.000, 1.000] | n/a |
| extra_incorrect_statement | 7 | nli-onnx | 7 | 1.036 [0.321, 1.894] | 0.429 [0.143, 0.718] | 0.000 [0.000, 0.000] d=99% | 0.000 [0.000, 0.000] d=99% | 1.000 [1.000, 1.000] | n/a |

## By question type

| Type | N | Grader | Scored | MAE | Within ±0.5 | QWK (normalised) | KP κ | Flag rate | Error rate, non-flagged |
|---|---|---|---|---|---|---|---|---|---|
| short | 41 | baseline-formula | 41 | 0.634 [0.470, 0.823] | 0.707 [0.561, 0.854] | 0.529 [0.261, 0.747] | n/a | n/a | n/a |
| short | 41 | lexical | 41 | 0.366 [0.213, 0.537] | 0.780 [0.659, 0.902] | 0.735 [0.492, 0.894] | 0.600 [0.454, 0.724] | 1.000 [1.000, 1.000] | n/a |
| short | 41 | nli-onnx | 41 | 0.293 [0.152, 0.457] | 0.854 [0.732, 0.951] | 0.771 [0.535, 0.921] | 0.668 [0.528, 0.788] | 0.902 [0.805, 0.976] | 0.250 [0.000, 0.800] d=97% |
| long | 46 | baseline-formula | 46 | 1.168 [0.924, 1.413] | 0.391 [0.261, 0.522] | 0.570 [0.375, 0.718] | n/a | n/a | n/a |
| long | 46 | lexical | 46 | 0.755 [0.478, 1.054] | 0.609 [0.457, 0.740] | 0.733 [0.540, 0.877] | 0.554 [0.419, 0.675] | 1.000 [1.000, 1.000] | n/a |
| long | 46 | nli-onnx | 46 | 0.696 [0.435, 0.957] | 0.609 [0.478, 0.761] | 0.788 [0.636, 0.905] | 0.581 [0.445, 0.705] | 0.978 [0.935, 1.000] | 0.000 [0.000, 0.000] d=64% |
| paragraph | 24 | baseline-formula | 24 | 1.708 [1.250, 2.167] | 0.208 [0.083, 0.375] | 0.629 [0.448, 0.774] | n/a | n/a | n/a |
| paragraph | 24 | lexical | 24 | 1.510 [0.823, 2.219] | 0.417 [0.208, 0.625] | 0.642 [0.354, 0.881] | 0.478 [0.324, 0.631] | 1.000 [1.000, 1.000] | n/a |
| paragraph | 24 | nli-onnx | 24 | 1.260 [0.708, 1.875] | 0.458 [0.250, 0.667] | 0.733 [0.443, 0.917] | 0.584 [0.435, 0.715] | 0.958 [0.875, 1.000] | 0.000 [0.000, 0.000] d=62% |

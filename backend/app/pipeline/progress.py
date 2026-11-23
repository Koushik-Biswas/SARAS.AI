"""Live processing progress of one answer sheet (Status page, PRD 9).

Every value here is set by a real pipeline event - a page rendered, a page read, an
AI request answered, an answer marked - never by a timer or an animation. The worker
stores a snapshot on Submission.progress (JSON) at each event it reaches on its own
thread; reading threads only update the in-memory object (under a lock) and the worker
flushes it.

Stages (the order the Status page shows them in):

1. preprocess      render, quality check, deskew, page images
2. recognition     handwriting recognition (vision model / text layer / local OCR)
3. identification  question numbers found on the sheet (or the AI split)
4. extraction      the text of every answer collected from its regions
5. rubric          rubric interpretation (key points, or the AI marking scheme)
6. analysis        semantic answer analysis (the AI marker and the offline checkers)
7. marking         question-wise mark allocation (code computes marks from verdicts)
8. verification    evidence checks and the selective second marking
9. results         totals, review routing, result generation
"""

from __future__ import annotations

import copy
import threading
import time
from datetime import datetime, timezone
from typing import Any

VERSION = 1
STAGES: tuple[tuple[str, str], ...] = (
    ("preprocess", "Document preprocessing"),
    ("recognition", "Handwriting recognition"),
    ("identification", "Question identification"),
    ("extraction", "Answer extraction"),
    ("rubric", "Rubric interpretation"),
    ("analysis", "Semantic answer analysis"),
    ("marking", "Question-wise mark allocation"),
    ("verification", "Evaluation verification"),
    ("results", "Result generation"),
)
STAGE_KEYS = tuple(k for k, _ in STAGES)
STATES = ("pending", "running", "done", "warning", "failed", "skipped")
COUNTERS = (
    "pages_total", "pages_preprocessed", "pages_blank", "pages_text_layer", "pages_read", "pages_unread",
    "questions_total", "answers_extracted", "answers_blank", "questions_marked", "uncertain",
    "verification_candidates", "reevaluated", "failed_operations",
    "ai_requests", "ai_cache_hits", "ai_failed_requests", "tokens_in", "tokens_out",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Progress:
    """Thread-safe progress record. ``snapshot()`` is what gets stored and served."""

    def __init__(self, pages_total: int = 0, questions_total: int = 0,
                 clock: Any = time.perf_counter) -> None:  # noqa: ANN401
        self._clock = clock
        self._lock = threading.Lock()
        self._t0 = clock()
        self._stage_t0: dict[str, float] = {}
        self.data: dict[str, Any] = {
            "version": VERSION,
            "started_at": _now_iso(),
            "updated_at": _now_iso(),
            "elapsed_seconds": 0.0,
            "current": None,
            "finished": False,
            "stages": [{"key": k, "label": label, "state": "pending", "seconds": None, "detail": ""}
                       for k, label in STAGES],
            "counts": {c: 0 for c in COUNTERS},
            "events": [],  # the last few notable events (failures, fallbacks), newest last
        }
        self.data["counts"]["pages_total"] = int(pages_total)
        self.data["counts"]["questions_total"] = int(questions_total)

    # ------------------------------------------------------------------ stages
    def _stage(self, key: str) -> dict:
        for st in self.data["stages"]:
            if st["key"] == key:
                return st
        raise KeyError(key)

    def start(self, key: str, detail: str | None = None) -> None:
        with self._lock:
            st = self._stage(key)
            if st["state"] != "running":
                self._stage_t0[key] = self._clock()
            st["state"] = "running"
            if detail is not None:
                st["detail"] = detail
            self.data["current"] = key

    def finish(self, key: str, state: str = "done", detail: str | None = None) -> None:
        if state not in STATES:
            raise ValueError(state)
        with self._lock:
            st = self._stage(key)
            t0 = self._stage_t0.pop(key, None)
            if t0 is not None:
                st["seconds"] = round((st["seconds"] or 0.0) + self._clock() - t0, 3)
            st["state"] = state
            if detail is not None:
                st["detail"] = detail
            if self.data["current"] == key:
                self.data["current"] = None

    def detail(self, key: str, detail: str) -> None:
        with self._lock:
            self._stage(key)["detail"] = detail

    def state(self, key: str) -> str:
        with self._lock:
            return self._stage(key)["state"]

    # ------------------------------------------------------------------ counters and events
    def bump(self, counter: str, n: int = 1) -> None:
        with self._lock:
            self.data["counts"][counter] = int(self.data["counts"].get(counter, 0)) + int(n)

    def set(self, counter: str, value: int) -> None:
        with self._lock:
            self.data["counts"][counter] = int(value)

    def get(self, counter: str) -> int:
        with self._lock:
            return int(self.data["counts"].get(counter, 0))

    def event(self, text: str, level: str = "info") -> None:
        """A short, text-free-of-student-data note ("Page 3: vision model unavailable")."""
        with self._lock:
            events = self.data["events"]
            events.append({"at": round(self._clock() - self._t0, 2), "level": level, "text": text[:160]})
            del events[:-12]

    def complete(self) -> None:
        with self._lock:
            self.data["finished"] = True
            self.data["current"] = None

    def snapshot(self) -> dict:
        with self._lock:
            self.data["elapsed_seconds"] = round(self._clock() - self._t0, 2)
            self.data["updated_at"] = _now_iso()
            return copy.deepcopy(self.data)

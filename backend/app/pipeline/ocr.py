"""Pluggable OCR engines (PRD 18: swap OCR models independently).

* TextLayerOCR — reads the PDF's own text layer (digitally written or typed
  sheets, demo data). Instant, no model.
* TrOCREngine — TrOCR handwritten checkpoint on detected text lines. Needs the
  optional ML extras (requirements-ml.txt); loaded lazily, once per process.
* "auto" picks the text layer when a page has one, otherwise TrOCR if installed,
  otherwise returns no text so the answer is flagged for manual transcription.
"""

from __future__ import annotations

import logging
import math
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from . import preprocess
from .pdf import RenderedPage

log = logging.getLogger(__name__)


@dataclass
class OcrLine:
    page_no: int
    bbox: tuple[int, int, int, int]
    text: str
    confidence: float


class OcrEngine(ABC):
    name: str = "base"

    @abstractmethod
    def read_page(self, page: RenderedPage, cleaned: np.ndarray | None) -> list[OcrLine]: ...


class TextLayerOCR(OcrEngine):
    name = "textlayer"

    def read_page(self, page: RenderedPage, cleaned: np.ndarray | None) -> list[OcrLine]:
        return [OcrLine(page.page_no, ln.bbox, ln.text, 0.99) for ln in page.text_lines]


class NullOCR(OcrEngine):
    """No recogniser available: emit detected line boxes with empty text so the
    teacher can transcribe them on the review screen."""

    name = "none"

    def read_page(self, page: RenderedPage, cleaned: np.ndarray | None) -> list[OcrLine]:
        if cleaned is None:
            return []
        return [OcrLine(page.page_no, box, "", 0.0) for box in preprocess.detect_lines(cleaned)]


class TrOCREngine(OcrEngine):
    name = "trocr"
    _lock = threading.Lock()

    def __init__(self, model_name: str, batch_size: int = 4) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._processor = None
        self._model = None

    @staticmethod
    def available() -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            return False
        return True

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            log.info("Loading %s", self.model_name)
            self._processor = TrOCRProcessor.from_pretrained(self.model_name)
            model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
            model.eval()
            try:  # int8 dynamic quantisation for the 4 GB CPU profile
                model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            except Exception:  # pragma: no cover - quantisation is best effort
                log.warning("Dynamic quantisation unavailable; running fp32")
            self._model = model

    def read_page(self, page: RenderedPage, cleaned: np.ndarray | None) -> list[OcrLine]:
        import torch
        from PIL import Image

        self._load()
        source = cleaned if cleaned is not None else page.image
        boxes = preprocess.detect_lines(source)
        out: list[OcrLine] = []
        for i in range(0, len(boxes), self.batch_size):
            batch = boxes[i:i + self.batch_size]
            crops = [Image.fromarray(page.image[y:y + h, x:x + w]).convert("RGB") for x, y, w, h in batch]
            pixel_values = self._processor(images=crops, return_tensors="pt").pixel_values
            with torch.no_grad():
                gen = self._model.generate(
                    pixel_values, max_new_tokens=64, num_beams=1,
                    output_scores=True, return_dict_in_generate=True,
                )
            texts = self._processor.batch_decode(gen.sequences, skip_special_tokens=True)
            scores = self._model.compute_transition_scores(gen.sequences, gen.scores, normalize_logits=True)
            for box, text, row in zip(batch, texts, scores):
                valid = row[torch.isfinite(row)]
                conf = float(math.exp(valid.mean().item())) if valid.numel() else 0.0
                out.append(OcrLine(page.page_no, box, text.strip(), max(0.0, min(1.0, conf))))
        return out


class AutoOCR(OcrEngine):
    name = "auto"

    def __init__(self, trocr: TrOCREngine | None) -> None:
        self.textlayer = TextLayerOCR()
        self.trocr = trocr
        self.null = NullOCR()
        self.last_used: str = "textlayer"

    def read_page(self, page: RenderedPage, cleaned: np.ndarray | None) -> list[OcrLine]:
        if page.text_lines:
            self.last_used = self.textlayer.name
            return self.textlayer.read_page(page, cleaned)
        if self.trocr is not None:
            self.last_used = self.trocr.name
            return self.trocr.read_page(page, cleaned)
        self.last_used = self.null.name
        return self.null.read_page(page, cleaned)


_engine: OcrEngine | None = None
_engine_lock = threading.Lock()


def get_ocr_engine(kind: str, trocr_model: str) -> OcrEngine:
    """Load once per worker process, not per job (PRD 18.2)."""
    global _engine
    with _engine_lock:
        if _engine is None or getattr(_engine, "_kind", None) != kind:
            trocr = TrOCREngine(trocr_model) if TrOCREngine.available() else None
            if kind == "textlayer":
                engine: OcrEngine = TextLayerOCR()
            elif kind == "trocr":
                if trocr is None:
                    raise RuntimeError("TrOCR requested but torch/transformers are not installed")
                engine = trocr
            else:
                engine = AutoOCR(trocr)
            engine._kind = kind  # type: ignore[attr-defined]
            _engine = engine
        return _engine


def ocr_capabilities() -> dict:
    return {"textlayer": True, "trocr": TrOCREngine.available()}

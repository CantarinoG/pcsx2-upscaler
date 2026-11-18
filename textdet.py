"""Decide whether a texture contains readable text, so text and everything else can go to different models.

Runs OCR (RapidOCR with PaddleOCR's PP-OCRv4 mobile detection + recognition, Latin and Chinese) and counts
the characters it can actually read. Measured on 229 hand-labelled dumps from two games:
  - "share of the image the detector marks as text" topped out at ~63% precision / 81% recall (6%);
    it confuses noise, fog, grilles and tail lights with text, and misses small signs;
  - reading the text (this module) reaches ~89% precision / 98% recall.
The rule favours recall: text sent to the material model gets garbled, while a non-text texture sent to the
text model still upscales fine.
"""
import logging
import re

import numpy as np
from PIL import Image, ImageOps

MIN_SCORE = 0.5    # recognition confidence for a piece of text to count
MIN_CHARS = 2      # readable letters/digits needed, summed over all text pieces (stacked neon signs read 1 per box)
SHORT_SIDE = 128   # dumps are tiny: enlarge so the short side is at least this (and by at least 2x)
MAX_SIDE = 1200
PAD = 0.15         # border added around the texture: text touching the edges (signs filling the texture) is
                   # otherwise missed


class TextDetector:
    def __init__(self, use_cuda: bool = True):
        from rapidocr import ModelType, OCRVersion, RapidOCR  # slow import; only needed when detecting

        logging.getLogger("RapidOCR").setLevel(logging.ERROR)
        self.ocr = RapidOCR(params={
            "Global.log_level": "error",
            "Det.ocr_version": OCRVersion.PPOCRV4, "Det.model_type": ModelType.MOBILE,
            "Rec.ocr_version": OCRVersion.PPOCRV4, "Rec.model_type": ModelType.MOBILE,
            "EngineConfig.onnxruntime.use_cuda": use_cuda,
        })

    @staticmethod
    def _prepare(img: Image.Image) -> np.ndarray:
        rgba = np.asarray(img.convert("RGBA")).copy()
        rgba[..., 3] = np.clip(rgba[..., 3].astype(np.int32) * 2, 0, 255)  # PS2 alpha: 128 = opaque
        flat = Image.alpha_composite(Image.new("RGBA", img.size, (128, 128, 128, 255)), Image.fromarray(rgba))
        w, h = img.size
        k = min(max(2.0, SHORT_SIDE / min(w, h)), MAX_SIDE / max(w, h))
        flat = flat.convert("RGB").resize((max(1, int(w * k)), max(1, int(h * k))), Image.Resampling.BICUBIC)
        flat = ImageOps.expand(flat, int(PAD * max(flat.size)), fill=(128, 128, 128))
        return np.ascontiguousarray(np.asarray(flat)[..., ::-1])  # BGR, as OpenCV/PaddleOCR expect

    def read(self, img: Image.Image) -> list[tuple[str, float]]:
        r = self.ocr(self._prepare(img), use_cls=False)
        return list(zip(r.txts or (), (float(s) for s in (r.scores or ()))))

    def has_text(self, img: Image.Image) -> bool:
        chars = sum(len(re.findall(r"\w", t)) for t, s in self.read(img) if s >= MIN_SCORE)
        return chars >= MIN_CHARS

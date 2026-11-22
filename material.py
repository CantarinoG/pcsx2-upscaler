"""Add photographed material detail to upscaled surface textures (roads, walls, grass, rock, ...).

An optional ONNX classifier (models/material.onnx, see README) says whether a texture is a surface and of which
family. A surface gets the fine luminance detail of that family's photo (materials/) laid over its upscale. Colour,
shading and layout stay the texture's own, and the detail backs off on strong edges such as joints and painted lines.
"""
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

import dds

PHOTOS_DIR = Path(__file__).resolve().parent / "materials"
# In the classifier's output order, after "not a surface": family -> (photo, fine log-luminance contrast added)
FAMILIES = {
    "grain": ("concrete.jpg", 0.119),    # asphalt, concrete, plaster, dirt
    "stone": ("rock.jpg", 0.168),        # rock and bark
    "masonry": ("concrete.jpg", 0.112),  # bricks and blocks: only grain, their layout is the texture's
    "grass": ("grass.jpg", 0.21),        # grass and moss
    "wood": ("wood.jpg", 0.14),
    "metal": ("metal.jpg", 0.098),
}
DIRECTIONAL = ("wood", "metal")  # the photo is turned to run the way the texture does
SURFACE_MIN = 0.5                # probability of being a surface needed to add detail
TARGET_SIDE = 1024               # photo grain needs room: surfaces are enlarged towards this short side,
MIN_SCALE, MAX_SCALE = 4, 8      # within these multiples of the dump
DETAIL_SIZE = 1.2                # detail finer than this many dump texels comes from the photo
LUMA = np.array([0.299, 0.587, 0.114], np.float32)


def output_scale(w: int, h: int) -> int:
    """How many times the dump's size a surface with added detail is saved at."""
    return int(np.clip(TARGET_SIDE // min(w, h), MIN_SCALE, MAX_SCALE))


def eligible(dump: Image.Image) -> bool:
    """Whether the classifier is asked at all: not tiny, huge, a thin strip, flat, black or a cut-out."""
    w, h = dump.size
    if min(w, h) < 32 or max(w, h) > 512 or max(w, h) > 4 * min(w, h):
        return False
    rgba = np.asarray(dump.convert("RGBA"), np.float32)
    y = rgba[..., :3] @ LUMA
    return bool(y.std() >= 2 and y.mean() >= 8 and not dds.is_cutout(rgba[..., 3]))


class MaterialClassifier:
    """Input: N x 3 x S x S RGB in 0..1, N views of one texture. Output: 1 x 7 probabilities, "not a surface"
    followed by FAMILIES."""

    def __init__(self, path: Path):
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0]
        side = self.input.shape[-1]
        self.side = side if isinstance(side, int) else 224
        out = self._run(np.zeros((2, 3, self.side, self.side), np.float32))
        if out.shape != (1, len(FAMILIES) + 1):
            raise ValueError(f"{path.name} outputs {out.shape}, expected (1, {len(FAMILIES) + 1}); see README")

    def _run(self, views: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {self.input.name: views})[0]

    def family(self, dump: Image.Image) -> str | None:
        """The surface's family, or None to leave the texture alone."""
        if not eligible(dump):
            return None
        rgb = dump.convert("RGB")
        w, h = rgb.size
        repeated = Image.new("RGB", (w * 2, h * 2))  # second view: shows how it tiles
        for x in (0, w):
            for y in (0, h):
                repeated.paste(rgb, (x, y))
        size = (self.side, self.side)
        views = np.stack([np.asarray(v.resize(size, Image.Resampling.BICUBIC), np.float32) / 255
                          for v in (rgb, repeated)])
        prob = self._run(views.transpose(0, 3, 1, 2))[0]
        if 1 - prob[0] < SURFACE_MIN:
            return None
        return list(FAMILIES)[int(prob[1:].argmax())]


def _pad(a: np.ndarray, n: int, modes: tuple[str, str]) -> np.ndarray:
    return np.pad(np.pad(a, ((n, n), (0, 0)), mode=modes[0]), ((0, 0), (n, n)), mode=modes[1])


def _blur(a: np.ndarray, sigma: float, modes: tuple[str, str] = ("wrap", "wrap")) -> np.ndarray:
    n = int(3 * sigma) + 1
    return cv2.GaussianBlur(_pad(a, n, modes), (0, 0), sigma)[n:-n, n:-n]


def _resize_tiling(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Resize a tiling image without breaking the tiling: the filter sees the opposite edge, not a border."""
    w, h = img.size
    px, py = w // 8, h // 8
    around = Image.fromarray(np.pad(np.asarray(img), ((py, py), (px, px)), mode="wrap"))
    return around.resize(size, Image.Resampling.LANCZOS, box=(px, py, px + w, py + h))


def _runs_vertically(a: np.ndarray) -> bool:
    small = cv2.resize(a, (128, 128), interpolation=cv2.INTER_AREA)
    return bool(np.abs(np.diff(small, axis=1)).mean() > np.abs(np.diff(small, axis=0)).mean())


class MaterialDetail:
    def __init__(self):
        self.photos = {}
        for name in {name for name, _ in FAMILIES.values()}:
            with Image.open(PHOTOS_DIR / name) as im:
                self.photos[name] = im.convert("L")

    def _photo(self, family: str, y: np.ndarray) -> np.ndarray:
        """The family's photo covering y's size, repeated a whole number of times so tiling textures still tile."""
        photo = self.photos[FAMILIES[family][0]]
        if family in DIRECTIONAL and _runs_vertically(y) != _runs_vertically(np.asarray(photo, np.float32)):
            photo = photo.transpose(Image.Transpose.ROTATE_90)
        h, w = y.shape
        nx, ny = max(1, round(w / min(w, h))), max(1, round(h / min(w, h)))
        tile = _resize_tiling(photo, (round(w / nx), round(h / ny)))
        covered = Image.fromarray(np.tile(np.asarray(tile), (ny, nx)))
        if covered.size != (w, h):
            covered = covered.resize((w, h), Image.Resampling.BICUBIC)
        return np.asarray(covered, np.float32)

    def apply(self, up: Image.Image, dump_size: tuple[int, int], family: str, modes: tuple[str, str]) -> Image.Image:
        """up: the upscale (4x, or 16x from the second pass). modes: the texture's edge padding, see edge_modes."""
        sw, sh = dump_size
        scale = max(up.width // sw, output_scale(sw, sh))
        size = (sw * scale, sh * scale)
        rgb = np.asarray(up.convert("RGB").resize(size, Image.Resampling.BICUBIC), np.float32)
        y = rgb @ LUMA

        photo = np.log(self._photo(family, y) + 8)
        detail = photo - _blur(photo, scale * DETAIL_SIZE)
        detail = np.clip(detail / (detail.std() + 1e-5), -3, 3)

        # back off on joints, lines and markings
        soft = _pad(_blur(y, scale / 2, modes), 1, modes)
        edge = np.hypot(cv2.Scharr(soft, -1, 1, 0), cv2.Scharr(soft, -1, 0, 1))[1:-1, 1:-1] / 32 * scale
        keep = _blur(np.clip(1.4 - edge / 60, 0.3, 1), scale / 2, modes)

        out = rgb * np.exp(detail * FAMILIES[family][1] * keep)[..., None]
        out = np.clip(out + 0.5, 0, 255).astype(np.uint8)
        if "A" not in up.getbands():
            return Image.fromarray(out, "RGB")
        # alpha is resized by itself (Pillow would weight colour by it) and, being PS2 alpha, kept in its range
        alpha = np.asarray(up.getchannel("A"))
        resized = np.asarray(Image.fromarray(alpha).resize(size, Image.Resampling.BICUBIC))
        return Image.fromarray(np.dstack([out, np.clip(resized, alpha.min(), alpha.max())]), "RGBA")

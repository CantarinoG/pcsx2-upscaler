"""Add photographed material detail to upscaled surface textures (roads, walls, grass, rock, ...).

An optional ONNX classifier (models/material.onnx, see README) says whether a texture is a surface and of which
family. A surface gets the fine luminance detail of one of that family's photos (materials/<family>/) laid over its
upscale. Which photo, and where it starts, follows from the texture's own hash: textures differ from each other, but
each gets the same result on every run. Colour, shading and layout stay the texture's own, and the detail backs off
on strong edges such as joints and painted lines.

Cut-outs (alpha with holes) have a classifier of their own (models/cutout.onnx), as their outline misleads one made
for whole textures. They only get detail inside their solid parts: holes and outline stay as they are.
"""
import hashlib
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

import dds

PHOTOS_DIR = Path(__file__).resolve().parent / "materials"
# In the classifier's output order, after "not a surface": family -> fine log-luminance contrast added
FAMILIES = {
    "asphalt": 0.119,
    "concrete": 0.119,  # and plaster, pavement
    "dirt": 0.119,      # soil, sand, gravel
    "stone": 0.168,     # rock and bark
    "masonry": 0.112,   # bricks and blocks: only grain, their layout is the texture's
    "grass": 0.21,      # grass and moss
    "wood": 0.14,
    "metal": 0.098,
}
DIRECTIONAL = ("wood", "metal")  # the photo is turned to run the way the texture does
SURFACE_MIN = 0.5                # probability of being a surface needed to add detail
TARGET_SIDE = 1024               # photo grain needs room: surfaces are enlarged towards this short side,
MIN_SCALE, MAX_SCALE = 4, 8      # within these multiples of the dump
DETAIL_SIZE = 1.2                # detail finer than this many dump texels comes from the photo
CUTOUT_MIN_CORE = 0.1            # cut-outs: share of the texture that must be solid, away from the outline
CUTOUT_MIN_HARD = 0.5            # share of the visible part that must be opaque (no smoke, glows, soft shadows)
CUTOUT_GRAIN_ALPHA = 110         # grain goes where alpha is at least this: solid parts are often a bit below opaque
CUTOUT_PATCHES = 3               # views from inside the solid part, at most
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


def texture_key(name: str) -> str:
    """The part of a dump's name that identifies its pixels, without palette and format: palette variants of a
    texture must get identical grain, or it would flicker when a game switches between them."""
    return name.split("-", 1)[0]


def _seed(*parts: str) -> int:
    return int.from_bytes(hashlib.sha1("/".join(parts).encode()).digest()[:8], "little")


def _core(solid: np.ndarray) -> np.ndarray:
    """The solid part without its outline, two texels in."""
    return cv2.erode(solid, np.ones((5, 5), np.uint8), borderType=cv2.BORDER_REPLICATE)


def cutout_core(dump: Image.Image) -> np.ndarray | None:
    """A cut-out's solid part if the cut-out classifier is to be asked about it, else None: not a cut-out, a bad
    size, mostly holes, soft, or flat."""
    w, h = dump.size
    if min(w, h) < 32 or max(w, h) > 512 or max(w, h) > 4 * min(w, h):
        return None
    rgba = np.asarray(dump.convert("RGBA"))
    alpha = rgba[..., 3]
    if not dds.is_cutout(alpha):
        return None
    solid = (alpha >= dds.CUTOUT_OPAQUE).astype(np.uint8)
    core = _core(solid)
    if core.mean() < CUTOUT_MIN_CORE or solid.sum() < CUTOUT_MIN_HARD * (alpha > dds.CUTOUT_CLEAR).sum():
        return None
    y = rgba[..., :3].astype(np.float32) @ LUMA
    return core if y[core > 0].std() >= 2 else None


def grain_mask(dump: Image.Image) -> np.ndarray | None:
    """Where a cut-out may get grain, None for other textures."""
    alpha = np.asarray(dump.convert("RGBA"))[..., 3]
    return (alpha >= CUTOUT_GRAIN_ALPHA).astype(np.uint8) if dds.is_cutout(alpha) else None


def cutout_views(dump: Image.Image, core: np.ndarray) -> list[Image.Image]:
    """What the cut-out classifier sees, outline removed in different ways: holes filled with spread colour, with
    the nearest solid texel and with the mean colour (each also repeated 2x2), then squares from inside."""
    rgb = np.asarray(dump.convert("RGB"))
    inside = core > 0

    weight = core.astype(np.float32)
    colour = rgb.astype(np.float32) * weight[..., None]
    spread, done = colour.copy(), inside.copy()
    for sigma in (1, 2, 4, 8, 16, 32, 64, 128):
        share = cv2.GaussianBlur(weight, (0, 0), sigma)
        blurred = cv2.GaussianBlur(colour, (0, 0), sigma)
        new = ~done & (share > 0.01)
        spread[new] = blurred[new] / share[new][:, None]
        done |= new

    labels = cv2.distanceTransformWithLabels((~inside).astype(np.uint8), cv2.DIST_L2, 5,
                                             labelType=cv2.DIST_LABEL_PIXEL)[1]
    colours = np.zeros((labels.max() + 1, 3), np.uint8)
    colours[labels[inside]] = rgb[inside]

    flat = rgb.copy()
    flat[~inside] = rgb[inside].mean(0).astype(np.uint8)

    views = []
    for filled in (np.clip(spread + 0.5, 0, 255).astype(np.uint8), colours[labels], flat):
        views += [Image.fromarray(filled), Image.fromarray(np.tile(filled, (2, 2, 1)))]

    free, opaque = core.copy(), Image.fromarray(rgb)
    for i in range(CUTOUT_PATCHES):  # each the largest square left
        room = cv2.distanceTransform(np.pad(free, 1), cv2.DIST_C, 3)[1:-1, 1:-1]
        y, x = np.unravel_index(room.argmax(), room.shape)
        r = int(room[y, x]) - 1
        if r < 4 and i:
            break
        r = max(r, 1)
        views.append(opaque.crop((x - r, y - r, x + r + 1, y + r + 1)))
        free[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = 0
    return views


class MaterialClassifier:
    """Input: N x 3 x S x S RGB in 0..1, N views of one texture. Output: 1 x (1 + len(FAMILIES)) probabilities,
    "not a surface" followed by FAMILIES."""

    PROBE_VIEWS = 2  # how many views the model is tried with on loading

    def __init__(self, path: Path):
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0]
        side = self.input.shape[-1]
        self.side = side if isinstance(side, int) else 224
        out = self._run(np.zeros((self.PROBE_VIEWS, 3, self.side, self.side), np.float32))
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
        return self._decide([rgb, repeated])

    def _decide(self, images: list[Image.Image]) -> str | None:
        size = (self.side, self.side)
        views = np.stack([np.asarray(v.resize(size, Image.Resampling.BICUBIC), np.float32) / 255 for v in images])
        prob = self._run(views.transpose(0, 3, 1, 2))[0]
        if 1 - prob[0] < SURFACE_MIN:
            return None
        return list(FAMILIES)[int(prob[1:].argmax())]


class CutoutClassifier(MaterialClassifier):
    """Like MaterialClassifier, for cut-outs. Input: the 7 to 9 views of cutout_views, in that order."""

    PROBE_VIEWS = 7

    def family(self, dump: Image.Image) -> str | None:
        core = cutout_core(dump)
        return None if core is None else self._decide(cutout_views(dump, core))


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


def _runs_vertically(a: np.ndarray, side: int = 128) -> bool:
    small = cv2.resize(a, (side, side), interpolation=cv2.INTER_AREA)
    return bool(np.abs(np.diff(small, axis=1)).mean() > np.abs(np.diff(small, axis=0)).mean())


class MaterialDetail:
    def __init__(self):
        self.photos: dict[str, dict[str, Image.Image]] = {}
        self.vertical: dict[tuple[str, str], bool] = {}  # which way each directional photo runs
        for family in FAMILIES:
            self.photos[family] = {}
            for path in sorted((PHOTOS_DIR / family).glob("*.jpg")):
                with Image.open(path) as im:
                    photo = self.photos[family][path.stem] = im.convert("L")
                if family in DIRECTIONAL:  # judged at 512: fine grain vanishes at 128
                    self.vertical[family, path.stem] = _runs_vertically(np.asarray(photo, np.float32), 512)
            if not self.photos[family]:
                raise FileNotFoundError(f"no photos in {PHOTOS_DIR / family}")

    def choose(self, family: str, key: str) -> str:
        """The photo for a texture. Each photo bids with a hash, so adding one only moves textures onto that one."""
        return max(self.photos[family], key=lambda name: _seed(key, name))

    def _photo(self, family: str, y: np.ndarray, key: str) -> np.ndarray:
        """The texture's photo covering y's size, repeated a whole number of times and shifted round the edges, so
        tiling textures still tile."""
        name = self.choose(family, key)
        photo = self.photos[family][name]
        if family in DIRECTIONAL and _runs_vertically(y) != self.vertical[family, name]:
            photo = photo.transpose(Image.Transpose.ROTATE_90)
        h, w = y.shape
        nx, ny = max(1, round(w / min(w, h))), max(1, round(h / min(w, h)))
        tile = _resize_tiling(photo, (round(w / nx), round(h / ny)))
        shift = _seed(key, "offset")
        rolled = np.roll(np.asarray(tile), (shift % tile.height, (shift >> 32) % tile.width), axis=(0, 1))
        covered = Image.fromarray(np.tile(rolled, (ny, nx)))
        if covered.size != (w, h):
            covered = covered.resize((w, h), Image.Resampling.BICUBIC)
        return np.asarray(covered, np.float32)

    def apply(self, up: Image.Image, dump_size: tuple[int, int], family: str, modes: tuple[str, str],
              key: str, mask: np.ndarray | None = None) -> Image.Image:
        """up: the upscale (4x, or 16x from the second pass). modes: the texture's edge padding, see edge_modes.
        key: see texture_key. mask: for cut-outs, see grain_mask."""
        sw, sh = dump_size
        scale = max(up.width // sw, output_scale(sw, sh))
        size = (sw * scale, sh * scale)
        rgb = np.asarray(up.convert("RGB").resize(size, Image.Resampling.BICUBIC), np.float32)
        y = rgb @ LUMA

        photo = np.log(self._photo(family, y, key) + 8)
        detail = photo - _blur(photo, scale * DETAIL_SIZE)
        detail = np.clip(detail / (detail.std() + 1e-5), -3, 3)

        # back off on joints, lines and markings
        soft = _pad(_blur(y, scale / 2, modes), 1, modes)
        edge = np.hypot(cv2.Scharr(soft, -1, 1, 0), cv2.Scharr(soft, -1, 0, 1))[1:-1, 1:-1] / 32 * scale
        keep = _blur(np.clip(1.4 - edge / 60, 0.3, 1), scale / 2, modes)

        out = rgb * np.exp(detail * FAMILIES[family] * keep)[..., None]
        out = np.clip(out + 0.5, 0, 255).astype(np.uint8)
        if mask is not None:  # fades out before the outline and never reaches past the solid part
            inner = cv2.resize(_core(mask).astype(np.float32), size)
            inner = cv2.GaussianBlur(inner, (0, 0), scale * 0.6, borderType=cv2.BORDER_REPLICATE)
            weight = (inner * (cv2.resize(mask.astype(np.float32), size) >= 0.999))[..., None]
            out = np.clip(rgb * (1 - weight) + out * weight + 0.5, 0, 255).astype(np.uint8)
        if "A" not in up.getbands():
            return Image.fromarray(out, "RGB")
        # alpha is resized by itself (Pillow would weight colour by it) and, being PS2 alpha, kept in its range
        alpha = np.asarray(up.getchannel("A"))
        resized = np.asarray(Image.fromarray(alpha).resize(size, Image.Resampling.BICUBIC))
        return Image.fromarray(np.dstack([out, np.clip(resized, alpha.min(), alpha.max())]), "RGBA")
